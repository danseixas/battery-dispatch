"""Two-market battery-dispatch simulation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from battery_dispatch.battery import (
    BatteryConfig,
    BatteryMode,
    BatteryState,
    DispatchResult,
    dispatch,
)
from battery_dispatch.data import HALF_HOURLY_INTERVAL
from battery_dispatch.economics import EconomicsAssumptions, analyse_dispatch
from battery_dispatch.strategy import PercentileStrategy


@dataclass(frozen=True)
class SimulationSummary:
    """Headline results for a one- or two-market dispatch simulation."""

    charge_threshold_gbp_per_mwh: float | None
    discharge_threshold_gbp_per_mwh: float | None
    import_cost_gbp: float
    export_revenue_gbp: float
    net_cashflow_gbp: float
    ending_energy_mwh: float
    analysis_period_years: float
    equivalent_cycles: float
    fixed_operating_cost_gbp: float
    operating_cashflow_gbp: float
    projected_capacity_mwh: float
    title: str = "Heuristic simulation"

    def to_text(self) -> str:
        """Format a concise summary for the command line."""
        return "\n".join(
            (
                self.title,
                "Charge threshold: "
                + _format_threshold(self.charge_threshold_gbp_per_mwh),
                "Discharge threshold: "
                + _format_threshold(self.discharge_threshold_gbp_per_mwh),
                f"Import cost: {self.import_cost_gbp:.2f} GBP",
                f"Export revenue: {self.export_revenue_gbp:.2f} GBP",
                f"Net cashflow: {self.net_cashflow_gbp:.2f} GBP",
                f"Equivalent cycles: {self.equivalent_cycles:.3f}",
                f"Fixed operating cost: {self.fixed_operating_cost_gbp:.2f} GBP",
                f"Operating cashflow: {self.operating_cashflow_gbp:.2f} GBP",
                f"Projected capacity: {self.projected_capacity_mwh:.3f} MWh",
                f"Ending stored energy: {self.ending_energy_mwh:.3f} MWh",
            )
        )


def _format_threshold(value: float | None) -> str:
    """Format a heuristic threshold when a run has one."""
    return "not used" if value is None else f"{value:.2f} GBP/MWh"


def _summarise_dispatches(
    dispatches: pd.DataFrame,
    *,
    battery: BatteryConfig,
    economics: EconomicsAssumptions,
    low_threshold: float | None,
    high_threshold: float | None,
    ending_state: BatteryState,
    title: str = "Heuristic simulation",
) -> tuple[pd.DataFrame, SimulationSummary]:
    """Calculate common financial and operating metrics from dispatch records."""
    import_cost = -float(dispatches.loc[dispatches["cashflow_gbp"] < 0, "cashflow_gbp"].sum())
    export_revenue = float(dispatches.loc[dispatches["cashflow_gbp"] > 0, "cashflow_gbp"].sum())
    economic_results = analyse_dispatch(
        dispatches,
        battery=battery,
        assumptions=economics,
        interval_hours=HALF_HOURLY_INTERVAL.total_seconds() / 3600,
    )
    summary = SimulationSummary(
        charge_threshold_gbp_per_mwh=low_threshold,
        discharge_threshold_gbp_per_mwh=high_threshold,
        import_cost_gbp=import_cost,
        export_revenue_gbp=export_revenue,
        net_cashflow_gbp=float(dispatches["cashflow_gbp"].sum()),
        ending_energy_mwh=ending_state.energy_mwh,
        analysis_period_years=economic_results.analysis_period_years,
        equivalent_cycles=economic_results.equivalent_cycles,
        fixed_operating_cost_gbp=economic_results.fixed_operating_cost_gbp,
        operating_cashflow_gbp=economic_results.operating_cashflow_gbp,
        projected_capacity_mwh=economic_results.projected_capacity_mwh,
        title=title,
    )
    return dispatches, summary


def simulate_two_markets(
    market_1_prices: pd.DataFrame,
    market_2_prices: pd.DataFrame,
    *,
    battery: BatteryConfig,
    strategy: PercentileStrategy,
    market_2_power_share: float = 0.5,
    initial_state: BatteryState | None = None,
    economics: EconomicsAssumptions | None = None,
) -> tuple[pd.DataFrame, SimulationSummary]:
    """Run both markets simultaneously with distributed power per half-hour.

    When both markets signal the same direction, Market 2 receives the
    configured share and Market 1 receives the remainder.
    Opposing signals are resolved using state-of-charge feasibility and distance
    from the strategy thresholds.
    This is a transparent allocation rule, not a profit optimiser.
    """
    if not 0 < market_2_power_share < 1:
        raise ValueError("The Market 2 power share must be between zero and one.")
    combined = _combine_market_prices(market_1_prices, market_2_prices)
    threshold_prices = pd.concat(
        [
            combined["price_gbp_per_mwh"],
            combined["market_2_price_gbp_per_mwh"],
        ],
        ignore_index=True,
    )
    low_threshold, high_threshold = strategy.thresholds(threshold_prices)
    state = initial_state or BatteryState(0.0)
    state.validate(battery)
    records: list[dict[str, object]] = []

    for _, hour in combined.groupby("hour_start", sort=True):
        market_2_price = float(hour["market_2_price_gbp_per_mwh"].iloc[0])
        market_2_mode = strategy.decide(
            market_2_price,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        for row in hour.itertuples(index=False):
            market_1_price = float(row.price_gbp_per_mwh)
            market_1_mode = strategy.decide(
                market_1_price,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
            )
            mode, market_1_power_mw, market_2_power = _allocate_interval_power(
                market_1_mode,
                market_2_mode,
                market_1_price=market_1_price,
                market_2_price=market_2_price,
                battery=battery,
                market_2_power_share=market_2_power_share,
                state=state,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
            )
            total_requested_power = market_1_power_mw + market_2_power
            result = dispatch(
                battery,
                state,
                mode=mode,
                requested_power_mw=total_requested_power,
                duration_hours=HALF_HOURLY_INTERVAL.total_seconds() / 3600,
            )
            records.append(
                _build_dispatch_record(
                    row.timestamp_utc,
                    market_1_price=market_1_price,
                    market_2_price=market_2_price,
                    mode=mode,
                    market_1_power_mw=market_1_power_mw,
                    market_2_power_mw=market_2_power,
                    result=result,
                )
            )
            state = result.state_after

    dispatches = pd.DataFrame.from_records(records)
    return _summarise_dispatches(
        dispatches,
        battery=battery,
        economics=economics or EconomicsAssumptions(),
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        ending_state=state,
    )


def _combine_market_prices(
    market_1_prices: pd.DataFrame,
    market_2_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Align one hourly Market 2 price with its two Market 1 intervals."""
    required_columns = {"timestamp_utc", "price_gbp_per_mwh"}
    for market_name, prices in (("Market 1", market_1_prices), ("Market 2", market_2_prices)):
        missing_columns = required_columns.difference(prices.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{market_name} prices are missing required columns: {missing}.")
        if prices.empty:
            raise ValueError(f"{market_name} prices cannot be empty.")

    market_1 = market_1_prices.copy()
    market_1["hour_start"] = market_1["timestamp_utc"].dt.floor("h")
    market_2 = market_2_prices.rename(
        columns={"price_gbp_per_mwh": "market_2_price_gbp_per_mwh"}
    )[["timestamp_utc", "market_2_price_gbp_per_mwh"]]
    combined = market_1.merge(
        market_2,
        left_on="hour_start",
        right_on="timestamp_utc",
        how="left",
        validate="many_to_one",
        suffixes=("", "_market_2"),
    ).sort_values("timestamp_utc").reset_index(drop=True)
    if combined["market_2_price_gbp_per_mwh"].isna().any():
        raise ValueError("Market 2 does not cover every requested Market 1 hour.")
    if not combined.groupby("hour_start").size().eq(2).all():
        raise ValueError("Market 1 must contain exactly two intervals for every Market 2 hour.")
    return combined


def _allocate_interval_power(
    market_1_mode: BatteryMode,
    market_2_mode: BatteryMode,
    *,
    market_1_price: float,
    market_2_price: float,
    battery: BatteryConfig,
    market_2_power_share: float,
    state: BatteryState,
    low_threshold: float,
    high_threshold: float,
) -> tuple[BatteryMode, float, float]:
    """Allocate power independently for one Market 1 half-hour interval."""
    if market_2_mode is BatteryMode.IDLE:
        if market_1_mode is BatteryMode.CHARGE:
            return market_1_mode, battery.max_charge_mw, 0.0
        if market_1_mode is BatteryMode.DISCHARGE:
            return market_1_mode, battery.max_discharge_mw, 0.0
        return BatteryMode.IDLE, 0.0, 0.0

    if market_1_mode is market_2_mode:
        market_2_power = (
            battery.max_charge_mw if market_2_mode is BatteryMode.CHARGE else battery.max_discharge_mw
        ) * market_2_power_share
        market_1_power = (
            battery.max_charge_mw if market_2_mode is BatteryMode.CHARGE else battery.max_discharge_mw
        ) - market_2_power
        return market_2_mode, market_1_power, market_2_power

    if market_1_mode is BatteryMode.IDLE:
        market_2_power = (
            battery.max_charge_mw
            if market_2_mode is BatteryMode.CHARGE
            else battery.max_discharge_mw
        )
        return market_2_mode, 0.0, market_2_power

    can_charge = state.energy_mwh < battery.capacity_mwh
    can_discharge = state.energy_mwh > 0.0
    if not can_charge:
        return BatteryMode.DISCHARGE, (
            battery.max_discharge_mw if market_1_mode is BatteryMode.DISCHARGE else 0.0
        ), (
            battery.max_discharge_mw if market_2_mode is BatteryMode.DISCHARGE else 0.0
        )
    if not can_discharge:
        return BatteryMode.CHARGE, (
            battery.max_charge_mw if market_1_mode is BatteryMode.CHARGE else 0.0
        ), (
            battery.max_charge_mw if market_2_mode is BatteryMode.CHARGE else 0.0
        )

    if market_1_mode is BatteryMode.CHARGE:
        charge_price = market_1_price
        discharge_price = market_2_price
        charge_is_market_1 = True
    else:
        charge_price = market_2_price
        discharge_price = market_1_price
        charge_is_market_1 = False
    charge_distance = low_threshold - charge_price
    discharge_distance = discharge_price - high_threshold

    if discharge_distance >= charge_distance:
        return BatteryMode.DISCHARGE, (
            battery.max_discharge_mw if not charge_is_market_1 else 0.0
        ), (
            battery.max_discharge_mw if charge_is_market_1 else 0.0
        )
    return BatteryMode.CHARGE, (
        battery.max_charge_mw if charge_is_market_1 else 0.0
    ), (
        battery.max_charge_mw if not charge_is_market_1 else 0.0
    )


def _build_dispatch_record(
    timestamp: pd.Timestamp,
    *,
    market_1_price: float,
    market_2_price: float,
    mode: BatteryMode,
    market_1_power_mw: float,
    market_2_power_mw: float,
    result: DispatchResult,
) -> dict[str, object]:
    """Represent both market allocations in one half-hourly output row."""
    requested_power = market_1_power_mw + market_2_power_mw
    accepted_ratio = result.accepted_power_mw / requested_power if requested_power else 0.0
    interval_hours = HALF_HOURLY_INTERVAL.total_seconds() / 3600
    market_1_grid_energy = market_1_power_mw * interval_hours * accepted_ratio
    market_2_grid_energy = market_2_power_mw * interval_hours * accepted_ratio
    sign = -1 if mode is BatteryMode.CHARGE else 1 if mode is BatteryMode.DISCHARGE else 0
    return {
        "timestamp_utc": timestamp,
        "mode": mode.value,
        "market_1_price_gbp_per_mwh": market_1_price,
        "market_2_price_gbp_per_mwh": market_2_price,
        "market_1_power_mw": market_1_power_mw * accepted_ratio,
        "market_2_power_mw": market_2_power_mw * accepted_ratio,
        "accepted_power_mw": result.accepted_power_mw,
        "grid_energy_mwh": result.grid_energy_mwh,
        "stored_energy_change_mwh": result.stored_energy_change_mwh,
        "energy_start_mwh": result.state_before.energy_mwh,
        "energy_end_mwh": result.state_after.energy_mwh,
        "market_1_cashflow_gbp": sign * market_1_grid_energy * market_1_price,
        "market_2_cashflow_gbp": sign * market_2_grid_energy * market_2_price,
        "cashflow_gbp": sign
        * (market_1_grid_energy * market_1_price + market_2_grid_energy * market_2_price),
    }
