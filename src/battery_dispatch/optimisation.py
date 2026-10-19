"""Perfect-foresight mixed-integer optimisation for the two-market model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from battery_dispatch.battery import BatteryConfig, BatteryMode, BatteryState
from battery_dispatch.economics import EconomicsAssumptions
from battery_dispatch.simulation import SimulationSummary, _combine_market_prices, _summarise_dispatches


@dataclass(frozen=True)
class OptimisationResult:
    """Dispatch results plus concise solver diagnostics."""

    dispatches: pd.DataFrame
    summary: SimulationSummary
    solver_message: str
    objective_cashflow_gbp: float


def optimise_two_markets(
    market_1_prices: pd.DataFrame,
    market_2_prices: pd.DataFrame,
    *,
    battery: BatteryConfig,
    initial_state: BatteryState | None = None,
    economics: EconomicsAssumptions | None = None,
    time_limit_seconds: float | None = None,
) -> OptimisationResult:
    """Maximise historic cashflow with one physical battery across both markets.

    The model has complete knowledge of the selected historic price window. It
    is therefore an upper-bound backtest, not a deployable trading rule. Market
    2 power is a single hourly commitment applied to both half-hours. Binary
    variables prevent charging and discharging in the same half-hour.
    """
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError as error:  # pragma: no cover - exercised by installation
        raise RuntimeError(
            "Two-market optimisation requires scipy. Install the project dependencies "
            "with `python -m pip install --editable '.[dev]'`."
        ) from error

    combined = _combine_market_prices(market_1_prices, market_2_prices)
    period_count = len(combined)
    if period_count % 2:
        raise ValueError("Optimisation requires complete Market 2 hours.")
    prices_1 = combined["price_gbp_per_mwh"].to_numpy(dtype=float)
    prices_2 = (
        combined.groupby("hour_start", sort=True)["market_2_price_gbp_per_mwh"]
        .first()
        .to_numpy(dtype=float)
    )
    if not np.isfinite(prices_1).all() or not np.isfinite(prices_2).all():
        raise ValueError("Market prices must be finite.")

    state = initial_state or BatteryState(0.0)
    state.validate(battery)
    hour_count = len(prices_2)
    interval_hours = 0.5

    # Blocks: Market 1 charge/discharge, Market 2 hourly charge/discharge,
    # state of charge, charge binary and discharge binary.
    c1 = 0
    d1 = c1 + period_count
    c2 = d1 + period_count
    d2 = c2 + hour_count
    soc = d2 + hour_count
    charge_on = soc + period_count
    discharge_on = charge_on + period_count
    variable_count = discharge_on + period_count

    objective = np.zeros(variable_count)
    objective[c1 : c1 + period_count] = interval_hours * prices_1
    objective[d1 : d1 + period_count] = -interval_hours * prices_1
    objective[c2 : c2 + hour_count] = prices_2
    objective[d2 : d2 + hour_count] = -prices_2

    lower = np.zeros(variable_count)
    upper = np.full(variable_count, np.inf)
    upper[c1 : c1 + period_count] = battery.max_charge_mw
    upper[d1 : d1 + period_count] = battery.max_discharge_mw
    upper[c2 : c2 + hour_count] = battery.max_charge_mw
    upper[d2 : d2 + hour_count] = battery.max_discharge_mw
    upper[soc : soc + period_count] = battery.capacity_mwh
    upper[charge_on : charge_on + period_count] = 1
    upper[discharge_on : discharge_on + period_count] = 1
    integrality = np.zeros(variable_count, dtype=int)
    integrality[charge_on : discharge_on + period_count] = 1

    # State transition plus three constraints per half-hour: charge cap,
    # discharge cap, and mutually exclusive operating direction.
    constraints = lil_matrix((4 * period_count, variable_count), dtype=float)
    lower_constraint = np.full(4 * period_count, -np.inf)
    upper_constraint = np.zeros(4 * period_count)
    for index in range(period_count):
        hour = index // 2
        state_row = 4 * index
        constraints[state_row, soc + index] = 1
        if index:
            constraints[state_row, soc + index - 1] = -1
        else:
            lower_constraint[state_row] = state.energy_mwh
            upper_constraint[state_row] = state.energy_mwh
        constraints[state_row, c1 + index] = -interval_hours * battery.charge_efficiency
        constraints[state_row, d1 + index] = interval_hours / battery.discharge_efficiency
        constraints[state_row, c2 + hour] = -interval_hours * battery.charge_efficiency
        constraints[state_row, d2 + hour] = interval_hours / battery.discharge_efficiency

        charge_row = state_row + 1
        constraints[charge_row, c1 + index] = 1
        constraints[charge_row, c2 + hour] = 1
        constraints[charge_row, charge_on + index] = -battery.max_charge_mw

        discharge_row = state_row + 2
        constraints[discharge_row, d1 + index] = 1
        constraints[discharge_row, d2 + hour] = 1
        constraints[discharge_row, discharge_on + index] = -battery.max_discharge_mw

        direction_row = state_row + 3
        constraints[direction_row, charge_on + index] = 1
        constraints[direction_row, discharge_on + index] = 1
        upper_constraint[direction_row] = 1

    options: dict[str, float] = {}
    if time_limit_seconds is not None:
        if time_limit_seconds <= 0:
            raise ValueError("Time limit must be positive.")
        options["time_limit"] = time_limit_seconds
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(constraints.tocsr(), lower_constraint, upper_constraint),
        options=options,
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Optimisation did not find an optimal solution: {result.message}")

    dispatches = _solution_to_dispatches(
        combined,
        result.x,
        c1=c1,
        d1=d1,
        c2=c2,
        d2=d2,
        soc=soc,
        initial_energy_mwh=state.energy_mwh,
    )
    _, summary = _summarise_dispatches(
        dispatches,
        battery=battery,
        economics=economics or EconomicsAssumptions(),
        low_threshold=None,
        high_threshold=None,
        ending_state=BatteryState(float(dispatches["energy_end_mwh"].iloc[-1])),
        title="Optimised historic simulation",
    )
    return OptimisationResult(
        dispatches=dispatches,
        summary=summary,
        solver_message=str(result.message),
        objective_cashflow_gbp=float(-result.fun),
    )


def _solution_to_dispatches(
    combined: pd.DataFrame,
    solution: np.ndarray,
    *,
    c1: int,
    d1: int,
    c2: int,
    d2: int,
    soc: int,
    initial_energy_mwh: float,
) -> pd.DataFrame:
    """Convert a solved vector to the same auditable rows as the heuristic."""
    interval_hours = 0.5
    records: list[dict[str, object]] = []
    energy_start = initial_energy_mwh
    for index, row in enumerate(combined.itertuples(index=False)):
        hour = index // 2
        market_1_charge = _clean(solution[c1 + index])
        market_1_discharge = _clean(solution[d1 + index])
        market_2_charge = _clean(solution[c2 + hour])
        market_2_discharge = _clean(solution[d2 + hour])
        charge_power = market_1_charge + market_2_charge
        discharge_power = market_1_discharge + market_2_discharge
        if charge_power > 1e-7:
            mode = BatteryMode.CHARGE
            market_1_power = market_1_charge
            market_2_power = market_2_charge
        elif discharge_power > 1e-7:
            mode = BatteryMode.DISCHARGE
            market_1_power = market_1_discharge
            market_2_power = market_2_discharge
        else:
            mode = BatteryMode.IDLE
            market_1_power = 0.0
            market_2_power = 0.0
        energy_end = _clean(solution[soc + index])
        market_1_grid_energy = market_1_power * interval_hours
        market_2_grid_energy = market_2_power * interval_hours
        sign = -1 if mode is BatteryMode.CHARGE else 1 if mode is BatteryMode.DISCHARGE else 0
        records.append(
            {
                "timestamp_utc": row.timestamp_utc,
                "mode": mode.value,
                "market_1_price_gbp_per_mwh": float(row.price_gbp_per_mwh),
                "market_2_price_gbp_per_mwh": float(row.market_2_price_gbp_per_mwh),
                "market_1_power_mw": market_1_power,
                "market_2_power_mw": market_2_power,
                "accepted_power_mw": market_1_power + market_2_power,
                "grid_energy_mwh": (market_1_power + market_2_power) * interval_hours,
                "stored_energy_change_mwh": energy_end - energy_start,
                "energy_start_mwh": energy_start,
                "energy_end_mwh": energy_end,
                "market_1_cashflow_gbp": sign * market_1_grid_energy * float(row.price_gbp_per_mwh),
                "market_2_cashflow_gbp": sign * market_2_grid_energy * float(row.market_2_price_gbp_per_mwh),
                "cashflow_gbp": sign
                * (
                    market_1_grid_energy * float(row.price_gbp_per_mwh)
                    + market_2_grid_energy * float(row.market_2_price_gbp_per_mwh)
                ),
            }
        )
        energy_start = energy_end
    return pd.DataFrame.from_records(records)


def _clean(value: float) -> float:
    """Remove solver tolerance noise from exported values."""
    return 0.0 if abs(value) < 1e-9 else float(value)
