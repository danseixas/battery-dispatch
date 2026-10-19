"""Economic metrics derived from a dispatch simulation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from battery_dispatch.battery import BatteryConfig


@dataclass(frozen=True)
class EconomicsAssumptions:
    """Cost and degradation assumptions supplied with the battery data."""

    annual_fixed_operating_cost_gbp: float = 5_000.0
    storage_degradation_percent_per_cycle: float = 0.001

    def __post_init__(self) -> None:
        if self.annual_fixed_operating_cost_gbp < 0:
            raise ValueError("Annual fixed operating cost cannot be negative.")
        if self.storage_degradation_percent_per_cycle < 0:
            raise ValueError("Storage degradation per cycle cannot be negative.")


@dataclass(frozen=True)
class EconomicResults:
    """Operating results over the simulated period."""

    analysis_period_years: float
    equivalent_cycles: float
    fixed_operating_cost_gbp: float
    operating_cashflow_gbp: float
    projected_capacity_mwh: float


def analyse_dispatch(
    dispatches: pd.DataFrame,
    *,
    battery: BatteryConfig,
    assumptions: EconomicsAssumptions,
    interval_hours: float,
) -> EconomicResults:
    """Calculate operating cost, cycles and a linear degradation projection.

    Equivalent cycles are internal energy discharged divided by initial storage
    capacity. Capacity degradation is reported after the run and does not
    change physical dispatch decisions in this baseline model.
    """
    required_columns = {"cashflow_gbp", "stored_energy_change_mwh", "timestamp_utc"}
    missing_columns = required_columns.difference(dispatches.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dispatches are missing required columns: {missing}.")
    if interval_hours <= 0:
        raise ValueError("Interval duration must be positive.")

    discharged_internal_energy_mwh = -float(
        dispatches.loc[
            dispatches["stored_energy_change_mwh"] < 0,
            "stored_energy_change_mwh",
        ].sum()
    )
    equivalent_cycles = discharged_internal_energy_mwh / battery.capacity_mwh
    timestamps = pd.to_datetime(dispatches["timestamp_utc"], utc=True)
    if timestamps.isna().any():
        raise ValueError("Dispatches contain an invalid or missing UTC timestamp.")
    hours_in_calendar_year = timestamps.map(
        lambda timestamp: 366 * 24 if timestamp.is_leap_year else 365 * 24
    )
    analysis_period_years = float((interval_hours / hours_in_calendar_year).sum())
    fixed_operating_cost = (
        assumptions.annual_fixed_operating_cost_gbp * analysis_period_years
    )
    degradation_fraction = assumptions.storage_degradation_percent_per_cycle / 100
    projected_capacity = battery.capacity_mwh * max(
        0.0,
        1 - equivalent_cycles * degradation_fraction,
    )
    return EconomicResults(
        analysis_period_years=analysis_period_years,
        equivalent_cycles=equivalent_cycles,
        fixed_operating_cost_gbp=fixed_operating_cost,
        operating_cashflow_gbp=float(dispatches["cashflow_gbp"].sum())
        - fixed_operating_cost,
        projected_capacity_mwh=projected_capacity,
    )
