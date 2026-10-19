from __future__ import annotations

import pandas as pd
import pytest

from battery_dispatch.battery import BatteryConfig
from battery_dispatch.economics import EconomicsAssumptions, analyse_dispatch


def test_economic_analysis_uses_internal_discharge_for_cycle_count() -> None:
    dispatches = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2020-01-01", periods=2, freq="30min", tz="UTC"),
            "cashflow_gbp": [-10.0, 90.25],
            "stored_energy_change_mwh": [0.95, -0.95],
        }
    )

    results = analyse_dispatch(
        dispatches,
        battery=BatteryConfig(2.0, 2.0, 4.0),
        assumptions=EconomicsAssumptions(
            annual_fixed_operating_cost_gbp=5_000.0,
            storage_degradation_percent_per_cycle=0.001,
        ),
        interval_hours=0.5,
    )

    assert results.equivalent_cycles == pytest.approx(0.2375)
    expected_cost = 5_000 * 1 / (366 * 24)
    assert results.fixed_operating_cost_gbp == pytest.approx(expected_cost)
    assert results.operating_cashflow_gbp == pytest.approx(80.25 - expected_cost)
    assert results.projected_capacity_mwh < 4.0
