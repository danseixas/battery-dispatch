from __future__ import annotations

import pandas as pd
import pytest

from battery_dispatch.battery import BatteryConfig, BatteryState
from battery_dispatch.simulation import simulate_two_markets
from battery_dispatch.strategy import PercentileStrategy


def test_market_allocation_divides_power_for_the_same_mode() -> None:
    market_1 = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2020-01-01", periods=2, freq="30min", tz="UTC"),
            "price_gbp_per_mwh": [10.0, 100.0],
        }
    )
    market_2 = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2020-01-01", periods=1, freq="h", tz="UTC"),
            "price_gbp_per_mwh": [100.0],
        }
    )

    dispatches, _ = simulate_two_markets(
        market_1,
        market_2,
        battery=BatteryConfig(2.0, 2.0, 4.0),
        strategy=PercentileStrategy(),
        initial_state=BatteryState(4.0),
    )

    assert dispatches["mode"].tolist() == ["discharge", "discharge"]
    assert dispatches["market_2_power_mw"].tolist() == [2.0, 1.0]
    assert dispatches["market_1_power_mw"].tolist() == [0.0, 1.0]
    assert dispatches["accepted_power_mw"].tolist() == [2.0, 2.0]
    assert (
        dispatches["market_1_power_mw"] + dispatches["market_2_power_mw"]
    ).le(2.0).all()
    assert dispatches["accepted_power_mw"].le(2.0).all()
    assert dispatches["energy_end_mwh"].between(0, 4).all()
    assert dispatches["cashflow_gbp"].sum() == pytest.approx(200.0)


def test_market_allocation_changes_market_1_mode_between_half_hours() -> None:
    market_1 = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2020-01-01", periods=2, freq="30min", tz="UTC"),
            "price_gbp_per_mwh": [10.0, 100.0],
        }
    )
    market_2 = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2020-01-01", periods=1, freq="h", tz="UTC"),
            "price_gbp_per_mwh": [50.0],
        }
    )

    dispatches, _ = simulate_two_markets(
        market_1,
        market_2,
        battery=BatteryConfig(2.0, 2.0, 4.0),
        strategy=PercentileStrategy(),
        initial_state=BatteryState(2.0),
    )

    assert dispatches["mode"].tolist() == ["charge", "discharge"]


def test_conflicting_modes_choose_the_more_extreme_feasible_signal() -> None:
    market_1 = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2020-01-01", periods=2, freq="30min", tz="UTC"),
            "price_gbp_per_mwh": [10.0, 100.0],
        }
    )
    market_2 = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2020-01-01", periods=1, freq="h", tz="UTC"),
            "price_gbp_per_mwh": [100.0],
        }
    )

    dispatches, _ = simulate_two_markets(
        market_1,
        market_2,
        battery=BatteryConfig(2.0, 2.0, 4.0),
        strategy=PercentileStrategy(),
        initial_state=BatteryState(2.0),
    )

    assert dispatches.loc[0, "mode"] == "charge"
    assert dispatches.loc[0, "market_1_power_mw"] == 2.0
    assert dispatches.loc[0, "market_2_power_mw"] == 0.0
