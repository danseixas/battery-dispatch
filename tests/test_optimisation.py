from __future__ import annotations

import pandas as pd
import pytest

from battery_dispatch.battery import BatteryConfig, BatteryState
from battery_dispatch.optimisation import optimise_two_markets


def test_optimisation_uses_best_charge_and_discharge_opportunities() -> None:
    timestamps = pd.date_range("2020-01-01", periods=2, freq="30min", tz="UTC")
    market_1 = pd.DataFrame(
        {"timestamp_utc": timestamps, "price_gbp_per_mwh": [10.0, 100.0]}
    )
    market_2 = pd.DataFrame(
        {"timestamp_utc": timestamps[:1], "price_gbp_per_mwh": [50.0]}
    )

    result = optimise_two_markets(
        market_1,
        market_2,
        battery=BatteryConfig(2.0, 2.0, 4.0),
        initial_state=BatteryState(0.0),
    )

    dispatches = result.dispatches
    assert dispatches["mode"].tolist() == ["charge", "discharge"]
    assert dispatches["market_1_power_mw"].tolist() == pytest.approx([2.0, 1.805])
    assert dispatches["market_2_power_mw"].eq(0.0).all()
    assert dispatches["accepted_power_mw"].le(2.0).all()
    assert result.summary.net_cashflow_gbp == pytest.approx(80.25)
