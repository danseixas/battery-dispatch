from __future__ import annotations

import pytest

from battery_dispatch.battery import BatteryConfig, BatteryMode, BatteryState, dispatch


@pytest.fixture
def battery() -> BatteryConfig:
    return BatteryConfig(
        max_charge_mw=2.0,
        max_discharge_mw=2.0,
        capacity_mwh=4.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
    )


def test_charge_applies_efficiency(battery: BatteryConfig) -> None:
    result = dispatch(
        battery,
        BatteryState(0.0),
        mode=BatteryMode.CHARGE,
        requested_power_mw=2.0,
        duration_hours=0.5,
    )

    assert result.grid_energy_mwh == pytest.approx(1.0)
    assert result.stored_energy_change_mwh == pytest.approx(0.95)
    assert result.state_after.energy_mwh == pytest.approx(0.95)


def test_charge_is_limited_by_remaining_capacity(battery: BatteryConfig) -> None:
    result = dispatch(
        battery,
        BatteryState(3.9),
        mode=BatteryMode.CHARGE,
        requested_power_mw=2.0,
        duration_hours=0.5,
    )

    assert result.accepted_power_mw == pytest.approx(0.2105263158)
    assert result.state_after.energy_mwh == pytest.approx(4.0)


def test_discharge_is_limited_by_available_energy(battery: BatteryConfig) -> None:
    result = dispatch(
        battery,
        BatteryState(0.5),
        mode=BatteryMode.DISCHARGE,
        requested_power_mw=2.0,
        duration_hours=0.5,
    )

    assert result.accepted_power_mw == pytest.approx(0.95)
    assert result.grid_energy_mwh == pytest.approx(0.475)
    assert result.state_after.energy_mwh == pytest.approx(0.0)


def test_power_is_limited_by_maximum_rate(battery: BatteryConfig) -> None:
    result = dispatch(
        battery,
        BatteryState(4.0),
        mode=BatteryMode.DISCHARGE,
        requested_power_mw=3.0,
        duration_hours=0.5,
    )

    assert result.accepted_power_mw == 2.0


def test_idle_keeps_energy_unchanged(battery: BatteryConfig) -> None:
    result = dispatch(
        battery,
        BatteryState(2.0),
        mode=BatteryMode.IDLE,
        requested_power_mw=0.0,
        duration_hours=0.5,
    )

    assert result.state_after == BatteryState(2.0)


def test_idle_operation_rejects_power_request(battery: BatteryConfig) -> None:
    with pytest.raises(ValueError, match="Idle dispatch"):
        dispatch(
            battery,
            BatteryState(2.0),
            mode=BatteryMode.IDLE,
            requested_power_mw=1.0,
            duration_hours=0.5,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "two"])
def test_config_rejects_non_finite_or_non_numeric_values(value: object) -> None:
    with pytest.raises(ValueError, match="finite number"):
        BatteryConfig(value, 2.0, 4.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "full"])
def test_state_rejects_non_finite_or_non_numeric_energy(
    battery: BatteryConfig, value: object
) -> None:
    with pytest.raises(ValueError, match="finite number"):
        BatteryState(value).validate(battery)  # type: ignore[arg-type]


@pytest.mark.parametrize("duration", [float("nan"), float("inf"), 0.0, -0.001])
def test_dispatch_rejects_invalid_duration(battery: BatteryConfig, duration: float) -> None:
    with pytest.raises(ValueError):
        dispatch(
            battery,
            BatteryState(0.0),
            mode=BatteryMode.CHARGE,
            requested_power_mw=1.0,
            duration_hours=duration,
        )


@pytest.mark.parametrize("power", [float("nan"), float("inf"), "one"])
def test_dispatch_rejects_invalid_requested_power(battery: BatteryConfig, power: object) -> None:
    with pytest.raises(ValueError, match="finite number"):
        dispatch(
            battery,
            BatteryState(0.0),
            mode=BatteryMode.CHARGE,
            requested_power_mw=power,  # type: ignore[arg-type]
            duration_hours=0.5,
        )
