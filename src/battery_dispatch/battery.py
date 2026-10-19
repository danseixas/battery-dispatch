"""Battery state transitions and physical operating limits."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
from numbers import Real


def _require_finite_number(value: object, *, name: str) -> float:
    """Return a finite real number or raise a clear validation error."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    return float(value)


class BatteryMode(StrEnum):
    """The mutually exclusive operating modes for one dispatch interval."""

    CHARGE = "charge"
    DISCHARGE = "discharge"
    IDLE = "idle"


@dataclass(frozen=True)
class BatteryConfig:
    """Physical battery characteristics in MW, MWh, and fractional efficiency."""

    max_charge_mw: float
    max_discharge_mw: float
    capacity_mwh: float
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95

    def __post_init__(self) -> None:
        _require_finite_number(self.max_charge_mw, name="Maximum charge rate")
        _require_finite_number(self.max_discharge_mw, name="Maximum discharge rate")
        _require_finite_number(self.capacity_mwh, name="Battery capacity")
        _require_finite_number(self.charge_efficiency, name="Charge efficiency")
        _require_finite_number(self.discharge_efficiency, name="Discharge efficiency")
        if self.max_charge_mw <= 0 or self.max_discharge_mw <= 0:
            raise ValueError("Maximum charge and discharge rates must be positive.")
        if self.capacity_mwh <= 0:
            raise ValueError("Battery capacity must be positive.")
        if not 0 < self.charge_efficiency <= 1:
            raise ValueError("Charge efficiency must be in the interval (0, 1].")
        if not 0 < self.discharge_efficiency <= 1:
            raise ValueError("Discharge efficiency must be in the interval (0, 1].")


@dataclass(frozen=True)
class BatteryState:
    """Energy currently stored inside the battery."""

    energy_mwh: float

    def validate(self, config: BatteryConfig) -> None:
        """Ensure energy is physically possible for the configured battery."""
        _require_finite_number(self.energy_mwh, name="Stored energy")
        if not 0 <= self.energy_mwh <= config.capacity_mwh:
            raise ValueError(
                "Stored energy must be between zero and the configured battery capacity."
            )


@dataclass(frozen=True)
class DispatchResult:
    """The feasible result of one requested battery operation."""

    mode: BatteryMode
    requested_power_mw: float
    accepted_power_mw: float
    duration_hours: float
    grid_energy_mwh: float
    stored_energy_change_mwh: float
    state_before: BatteryState
    state_after: BatteryState

    def __post_init__(self) -> None:
        if not isinstance(self.mode, BatteryMode):
            raise ValueError("Dispatch mode must be a BatteryMode.")
        for name, value in (
            ("Requested power", self.requested_power_mw),
            ("Accepted power", self.accepted_power_mw),
            ("Dispatch duration", self.duration_hours),
            ("Grid energy", self.grid_energy_mwh),
            ("Stored energy change", self.stored_energy_change_mwh),
        ):
            _require_finite_number(value, name=name)
        if self.requested_power_mw < 0 or self.accepted_power_mw < 0:
            raise ValueError("Dispatch powers must be non-negative.")
        if self.accepted_power_mw > self.requested_power_mw:
            raise ValueError("Accepted power cannot exceed requested power.")
        if self.duration_hours <= 0:
            raise ValueError("Dispatch duration must be positive.")
        if self.grid_energy_mwh < 0:
            raise ValueError("Grid energy must be non-negative.")
        if not isinstance(self.state_before, BatteryState) or not isinstance(
            self.state_after, BatteryState
        ):
            raise ValueError("Dispatch states must be BatteryState instances.")


def dispatch(
    config: BatteryConfig,
    state: BatteryState,
    *,
    mode: BatteryMode,
    requested_power_mw: float,
    duration_hours: float,
) -> DispatchResult:
    """Apply a requested operation, clipping it to physical battery limits.

    Charge power is measured at the grid import point. Discharge power is
    measured at the grid export point. Therefore charge efficiency determines
    energy stored from an import, while discharge efficiency determines energy
    removed from storage to support an export.
    """
    state.validate(config)
    duration = _require_finite_number(duration_hours, name="Dispatch duration")
    requested_power = _require_finite_number(
        requested_power_mw, name="Requested power"
    )
    if duration <= 0:
        raise ValueError("Dispatch duration must be positive.")
    if requested_power < 0:
        raise ValueError("Requested power must be non-negative.")

    if mode is BatteryMode.IDLE:
        if requested_power != 0:
            raise ValueError("Idle dispatch must request zero power.")
        return DispatchResult(
            mode=mode,
            requested_power_mw=0,
            accepted_power_mw=0,
            duration_hours=duration,
            grid_energy_mwh=0,
            stored_energy_change_mwh=0,
            state_before=state,
            state_after=state,
        )

    if mode is BatteryMode.CHARGE:
        capacity_limited_power = (
            (config.capacity_mwh - state.energy_mwh)
            / (duration * config.charge_efficiency)
        )
        accepted_power = min(
            requested_power,
            config.max_charge_mw,
            capacity_limited_power,
        )
        grid_energy = accepted_power * duration
        stored_energy_change = grid_energy * config.charge_efficiency
    elif mode is BatteryMode.DISCHARGE:
        energy_limited_power = (
            state.energy_mwh * config.discharge_efficiency / duration
        )
        accepted_power = min(
            requested_power,
            config.max_discharge_mw,
            energy_limited_power,
        )
        grid_energy = accepted_power * duration
        stored_energy_change = -grid_energy / config.discharge_efficiency
    else:
        raise ValueError(f"Unsupported battery mode: {mode!r}")

    state_after = BatteryState(
        energy_mwh=min(
            config.capacity_mwh,
            max(0.0, state.energy_mwh + stored_energy_change),
        )
    )
    return DispatchResult(
        mode=mode,
        requested_power_mw=requested_power,
        accepted_power_mw=accepted_power,
        duration_hours=duration,
        grid_energy_mwh=grid_energy,
        stored_energy_change_mwh=state_after.energy_mwh - state.energy_mwh,
        state_before=state,
        state_after=state_after,
    )
