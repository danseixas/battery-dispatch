"""Simple dispatch rules for a single electricity market."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from battery_dispatch.battery import BatteryMode


@dataclass(frozen=True)
class PercentileStrategy:
    """Charge below one price percentile and discharge above another.

    This historical-backtest rule calculates both thresholds from the full
    selected price window. It is deliberately simple and has look-ahead bias;
    it is a baseline rather than a production trading strategy.
    """

    charge_quantile: float = 0.25
    discharge_quantile: float = 0.75

    def __post_init__(self) -> None:
        if not 0 <= self.charge_quantile <= 1:
            raise ValueError("The charge quantile must be between zero and one.")
        if not 0 <= self.discharge_quantile <= 1:
            raise ValueError("The discharge quantile must be between zero and one.")
        if self.charge_quantile >= self.discharge_quantile:
            raise ValueError("The charge quantile must be lower than the discharge quantile.")

    def thresholds(self, prices: pd.Series) -> tuple[float, float]:
        """Calculate charge and discharge thresholds for a price window."""
        if prices.empty:
            raise ValueError("At least one price is required to calculate thresholds.")
        return (
            float(prices.quantile(self.charge_quantile)),
            float(prices.quantile(self.discharge_quantile)),
        )

    def decide(self, price_gbp_per_mwh: float, *, low_threshold: float, high_threshold: float) -> BatteryMode:
        """Choose one mutually exclusive battery mode for a market interval."""
        if price_gbp_per_mwh <= low_threshold:
            return BatteryMode.CHARGE
        if price_gbp_per_mwh >= high_threshold:
            return BatteryMode.DISCHARGE
        return BatteryMode.IDLE
