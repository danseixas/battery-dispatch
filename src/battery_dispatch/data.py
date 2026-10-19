"""Market-data ingestion and timestamp normalisation."""

from __future__ import annotations

from dataclasses import dataclass
# import logging
from pathlib import Path

import pandas as pd


HALF_HOURLY_INTERVAL = pd.Timedelta(minutes=30)
HOURLY_INTERVAL = pd.Timedelta(hours=1)
DEFAULT_MARKET_1_SHEET = "Half-hourly data"
DEFAULT_MARKET_2_SHEET = "Hourly data"


class MarketDataError(ValueError):
    """Raised when a market price file cannot be safely interpreted."""


@dataclass(frozen=True)
class MarketDataQualityReport:
    """Summary of the checks performed while reading Market 1 prices."""

    rows: int
    start_utc: pd.Timestamp
    end_utc: pd.Timestamp
    negative_price_rows: int
    dst_transition_timestamp_rows: int
    duplicate_timestamp_rows: int
    missing_intervals: int

    def to_text(self) -> str:
        """Format a concise, stable report for command-line use."""
        return "\n".join(
            (
                "Market 1 validation",
                f"Rows: {self.rows}",
                "Canonical UTC period: "
                f"{self.start_utc.isoformat()} to {self.end_utc.isoformat()}",
                f"Negative-price rows: {self.negative_price_rows}",
                "Rows explained by a daylight-saving transition: "
                f"{self.dst_transition_timestamp_rows}",
                f"Unexpected duplicate timestamp rows: {self.duplicate_timestamp_rows}",
                f"Unexpected missing half-hour intervals: {self.missing_intervals}",
            )
        )


def load_market_1(
    workbook: Path | str,
    *,
    sheet_name: str = DEFAULT_MARKET_1_SHEET,
) -> tuple[pd.DataFrame, MarketDataQualityReport]:
    """Load the supplied Market 1 worksheet and return its interval prices.

    The workbook uses 48 rows per civil date. At the March UK clock change it
    repeats the 02:00 and 02:30 labels to retain that count. The canonical UTC
    timeline therefore follows source-row order, and source labels are retained
    separately for traceability.
    """
    # logger.debug("Reading Market 1 workbook=%s sheet=%s", workbook, sheet_name)
    try:
        raw_data = pd.read_excel(workbook, sheet_name=sheet_name)
    except (FileNotFoundError, OSError) as error:
        raise MarketDataError(f"Could not read workbook: {workbook}") from error
    except ValueError as error:
        raise MarketDataError(f"Could not read worksheet {sheet_name!r}") from error

    return normalise_market_1(raw_data)


def load_market_2(
    workbook: Path | str,
    *,
    sheet_name: str = DEFAULT_MARKET_2_SHEET,
) -> pd.DataFrame:
    """Load and validate the supplied hourly Market 2 worksheet."""
    try:
        raw_data = pd.read_excel(workbook, sheet_name=sheet_name)
    except (FileNotFoundError, OSError) as error:
        raise MarketDataError(f"Could not read workbook: {workbook}") from error
    except ValueError as error:
        raise MarketDataError(f"Could not read worksheet {sheet_name!r}") from error

    if raw_data.empty:
        raise MarketDataError("Market 2 data must not be empty.")
    if raw_data.shape[1] < 2:
        raise MarketDataError("Market 2 data must have a timestamp and a price column.")
    timestamp = pd.to_datetime(raw_data.iloc[:, 0], errors="coerce")
    price = pd.to_numeric(raw_data.iloc[:, 1], errors="coerce")
    if timestamp.isna().any():
        raise MarketDataError("Market 2 contains an invalid or missing timestamp.")
    if price.isna().any():
        raise MarketDataError("Market 2 contains an invalid or missing price.")
    if getattr(timestamp.dt, "tz", None) is not None:
        raise MarketDataError("Market 2 timestamps must be timezone-naive local UK times.")
    _check_market_2_timestamp_sequence(timestamp)

    first_timestamp_utc = timestamp.iloc[0].tz_localize("Europe/London").tz_convert("UTC")
    timestamp_utc = pd.Series(
        pd.date_range(
            start=first_timestamp_utc,
            periods=len(timestamp),
            freq=HOURLY_INTERVAL,
        ),
        index=timestamp.index,
    )
    return pd.DataFrame(
        {
            "source_timestamp_local": timestamp,
            "timestamp_utc": timestamp_utc,
            "price_gbp_per_mwh": price.astype(float),
        }
    )


def _check_market_2_timestamp_sequence(timestamps: pd.Series) -> None:
    """Validate hourly Market 2 labels while allowing the UK spring DST jump."""
    for previous, current in zip(timestamps.iloc[:-1], timestamps.iloc[1:], strict=True):
        difference = current - previous
        if abs(difference - HOURLY_INTERVAL) <= pd.Timedelta(milliseconds=1):
            continue
        if _is_source_market_2_spring_dst_transition(previous, current):
            continue
        raise MarketDataError(
            "Market 2 timestamps must advance hourly, except for the expected "
            "UK spring daylight-saving transition."
        )


def _is_source_market_2_spring_dst_transition(
    previous: pd.Timestamp, current: pd.Timestamp
) -> bool:
    """Return whether a 00:00 to 02:00 jump is the Market 2 DST convention."""
    return (
        previous.date() == current.date() == _uk_spring_clock_change(previous.year).date()
        and previous.hour == 0
        and previous.minute == 0
        and current.hour == 2
        and current.minute == 0
    )


def normalise_market_1(
    raw_data: pd.DataFrame,
) -> tuple[pd.DataFrame, MarketDataQualityReport]:
    """Validate Market 1 data and construct a continuous interval timeline."""
    if raw_data.shape[1] < 2:
        raise MarketDataError("Market 1 data must have a timestamp and a price column.")

    timestamp = pd.to_datetime(raw_data.iloc[:, 0], errors="coerce")
    price = pd.to_numeric(raw_data.iloc[:, 1], errors="coerce")
    if timestamp.isna().any():
        raise MarketDataError("Market 1 contains an invalid or missing timestamp.")
    if price.isna().any():
        raise MarketDataError("Market 1 contains an invalid or missing price.")
    if getattr(timestamp.dt, "tz", None) is not None:
        raise MarketDataError("Market 1 timestamps must be timezone-naive local UK times.")

    dst_rows, duplicate_rows, missing_intervals = _check_market_1_timestamp_sequence(
        timestamp
    )
    first_timestamp_utc = timestamp.iloc[0].tz_localize("Europe/London").tz_convert("UTC")
    timestamp_utc = pd.Series(
        pd.date_range(
            start=first_timestamp_utc,
            periods=len(timestamp),
            freq=HALF_HOURLY_INTERVAL,
        ),
        index=timestamp.index,
    )

    normalised = pd.DataFrame(
        {
            "source_timestamp_local": timestamp,
            "timestamp_utc": timestamp_utc,
            "price_gbp_per_mwh": price.astype(float),
        }
    )
    report = MarketDataQualityReport(
        rows=len(normalised),
        start_utc=normalised["timestamp_utc"].iloc[0],
        end_utc=normalised["timestamp_utc"].iloc[-1],
        negative_price_rows=int((normalised["price_gbp_per_mwh"] < 0).sum()),
        dst_transition_timestamp_rows=dst_rows,
        duplicate_timestamp_rows=duplicate_rows,
        missing_intervals=missing_intervals,
    )
    return normalised, report


def _check_market_1_timestamp_sequence(
    timestamps: pd.Series,
) -> tuple[int, int, int]:
    """Classify source timestamp changes without mistaking DST labels for gaps."""
    dst_rows = 0
    duplicate_rows = 0
    missing_intervals = 0

    for previous, current in zip(timestamps.iloc[:-1], timestamps.iloc[1:], strict=True):
        difference = current - previous
        if difference == HALF_HOURLY_INTERVAL:
            continue
        if _is_source_spring_dst_transition(previous, current):
            dst_rows += 1
            continue
        if _is_source_spring_dst_repeated_label(previous, current):
            dst_rows += 1
            continue
        if difference == pd.Timedelta(0):
            duplicate_rows += 1
            # logger.warning("Unexpected duplicate Market 1 timestamp: %s", current)
        elif difference > HALF_HOURLY_INTERVAL:
            missing_count = int(difference / HALF_HOURLY_INTERVAL) - 1
            missing_intervals += missing_count
        else:
            raise MarketDataError(
                "Market 1 timestamps move backwards outside the expected UK "
                "daylight-saving transition."
            )

    return dst_rows, duplicate_rows, missing_intervals


def _is_source_spring_dst_transition(previous: pd.Timestamp, current: pd.Timestamp) -> bool:
    """Return whether a 00:30 to 02:00 jump is the source's DST convention."""
    return (
        previous.date() == current.date() == _uk_spring_clock_change(previous.year).date()
        and previous.hour == 0
        and previous.minute == 30
        and current.hour == 2
        and current.minute == 0
    )


def _is_source_spring_dst_repeated_label(
    previous: pd.Timestamp, current: pd.Timestamp
) -> bool:
    """Return whether a repeated 02:00 follows the source's DST convention."""
    return (
        previous.date() == current.date() == _uk_spring_clock_change(previous.year).date()
        and previous.hour == 2
        and previous.minute == 30
        and current.hour == 2
        and current.minute == 0
    )


def _uk_spring_clock_change(year: int) -> pd.Timestamp:
    """Return the final Sunday in March for a given year."""
    march_31 = pd.Timestamp(year=year, month=3, day=31)
    return march_31 - pd.Timedelta(days=(march_31.weekday() + 1) % 7)
