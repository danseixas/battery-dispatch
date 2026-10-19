from __future__ import annotations

import pandas as pd
import pytest

from battery_dispatch.data import MarketDataError, load_market_2, normalise_market_1


def test_source_spring_clock_change_is_not_reported_as_a_duplicate_or_gap() -> None:
    source = pd.DataFrame(
        {
            "timestamp": [
                "2020-03-29 00:00",
                "2020-03-29 00:30",
                "2020-03-29 02:00",
                "2020-03-29 02:30",
                "2020-03-29 02:00",
                "2020-03-29 02:30",
                "2020-03-29 03:00",
            ],
            "price": [10, 11, 12, 13, 14, 15, 16],
        }
    )

    prices, report = normalise_market_1(source)

    assert prices["timestamp_utc"].diff().dropna().eq(pd.Timedelta(minutes=30)).all()
    assert report.dst_transition_timestamp_rows == 2
    assert report.duplicate_timestamp_rows == 0
    assert report.missing_intervals == 0


def test_non_dst_gap_is_reported() -> None:
    source = pd.DataFrame(
        {
            "timestamp": ["2020-01-01 00:00", "2020-01-01 00:30", "2020-01-01 01:30"],
            "price": [10, 11, 12],
        }
    )

    _, report = normalise_market_1(source)

    assert report.missing_intervals == 1


def test_invalid_price_is_rejected() -> None:
    source = pd.DataFrame({"timestamp": ["2020-01-01 00:00"], "price": [None]})

    with pytest.raises(MarketDataError, match="missing price"):
        normalise_market_1(source)


def test_load_market_2_returns_hourly_utc_timeline(monkeypatch: pytest.MonkeyPatch) -> None:
    source = pd.DataFrame(
        {
            "timestamp": [
                "2020-01-01 00:00",
                "2020-01-01 01:00",
                "2020-01-01 02:00",
            ],
            "price": [10, 11, 12],
        }
    )
    monkeypatch.setattr("battery_dispatch.data.pd.read_excel", lambda *args, **kwargs: source)

    prices = load_market_2("market-data.xlsx")

    assert prices["source_timestamp_local"].tolist() == list(
        pd.to_datetime(source["timestamp"])
    )
    assert prices["timestamp_utc"].diff().dropna().eq(pd.Timedelta(hours=1)).all()
    assert prices["price_gbp_per_mwh"].tolist() == [10.0, 11.0, 12.0]


def test_load_market_2_accepts_spring_clock_change(monkeypatch: pytest.MonkeyPatch) -> None:
    source = pd.DataFrame(
        {
            "timestamp": [
                "2020-03-29 00:00",
                "2020-03-29 02:00",
                "2020-03-29 03:00",
            ],
            "price": [10, 11, 12],
        }
    )
    monkeypatch.setattr("battery_dispatch.data.pd.read_excel", lambda *args, **kwargs: source)

    prices = load_market_2("market-data.xlsx")

    assert prices["timestamp_utc"].diff().dropna().eq(pd.Timedelta(hours=1)).all()


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            pd.DataFrame({"timestamp": ["2020-01-01 00:00"]}),
            "timestamp and a price column",
        ),
        (
            pd.DataFrame({"timestamp": ["not-a-timestamp"], "price": [10]}),
            "invalid or missing timestamp",
        ),
        (
            pd.DataFrame({"timestamp": ["2020-01-01 00:00"], "price": [None]}),
            "missing price",
        ),
        (
            pd.DataFrame(
                {
                    "timestamp": ["2020-01-01 00:00", "2020-01-01 02:00"],
                    "price": [10, 11],
                }
            ),
            "advance hourly",
        ),
        (
            pd.DataFrame(
                {
                    "timestamp": ["2020-01-01 01:00+00:00"],
                    "price": [10],
                }
            ),
            "timezone-naive",
        ),
    ],
)
def test_load_market_2_rejects_invalid_data(
    monkeypatch: pytest.MonkeyPatch,
    source: pd.DataFrame,
    message: str,
) -> None:
    monkeypatch.setattr("battery_dispatch.data.pd.read_excel", lambda *args, **kwargs: source)

    with pytest.raises(MarketDataError, match=message):
        load_market_2("market-data.xlsx")


def test_load_market_2_rejects_empty_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "battery_dispatch.data.pd.read_excel",
        lambda *args, **kwargs: pd.DataFrame(columns=["timestamp", "price"]),
    )

    with pytest.raises(MarketDataError, match="must not be empty"):
        load_market_2("market-data.xlsx")
