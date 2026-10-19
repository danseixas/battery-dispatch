from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from battery_dispatch.cli import main


def test_main_succeeds_without_arguments() -> None:
    assert main([]) == 0


def test_help_exits_successfully(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as result:
        main(["--help"])

    assert result.value.code == 0
    assert "Model battery dispatch" in capsys.readouterr().out


def test_run_two_markets_creates_dispatch_and_summary(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    workbook = tmp_path / "market-data.xlsx"
    output_dir = tmp_path / "run"
    market_1 = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=4, freq="30min"),
            "price": [10.0, 40.0, 80.0, 100.0],
        }
    )
    market_2 = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=2, freq="h"),
            "price": [20.0, 90.0],
        }
    )
    with pd.ExcelWriter(workbook) as writer:
        market_1.to_excel(writer, sheet_name="Half-hourly data", index=False)
        market_2.to_excel(writer, sheet_name="Hourly data", index=False)

    assert main(
        [
            "run-two-markets",
            str(workbook),
            "--periods",
            "4",
            "--start-index",
            "0",
            "--initial-energy-mwh",
            "1.0",
            "--market-2-power-share",
            "0.4",
            "--annual-fixed-operating-cost-gbp",
            "6000",
            "--storage-degradation-percent-per-cycle",
            "0.002",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    dispatch_path = output_dir / "dispatch.csv"
    summary_path = output_dir / "summary.json"
    dispatches = pd.read_csv(dispatch_path)
    results = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata = results["metadata"]
    summary = results["summary"]

    assert len(dispatches) == 4
    assert {
        "timestamp_utc",
        "mode",
        "market_1_power_mw",
        "market_2_power_mw",
        "energy_start_mwh",
        "energy_end_mwh",
        "cashflow_gbp",
    }.issubset(dispatches.columns)
    assert dispatches["accepted_power_mw"].le(2.0).all()
    assert metadata["input_sha256"] == hashlib.sha256(workbook.read_bytes()).hexdigest()
    assert metadata["start_index"] == 0
    assert metadata["periods"] == 4
    assert metadata["initial_energy_mwh"] == 1.0
    assert metadata["market_2_power_share"] == 0.4
    assert metadata["economics"]["annual_fixed_operating_cost_gbp"] == 6000.0
    assert metadata["economics"]["storage_degradation_percent_per_cycle"] == 0.002
    assert isinstance(metadata["elapsed_seconds"], (int, float))
    assert metadata["elapsed_seconds"] >= 0
    assert summary["ending_energy_mwh"] == pytest.approx(
        dispatches["energy_end_mwh"].iloc[-1]
    )

    output = capsys.readouterr().out
    assert "Heuristic simulation" in output
    assert f"Detailed dispatch: {dispatch_path}" in output
    assert f"Run summary: {summary_path}" in output


def test_optimisation_summary_includes_elapsed_time(
    tmp_path,
) -> None:
    workbook = tmp_path / "market-data.xlsx"
    market_1 = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=2, freq="30min"),
            "price": [10.0, 100.0],
        }
    )
    market_2 = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=1, freq="h"),
            "price": [50.0],
        }
    )
    with pd.ExcelWriter(workbook) as writer:
        market_1.to_excel(writer, sheet_name="Half-hourly data", index=False)
        market_2.to_excel(writer, sheet_name="Hourly data", index=False)

    output_dir = tmp_path / "optimised-run"
    assert main(
        [
            "optimise-two-markets",
            str(workbook),
            "--periods",
            "2",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    results = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    elapsed_seconds = results["metadata"]["elapsed_seconds"]
    assert isinstance(elapsed_seconds, (int, float))
    assert elapsed_seconds >= 0


def test_optimise_two_markets_dispatches_through_both_markets(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    workbook = tmp_path / "market-data.xlsx"
    market_1 = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=4, freq="30min"),
            "price": [100.0, 100.0, 200.0, 200.0],
        }
    )
    market_2 = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=2, freq="h"),
            "price": [10.0, 10.0],
        }
    )
    with pd.ExcelWriter(workbook) as writer:
        market_1.to_excel(writer, sheet_name="Half-hourly data", index=False)
        market_2.to_excel(writer, sheet_name="Hourly data", index=False)

    output_dir = tmp_path / "optimised-run"
    assert main(
        [
            "optimise-two-markets",
            str(workbook),
            "--periods",
            "4",
            "--initial-energy-mwh",
            "0",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    dispatch_path = output_dir / "dispatch.csv"
    summary_path = output_dir / "summary.json"
    dispatches = pd.read_csv(dispatch_path)
    results = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata = results["metadata"]
    summary = results["summary"]

    assert len(dispatches) == 4
    assert dispatches["mode"].tolist() == [
        "charge",
        "charge",
        "discharge",
        "discharge",
    ]
    assert dispatches.loc[:1, "market_2_power_mw"].tolist() == pytest.approx(
        [2.0, 2.0]
    )
    assert dispatches.loc[:1, "market_1_power_mw"].eq(0.0).all()
    assert dispatches.loc[2:, "market_1_power_mw"].sum() > 0.0
    assert dispatches.loc[2:, "market_2_power_mw"].eq(0.0).all()
    assert dispatches["accepted_power_mw"].le(2.0).all()
    assert dispatches["energy_start_mwh"].between(0.0, 4.0).all()
    assert dispatches["energy_end_mwh"].between(0.0, 4.0).all()
    assert dispatches["cashflow_gbp"].sum() == pytest.approx(341.0)
    assert summary["net_cashflow_gbp"] == pytest.approx(
        dispatches["cashflow_gbp"].sum()
    )
    assert summary["ending_energy_mwh"] == pytest.approx(
        dispatches["energy_end_mwh"].iloc[-1]
    )
    assert metadata["approach"] == "mixed-integer perfect-foresight optimisation"
    assert metadata["input_sha256"] == hashlib.sha256(workbook.read_bytes()).hexdigest()
    assert metadata["periods"] == 4
    assert metadata["initial_energy_mwh"] == 0.0
    assert metadata["solver_message"]
    assert metadata["elapsed_seconds"] >= 0

    output = capsys.readouterr().out
    assert "Optimised historic simulation" in output
    assert "Solver:" in output
    assert f"Detailed dispatch: {dispatch_path}" in output
    assert f"Run summary: {summary_path}" in output
