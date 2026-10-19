from __future__ import annotations

import json

import pandas as pd

from battery_dispatch.outputs import write_simulation_results
from battery_dispatch.simulation import SimulationSummary


def test_write_simulation_results_creates_reusable_artifacts(tmp_path) -> None:
    dispatches = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2020-01-01", periods=1, freq="30min", tz="UTC"),
            "cashflow_gbp": [12.5],
        }
    )
    summary = SimulationSummary(
        charge_threshold_gbp_per_mwh=10.0,
        discharge_threshold_gbp_per_mwh=20.0,
        import_cost_gbp=5.0,
        export_revenue_gbp=17.5,
        net_cashflow_gbp=12.5,
        ending_energy_mwh=0.0,
        analysis_period_years=1.0,
        equivalent_cycles=2.0,
        fixed_operating_cost_gbp=5_000.0,
        operating_cashflow_gbp=7.5,
        projected_capacity_mwh=3.99992,
    )

    dispatch_path, summary_path = write_simulation_results(
        dispatches,
        summary,
        output_dir=tmp_path / "run",
        metadata={"input_sha256": "abc123", "start_index": 0},
    )

    restored_dispatches = pd.read_csv(dispatch_path)
    restored_summary = json.loads(summary_path.read_text())
    assert restored_dispatches.loc[0, "cashflow_gbp"] == 12.5
    assert restored_summary["metadata"]["input_sha256"] == "abc123"
    assert restored_summary["summary"]["net_cashflow_gbp"] == 12.5
