"""Writing reproducible simulation artifacts."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from battery_dispatch.simulation import SimulationSummary


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a source workbook."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_simulation_results(
    dispatches: pd.DataFrame,
    summary: SimulationSummary,
    *,
    output_dir: Path,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    """Write detailed dispatch and deterministic run metadata to an output folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dispatch_path = output_dir / "dispatch.csv"
    summary_path = output_dir / "summary.json"

    dispatches.to_csv(dispatch_path, index=False, date_format="%Y-%m-%dT%H:%M:%S%z")
    summary_path.write_text(
        json.dumps(
            {"metadata": metadata, "summary": asdict(summary)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return dispatch_path, summary_path
