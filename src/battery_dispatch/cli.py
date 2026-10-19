"""Command-line interface for battery-dispatch."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
import time

from battery_dispatch import __version__
from battery_dispatch.battery import BatteryConfig, BatteryState
from battery_dispatch.data import load_market_1, load_market_2
from battery_dispatch.economics import EconomicsAssumptions
from battery_dispatch.outputs import sha256_file, write_simulation_results
from battery_dispatch.optimisation import optimise_two_markets
from battery_dispatch.simulation import simulate_two_markets
from battery_dispatch.strategy import PercentileStrategy


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command-line parser."""
    parser = argparse.ArgumentParser(
        prog="battery-dispatch",
        description="Model battery dispatch across electricity markets.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")
    run_two_markets = subparsers.add_parser(
        "run-two-markets",
        help="Run the two-market heuristic with power allocation.",
    )
    run_two_markets.add_argument("workbook", type=Path)
    run_two_markets.add_argument("--market-1-sheet", default="Half-hourly data")
    run_two_markets.add_argument("--market-2-sheet", default="Hourly data")
    run_two_markets.add_argument("--periods", type=int)
    run_two_markets.add_argument("--start-index", type=int, default=0)
    run_two_markets.add_argument("--initial-energy-mwh", type=float, default=0.0)
    run_two_markets.add_argument("--charge-quantile", type=float, default=0.25)
    run_two_markets.add_argument("--discharge-quantile", type=float, default=0.75)
    run_two_markets.add_argument("--market-2-power-share", type=float, default=0.5)
    run_two_markets.add_argument(
        "--annual-fixed-operating-cost-gbp", type=float, default=5_000.0
    )
    run_two_markets.add_argument(
        "--storage-degradation-percent-per-cycle", type=float, default=0.001
    )
    run_two_markets.add_argument("--output-dir", type=Path, required=True)
    optimise = subparsers.add_parser(
        "optimise-two-markets",
        help="Optimise a complete two-market historic window with perfect foresight.",
    )
    optimise.add_argument("workbook", type=Path)
    optimise.add_argument("--market-1-sheet", default="Half-hourly data")
    optimise.add_argument("--market-2-sheet", default="Hourly data")
    optimise.add_argument("--periods", type=int, default=336)
    optimise.add_argument("--start-index", type=int, default=0)
    optimise.add_argument("--initial-energy-mwh", type=float, default=0.0)
    optimise.add_argument("--time-limit-seconds", type=float)
    optimise.add_argument(
        "--annual-fixed-operating-cost-gbp", type=float, default=5_000.0
    )
    optimise.add_argument(
        "--storage-degradation-percent-per-cycle", type=float, default=0.001
    )
    optimise.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return an exit status."""
    arguments = build_parser().parse_args(argv)
    started_at = time.perf_counter()

    if arguments.command == "run-two-markets":
        if arguments.periods is not None and arguments.periods <= 0:
            raise ValueError("The number of periods must be positive.")
        if arguments.start_index < 0:
            raise ValueError("The start index must be zero or greater.")
        market_1, _ = load_market_1(arguments.workbook, sheet_name=arguments.market_1_sheet)
        market_2 = load_market_2(arguments.workbook, sheet_name=arguments.market_2_sheet)
        end_index = (
            arguments.start_index + arguments.periods
            if arguments.periods is not None
            else len(market_1)
        )
        market_1_window = market_1.iloc[arguments.start_index:end_index].copy()
        if arguments.periods is not None and len(market_1_window) != arguments.periods:
            raise ValueError("The requested window extends beyond the available Market 1 data.")
        config = BatteryConfig(2.0, 2.0, 4.0)
        strategy = PercentileStrategy(
            charge_quantile=arguments.charge_quantile,
            discharge_quantile=arguments.discharge_quantile,
        )
        economics = EconomicsAssumptions(
            annual_fixed_operating_cost_gbp=arguments.annual_fixed_operating_cost_gbp,
            storage_degradation_percent_per_cycle=(
                arguments.storage_degradation_percent_per_cycle
            ),
        )
        dispatches, summary = simulate_two_markets(
            market_1_window,
            market_2,
            battery=config,
            strategy=strategy,
            market_2_power_share=arguments.market_2_power_share,
            initial_state=BatteryState(arguments.initial_energy_mwh),
            economics=economics,
        )
        dispatch_path, summary_path = write_simulation_results(
            dispatches,
            summary,
            output_dir=arguments.output_dir,
            metadata={
                "battery": asdict(config),
                "economics": asdict(economics),
                "initial_energy_mwh": arguments.initial_energy_mwh,
                "input_sha256": sha256_file(arguments.workbook),
                "market_1_sheet": arguments.market_1_sheet,
                "market_2_power_share": arguments.market_2_power_share,
                "market_2_sheet": arguments.market_2_sheet,
                "periods": arguments.periods,
                "source_workbook": str(arguments.workbook),
                "start_index": arguments.start_index,
                "strategy": asdict(strategy),
                "elapsed_seconds": time.perf_counter() - started_at,
            },
        )
        print(summary.to_text())
        print(f"Detailed dispatch: {dispatch_path}")
        print(f"Run summary: {summary_path}")

    if arguments.command == "optimise-two-markets":
        if arguments.periods <= 0 or arguments.periods % 2:
            raise ValueError("The number of periods must be a positive, even number.")
        if arguments.start_index < 0 or arguments.start_index % 2:
            raise ValueError("The start index must be a non-negative, even number.")
        market_1, _ = load_market_1(arguments.workbook, sheet_name=arguments.market_1_sheet)
        market_2 = load_market_2(arguments.workbook, sheet_name=arguments.market_2_sheet)
        end_index = arguments.start_index + arguments.periods
        market_1_window = market_1.iloc[arguments.start_index:end_index].copy()
        if len(market_1_window) != arguments.periods:
            raise ValueError("The requested window extends beyond the available Market 1 data.")
        config = BatteryConfig(2.0, 2.0, 4.0)
        economics = EconomicsAssumptions(
            annual_fixed_operating_cost_gbp=arguments.annual_fixed_operating_cost_gbp,
            storage_degradation_percent_per_cycle=(
                arguments.storage_degradation_percent_per_cycle
            ),
        )
        result = optimise_two_markets(
            market_1_window,
            market_2,
            battery=config,
            initial_state=BatteryState(arguments.initial_energy_mwh),
            economics=economics,
            time_limit_seconds=arguments.time_limit_seconds,
        )
        dispatch_path, summary_path = write_simulation_results(
            result.dispatches,
            result.summary,
            output_dir=arguments.output_dir,
            metadata={
                "approach": "mixed-integer perfect-foresight optimisation",
                "battery": asdict(config),
                "economics": asdict(economics),
                "initial_energy_mwh": arguments.initial_energy_mwh,
                "input_sha256": sha256_file(arguments.workbook),
                "market_1_sheet": arguments.market_1_sheet,
                "market_2_sheet": arguments.market_2_sheet,
                "periods": arguments.periods,
                "solver_message": result.solver_message,
                "source_workbook": str(arguments.workbook),
                "start_index": arguments.start_index,
                "time_limit_seconds": arguments.time_limit_seconds,
                "elapsed_seconds": time.perf_counter() - started_at,
            },
        )
        print(result.summary.to_text())
        print(f"Solver: {result.solver_message}")
        print(f"Detailed dispatch: {dispatch_path}")
        print(f"Run summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
