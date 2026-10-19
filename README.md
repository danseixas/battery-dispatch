# Battery Dispatch

Small Python package for modelling the dispatch of an electricity battery.

## Setup

Requires Python 3.11 or newer.

```bash
python -m venv .venv
```

Activate the virtual environment before running the commands in the sections
below. Use the command for your operating system and shell:

### Linux and macOS

```bash
source .venv/bin/activate
```

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, allow locally created scripts for
your user account and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Windows Command Prompt

```bat
.venv\Scripts\activate.bat
```

### Windows with Git Bash

```bash
source .venv/Scripts/activate
```

On Windows, use `py -3.11 -m venv .venv` instead of `python -m venv .venv`
if the `python` command is not available. The same substitution can be used
with `python -m pip` after creating the environment.

After activating the environment, install the project and its development
dependencies:

```bash
python -m pip install --editable '.[dev]'
```

## Run

```bash
battery-dispatch --help
```

## Run both markets

The two-market simulation evaluates both markets in every half-hour and can
distribute the available battery power between them. The default reserves 50% of
power for a qualifying Market 2 commitment and uses the remainder in Market 1.
The Market 2 share remains constant across the hour, and total power remains
at or below 2 MW in each half-hour. Change the share with
`--market-2-power-share`.

```bash
battery-dispatch run-two-markets /path/to/data.xlsx \
  --output-dir outputs/two-markets-full-history
```

The command creates two files in the output directory:

- `dispatch.csv` contains one row per half-hour, including the selected mode,
  prices and allocated power for both markets, battery energy before and after
  dispatch, and the resulting market cashflows. It shows how the battery was
  operated throughout the selected period.
- `summary.json` contains the run metadata and headline results. The metadata
  records the input workbook's SHA-256 digest, selected sheets, run parameters,
  battery configuration, and strategy/economic assumptions so the run can be
  identified and reproduced. It also records `elapsed_seconds`, the time spent
  loading the input and calculating the result, excluding artifact writing. The
  summary reports price thresholds, import cost,
  export revenue, net cashflow, equivalent cycles, operating cost, operating
  cashflow, ending stored energy, and projected capacity after degradation.

These files describe the heuristic backtest and its assumption-based
degradation projection; they are not a profit guarantee or a forecast of future
market performance.

## Optimise a historic window

The final stage uses mixed-integer optimisation to maximise cashflow over a
selected historic window. It has perfect knowledge of that window, so it is an
upper-bound backtest rather than a trading rule. The default window is one week
(336 half-hours), which keeps runs short and makes each result easy to inspect.
The window must begin on an hourly boundary and contain an even number of
half-hours because Market 2 accepts a single power commitment for each hour.

```bash
battery-dispatch optimise-two-markets /path/to/data.xlsx \
  --output-dir outputs/optimised-week
```

The output uses the same auditable `dispatch.csv` and `summary.json` files as
the heuristic. In addition, the JSON metadata records the solver result and the
fact that the run used perfect foresight.

## Test

```bash
pytest
```
