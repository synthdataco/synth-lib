"""Generate predictions from a modeling.py over a prompt grid — the reusable evaluation core.

DELIBERATELY SELF-CONTAINED: imports nothing from synth_lib, only numpy/pandas/stdlib. It is
copied into a champion's clone and executed inside a --network none sandbox whose venv holds only
the champion's own pins, so a synth_lib import here would break every verdict.
That is what lets it serve three duties with one implementation:
  - the verdict runner copies this single file into a cloned champion workspace and runs it
    inside the --network none sandbox (synth_lib/benchmark/verdict/run_verdict.py);
  - the CI contract gate runs it on the host against any miner exposing simulate();
  - an operator can run it standalone against any modeling.py.

No-lookahead guarantee: for each prompt at time t the model receives ONLY minutes <= t —
`frame.loc[t - 7d : t]` — regardless of how much data the frame holds. The model's sole market
input is that DataFrame; simulate() takes no data root.

Predictions land one file per day, under
`<out>/<asset>/<time_length>_<time_increment>/date=<YYYY-MM-DD>.npy` — float32, shaped
(prompts, simulations, steps) — beside a `.json` naming the prompt start_times it holds. A day
whose file already covers every requested start_time is skipped, so pointing --out-dir at a
directory that survives the run makes a re-score reuse the paths instead of regenerating them.

Usage:
  python generate_predictions.py --modeling agent/modeling.py --asset BTC \\
      --window-start 2026-07-30 --window-end 2026-08-03 \\
      --data-root market_data --out-dir predictions \\
      --time-increment 300 --time-length 86400 --cadence-minutes 60
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

CONTEXT_MINUTES = 7 * 24 * 60
# Duplicated from synth_lib.preparation.config rather than imported: this file runs inside a
# champion's own venv, which does not have synth_lib.
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume", "trade_count"]
# The validator's real serving size (PromptConfig.num_simulations). Empirical CRPS is biased
# upward for small N, so scoring at fewer paths than the field unfairly penalizes the candidate.
DEFAULT_NUM_SIMULATIONS = 1000
STORE_SUBDIR = "prices"
# float32 is what the validator stores miner predictions as, so the field's CRPS was computed at
# this precision too. It also quarters the cache: a day of crypto-1h is 35 MB instead of 170.
PATH_DTYPE = "float32"


def day_files(out_dir: Path, asset: str, time_length: int, time_increment: int, day: str) -> tuple[Path, Path]:
    """The (paths, index) pair for one day of one prompt format."""
    root = out_dir / asset / f"{time_length}_{time_increment}"
    return root / f"date={day}.npy", root / f"date={day}.json"


def covered_start_times(index_path: Path) -> set[str]:
    """Prompt start_times the day file already holds, or nothing when it has none."""
    if not index_path.exists():
        return set()
    try:
        return set(json.loads(index_path.read_text())["start_times"])
    except (ValueError, KeyError):
        return set()  # truncated by an interrupted write; regenerate the day


def load_simulate(modeling_path: Path) -> Callable:
    spec = importlib.util.spec_from_file_location(f"candidate_{modeling_path.parent.name}", modeling_path)
    assert spec is not None and spec.loader is not None  # always true for a .py file path
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.simulate


def store_root(data_root: Path, asset: str) -> Path:
    root = data_root / STORE_SUBDIR / asset / "1m"
    if not root.exists():
        raise FileNotFoundError(f"no minute store for {asset}: {root} does not exist")
    return root


def load_minute_prices(data_root: Path, asset: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Minute OHLCV over [start, end] from daily partitions. Missing partitions raise: a silent
    hole here would shrink every context that spans it without anyone noticing."""
    root = store_root(data_root, asset)
    frames = []
    day = start.date()
    while day <= end.date():
        path = root / f"date={day.isoformat()}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing partition {path}")
        try:
            frames.append(pd.read_parquet(path, columns=["timestamp", *OHLCV_COLUMNS]))
        except ValueError as exc:  # pyarrow's ArrowInvalid, on a column the file lacks
            raise ValueError(
                f"{path} predates the OHLCV columns. Re-ingest with --force-refresh: ingest_day "
                f"skips settled partitions that already exist."
            ) from exc
        day += timedelta(days=1)
    frame = pd.concat(frames, ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.set_index("timestamp").sort_index()
    return frame.loc[start:end]


def prompt_grid(
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    cadence_minutes: int,
    prompt_times: list[pd.Timestamp] | None = None,
) -> list[pd.Timestamp]:
    """Prompts in [start, end). Filtered with `t < window_end` rather than [:-1] so a window_end
    not aligned to the cadence does not drop the last valid prompt.

    `prompt_times` replaces the even grid with an explicit list — the start_times the validator
    actually prompted at, which sit at arbitrary minutes."""
    grid = (
        prompt_times
        if prompt_times is not None
        else pd.date_range(window_start, window_end, freq=f"{cadence_minutes}min", tz="UTC")
    )
    return [t for t in grid if window_start <= t < window_end]


def generate_day(
    simulate_fn: Callable,
    asset: str,
    day: str,
    times: list[pd.Timestamp],
    price_frame: pd.DataFrame,
    out_dir: Path,
    time_increment: int,
    time_length: int,
    num_simulations: int,
) -> int:
    """Generate one day's prompts and write the pair. Returns how many were generated, 0 if the
    day was already covered.

    A day is written whole: if anything is missing it is all regenerated, so a file always holds
    exactly the start_times its index names and an interrupted run cannot leave a half-day behind.
    """
    paths_file, index_file = day_files(out_dir, asset, time_length, time_increment, day)
    wanted = [t.isoformat() for t in times]
    if not set(wanted) - covered_start_times(index_file):
        return 0

    runs = []
    for t in times:
        # THE no-lookahead line: only minutes <= t reach the model, whatever the frame holds.
        context = price_frame.loc[t - pd.Timedelta(minutes=CONTEXT_MINUTES) : t]
        out = simulate_fn(
            asset=asset,
            start_time=t.isoformat(),
            time_increment=time_increment,
            time_length=time_length,
            num_simulations=num_simulations,
            context_prices=context,
        )
        runs.append(np.asarray(out[2:], dtype=PATH_DTYPE))

    paths_file.parent.mkdir(parents=True, exist_ok=True)
    stacked = np.stack(runs)
    np.save(paths_file, stacked)
    index_file.write_text(
        json.dumps(
            {
                "start_times": wanted,
                "asset": asset,
                "time_increment": time_increment,
                "time_length": time_length,
                "num_simulations": int(stacked.shape[1]),
                "num_steps": int(stacked.shape[2]),
            }
        )
    )
    return len(times)


def generate(
    simulate_fn: Callable,
    asset: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    price_frame: pd.DataFrame,
    out_dir: Path,
    cadence_minutes: int,
    time_increment: int,
    time_length: int,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    prompt_times: list[pd.Timestamp] | None = None,
) -> tuple[int, int]:
    """Returns (generated, reused) prompt counts."""
    by_day: dict[str, list[pd.Timestamp]] = {}
    for t in prompt_grid(window_start, window_end, cadence_minutes, prompt_times):
        by_day.setdefault(t.strftime("%Y-%m-%d"), []).append(t)

    generated = reused = 0
    for day, times in sorted(by_day.items()):
        made = generate_day(
            simulate_fn, asset, day, times, price_frame, out_dir, time_increment, time_length, num_simulations
        )
        generated += made
        reused += 0 if made else len(times)
    return generated, reused


def _utc(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modeling", required=True, type=Path)
    ap.add_argument("--asset", required=True)
    ap.add_argument("--window-start", required=True)
    ap.add_argument("--window-end", required=True)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--cadence-minutes", type=int, required=True)
    ap.add_argument("--time-increment", type=int, default=300)
    ap.add_argument("--time-length", type=int, default=86_400)
    ap.add_argument("--num-simulations", type=int, default=DEFAULT_NUM_SIMULATIONS)
    ap.add_argument(
        "--prompt-times",
        type=Path,
        default=None,
        help="JSON list of ISO start_times to generate at, instead of the --cadence-minutes grid",
    )
    args = ap.parse_args()

    start, end = _utc(args.window_start), _utc(args.window_end)
    prompt_times = None
    if args.prompt_times is not None:
        prompt_times = [_utc(t) for t in json.loads(args.prompt_times.read_text())]
    prices = load_minute_prices(args.data_root, args.asset, start - pd.Timedelta(minutes=CONTEXT_MINUTES), end)
    generated, reused = generate(
        load_simulate(args.modeling),
        args.asset,
        start,
        end,
        prices,
        args.out_dir,
        cadence_minutes=args.cadence_minutes,
        time_increment=args.time_increment,
        time_length=args.time_length,
        num_simulations=args.num_simulations,
        prompt_times=prompt_times,
    )
    print(f"{args.asset} tl={args.time_length}: {generated} predictions ({reused} reused)")


if __name__ == "__main__":
    main()
