"""Generate predictions from a modeling.py over a prompt grid — the reusable evaluation core.

DELIBERATELY SELF-CONTAINED: imports nothing from synth_lib, only numpy/pandas/stdlib. It is
copied into a champion's clone and executed inside a --network none sandbox whose venv holds only
the champion's own pins, so a synth_lib import here would break every verdict.
That is what lets it serve three duties with one implementation:
  - the verdict runner copies this single file into a cloned champion workspace and runs it
    inside the --network none sandbox (synth_lib/benchmark/verdict/run_verdict.py);
  - the CI contract gate runs it on the host against any miner exposing simulate();
  - an operator can run it standalone against any modeling.py.

No-lookahead guarantee: for each prompt at time t the model receives ONLY prices <= t —
`series.loc[t - 7d : t]` — regardless of how much data the frame holds. The model's sole market
input is that Series; simulate() takes no data root.

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

import pandas as pd

CONTEXT_MINUTES = 7 * 24 * 60
# The validator's real serving size (PromptConfig.num_simulations). Empirical CRPS is biased
# upward for small N, so scoring at fewer paths than the field unfairly penalizes the candidate.
DEFAULT_NUM_SIMULATIONS = 1000
STORE_SUBDIR = "prices"


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


def load_minute_prices(data_root: Path, asset: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Minute closes over [start, end] from daily partitions. Missing partitions raise: a silent
    hole here would shrink every context that spans it without anyone noticing."""
    root = store_root(data_root, asset)
    frames = []
    day = start.date()
    while day <= end.date():
        path = root / f"date={day.isoformat()}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing partition {path}")
        frames.append(pd.read_parquet(path, columns=["timestamp", "close"]))
        day += timedelta(days=1)
    frame = pd.concat(frames, ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    series = frame.set_index("timestamp")["close"].sort_index()
    return series.loc[start:end]


def prompt_grid(window_start: pd.Timestamp, window_end: pd.Timestamp, cadence_minutes: int) -> list[pd.Timestamp]:
    """Prompts in [start, end). Filtered with `t < window_end` rather than [:-1] so a window_end
    not aligned to the cadence does not drop the last valid prompt."""
    grid = pd.date_range(window_start, window_end, freq=f"{cadence_minutes}min", tz="UTC")
    return [t for t in grid if t < window_end]


def generate(
    simulate_fn: Callable,
    asset: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    price_series: pd.Series,
    out_dir: Path,
    cadence_minutes: int,
    time_increment: int,
    time_length: int,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for t in prompt_grid(window_start, window_end, cadence_minutes):
        # THE no-lookahead line: only prices <= t reach the model, whatever the frame holds.
        context = price_series.loc[t - pd.Timedelta(minutes=CONTEXT_MINUTES) : t]
        out = simulate_fn(
            asset=asset,
            start_time=t.isoformat(),
            time_increment=time_increment,
            time_length=time_length,
            num_simulations=num_simulations,
            context_prices=context,
        )
        paths = [list(map(float, p)) for p in out[2:]]
        payload = {
            "start_timestamp": t.isoformat(),
            "asset": asset,
            "time_increment": time_increment,
            "time_length": time_length,
            "num_simulations": num_simulations,
            "num_steps": len(paths[0]),
            "paths": paths,
        }
        name = t.strftime("%Y-%m-%d_%H:%M:%SZ") + f"_{asset}_{time_length}.json"
        (out_dir / name).write_text(json.dumps(payload))
        count += 1
    return count


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
    args = ap.parse_args()

    start, end = _utc(args.window_start), _utc(args.window_end)
    series = load_minute_prices(args.data_root, args.asset, start - pd.Timedelta(minutes=CONTEXT_MINUTES), end)
    n = generate(
        load_simulate(args.modeling),
        args.asset,
        start,
        end,
        series,
        args.out_dir,
        cadence_minutes=args.cadence_minutes,
        time_increment=args.time_increment,
        time_length=args.time_length,
        num_simulations=args.num_simulations,
    )
    print(f"{args.asset} tl={args.time_length}: {n} predictions")


if __name__ == "__main__":
    main()
