"""Generation subprocess: loads a modeling.py by path (champion OR baseline) and generates
predictions for an asset. Launched by run_campaign verdict, inside the network-cut sandbox."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from synth_lib.preparation.config import (  # type: ignore[import-untyped]
    LEGACY_STORE_SUBDIR,
    STORE_SUBDIR,
)
from synth_lib.preparation.minute_price_store import MinutePriceStore  # type: ignore[import-untyped]

from synth_lib.benchmark.nomination import load_simulate
from synth_lib.benchmark.verdict.generate import CONTEXT_MINUTES, generate_predictions


def _to_utc(x: str) -> pd.Timestamp:
    """pd.Timestamp(x, tz="UTC") semantically, while working around its trap: if x is already tz-aware,
    passing tz="UTC" raises (`Cannot pass a datetime or Timestamp with tzinfo with the tz parameter`)
    depending on the pandas version — so we only force it if the parsed value is naive."""
    ts = pd.Timestamp(x)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modeling", required=True, type=Path)
    ap.add_argument("--asset", required=True)
    ap.add_argument("--window-start", required=True)
    ap.add_argument("--window-end", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path, help="market_data root for the verdict")
    ap.add_argument(
        "--cadence-minutes",
        type=int,
        default=60,
        help="Spacing between generated prompts. The CRYPTO_1H field runs at ~144 prompts/day "
        "(cadence ~10min): a denser cadence than 24h (--cadence-minutes 60) is expected "
        "to match the real 1h frequency.",
    )
    ap.add_argument(
        "--num-simulations",
        type=int,
        default=1000,
        help="Paths per prompt. The validator (PromptConfig.num_simulations) uses 1000; lowering "
        "this is only for quick local checks, since it biases empirical CRPS upward relative to "
        "the field, which was scored at 1000.",
    )
    ap.add_argument("--time-increment", type=int, default=300, help="seconds between points in a path")
    ap.add_argument("--time-length", type=int, default=86_400, help="horizon in seconds (3600 for CRYPTO_1H)")
    args = ap.parse_args()

    start = _to_utc(args.window_start)
    end = _to_utc(args.window_end)
    # Canonical `prices/` subdir, legacy `pyth/` when that is what the snapshot holds. Not
    # hardcoded: a snapshot built from a freshly ingested store uses the new name, and a hardcoded
    # path would silently read nothing. default_store_root() cannot be reused here — it resolves
    # relative to the cwd, and --data-root is operator-supplied.
    root = args.data_root / STORE_SUBDIR / args.asset / "1m"
    if not root.exists():
        root = args.data_root / LEGACY_STORE_SUBDIR / args.asset / "1m"
    store = MinutePriceStore(args.asset, root=root)
    # load_range() returns a DataFrame with a "timestamp" column (default RangeIndex,
    # not time-indexed — verified at runtime): we re-index it ourselves before
    # extracting "close" so that generate_predictions's .loc[t - CONTEXT : t] slicing
    # works on a DatetimeIndex.
    frame = store.load_range(start - pd.Timedelta(minutes=CONTEXT_MINUTES), end)
    prices = frame.set_index("timestamp")["close"]
    n = generate_predictions(
        load_simulate(args.modeling),
        args.asset,
        start,
        end,
        args.out_dir,
        prices,
        args.cadence_minutes,
        args.num_simulations,
        args.time_increment,
        args.time_length,
    )
    print(f"generated: {n}")


if __name__ == "__main__":
    main()
