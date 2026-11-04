"""Freezing market_data up to the cutoff: hard-links + sha256 manifest."""

from __future__ import annotations

import re
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from synth_lib.backtester.field_store import SCORES_DIR
from synth_lib.benchmark.manifest import link_partitions, sha256_file, write_manifest
from synth_lib.preparation.config import STORE_SUBDIR
from synth_lib.preparation.event_calendar.events import (
    CALENDAR_DIR,
    CALENDAR_ENV,
    LEDGER_FILE,
    VIEW_FILE,
    known_at,
    read_ledger,
    write_atomic,
)

_PARTITION_RE = re.compile(r"^date=(\d{4}-\d{2}-\d{2})\.parquet$")


def _partition_day(path: Path) -> date | None:
    m = _PARTITION_RE.match(path.name)
    return date.fromisoformat(m.group(1)) if m else None


def build_snapshot(source_root: Path, dest_root: Path, cutoff: date, start: date | None = None) -> dict[str, str]:
    """Hard-links every partition in [start, cutoff] (all assets) and writes manifest.json.

    `start` decouples what is on disk from what a campaign sees: keep 9 months of ingested
    market_data and snapshot only the 2 months a campaign should train on. None = no lower
    bound, i.e. everything up to the cutoff. A rebuilt snapshot keeps its partitions and its frozen
    calendar."""
    manifest = link_partitions(source_root, dest_root, "date=*.parquet", _partition_day, cutoff, start)
    if not manifest:
        window = f"in [{start}, {cutoff}]" if start is not None else f"<= {cutoff}"
        raise ValueError(f"no partition {window} under {source_root}")
    ledger = source_root / CALENDAR_DIR / LEDGER_FILE
    view = dest_root / CALENDAR_DIR / VIEW_FILE
    if ledger.exists() and not view.exists():
        # Known by the end of the cutoff day, the last day of prices the snapshot holds.
        at = pd.Timestamp(cutoff + timedelta(days=1), tz="UTC") - pd.Timedelta(microseconds=1)
        write_atomic(known_at(read_ledger(ledger), at), view)
    if view.exists():
        manifest[str(view.relative_to(dest_root))] = sha256_file(view)
    write_manifest(dest_root, manifest)
    return manifest


def render_data_md(snapshot_root: Path) -> str:
    """DATA.md for the agents: the MEASURED truth of this exact snapshot.

    Hand-written data prose rots between campaigns (a stale 'SP500 has 11 scorable prompts'
    survived one campaign too many; pre-history NaN skeletons earned a 'synthetic SP500'
    theory from an agent). Everything quantitative here is computed from the snapshot at
    setup time, so the constitution can point at this file and never disagree with reality."""
    lines = [
        "# DATA.md — the shape of your market data (generated from THIS snapshot at setup)",
        "",
        "Each day is one parquet file, `date=YYYY-MM-DD.parquet`, a full 1440-row minute grid",
        "with columns `open, high, low, close, volume, trade_count`. `close` is the scored series.",
        "`close` is the venue's 1-minute candle close (the last trade of that minute); the other",
        "columns describe the same minute. For the Hyperliquid-routed assets each minute is",
        "aggregated from the venue's 1-second candles (first open, highest high, lowest low, last",
        "close), the same values as the venue's 1-minute candle.",
        "`source`/`ingested_at`/`is_final` are ingestion provenance — ignore them. Semantics:",
        "",
        "- **`trade_count` 0** = no trade that minute: the venue prints a flat candle (open, high, low",
        "  and close at the previous close, volume 0), and the validator scores against it. NOT market",
        "  hours: these venues trade 24/7. Trade density varies by asset and era — see the table.",
        "- **NaN close** = no price at all: before the market's first trade, or a hole in the feed.",
        "- **Missing file** = before that market's first trade. Nothing existed; nothing is hidden.",
        "- Scoring drops NaN per prompt. Never interpolate: inventing prices in the gaps is how",
        "  you fool your own validation.",
        "",
        "Per-asset coverage, measured from this snapshot (untraded% = share of minutes without a",
        "trade, flat or NaN, among the asset's real days; first/last 30d columns show density drift —",
        "early months of newer listings are genuinely thin, low realized volatility there is real):",
        "",
        "| asset | first day | last day | real days | untraded% total | untraded% first 30d | untraded% last 30d |",
        "|---|---|---|---|---|---|---|",
    ]
    store = snapshot_root / STORE_SUBDIR
    for asset_dir in sorted(store.iterdir()) if store.is_dir() else ():
        row = _coverage_row(asset_dir)
        if row is not None:
            lines.append(row)
    lines += _timestamp_lines(store)
    lines += _offline_bundle_lines(snapshot_root / "offline_data")
    lines += _realized_lines(snapshot_root / "realized")
    lines += _calendar_lines(snapshot_root / CALENDAR_DIR / VIEW_FILE, store)
    return "\n".join(lines) + "\n"


def _calendar_lines(path: Path, store: Path) -> list[str]:
    """The scheduled events the snapshot froze at its cutoff: how many of each type, and the types
    released at the same minute."""
    if not path.exists():
        return []
    view = pd.read_parquet(path)
    days = sorted(p.name[5:15] for p in store.glob("*/1m/date=*.parquet"))
    if view.empty or not days:
        return []
    end = pd.Timestamp(days[-1], tz="UTC") + pd.Timedelta(days=1)
    lines = [
        "",
        f"**Scheduled events** (`{CALENDAR_DIR}/{VIEW_FILE}`; your champion reads it at `${CALENDAR_ENV}`):",
        f"{len(view)} events, {view['time_utc'].min():%Y-%m-%d} → {view['time_utc'].max():%Y-%m-%d}: every date",
        "published by the cutoff, upcoming ones included. Columns: `time_utc`, `event`, `scope` (`all`, or the",
        "assets it concerns, comma-separated), `detail`, `source`. Times are scheduled New York times in UTC,",
        "so they move an hour with daylight saving: US macro data at 08:30, FOMC statements at 14:00, the EIA",
        "petroleum report Wednesdays at 10:30 (holiday weeks differ). Earnings are anchored at the session,",
        "08:00 before the open and 16:00 after the close; an NYSE closure at the 09:30 open that does not",
        "happen, an early close at 13:00. A date published or moved after the cutoff is not here.",
        "",
        "Events of each type before and after the snapshot's last day, and the other types released at the",
        "same minute, whose effects cannot be told apart from it there:",
        "",
        "| event | before cutoff | after cutoff | at the same time as |",
        "|---|---|---|---|",
    ]
    types_at = view.groupby("time_utc")["event"].agg(set)
    for event, rows in view.groupby("event"):
        past = rows[rows["time_utc"] < end]
        shared = Counter(other for t in rows["time_utc"] for other in types_at[t] - {event})
        same = ", ".join(f"{other} {n}/{len(rows)}" for other, n in sorted(shared.items())) or "—"
        lines.append(f"| {event} | {len(past)} | {len(rows) - len(past)} | {same} |")
    return lines


def _coverage_row(asset_dir: Path) -> str | None:
    per_day: list[tuple[str, int, int]] = []  # (day, untraded_cells, total_cells)
    for part in sorted((asset_dir / "1m").glob("date=*.parquet")):
        try:
            frame = pd.read_parquet(part)
        except Exception:  # unreadable/placeholder partition: not coverage, skip it
            continue
        if "close" not in frame or not frame["close"].notna().any():
            continue
        untraded = frame["close"].isna()
        if "trade_count" in frame:
            untraded |= frame["trade_count"].eq(0)
        per_day.append((part.name[5:-8], int(untraded.sum()), len(frame)))
    if not per_day:
        return None

    def pct(days: list[tuple[str, int, int]]) -> str:
        total = sum(d[2] for d in days)
        return f"{100.0 * sum(d[1] for d in days) / total:.1f}%" if total else "-"

    return (
        f"| {asset_dir.name} | {per_day[0][0]} | {per_day[-1][0]} | {len(per_day)} "
        f"| {pct(per_day)} | {pct(per_day[:30])} | {pct(per_day[-30:])} |"
    )


def _timestamp_lines(store: Path) -> list[str]:
    """The unit the partitions' timestamps load in: an integer view of them counts in it."""
    part = next(iter(sorted(store.glob("*/1m/date=*.parquet"))), None)
    if part is None:
        return []
    try:
        dtype = pd.read_parquet(part, columns=["timestamp"])["timestamp"].dtype
    except Exception:  # unreadable/placeholder partition
        return []
    return [
        "",
        f'`timestamp` loads as `{dtype}`. An integer view of it (`.view("int64")`) counts in that unit:',
        "derive minute indices from the dtype, never by assuming nanoseconds.",
    ]


def _offline_bundle_lines(offline: Path) -> list[str]:
    if not offline.is_dir():
        return []
    spans = []
    # Day partitions, or the single files of a bundle built before them.
    for f in sorted([*offline.glob(f"{SCORES_DIR}/*/*/date=*.parquet"), *offline.glob("miner_scores_*.parquet")]):
        try:
            scored = pd.read_parquet(f, columns=["scored_time"])["scored_time"]
        except Exception:
            continue
        if len(scored):
            spans.append((scored.min(), scored.max()))
    if not spans:
        return []
    lo = min(s[0] for s in spans)
    hi = max(s[1] for s in spans)
    return [
        "",
        "**Offline field-scores bundle** (`SYNTH_BACKTESTER_OFFLINE_DATA_ROOT`): scored",
        f"prompts covered **{lo:%Y-%m-%d} → {hi:%Y-%m-%d}**. Rank backtests outside this",
        "range have no field to compare against — do not query the live API for more.",
    ]


def _realized_lines(realized: Path) -> list[str]:
    """The span of cached validator paths: what a prompt with gaps can fall back on."""
    days = sorted(f.stem[5:] for f in realized.rglob("date=*.parquet"))
    if not days:
        return []
    assets = sorted(p.name for p in realized.iterdir() if p.is_dir())
    lines = [
        "",
        f"**Cached realized paths** ({len(assets)} assets): **{days[0]} → {days[-1]}**. Where a",
        "prompt's slice of the price store contains NaN, scoring substitutes the validator's own",
        "array for that prompt. Outside this range it cannot, and the prompt is scored on the",
        "points that remain — which is a smaller CRPS from a thinner reference, not a better model.",
    ]
    flat = _flat_step_shares(realized)
    if flat:
        lines += [
            "",
            "Steps with no price change in those paths, the arrays CRPS is computed against: "
            + ", ".join(f"{name} {share:.1%}" for name, share in flat)
            + ".",
        ]
    return lines


def _flat_step_shares(realized: Path) -> list[tuple[str, float]]:
    """Per asset and step size, the share of consecutive points of the cached paths at the same price,
    largest first."""
    shares = []
    for prompt_dir in sorted(realized.glob("*/*_*")):
        flat = steps = 0
        for part in prompt_dir.glob("date=*.parquet"):
            try:
                paths = pd.read_parquet(part, columns=["prices"])["prices"]
            except Exception:  # unreadable/placeholder partition
                continue
            for path in paths:
                diffs = np.diff(np.asarray(path, dtype=float))
                diffs = diffs[np.isfinite(diffs)]
                flat += int((diffs == 0).sum())
                steps += diffs.size
        if steps:
            increment = int(prompt_dir.name.split("_")[1])
            shares.append((f"{prompt_dir.parent.name} ({increment // 60} min)", flat / steps))
    return sorted(shares, key=lambda item: -item[1])
