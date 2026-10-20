"""Freezing market_data up to the cutoff: hard-links + sha256 manifest."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from synth_lib.preparation.config import STORE_SUBDIR
from synth_lib.preparation.event_calendar.events import (
    CALENDAR_DIR,
    CALENDAR_ENV,
    LEDGER_FILE,
    VIEW_FILE,
    concerns,
    known_at,
    read_ledger,
    write_atomic,
)


def sha256_file(path: Path) -> str:
    """SHA256 digest of a file, in chunks.

    Deliberately self-contained: the benchmark depends on NO operator-private code, which keeps it
    portable (server deployment, any git base) and avoids having to run mypy over modules
    outside its scope."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


_PARTITION_RE = re.compile(r"^date=(\d{4}-\d{2}-\d{2})\.parquet$")


def build_snapshot(source_root: Path, dest_root: Path, cutoff: date, start: date | None = None) -> dict[str, str]:
    """Hard-links every partition in [start, cutoff] (all assets) and writes manifest.json.

    `start` decouples what is on disk from what a campaign sees: keep 9 months of ingested
    market_data and snapshot only the 2 months a campaign should train on. None = no lower
    bound, i.e. everything up to the cutoff. A rebuilt snapshot keeps its partitions and its frozen
    calendar."""
    manifest: dict[str, str] = {}
    for src in sorted(Path(source_root).rglob("date=*.parquet")):
        m = _PARTITION_RE.match(src.name)
        if m is None:
            continue
        day = date.fromisoformat(m.group(1))
        if day > cutoff or (start is not None and day < start):
            continue
        rel = src.relative_to(source_root)
        dst = dest_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.link(src, dst)
        manifest[str(rel)] = sha256_file(dst)
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
    (dest_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
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
        "columns describe the same minute. For the Hyperliquid-routed assets open/high/low are",
        "derived from 1-second closes, so a minute's range is sampled rather than continuous.",
        "`source`/`ingested_at`/`is_final` are ingestion provenance — ignore them. Semantics:",
        "",
        "- **NaN close** = no trade that minute (thin market or feed gap). NOT market hours:",
        "  these venues trade 24/7. Real density varies by asset and era — see the table.",
        "- **Missing file** = before that market's first trade. Nothing existed; nothing is hidden.",
        "- Scoring drops NaN per prompt. Never interpolate: inventing prices in the gaps is how",
        "  you fool your own validation.",
        "",
        "Per-asset coverage, measured from this snapshot (nan% = share of NaN minutes among the",
        "asset's real days; first/last 30d columns show density drift — early months of newer",
        "listings are genuinely thin, low realized volatility there is real):",
        "",
        "| asset | first day | last day | real days | nan% total | nan% first 30d | nan% last 30d |",
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


# Asset groups the calendar section measures volatility over.
EVENT_GROUPS = {
    "crypto": ("BTC", "ETH", "SOL", "XRP", "HYPE"),
    "stocks": ("NVDAX", "TSLAX", "AAPLX", "GOOGLX", "SPCX"),
    "XAU": ("XAU",),
    "WTIOIL": ("WTIOIL",),
    "SP500": ("SP500",),
}


def _calendar_lines(path: Path, store: Path) -> list[str]:
    """The scheduled events the snapshot froze at its cutoff, and the volatility each type brought."""
    if not path.exists():
        return []
    view = pd.read_parquet(path)
    days = sorted(p.name[5:15] for p in store.glob("*/1m/date=*.parquet"))
    if view.empty or not days:
        return []
    end = pd.Timestamp(days[-1], tz="UTC") + pd.Timedelta(days=1)
    returns = {asset: _abs_returns(store / asset / "1m") for group in EVENT_GROUPS.values() for asset in group}
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
        "Volatility in the hour after an event, over the same hour a week before and after, median over the",
        "snapshot's events of that type (— where the type concerns no asset of the group):",
        "",
        "| event | before cutoff | after cutoff | " + " | ".join(EVENT_GROUPS) + " |",
        "|---|---|---|" + "---|" * len(EVENT_GROUPS),
    ]
    for event, rows in view.groupby("event"):
        past = rows[rows["time_utc"] < end]
        cells = []
        for assets in EVENT_GROUPS.values():
            ratios = [
                ratio
                for asset in assets
                for t in past.loc[concerns(past, asset), "time_utc"]
                if (ratio := _hour_ratio(returns[asset], t)) is not None
            ]
            cells.append(f"{np.median(ratios):.2f}×" if len(ratios) >= 3 else "—")
        lines.append(f"| {event} | {len(past)} | {len(rows) - len(past)} | " + " | ".join(cells) + " |")
    return lines


def _abs_returns(asset_dir: Path) -> pd.Series:
    """Absolute 1-minute log returns of an asset's closes, NaN where a minute did not trade."""
    frames = []
    for part in sorted(asset_dir.glob("date=*.parquet")):
        try:
            frames.append(pd.read_parquet(part, columns=["timestamp", "close"]))
        except Exception:  # unreadable/placeholder partition
            continue
    if not frames:
        return pd.Series(dtype=float)
    closes = pd.concat(frames, ignore_index=True)
    closes["timestamp"] = pd.to_datetime(closes["timestamp"], utc=True)
    series = closes.set_index("timestamp")["close"].sort_index()
    return np.log(series.where(series > 0)).diff().abs()


def _hour_ratio(returns: pd.Series, t: pd.Timestamp) -> float | None:
    hour = pd.Timedelta(minutes=59)
    week = pd.Timedelta(days=7)
    after = returns.loc[t : t + hour].mean()
    base = pd.concat([returns.loc[t - week : t - week + hour], returns.loc[t + week : t + week + hour]]).mean()
    if not (np.isfinite(after) and np.isfinite(base)) or base <= 0:
        return None
    return float(after / base)


def _coverage_row(asset_dir: Path) -> str | None:
    per_day: list[tuple[str, int, int]] = []  # (day, nan_cells, total_cells)
    for part in sorted((asset_dir / "1m").glob("date=*.parquet")):
        try:
            closes = pd.read_parquet(part, columns=["close"])["close"]
        except Exception:  # unreadable/placeholder partition: not coverage, skip it
            continue
        if closes.notna().any():
            per_day.append((part.name[5:-8], int(closes.isna().sum()), len(closes)))
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
    for f in sorted(offline.glob("miner_scores_*.parquet")):
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


def verify_snapshot(dest_root: Path) -> bool:
    entries = json.loads((dest_root / "manifest.json").read_text())
    return all(sha256_file(dest_root / rel) == digest for rel, digest in entries.items())
