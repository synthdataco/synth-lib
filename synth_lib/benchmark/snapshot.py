"""Freezing market_data up to the cutoff: hard-links + sha256 manifest."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import date
from pathlib import Path

import pandas as pd

from synth_lib.preparation.config import STORE_SUBDIR


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
    bound, i.e. everything up to the cutoff."""
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
        "Each day is one parquet file, `date=YYYY-MM-DD.parquet`, a full 1440-row minute grid.",
        "Only `close` carries real data: the venue's 1-minute candle close (the last trade of",
        "that minute). `open/high/low/volume` columns may exist but are NA in this snapshot.",
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
    lines += _offline_bundle_lines(snapshot_root / "offline_data")
    return "\n".join(lines) + "\n"


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


def verify_snapshot(dest_root: Path) -> bool:
    entries = json.loads((dest_root / "manifest.json").read_text())
    return all(sha256_file(dest_root / rel) == digest for rel, digest in entries.items())
