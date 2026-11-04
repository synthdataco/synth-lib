"""Freezing files up to a cutoff: hard links and a sha256 manifest. Standard library only."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Callable


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


def link_partitions(
    source_root: Path,
    dest_root: Path,
    pattern: str,
    day_of: Callable[[Path], date | None],
    cutoff: date,
    start: date | None = None,
) -> dict[str, str]:
    """Hard-links every file under source_root matching `pattern` whose day is in [start, cutoff], and
    returns its manifest entries, {path relative to source_root: sha256}. `day_of` gives a file's
    day, or None to skip it. A file already linked is kept."""
    manifest: dict[str, str] = {}
    for src in sorted(Path(source_root).rglob(pattern)):
        day = day_of(src)
        if day is None or day > cutoff or (start is not None and day < start):
            continue
        rel = src.relative_to(source_root)
        dst = dest_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.link(src, dst)
        manifest[str(rel)] = sha256_file(dst)
    return manifest


def write_manifest(dest_root: Path, manifest: dict[str, str]) -> None:
    """The manifest.json verify_snapshot checks."""
    (dest_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def verify_snapshot(dest_root: Path) -> bool:
    entries = json.loads((dest_root / "manifest.json").read_text())
    return all(sha256_file(dest_root / rel) == digest for rel, digest in entries.items())
