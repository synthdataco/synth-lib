"""default_store_root: canonical `prices/` directory with a legacy `pyth/` fallback.

The directory was called `pyth` when Pyth was the only source. It never described provenance —
that lives in each partition's `source` column — and after the Pyth exit the name actively misleads
operators into thinking the store is Pyth-backed. Renaming outright would orphan every existing
store, so the default prefers `prices/` and falls back to `pyth/` when that is what is on disk.
"""

from pathlib import Path

from synth_lib.preparation.config import LEGACY_STORE_SUBDIR, STORE_SUBDIR, default_store_root
from synth_lib.preparation.minute_price_store import MinutePriceStore


def test_fresh_store_uses_the_canonical_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert default_store_root("BTC") == Path("market_data") / STORE_SUBDIR / "BTC" / "1m"


def test_existing_legacy_store_is_reused(tmp_path, monkeypatch):
    """An operator with months of ingested data must not have to move it."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "market_data" / LEGACY_STORE_SUBDIR / "BTC" / "1m").mkdir(parents=True)
    assert default_store_root("BTC") == Path("market_data") / LEGACY_STORE_SUBDIR / "BTC" / "1m"


def test_canonical_wins_when_both_exist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "market_data" / LEGACY_STORE_SUBDIR / "BTC" / "1m").mkdir(parents=True)
    (tmp_path / "market_data" / STORE_SUBDIR / "BTC" / "1m").mkdir(parents=True)
    assert default_store_root("BTC") == Path("market_data") / STORE_SUBDIR / "BTC" / "1m"


def test_choice_is_made_per_store_not_per_asset(tmp_path, monkeypatch):
    """A legacy store holding only BTC must still serve a never-ingested asset from the same
    tree — otherwise one ingest run splits the store across both directory names."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "market_data" / LEGACY_STORE_SUBDIR / "BTC" / "1m").mkdir(parents=True)
    assert default_store_root("SP500") == Path("market_data") / LEGACY_STORE_SUBDIR / "SP500" / "1m"


def test_minute_price_store_picks_up_the_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "market_data" / LEGACY_STORE_SUBDIR / "XAU" / "1m").mkdir(parents=True)
    assert MinutePriceStore("XAU").root == Path("market_data") / LEGACY_STORE_SUBDIR / "XAU" / "1m"


def test_explicit_root_still_wins(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "market_data" / LEGACY_STORE_SUBDIR / "XAU" / "1m").mkdir(parents=True)
    assert MinutePriceStore("XAU", root=tmp_path / "elsewhere").root == tmp_path / "elsewhere"
