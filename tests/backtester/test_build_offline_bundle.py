"""Tests for the offline-bundle builder — run with: uv run pytest tests/backtester/ -v"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from synth_lib.backtester.scripts.build_offline_bundle import MANIFEST_NAME, build_bundle

UTC = timezone.utc
MODULE = "synth_lib.backtester.scripts.build_offline_bundle"
EVAL_END = datetime(2026, 9, 16, tzinfo=UTC)


def _build(out: Path, days: int, eval_end: datetime = EVAL_END) -> None:
    scores = pd.DataFrame(
        {
            "miner_uid": [1, 2],
            "asset": ["BTC", "BTC"],
            "crps": [1.0, 2.0],
            "scored_time": [eval_end - timedelta(hours=1)] * 2,
            "time_length": [86400, 86400],
            "time_increment": [300, 300],
        }
    )
    with (
        patch(f"{MODULE}.get_miner_scores", return_value=scores),
        patch(f"{MODULE}.get_rewards_history", return_value=pd.DataFrame({"updated_at": [eval_end]})),
        patch(f"{MODULE}.get_daily_miner_pool_usd", return_value=pd.Series([1.0], index=[eval_end.date()])),
    ):
        build_bundle(
            slug="crypto-24h",
            days=days,
            eval_end=eval_end,
            assets=["BTC"],
            chunk_days=60.0,
            out=out,
            realized_paths=False,
        )


class TestBundleWindowManifest:
    def test_every_bundled_file_records_its_window(self, tmp_path):
        _build(tmp_path, days=17)

        manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
        assert set(manifest) == {
            "miner_scores_BTC_crypto-24h.parquet",
            "rewards_history_crypto-24h.parquet",
            "miner_pool_usd.parquet",
        }
        scores = manifest["miner_scores_BTC_crypto-24h.parquet"]
        assert scores["start"].startswith("2026-08-29") and scores["end"].startswith("2026-09-16")

    def test_the_same_window_still_resumes(self, tmp_path):
        """Skipping an already-written file is the point of the bundle; the check must not break it."""
        _build(tmp_path, days=17)
        _build(tmp_path, days=17)  # no exception

    def test_a_narrower_window_is_covered(self, tmp_path):
        _build(tmp_path, days=17)
        _build(tmp_path, days=10)

    def test_a_wider_window_is_refused(self, tmp_path):
        """The filename carries the asset and the competition but not the dates, so without this a
        bundle built for one window scores another over whatever days the two share."""
        _build(tmp_path, days=10)

        with pytest.raises(SystemExit) as excinfo:
            _build(tmp_path, days=17)
        message = str(excinfo.value)
        assert "2026-09-05" in message and "2026-08-29" in message
        assert "fresh --out" in message

    def test_a_bundle_without_a_manifest_is_refused(self, tmp_path):
        """Every bundle built before window tracking looks exactly like a correct one."""
        _build(tmp_path, days=17)
        (tmp_path / MANIFEST_NAME).unlink()

        with pytest.raises(SystemExit, match="predates window tracking"):
            _build(tmp_path, days=17)
