"""Tests for the offline-bundle builder — run with: uv run pytest tests/backtester/ -v"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from synth_lib.backtester.field_store import day_path, pool_dir, rewards_dir, scores_dir
from synth_lib.backtester.loading import get_miner_scores, get_rewards_history
from synth_lib.backtester.scripts.build_offline_bundle import build_bundle

UTC = timezone.utc
MODULE = "synth_lib.backtester.scripts.build_offline_bundle"
EVAL_END = datetime(2026, 9, 16, tzinfo=UTC)


class _Api:
    """Hourly rows over each requested range, recording the requests."""

    def __init__(self):
        self.scores: list[tuple[datetime, datetime]] = []
        self.rewards: list[tuple[datetime, datetime]] = []

    def get_miner_scores(self, start, end, asset, time_length, time_increment):
        self.scores.append((start, end))
        scored = pd.date_range(start, end, freq="1h", inclusive="left")
        return pd.DataFrame(
            {
                "miner_uid": 1,
                "asset": asset,
                "crps": 1.0,
                "scored_time": scored,
                "time_length": time_length,
                "time_increment": time_increment,
                "start_time": scored - pd.Timedelta(seconds=time_length),
            }
        )

    def get_rewards_history(self, start, end, prompt_name=None):
        self.rewards.append((start, end))
        return pd.DataFrame({"updated_at": pd.date_range(start, end, freq="1h", inclusive="left"), "miner_uid": 1})

    def get_daily_miner_pool_usd(self, start, end):
        days = pd.date_range(pd.Timestamp(start).floor("D"), end, freq="1D", inclusive="left")
        return pd.Series(1.0, index=days)


def _build(out: Path, days: int, eval_end: datetime = EVAL_END, force_refresh: bool = False) -> _Api:
    api = _Api()
    with (
        patch(f"{MODULE}.get_miner_scores", side_effect=api.get_miner_scores),
        patch(f"{MODULE}.get_rewards_history", side_effect=api.get_rewards_history),
        patch(f"{MODULE}.get_daily_miner_pool_usd", side_effect=api.get_daily_miner_pool_usd),
        patch(f"{MODULE}.REQUEST_SPACING_SECONDS", 0),
    ):
        build_bundle(
            slug="crypto-24h",
            days=days,
            eval_end=eval_end,
            assets=["BTC"],
            out=out,
            realized_paths=False,
            force_refresh=force_refresh,
        )
    return api


def _days(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.glob("date=*.parquet"))


class TestDayPartitions:
    def test_each_day_lands_in_a_partition_named_by_its_date(self, tmp_path):
        _build(tmp_path, days=2)
        # Scores from one pad hour before the window; the window ends at midnight.
        assert _days(scores_dir(tmp_path, "crypto-24h", "BTC")) == [
            "date=2026-09-13.parquet",
            "date=2026-09-14.parquet",
            "date=2026-09-15.parquet",
        ]
        # Rewards reach 25 h past the window: the last day stops there and is kept partial.
        rewards = _days(rewards_dir(tmp_path, "crypto-24h"))
        assert rewards[0] == "date=2026-09-12.parquet" and rewards[-1] == "date=2026-09-17.partial.parquet"
        assert len(pd.read_parquet(day_path(rewards_dir(tmp_path, "crypto-24h"), date(2026, 9, 17), True))) == 1
        assert _days(pool_dir(tmp_path))[-1] == "date=2026-09-17.partial.parquet"

    def test_the_same_window_fetches_no_complete_day_again(self, tmp_path):
        _build(tmp_path, days=2)
        api = _build(tmp_path, days=2)
        assert api.scores == []
        assert api.rewards == [(datetime(2026, 9, 17, tzinfo=UTC), datetime(2026, 9, 17, 1, tzinfo=UTC))]

    def test_a_wider_window_fetches_only_its_new_days(self, tmp_path):
        _build(tmp_path, days=2)
        api = _build(tmp_path, days=4)
        assert [s.date() for s, _ in api.scores] == [date(2026, 9, 11), date(2026, 9, 12)]

    def test_force_refresh_fetches_every_day_again(self, tmp_path):
        _build(tmp_path, days=2)
        assert len(_build(tmp_path, days=2, force_refresh=True).scores) == 3

    def test_the_loader_reads_the_partitions_of_its_range(self, tmp_path, monkeypatch):
        _build(tmp_path, days=2)
        monkeypatch.setenv("SYNTH_BACKTESTER_OFFLINE_DATA_ROOT", str(tmp_path))
        scores = get_miner_scores(datetime(2026, 9, 14, tzinfo=UTC), EVAL_END, "BTC", 86400, 300)
        assert len(scores) == 48 and scores["scored_time"].min() == pd.Timestamp("2026-09-14", tz="UTC")
        rewards = get_rewards_history(EVAL_END, EVAL_END + timedelta(hours=1), prompt_name="crypto-24h")
        assert len(rewards) == 1

    def test_the_loader_still_reads_a_bundle_of_single_files(self, tmp_path, monkeypatch):
        scored = pd.date_range("2026-09-14", periods=3, freq="1h", tz="UTC")
        pd.DataFrame({"miner_uid": 1, "crps": 1.0, "scored_time": scored}).to_parquet(
            tmp_path / "miner_scores_BTC_crypto-24h.parquet"
        )
        monkeypatch.setenv("SYNTH_BACKTESTER_OFFLINE_DATA_ROOT", str(tmp_path))
        assert len(get_miner_scores(scored[0].to_pydatetime(), EVAL_END, "BTC", 86400, 300)) == 3
