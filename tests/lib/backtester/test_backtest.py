"""Tests for app.lib.backtester.backtest — run with: uv run pytest app/lib/backtester/test_backtest.py -v"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.lib.backtester.backtest import (
    BacktestResult,
    HF_CRPS_FORMULA_CHANGE_DATE,
    _BacktestMinerDataHandler,
    _find_prediction_file,
    _maybe_warn_hf_crps_formula_change,
    _parse_prediction_filename_time,
    _score_single_prompt,
    backtest,
    _compute_prompt_scores_for_group,
    _compute_relative_crps,
    calculate_smoothed_scores,
    compute_combined_smoothed_scores,
    download_price_data,
    load_prediction,
    plot_crps_by_day,
    plot_crps_by_hour,
    plot_crps_over_time,
    plot_crps_ratio_distribution,
    plot_total_rank_evolution,
    plot_weekly_percentile,
)
from synth.validator.prompt_config import HIGH_FREQUENCY, LOW_FREQUENCY

UTC = timezone.utc

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ASSET = "BTC"
TIME_LENGTH = 86_400
TIME_INCREMENT = 300
NUM_STEPS = TIME_LENGTH // TIME_INCREMENT  # 288
NUM_SIMULATIONS = 10
MINER_ID = 999

# Two prompt times, 6 hours apart
T0 = datetime(2026, 3, 28, 0, 0, 0, tzinfo=UTC)
T1 = datetime(2026, 3, 28, 6, 0, 0, tzinfo=UTC)
SCORED_T0 = T0 + timedelta(seconds=TIME_LENGTH)
SCORED_T1 = T1 + timedelta(seconds=TIME_LENGTH)


def _make_prediction(start_time: datetime, price: float = 100_000.0) -> dict:
    """Create a minimal valid prediction dict."""
    paths = np.full((NUM_SIMULATIONS, NUM_STEPS + 1), price).tolist()
    return {
        "start_timestamp": int(start_time.timestamp()),
        "asset": ASSET,
        "time_increment": TIME_INCREMENT,
        "time_length": TIME_LENGTH,
        "num_simulations": NUM_SIMULATIONS,
        "num_steps": NUM_STEPS,
        "paths": paths,
    }


def _make_price_df(start: datetime, end: datetime, price: float = 100_000.0) -> pd.DataFrame:
    """Create a minute-resolution price DataFrame covering start..end."""
    idx = pd.date_range(start - timedelta(hours=3), end + timedelta(hours=3), freq="1min", tz=UTC)
    return pd.DataFrame({"close": price}, index=idx)


def _make_scores_df() -> pd.DataFrame:
    """Fake miner scores as returned by get_miner_scores (two existing miners per prompt)."""
    rows = []
    for scored_time, start_time in [(SCORED_T0, T0), (SCORED_T1, T1)]:
        for uid in [1, 2]:
            rows.append(
                {
                    "miner_uid": uid,
                    "asset": ASSET,
                    "crps": 500.0 + uid * 10,
                    "scored_time": scored_time,
                    "time_length": TIME_LENGTH,
                    "time_increment": TIME_INCREMENT,
                    "start_time": start_time,
                }
            )
    return pd.DataFrame(rows)


def _make_rewards_df() -> pd.DataFrame:
    """Fake rewards history as returned by get_rewards_history."""
    rows = []
    for scored_time in [SCORED_T0, SCORED_T1]:
        updated_at = scored_time + timedelta(minutes=5)
        for uid in [1, 2]:
            rows.append(
                {
                    "miner_uid": uid,
                    "smoothed_score": 100.0,
                    "reward_weight": 0.5,
                    "prompt_name": "low",
                    "updated_at": updated_at,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def predictions_dir(tmp_path: Path) -> Path:
    """Write two prediction files into a temp directory."""
    for t in [T0, T1]:
        fname = t.strftime("%Y-%m-%d_%H:%M:%SZ") + f"_{ASSET}_{TIME_LENGTH}.json"
        (tmp_path / fname).write_text(json.dumps(_make_prediction(t)))
    return tmp_path


@pytest.fixture()
def single_prediction_dir(tmp_path: Path) -> Path:
    """Write only ONE prediction file (T0), leaving T1 without a prediction."""
    fname = T0.strftime("%Y-%m-%d_%H:%M:%SZ") + f"_{ASSET}_{TIME_LENGTH}.json"
    (tmp_path / fname).write_text(json.dumps(_make_prediction(T0)))
    return tmp_path


# ---------------------------------------------------------------------------
# Unit tests — pure functions, no mocking needed
# ---------------------------------------------------------------------------


# Parsing of prediction filenames like "2026-03-28_00:00:00Z_BTC_86400.json" into datetime objects.
class TestParseFilenameTime:
    def test_valid(self) -> None:
        p = Path("2026-03-28_00:00:00Z_BTC_86400.json")
        assert _parse_prediction_filename_time(p) == datetime(2026, 3, 28, 0, 0, 0, tzinfo=UTC)

    def test_invalid_short(self) -> None:
        assert _parse_prediction_filename_time(Path("bad.json")) is None


# Matching a scored prompt time to the closest prediction file (by asset, time_length, and tolerance).
class TestFindPredictionFile:
    def test_outside_tolerance(self, predictions_dir: Path) -> None:
        files = list(predictions_dir.glob("*.json"))
        far_time = T0 + timedelta(hours=2)
        found = _find_prediction_file(files, far_time, ASSET, TIME_LENGTH, tolerance_minutes=30)
        assert found is None

    def test_wrong_asset(self, predictions_dir: Path) -> None:
        files = list(predictions_dir.glob("*.json"))
        assert _find_prediction_file(files, T0, "ETH", TIME_LENGTH) is None

    def test_closest_of_multiple_within_tolerance(self, tmp_path: Path) -> None:
        """When multiple files are within tolerance, the closest should be selected."""
        fname_exact = T0.strftime("%Y-%m-%d_%H:%M:%SZ") + f"_{ASSET}_{TIME_LENGTH}.json"
        t_near = T0 + timedelta(minutes=15)
        fname_near = t_near.strftime("%Y-%m-%d_%H:%M:%SZ") + f"_{ASSET}_{TIME_LENGTH}.json"
        (tmp_path / fname_exact).write_text("{}")
        (tmp_path / fname_near).write_text("{}")

        files = list(tmp_path.glob("*.json"))
        query_time = T0 + timedelta(minutes=5)
        found = _find_prediction_file(files, query_time, ASSET, TIME_LENGTH, tolerance_minutes=30)
        assert found is not None
        assert "00:00:00Z" in found.name


# Loading prediction JSON files in both formats: flat (notebook) and ArtifactManager.
class TestLoadPrediction:
    def test_flat_format(self, tmp_path: Path) -> None:
        pred = _make_prediction(T0)
        path = tmp_path / "pred.json"
        path.write_text(json.dumps(pred))
        loaded = load_prediction(path)
        assert loaded["num_simulations"] == NUM_SIMULATIONS
        assert loaded["num_steps"] == NUM_STEPS
        assert len(loaded["paths"]) == NUM_SIMULATIONS

    def test_artifact_format(self, tmp_path: Path) -> None:
        artifact = {
            "simulation_input": {
                "start_time": T0.isoformat(),
                "asset": ASSET,
                "time_increment": TIME_INCREMENT,
                "time_length": TIME_LENGTH,
                "num_simulations": NUM_SIMULATIONS,
            },
            "prediction": ["meta0", "meta1"] + [[100_000.0] * (NUM_STEPS + 1)] * NUM_SIMULATIONS,
        }
        path = tmp_path / "pred.json"
        path.write_text(json.dumps(artifact))
        loaded = load_prediction(path)
        assert loaded["asset"] == ASSET
        assert len(loaded["paths"]) == NUM_SIMULATIONS


class TestComputePromptScores:
    def test_handles_negative_one(self) -> None:
        crps = pd.Series([100.0, -1, 300.0])
        result = _compute_prompt_scores_for_group(crps)
        # -1 entries get replaced by 90th percentile, then shifted
        assert result.iloc[0] == 0.0

    def test_all_negative_one_returns_zeros(self) -> None:
        """When all scores are -1, compute_prompt_scores returns None; wrapper should return zeros."""
        crps = pd.Series([-1.0, -1.0, -1.0])
        result = _compute_prompt_scores_for_group(crps)
        assert len(result) == 3
        assert all(result == 0.0)


# Fake MinerDataHandler that maps miner_uid == miner_id without requiring a real DB.
class TestBacktestMinerDataHandler:
    def test_identity_mapping(self) -> None:
        handler = _BacktestMinerDataHandler()
        data = [{"miner_id": 1}, {"miner_id": 42}, {"miner_id": 999}]
        result = handler.populate_miner_uid_in_miner_data(data)
        for row in result:
            assert row["miner_uid"] == row["miner_id"]


# Smoothed score computation using synth's compute_smoothed_score:
# per-asset weighting, rolling window, softmax normalization → reward weights.
class TestCalculateSmoothedScores:
    def test_window_cutoff(self) -> None:
        """Scores older than cutoff_days should not contribute."""
        scores = _make_scores_df()
        scores["new_prompt_scores"] = 10.0
        # Add an old score well outside the window
        old = scores.iloc[:1].copy()
        old["scored_time"] = SCORED_T0 - timedelta(days=30)
        old["start_time"] = T0 - timedelta(days=30)
        scores = pd.concat([old, scores], ignore_index=True)

        rewards = _make_rewards_df()
        result_10d = calculate_smoothed_scores(scores, rewards, cutoff_days=10)
        result_60d = calculate_smoothed_scores(scores, rewards, cutoff_days=60)
        # With 60-day window the old score is included, so smoothed scores should be higher
        max_10 = result_10d["new_smoothed_score"].max()
        max_60 = result_60d["new_smoothed_score"].max()
        assert max_60 >= max_10

    def test_handles_miner_id_column_in_input(self) -> None:
        """all_scores may contain both miner_uid and miner_id; should not cause duplicate column errors."""
        scores = _make_scores_df()
        scores["new_prompt_scores"] = 10.0
        scores["miner_id"] = scores["miner_uid"]  # simulate backtest() output
        rewards = _make_rewards_df()
        result = calculate_smoothed_scores(scores, rewards)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Integration tests — mock external I/O, exercise full backtest() pipeline
# ---------------------------------------------------------------------------


FAKE_CRPS = 42.0


# Full backtest() pipeline with mocked external I/O (Synth API, price data, CRPS).
# Verifies that predictions are discovered, scored, and aggregated into a valid BacktestResult.
class TestBacktestIntegration:
    @patch("app.lib.backtester.backtest.calculate_crps_for_miner")
    @patch("app.lib.backtester.backtest.download_price_data")
    @patch("app.lib.backtester.backtest.get_rewards_history")
    @patch("app.lib.backtester.backtest.get_miner_scores")
    def test_full_pipeline(
        self,
        mock_scores: object,
        mock_rewards: object,
        mock_prices: object,
        mock_crps: object,
        predictions_dir: Path,
    ) -> None:
        """End-to-end: result shape, summary contract, column contracts, miner merging, no NaN miner_uids."""
        custom_id = 42
        mock_scores.return_value = _make_scores_df()  # type: ignore[union-attr]
        mock_rewards.return_value = _make_rewards_df()  # type: ignore[union-attr]
        mock_prices.return_value = _make_price_df(T0, SCORED_T1)  # type: ignore[union-attr]
        mock_crps.return_value = (FAKE_CRPS, [{"interval": "stub"}])  # type: ignore[union-attr]

        result = backtest(
            miner_name="test_miner",
            asset=ASSET,
            time_length=TIME_LENGTH,
            n_backtest_days=5,
            miner_id=custom_id,
            predictions_dir=predictions_dir,
        )

        # Type + summary
        assert isinstance(result, BacktestResult)
        assert result.miner_name == "test_miner"
        assert result.summary["miner_id"] == custom_id
        assert result.summary["num_prompts"] > 0
        assert result.summary["mean_crps"] == pytest.approx(FAKE_CRPS)
        assert result.summary["final_smoothed_score"] is not None

        # prompt_df contract
        expected_cols = {
            "miner_uid", "crps", "scored_time", "asset",
            "time_length", "time_increment", "start_time", "new_prompt_scores",
        }
        assert expected_cols.issubset(set(result.prompt_df.columns))
        assert result.prompt_df["miner_uid"].isna().sum() == 0
        # LLM miner merged with existing miners 1, 2
        assert {1, 2, custom_id}.issubset(set(result.prompt_df["miner_uid"].unique()))

        # smoothed_scores contract
        assert not result.smoothed_scores.empty
        assert "reward_weight" in result.smoothed_scores.columns
        assert result.smoothed_scores["miner_uid"].isna().sum() == 0
        miner_rw = result.smoothed_scores.loc[result.smoothed_scores["miner_uid"] == custom_id, "reward_weight"]
        assert len(miner_rw) > 0 and all(miner_rw > 0)

    # -- Partial predictions fill with crps=-1 --

    @patch("app.lib.backtester.backtest.calculate_crps_for_miner")
    @patch("app.lib.backtester.backtest.download_price_data")
    @patch("app.lib.backtester.backtest.get_rewards_history")
    @patch("app.lib.backtester.backtest.get_miner_scores")
    def test_partial_predictions_fill_crps_minus_one(
        self,
        mock_scores: object,
        mock_rewards: object,
        mock_prices: object,
        mock_crps: object,
        single_prediction_dir: Path,
    ) -> None:
        """When only some scoring windows have predictions, missing ones get crps=-1."""
        mock_scores.return_value = _make_scores_df()  # type: ignore[union-attr]
        mock_rewards.return_value = _make_rewards_df()  # type: ignore[union-attr]
        mock_prices.return_value = _make_price_df(T0, SCORED_T1)  # type: ignore[union-attr]
        mock_crps.return_value = (FAKE_CRPS, [{"interval": "stub"}])  # type: ignore[union-attr]

        result = backtest(
            miner_name="test_miner",
            asset=ASSET,
            time_length=TIME_LENGTH,
            n_backtest_days=5,
            miner_id=MINER_ID,
            predictions_dir=single_prediction_dir,
        )

        miner_scores = result.prompt_df.loc[result.prompt_df["miner_uid"] == MINER_ID]
        assert (miner_scores["crps"] != -1).any(), "Expected at least one valid CRPS score"
        assert result.summary["num_prompts"] >= 1

    # -- Task 5: Price data gap raises ValueError --

    @patch("app.lib.backtester.backtest.calculate_crps_for_miner")
    @patch("app.lib.backtester.backtest.download_price_data")
    @patch("app.lib.backtester.backtest.get_rewards_history")
    @patch("app.lib.backtester.backtest.get_miner_scores")
    def test_price_data_gap_raises_valueerror(
        self,
        mock_scores: object,
        mock_rewards: object,
        mock_prices: object,
        mock_crps: object,
        predictions_dir: Path,
    ) -> None:
        """When price data has gaps (wrong number of steps), ValueError is raised."""
        mock_scores.return_value = _make_scores_df()  # type: ignore[union-attr]
        mock_rewards.return_value = _make_rewards_df()  # type: ignore[union-attr]
        short_idx = pd.date_range(T0, periods=10, freq="1min", tz=UTC)
        mock_prices.return_value = pd.DataFrame({"close": 100_000.0}, index=short_idx)  # type: ignore[union-attr]
        mock_crps.return_value = (42.0, [])  # type: ignore[union-attr]

        with pytest.raises(ValueError, match="Price data length mismatch"):
            backtest(
                miner_name="test_miner",
                asset=ASSET,
                time_length=TIME_LENGTH,
                n_backtest_days=5,
                miner_id=MINER_ID,
                predictions_dir=predictions_dir,
            )


# ---------------------------------------------------------------------------
# Plot smokes: one parametrized class covers every single-result plot_* function
# ---------------------------------------------------------------------------


def _make_spread_result() -> BacktestResult:
    """Multi-week, multi-day, multi-hour BacktestResult sufficient for every plot_* smoke."""
    rows = []
    base = datetime(2026, 3, 9, 0, 0, 0, tzinfo=UTC)  # Monday, ISO week 11
    times = [base + timedelta(days=i, hours=(i * 5) % 24) for i in range(10)]
    scored_times = [t + timedelta(seconds=TIME_LENGTH) for t in times]
    for st in scored_times:
        for uid in range(1, 6):
            rows.append({
                "miner_uid": uid,
                "asset": ASSET,
                "crps": 400.0 + uid * 40,
                "scored_time": st,
                "time_length": TIME_LENGTH,
                "time_increment": TIME_INCREMENT,
            })
        rows.append({
            "miner_uid": MINER_ID,
            "asset": ASSET,
            "crps": 500.0,
            "scored_time": st,
            "time_length": TIME_LENGTH,
            "time_increment": TIME_INCREMENT,
        })
    return BacktestResult(
        miner_name="test_miner",
        prompt_df=pd.DataFrame(rows),
        smoothed_scores=pd.DataFrame(),
        summary={
            "miner_id": MINER_ID,
            "miner_name": "test_miner",
            "num_prompts": len(scored_times),
            "mean_crps": 500.0,
            "final_smoothed_score": None,
        },
    )


SIMPLE_PLOT_FNS = [
    (plot_crps_over_time, "crps_over_time"),
    (plot_crps_by_hour, "crps_by_hour"),
    (plot_crps_by_day, "crps_by_day"),
    (plot_crps_ratio_distribution, "crps_ratio_dist"),
    (plot_weekly_percentile, "weekly_percentile"),
]


class TestPlotSmokes:
    """One parametrized class covers every plot_* function that consumes a BacktestResult."""

    @pytest.mark.parametrize("plot_fn,expected_fragment", SIMPLE_PLOT_FNS)
    def test_saves_png(self, plot_fn, expected_fragment: str, tmp_path: Path) -> None:
        result = _make_spread_result()
        chart_path = plot_fn(result, output_dir=tmp_path)
        assert chart_path.exists()
        assert chart_path.suffix == ".png"
        assert expected_fragment in chart_path.name


# ---------------------------------------------------------------------------
# Helper: _compute_relative_crps
# ---------------------------------------------------------------------------


class TestComputeRelativeCrps:
    """Tests for _compute_relative_crps helper."""

    def _make_result_with_spread(self) -> BacktestResult:
        """Build a BacktestResult with 7 scored prompts spanning multiple days/hours/weeks."""
        rows = []
        base = datetime(2026, 3, 23, 0, 0, 0, tzinfo=UTC)  # Monday
        times = [
            base,  # Mon 00:00, week 13
            base + timedelta(hours=6),  # Mon 06:00, week 13
            base + timedelta(days=1, hours=12),  # Tue 12:00, week 13
            base + timedelta(days=2, hours=18),  # Wed 18:00, week 13
            base + timedelta(days=5, hours=3),  # Sat 03:00, week 13
            base + timedelta(days=7, hours=9),  # Mon 09:00, week 14
            base + timedelta(days=8, hours=15),  # Tue 15:00, week 14
        ]
        scored_times = [t + timedelta(seconds=TIME_LENGTH) for t in times]

        for st in scored_times:
            # 5 other miners with CRPS from 400 to 600
            for uid in range(1, 6):
                rows.append(
                    {
                        "miner_uid": uid,
                        "asset": ASSET,
                        "crps": 400.0 + uid * 40,
                        "scored_time": st,
                        "time_length": TIME_LENGTH,
                        "time_increment": TIME_INCREMENT,
                    }
                )
            # Our miner — CRPS of 500 (equal to uid=2.5, so roughly median)
            rows.append(
                {
                    "miner_uid": MINER_ID,
                    "asset": ASSET,
                    "crps": 500.0,
                    "scored_time": st,
                    "time_length": TIME_LENGTH,
                    "time_increment": TIME_INCREMENT,
                }
            )
        return BacktestResult(
            miner_name="test_miner",
            prompt_df=pd.DataFrame(rows),
            smoothed_scores=pd.DataFrame(),
            summary={
                "miner_id": MINER_ID,
                "miner_name": "test_miner",
                "num_prompts": 7,
                "mean_crps": 500.0,
                "final_smoothed_score": None,
            },
        )

    def test_crps_ratio_computation(self) -> None:
        result = self._make_result_with_spread()
        df = _compute_relative_crps(result)
        # Others: 440, 480, 520, 560, 600 → median = 520. Ours = 500.
        expected_ratio = 500.0 / 520.0
        assert df["crps_ratio"].iloc[0] == pytest.approx(expected_ratio)

    def test_filters_crps_negative_one(self) -> None:
        result = self._make_result_with_spread()
        # Set one prompt to crps=-1 for our miner
        mask = result.prompt_df["miner_uid"] == MINER_ID
        first_idx = result.prompt_df.loc[mask].index[0]
        result.prompt_df.loc[first_idx, "crps"] = -1
        df = _compute_relative_crps(result)
        assert len(df) == 6  # one fewer row

    def test_time_columns_derived_correctly(self) -> None:
        result = self._make_result_with_spread()
        df = _compute_relative_crps(result)
        first = df.sort_values("scored_time").iloc[0]
        # First scored_time is 2026-03-24 00:00 UTC (base + 86400s)
        assert first["hour"] == 0
        assert first["day_of_week"] == 1  # Tuesday (March 24, 2026)


# ---------------------------------------------------------------------------
# Task 1: _score_single_prompt worker function
# ---------------------------------------------------------------------------


class TestScoreSinglePrompt:
    """Unit tests for the parallelizable prompt-scoring worker."""

    def test_returns_crps_for_valid_prediction(self, predictions_dir: Path) -> None:
        """Worker loads prediction from disk, computes CRPS, returns result dict."""
        fname = T0.strftime("%Y-%m-%d_%H:%M:%SZ") + f"_{ASSET}_{TIME_LENGTH}.json"
        file_path = predictions_dir / fname
        real_prices = [100_000.0] * (NUM_STEPS + 1)

        result = _score_single_prompt(
            file_path=file_path,
            start_time=T0,
            asset_val=ASSET,
            scored_time=SCORED_T0,
            time_len=TIME_LENGTH,
            time_incr=TIME_INCREMENT,
            real_prices=real_prices,
            scoring_intervals={"5min": 300, "30min": 1800, "3hour": 10800, "24hour_abs": 86400},
            miner_id=MINER_ID,
        )

        assert result["crps"] != -1
        assert result["asset"] == ASSET
        assert result["start_time"] == T0
        assert result["scored_time"] == SCORED_T0
        assert result["miner_uid"] == MINER_ID

    def test_returns_negative_one_when_no_file(self) -> None:
        """Missing prediction files get crps=-1."""
        result = _score_single_prompt(
            file_path=None,
            start_time=T0,
            asset_val=ASSET,
            scored_time=SCORED_T0,
            time_len=TIME_LENGTH,
            time_incr=TIME_INCREMENT,
            real_prices=[],
            scoring_intervals={},
            miner_id=MINER_ID,
        )

        assert result["crps"] == -1


# ---------------------------------------------------------------------------
# Task 2: Parallel scoring — backtest() with scoring_executor
# ---------------------------------------------------------------------------


class TestBacktestParallelScoring:
    """Verify backtest() produces identical results with and without scoring_executor."""

    @patch("app.lib.backtester.backtest.download_price_data")
    @patch("app.lib.backtester.backtest.get_rewards_history")
    @patch("app.lib.backtester.backtest.get_miner_scores")
    def test_parallel_matches_sequential(
        self,
        mock_scores: object,
        mock_rewards: object,
        mock_prices: object,
        predictions_dir: Path,
    ) -> None:
        """backtest() with scoring_executor produces the same summary as without."""
        mock_scores.return_value = _make_scores_df()
        mock_rewards.return_value = _make_rewards_df()
        mock_prices.return_value = _make_price_df(T0, SCORED_T1)

        result_seq = backtest(
            miner_name="test_miner",
            asset=ASSET,
            time_length=TIME_LENGTH,
            n_backtest_days=5,
            miner_id=MINER_ID,
            predictions_dir=predictions_dir,
            scoring_executor=None,
        )

        mock_scores.return_value = _make_scores_df()
        mock_rewards.return_value = _make_rewards_df()
        mock_prices.return_value = _make_price_df(T0, SCORED_T1)

        with ProcessPoolExecutor(max_workers=2) as executor:
            result_par = backtest(
                miner_name="test_miner",
                asset=ASSET,
                time_length=TIME_LENGTH,
                n_backtest_days=5,
                miner_id=MINER_ID,
                predictions_dir=predictions_dir,
                scoring_executor=executor,
            )

        assert result_seq.summary["num_prompts"] == result_par.summary["num_prompts"]
        assert result_seq.summary["mean_crps"] == pytest.approx(result_par.summary["mean_crps"])


# ---------------------------------------------------------------------------
# download_price_data — per-asset parquet routing
# ---------------------------------------------------------------------------


class TestDownloadPriceDataPerAsset:
    """download_price_data must read from market_data/pyth/{asset}/1m, not a hardcoded path."""

    @staticmethod
    def _write_partition(root: Path, day: datetime, price: float) -> None:
        root.mkdir(parents=True, exist_ok=True)
        idx = pd.date_range(day.replace(hour=0, minute=0), day.replace(hour=23, minute=59), freq="1min", tz=UTC)
        df = pd.DataFrame(
            {
                "timestamp": idx,
                "close": price,
                "source": "test",
                "ingested_at": idx,
                "is_final": True,
            }
        )
        df.to_parquet(root / f"date={day.date().isoformat()}.parquet", index=False)

    def test_loads_eth_prices_not_btc(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Requesting ETH must return ETH prices even when a BTC partition exists for the same date."""
        monkeypatch.chdir(tmp_path)
        day = datetime(2026, 3, 20, tzinfo=UTC)

        btc_root = tmp_path / "market_data" / "pyth" / "BTC" / "1m"
        eth_root = tmp_path / "market_data" / "pyth" / "ETH" / "1m"
        self._write_partition(btc_root, day, price=70_000.0)
        self._write_partition(eth_root, day, price=2_000.0)

        start = day.replace(hour=1)
        end = day.replace(hour=2)
        result = download_price_data(start, end, "ETH", freq="1min")

        assert float(result["close"].iloc[0]) == pytest.approx(2_000.0)
        assert float(result["close"].iloc[-1]) == pytest.approx(2_000.0)


# ---------------------------------------------------------------------------
# get_daily_miner_pool_usd
# ---------------------------------------------------------------------------


class TestGetDailyMinerPoolUsd:
    """Parses /rewards/historical response into a date-indexed USD series."""

    @patch("app.lib.backtester.backtest._http_get")
    def test_empty_response_returns_empty_series(self, mock_get: object) -> None:
        class FakeResponse:
            status_code = 200

            def raise_for_status(self) -> None:
                pass

            def json(self) -> list[dict]:
                return []

        mock_get.return_value = FakeResponse()  # type: ignore[attr-defined]

        from app.lib.backtester.backtest import get_daily_miner_pool_usd

        result = get_daily_miner_pool_usd(
            datetime(2026, 4, 10, tzinfo=UTC),
            datetime(2026, 4, 12, tzinfo=UTC),
        )

        assert isinstance(result, pd.Series)
        assert len(result) == 0

    @patch("app.lib.backtester.backtest._http_get")
    def test_paginates_multi_chunk_window(self, mock_get: object) -> None:
        """A 14-day window paginates into 3 chunks (6 + 6 + 2 days); cursor advances and rows merge."""

        class FakeResponse:
            def __init__(self, rows: list[dict]) -> None:
                self.status_code = 200
                self._rows = rows

            def raise_for_status(self) -> None:
                pass

            def json(self) -> list[dict]:
                return self._rows

        def _row(date_str: str, usd: float) -> dict:
            return {
                "date": f"{date_str}T00:00:00Z",
                "rewards": [{"asset": "USD", "amount": usd}],
            }

        mock_get.side_effect = [  # type: ignore[attr-defined]
            FakeResponse([_row("2026-04-01", 1000.0), _row("2026-04-06", 1500.0)]),
            FakeResponse([_row("2026-04-07", 2000.0), _row("2026-04-12", 2500.0)]),
            FakeResponse([_row("2026-04-13", 3000.0), _row("2026-04-14", 3500.0)]),
        ]

        from app.lib.backtester.backtest import get_daily_miner_pool_usd

        result = get_daily_miner_pool_usd(
            datetime(2026, 4, 1, tzinfo=UTC),
            datetime(2026, 4, 15, tzinfo=UTC),
        )

        # All three chunks were fetched
        assert mock_get.call_count == 3  # type: ignore[attr-defined]

        # Cursor advances correctly across chunks — verify each call's from/to params
        params_per_call = [c.kwargs["params"] for c in mock_get.call_args_list]  # type: ignore[attr-defined]
        assert params_per_call[0]["from"] == "2026-04-01T00:00:00Z"
        assert params_per_call[0]["to"] == "2026-04-07T00:00:00Z"
        assert params_per_call[1]["from"] == "2026-04-07T00:00:00Z"
        assert params_per_call[1]["to"] == "2026-04-13T00:00:00Z"
        assert params_per_call[2]["from"] == "2026-04-13T00:00:00Z"
        assert params_per_call[2]["to"] == "2026-04-15T00:00:00Z"

        # All 6 rows across the 3 chunks merged into a sorted Series
        assert len(result) == 6
        assert list(result.index) == [
            pd.Timestamp(d, tz="UTC")
            for d in [
                "2026-04-01",
                "2026-04-06",
                "2026-04-07",
                "2026-04-12",
                "2026-04-13",
                "2026-04-14",
            ]
        ]
        assert result.loc[pd.Timestamp("2026-04-01", tz="UTC")] == pytest.approx(1000.0)
        assert result.loc[pd.Timestamp("2026-04-14", tz="UTC")] == pytest.approx(3500.0)


# ---------------------------------------------------------------------------
# compute_combined_smoothed_scores
# ---------------------------------------------------------------------------


class TestComputeCombinedSmoothedScores:
    """Combined-across-assets smoothed scores via synth.compute_smoothed_score.

    Verifies asset coefficients ARE applied: a miner who performs well on XAU
    (coef 1.74) should beat a miner who performs well on BTC (coef 1.0) when
    both have identical paired high/low prompt_scores.
    """

    @staticmethod
    def _make_result(
        asset: str,
        scores_per_miner: dict[int, float],
        scored_time: datetime,
        updated_at: pd.Timestamp,
    ) -> BacktestResult:
        rows = [
            {
                "miner_uid": uid,
                "asset": asset,
                "crps": val,
                "new_prompt_scores": val,
                "scored_time": scored_time,
                "time_length": TIME_LENGTH,
                "time_increment": TIME_INCREMENT,
                "start_time": scored_time - timedelta(seconds=TIME_LENGTH),
            }
            for uid, val in scores_per_miner.items()
        ]
        smoothed = pd.DataFrame([{"updated_at": updated_at}])
        return BacktestResult(
            miner_name="t",
            prompt_df=pd.DataFrame(rows),
            smoothed_scores=smoothed,
            summary={
                "miner_id": 999,
                "miner_name": "t",
                "num_prompts": len(rows),
                "mean_crps": 1.0,
                "final_smoothed_score": None,
            },
        )

    def test_xau_dominance_over_btc(self) -> None:
        """Miner B (good on high-coef XAU) beats Miner A (good on low-coef BTC)."""
        from app.lib.backtester.backtest import compute_combined_smoothed_scores

        updated = pd.Timestamp("2026-04-15 12:00:00", tz="UTC")
        st = datetime(2026, 4, 15, 10, 0, 0, tzinfo=UTC)

        r_btc = self._make_result("BTC", {1: 100.0, 2: 500.0}, st, updated)
        r_xau = self._make_result("XAU", {1: 500.0, 2: 100.0}, st, updated)

        combined = compute_combined_smoothed_scores([r_btc, r_xau])

        rows_at_updated = combined.loc[combined["updated_at"] == updated]
        assert len(rows_at_updated) == 2

        rw = rows_at_updated.set_index("miner_uid")["reward_weight"]
        # Lower smoothed_score is better; XAU weight (1.74) means miner 2 wins
        assert rw.loc[2] > rw.loc[1], (
            f"Expected miner 2 (good on XAU coef 1.74) to beat miner 1 " f"(good on BTC coef 1.0). Got {rw.to_dict()}"
        )

    def test_output_columns_and_sums(self) -> None:
        """reward_weight across miners at a single timestamp sums to
        smoothed_score_coefficient (0.5 for LOW_FREQUENCY)."""
        from synth.validator.prompt_config import LOW_FREQUENCY
        from app.lib.backtester.backtest import compute_combined_smoothed_scores

        updated = pd.Timestamp("2026-04-15 12:00:00", tz="UTC")
        st = datetime(2026, 4, 15, 10, 0, 0, tzinfo=UTC)
        r_btc = self._make_result("BTC", {1: 100.0, 2: 500.0, 3: 300.0}, st, updated)

        combined = compute_combined_smoothed_scores([r_btc])

        rw_sum = combined.loc[combined["updated_at"] == updated, "reward_weight"].sum()
        assert rw_sum == pytest.approx(LOW_FREQUENCY.smoothed_score_coefficient, abs=1e-6)


# ---------------------------------------------------------------------------
# plot_total_rank_evolution — consumes combined smoothed-scores DataFrame
# ---------------------------------------------------------------------------


class TestPlotTotalRankEvolution:
    """Smoke test for the cross-asset rank chart driven by compute_combined_smoothed_scores."""

    @staticmethod
    def _make_result_with_crps(asset: str, miner_id: int) -> BacktestResult:
        """Minimal BacktestResult with both prompt_df (CRPS) and smoothed_scores timestamps."""
        st0 = datetime(2026, 4, 15, 10, 0, 0, tzinfo=UTC)
        st1 = datetime(2026, 4, 15, 11, 0, 0, tzinfo=UTC)
        ts0 = pd.Timestamp("2026-04-15 12:00:00", tz="UTC")
        ts1 = pd.Timestamp("2026-04-15 13:00:00", tz="UTC")

        prompt_rows = []
        for st in [st0, st1]:
            for uid in [1, 2, miner_id]:
                prompt_rows.append(
                    {
                        "miner_uid": uid,
                        "asset": asset,
                        "crps": 100.0 + uid,
                        "new_prompt_scores": 100.0 + uid,
                        "scored_time": st,
                        "time_length": TIME_LENGTH,
                        "time_increment": TIME_INCREMENT,
                        "start_time": st - timedelta(seconds=TIME_LENGTH),
                    }
                )

        smoothed = pd.DataFrame([{"updated_at": ts0}, {"updated_at": ts1}])
        return BacktestResult(
            miner_name="test_miner",
            prompt_df=pd.DataFrame(prompt_rows),
            smoothed_scores=smoothed,
            summary={
                "miner_id": miner_id,
                "miner_name": "test_miner",
                "num_prompts": len(prompt_rows),
                "mean_crps": 100.0,
                "final_smoothed_score": None,
            },
        )

    def test_saves_png(self, tmp_path: Path) -> None:
        r_btc = self._make_result_with_crps("BTC", MINER_ID)
        r_eth = self._make_result_with_crps("ETH", MINER_ID)

        combined = compute_combined_smoothed_scores([r_btc, r_eth])
        assert not combined.empty

        chart_path = plot_total_rank_evolution(
            [r_btc, r_eth],
            combined=combined,
            profile_label="low",
            output_dir=tmp_path,
        )
        assert chart_path.exists()
        assert chart_path.suffix == ".png"
        assert "rank_evolution_TOTAL_low" in chart_path.name


# ---------------------------------------------------------------------------
# _compute_earnings_df — per-round USD formula
# ---------------------------------------------------------------------------


class TestComputeEarningsDf:
    """usd_per_round = share_of_round * daily_pool / rounds_per_day."""

    def test_formula(self) -> None:
        """Two rounds on day 1 sharing a $5000 pool, one round on day 2 with $4000 pool."""
        from synth.validator.prompt_config import LOW_FREQUENCY
        from app.lib.backtester.backtest import _compute_earnings_df

        combined = pd.DataFrame(
            [
                {
                    "updated_at": pd.Timestamp("2026-04-10 08:00", tz="UTC"),
                    "miner_uid": 999,
                    "new_smoothed_score": 1.0,
                    "reward_weight": 0.1,
                },
                {
                    "updated_at": pd.Timestamp("2026-04-10 16:00", tz="UTC"),
                    "miner_uid": 999,
                    "new_smoothed_score": 1.0,
                    "reward_weight": 0.2,
                },
                {
                    "updated_at": pd.Timestamp("2026-04-11 08:00", tz="UTC"),
                    "miner_uid": 999,
                    "new_smoothed_score": 1.0,
                    "reward_weight": 0.05,
                },
            ]
        )
        daily_pool = pd.Series(
            {
                pd.Timestamp("2026-04-10", tz="UTC"): 5000.0,
                pd.Timestamp("2026-04-11", tz="UTC"): 4000.0,
            }
        )

        earnings = _compute_earnings_df(
            combined,
            miner_id=999,
            prompt_config=LOW_FREQUENCY,
            daily_pool_usd=daily_pool,
        )

        # usd_per_round = reward_weight * daily_pool / rounds_per_day
        # (reward_weight is already the profile's on-chain share — it sums to
        # smoothed_score_coefficient across miners per round, matching synth's
        # combine_moving_averages.)
        # Day 1: 2 rounds, pool $5000
        #   r1: reward_weight 0.1, usd = 0.1 * 5000 / 2 = 250
        #   r2: reward_weight 0.2, usd = 0.2 * 5000 / 2 = 500
        # Day 2: 1 round, pool $4000
        #   r3: reward_weight 0.05, usd = 0.05 * 4000 / 1 = 200
        assert earnings["usd_per_round"].tolist() == pytest.approx([250.0, 500.0, 200.0])
        assert earnings["usd_cumulative"].tolist() == pytest.approx([250.0, 750.0, 950.0])

    def test_drops_rounds_on_missing_pool_days(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Rounds on dates without pool data are dropped with a stdout warning."""
        from synth.validator.prompt_config import LOW_FREQUENCY
        from app.lib.backtester.backtest import _compute_earnings_df

        combined = pd.DataFrame(
            [
                {
                    "updated_at": pd.Timestamp("2026-04-10 08:00", tz="UTC"),
                    "miner_uid": 999,
                    "new_smoothed_score": 1.0,
                    "reward_weight": 0.1,
                },
                {
                    "updated_at": pd.Timestamp("2026-04-11 08:00", tz="UTC"),
                    "miner_uid": 999,
                    "new_smoothed_score": 1.0,
                    "reward_weight": 0.2,
                },
                {
                    "updated_at": pd.Timestamp("2026-04-12 08:00", tz="UTC"),
                    "miner_uid": 999,
                    "new_smoothed_score": 1.0,
                    "reward_weight": 0.15,
                },
            ]
        )
        # 2026-04-11 pool missing (API gap)
        daily_pool = pd.Series(
            {
                pd.Timestamp("2026-04-10", tz="UTC"): 5000.0,
                pd.Timestamp("2026-04-12", tz="UTC"): 4000.0,
            }
        )

        earnings = _compute_earnings_df(
            combined,
            miner_id=999,
            prompt_config=LOW_FREQUENCY,
            daily_pool_usd=daily_pool,
        )

        # The 2026-04-11 round is dropped; only the two days with pool data remain.
        assert len(earnings) == 2
        assert set(earnings["date"].dt.strftime("%Y-%m-%d")) == {"2026-04-10", "2026-04-12"}
        # Cumulative totals reflect ONLY days with pool data.
        # usd_per_round = reward_weight * daily_pool / rounds_per_day
        #   2026-04-10: reward_weight 0.1, usd = 0.1 * 5000 / 1 = 500
        #   2026-04-12: reward_weight 0.15, usd = 0.15 * 4000 / 1 = 600
        assert earnings["usd_per_round"].tolist() == pytest.approx([500.0, 600.0])
        assert earnings["usd_cumulative"].iloc[-1] == pytest.approx(1100.0)

        captured = capsys.readouterr()
        assert "2026-04-11" in captured.out
        assert "dropping" in captured.out.lower()

# ---------------------------------------------------------------------------
# plot_estimated_earnings
# ---------------------------------------------------------------------------


class TestPlotEstimatedEarnings:
    """Smoke test: three-panel chart saves a PNG and includes the expected label."""

    @staticmethod
    def _make_result_with_crps(asset: str, miner_id: int) -> BacktestResult:
        st0 = datetime(2026, 4, 15, 10, 0, 0, tzinfo=UTC)
        st1 = datetime(2026, 4, 16, 10, 0, 0, tzinfo=UTC)
        ts0 = pd.Timestamp("2026-04-15 12:00:00", tz="UTC")
        ts1 = pd.Timestamp("2026-04-16 12:00:00", tz="UTC")

        rows = []
        for st in [st0, st1]:
            for uid in [1, 2, miner_id]:
                rows.append(
                    {
                        "miner_uid": uid,
                        "asset": asset,
                        "crps": 100.0 + uid,
                        "new_prompt_scores": 100.0 + uid,
                        "scored_time": st,
                        "time_length": TIME_LENGTH,
                        "time_increment": TIME_INCREMENT,
                        "start_time": st - timedelta(seconds=TIME_LENGTH),
                    }
                )
        smoothed = pd.DataFrame([{"updated_at": ts0}, {"updated_at": ts1}])
        return BacktestResult(
            miner_name="test_miner",
            prompt_df=pd.DataFrame(rows),
            smoothed_scores=smoothed,
            summary={
                "miner_id": miner_id,
                "miner_name": "test_miner",
                "num_prompts": len(rows),
                "mean_crps": 100.0,
                "final_smoothed_score": None,
            },
        )

    @patch("app.lib.backtester.backtest.get_daily_miner_pool_usd")
    def test_saves_png(self, mock_pool: object, tmp_path: Path) -> None:
        from synth.validator.prompt_config import LOW_FREQUENCY
        from app.lib.backtester.backtest import (
            compute_combined_smoothed_scores,
            plot_estimated_earnings,
        )

        mock_pool.return_value = pd.Series(
            {  # type: ignore[attr-defined]
                pd.Timestamp("2026-04-15", tz="UTC"): 5000.0,
                pd.Timestamp("2026-04-16", tz="UTC"): 4000.0,
            }
        )

        r_btc = self._make_result_with_crps("BTC", MINER_ID)
        r_eth = self._make_result_with_crps("ETH", MINER_ID)
        combined = compute_combined_smoothed_scores([r_btc, r_eth])

        chart_path = plot_estimated_earnings(
            [r_btc, r_eth],
            prompt_config=LOW_FREQUENCY,
            combined=combined,
            output_dir=tmp_path,
        )

        assert chart_path.exists()
        assert chart_path.suffix == ".png"
        assert "estimated_earnings_low" in chart_path.name

    @patch("app.lib.backtester.backtest.get_daily_miner_pool_usd")
    def test_partial_coverage_warning(self, mock_pool: object, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Backtesting fewer assets than the profile's asset_list prints a partial-coverage tag."""
        from synth.validator.prompt_config import LOW_FREQUENCY
        from app.lib.backtester.backtest import (
            compute_combined_smoothed_scores,
            plot_estimated_earnings,
        )

        mock_pool.return_value = pd.Series(
            {  # type: ignore[attr-defined]
                pd.Timestamp("2026-04-15", tz="UTC"): 5000.0,
                pd.Timestamp("2026-04-16", tz="UTC"): 4000.0,
            }
        )

        # LOW_FREQUENCY has 12 assets; we backtest only 2 → partial coverage
        r_btc = self._make_result_with_crps("BTC", MINER_ID)
        r_eth = self._make_result_with_crps("ETH", MINER_ID)
        combined = compute_combined_smoothed_scores([r_btc, r_eth])

        plot_estimated_earnings(
            [r_btc, r_eth],
            prompt_config=LOW_FREQUENCY,
            combined=combined,
            output_dir=tmp_path,
        )

        captured = capsys.readouterr()
        total_assets = len(LOW_FREQUENCY.asset_list)
        assert f"PARTIAL COVERAGE (2/{total_assets} assets)" in captured.out


class TestHFCrpsFormulaWarning:
    """Gate logic for the 2026-03-11 HF CRPS formula change warning."""

    CUTOFF = HF_CRPS_FORMULA_CHANGE_DATE  # 2026-03-11
    SAFE_SIM_REG = CUTOFF + timedelta(days=HIGH_FREQUENCY.window_days)  # 2026-03-14

    def _call(self, **kwargs):
        defaults = dict(
            prompt_config=HIGH_FREQUENCY,
            n_backtest_days=7,
            eval_end=None,
            simulate_registration=None,
            simulate_deregistration=None,
        )
        defaults.update(kwargs)
        return _maybe_warn_hf_crps_formula_change(**defaults)

    def _assert_warns(self, **kwargs) -> str:
        with pytest.warns(UserWarning, match="HIGH_FREQUENCY backtest window starts") as record:
            self._call(**kwargs)
        assert len(record) == 1
        return str(record[0].message)

    def _assert_silent(self, **kwargs) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            self._call(**kwargs)

    # 1. HF + window pre-cutoff → WARN
    def test_hf_window_pre_cutoff_warns(self) -> None:
        msg = self._assert_warns(eval_end=datetime(2026, 3, 1, tzinfo=UTC), n_backtest_days=30)
        assert "2026-03-11" in msg
        assert "simulate_registration=2026-03-14" in msg

    # 2. HF + window post-cutoff → silence
    def test_hf_window_post_cutoff_silent(self) -> None:
        self._assert_silent(eval_end=datetime(2026, 4, 1, tzinfo=UTC), n_backtest_days=10)

    # 3. LF (always, regardless of window) → silence
    def test_lf_always_silent(self) -> None:
        self._assert_silent(
            prompt_config=LOW_FREQUENCY,
            eval_end=datetime(2025, 1, 1, tzinfo=UTC),
            n_backtest_days=365,
        )

    # 4. HF + sim_reg=2026-03-14 (cutoff + window_days) → silence
    def test_hf_sim_reg_at_safe_date_silent(self) -> None:
        self._assert_silent(
            simulate_registration=self.SAFE_SIM_REG,
            eval_end=datetime(2026, 4, 1, tzinfo=UTC),
        )

    # 5. HF + sim_reg=2026-03-11 (cutoff exact, smoothing reaches into pre-fix data) → WARN
    def test_hf_sim_reg_at_cutoff_warns(self) -> None:
        self._assert_warns(
            simulate_registration=self.CUTOFF,
            eval_end=datetime(2026, 4, 1, tzinfo=UTC),
        )

    # 6. HF + sim_dereg post-cutoff with window fully post-cutoff → silence
    def test_hf_sim_dereg_post_cutoff_silent(self) -> None:
        self._assert_silent(
            simulate_deregistration=datetime(2026, 3, 25, tzinfo=UTC),
            n_backtest_days=7,
        )

    # 7. HF + sim_dereg pre-cutoff → WARN
    def test_hf_sim_dereg_pre_cutoff_warns(self) -> None:
        self._assert_warns(
            simulate_deregistration=datetime(2026, 3, 8, tzinfo=UTC),
            n_backtest_days=5,
        )

