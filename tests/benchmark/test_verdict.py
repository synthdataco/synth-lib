import numpy as np
import pandas as pd
import pytest

from synth_lib.benchmark.verdict.generate import generate_predictions, prompt_grid
from synth_lib.benchmark.verdict.evaluate import final_rank, write_verdict


def _flat_simulate(asset, start_time, time_increment=300, time_length=86400, num_simulations=100, context_prices=None):
    steps = time_length // time_increment + 1
    base = float(context_prices.iloc[-1])
    return (start_time, time_increment, *[[base] * steps for _ in range(num_simulations)])


def test_prompt_grid_hourly():
    start = pd.Timestamp("2026-08-01T00:00:00Z")
    end = pd.Timestamp("2026-08-02T00:00:00Z")
    grid = prompt_grid(start, end, cadence_minutes=60)
    assert len(grid) == 24 and grid[0] == start and grid[-1] == end - pd.Timedelta(hours=1)

    # window_end not aligned with the cadence (02:30): the 02:00 prompt is valid (starts before
    # window_end) and must not be lost to a naive [:-1] that assumes alignment.
    misaligned_end = pd.Timestamp("2026-08-01T02:30:00Z")
    misaligned_grid = prompt_grid(start, misaligned_end, cadence_minutes=60)
    assert misaligned_grid == [
        pd.Timestamp("2026-08-01T00:00:00Z"),
        pd.Timestamp("2026-08-01T01:00:00Z"),
        pd.Timestamp("2026-08-01T02:00:00Z"),
    ]


def test_generate_predictions_1h_shape(tmp_path):
    idx = pd.date_range("2026-07-25T00:00:00Z", "2026-08-01T04:00:00Z", freq="1min", tz="UTC")
    prices = pd.Series(np.linspace(100.0, 110.0, len(idx)), index=idx)
    out = tmp_path / "predictions"
    n = generate_predictions(
        simulate_fn=_flat_simulate,
        asset="BTC",
        window_start=pd.Timestamp("2026-08-01T00:00:00Z"),
        window_end=pd.Timestamp("2026-08-01T02:00:00Z"),
        out_dir=out,
        price_series=prices,
        cadence_minutes=60,
        num_simulations=5,
        time_increment=60,
        time_length=3600,
    )
    assert n == 2
    files = sorted(out.iterdir())
    assert files[0].name.endswith("_BTC_3600.json")
    import json

    payload = json.loads(files[0].read_text())
    assert payload["num_steps"] == 61


def test_generate_predictions_writes_standard_artifacts(tmp_path):
    idx = pd.date_range("2026-07-25T00:00:00Z", "2026-08-02T00:00:00Z", freq="1min", tz="UTC")
    prices = pd.Series(np.linspace(100.0, 110.0, len(idx)), index=idx)
    out = tmp_path / "predictions"
    n = generate_predictions(
        simulate_fn=_flat_simulate,
        asset="BTC",
        window_start=pd.Timestamp("2026-08-01T00:00:00Z"),
        window_end=pd.Timestamp("2026-08-01T03:00:00Z"),
        out_dir=out,
        price_series=prices,
        cadence_minutes=60,
        num_simulations=10,
    )
    assert n == 3
    files = sorted(out.iterdir())
    assert files[0].name == "2026-08-01_00:00:00Z_BTC_86400.json"
    import json

    payload = json.loads(files[0].read_text())
    assert set(payload) == {
        "start_timestamp",
        "asset",
        "time_increment",
        "time_length",
        "num_simulations",
        "num_steps",
        "paths",
    }
    assert payload["num_simulations"] == 10 and len(payload["paths"]) == 10


def test_generate_predictions_default_num_simulations_is_1000(tmp_path):
    """generate_predictions' default must match the validator's PromptConfig.num_simulations
    (1000): the Synth field's CRPS was computed from 1000 sampled paths, and empirical CRPS is
    biased upward for small N — scoring a champion with fewer paths than the field would
    unfairly penalize it in the verdict. Uses the 1h/61-point shape to keep this light."""
    idx = pd.date_range("2026-07-25T00:00:00Z", "2026-08-01T04:00:00Z", freq="1min", tz="UTC")
    prices = pd.Series(np.linspace(100.0, 110.0, len(idx)), index=idx)
    out = tmp_path / "predictions"
    n = generate_predictions(
        simulate_fn=_flat_simulate,
        asset="BTC",
        window_start=pd.Timestamp("2026-08-01T00:00:00Z"),
        window_end=pd.Timestamp("2026-08-01T01:00:00Z"),
        out_dir=out,
        price_series=prices,
        cadence_minutes=60,
        time_increment=60,
        time_length=3600,
    )
    assert n == 1
    import json

    payload = json.loads(next(out.iterdir()).read_text())
    assert payload["num_simulations"] == 1000
    assert len(payload["paths"]) == 1000


def test_reward_metrics_simulated_emissions():
    from synth_lib.benchmark.verdict.evaluate import reward_metrics

    t1 = pd.Timestamp("2026-08-01T00:00:00Z")
    t2 = pd.Timestamp("2026-08-02T00:00:00Z")
    scores = pd.DataFrame(
        {
            "updated_at": [t1] * 3 + [t2] * 3,
            "miner_uid": [1, 2, 999, 1, 2, 999],
            "reward_weight": [0.5, 0.3, 0.2, 0.3, 0.3, 0.4],
        }
    )
    # window totals: uid1 = 0.8 (top other), uid2 = 0.6, candidate = 0.6
    m = reward_metrics(scores, miner_id=999)
    assert m["reward_share"] == pytest.approx(0.6 / 2.0)
    assert m["reward_vs_top"] == pytest.approx(0.6 / 0.8)
    assert m["reward_rank"] == 2  # only uid1's total strictly beats the candidate's
    assert m["beats_field"] is False

    # candidate absent from the field -> None, never a crash
    assert reward_metrics(scores[scores["miner_uid"] != 999], miner_id=999) is None


def test_reward_metrics_beating_the_field_exceeds_one():
    from synth_lib.benchmark.verdict.evaluate import reward_metrics

    t = pd.Timestamp("2026-08-01T00:00:00Z")
    scores = pd.DataFrame({"updated_at": [t] * 3, "miner_uid": [1, 2, 999], "reward_weight": [0.2, 0.3, 0.5]})
    m = reward_metrics(scores, miner_id=999)
    assert m["reward_vs_top"] == pytest.approx(0.5 / 0.3)  # > 1 stays visible, uncapped
    assert m["reward_rank"] == 1
    assert m["beats_field"] is True


def test_asset_rank_stats_best_and_mean():
    from synth_lib.benchmark.verdict.evaluate import asset_rank_stats

    t1 = pd.Timestamp("2026-08-01T00:00:00Z")
    t2 = pd.Timestamp("2026-08-02T00:00:00Z")
    scores = pd.DataFrame(
        {
            "updated_at": [t1] * 3 + [t2] * 3,
            "miner_uid": [1, 2, 999, 1, 2, 999],
            "reward_weight": [0.5, 0.3, 0.2, 0.1, 0.2, 0.7],  # 999: rank 3 at t1, rank 1 at t2
        }
    )
    stats = asset_rank_stats(scores, miner_id=999)
    assert stats == {"best_rank": 1, "mean_rank": 2.0, "field_size": 3}
    assert asset_rank_stats(pd.DataFrame(columns=["miner_uid", "updated_at", "reward_weight"]), 999) is None
    assert asset_rank_stats(scores[scores["miner_uid"] != 999], miner_id=999) is None


def test_final_rank_at_last_round():
    t1 = pd.Timestamp("2026-08-01T00:00:00Z")
    t2 = pd.Timestamp("2026-08-02T00:00:00Z")
    scores = pd.DataFrame(
        {
            "updated_at": [t1] * 3 + [t2] * 4,
            "miner_uid": [1, 2, 999, 1, 2, 3, 999],
            "reward_weight": [0.5, 0.3, 0.2, 0.1, 0.4, 0.3, 0.2],
        }
    )
    # last round (t2): weights 0.4 > 0.3 > 0.2 > 0.1 => miner 999 is 3rd of 4
    assert final_rank(scores, miner_id=999) == (3, 4)


class _FakeResult:
    """Stand-in for synth_lib.backtester.backtest.BacktestResult.

    Only .summary is read by evaluate_candidate directly; .prompt_df and .smoothed_scores
    exist purely to satisfy the BacktestResult shape passed on to (the stubbed)
    compute_combined_smoothed_scores, which in these tests never reads them.
    """

    def __init__(self, mean_crps: float = 1.0, num_prompts: int = 24):
        self.prompt_df = pd.DataFrame()
        self.smoothed_scores = pd.DataFrame()
        self.summary = {"mean_crps": mean_crps, "num_prompts": num_prompts}


def _fake_combined(miner_ranks: dict[int, float]):
    """Builds a compute_combined_smoothed_scores stub returning a fixed reward_weight ranking."""
    t1 = pd.Timestamp("2026-08-01T00:00:00Z")

    def fake(results, competition=None, cutoff_days=None):
        assert cutoff_days is None  # must let it default to competition.window_days
        if not results:
            return pd.DataFrame(columns=["updated_at", "miner_uid", "reward_weight"])
        return pd.DataFrame(
            {
                "updated_at": [t1] * len(miner_ranks),
                "miner_uid": list(miner_ranks.keys()),
                "reward_weight": list(miner_ranks.values()),
            }
        )

    return fake


def test_evaluate_candidate_covers_three_competitions(monkeypatch, tmp_path):
    # CRYPTO_1H shares BTC/ETH/SOL/XRP/HYPE with CRYPTO_24H at a different time_length — both
    # time_lengths must actually be requested. SPYX (com-equ-24h) is made to fail to exercise
    # fault isolation: it must be excluded from the results list handed to
    # compute_combined_smoothed_scores for that competition, but still surfaced in
    # assets_failed/per_asset.
    import synth_lib.benchmark.verdict.evaluate as ev

    backtest_calls: list[tuple[str, int]] = []

    def fake_backtest(
        *, miner_name, asset, time_length, time_increment, n_backtest_days, predictions_dir, eval_end, competition
    ):
        backtest_calls.append((asset, time_length))
        if asset == "SPYX":
            raise RuntimeError("missing data for SPYX")
        return _FakeResult()

    combined_calls: list[tuple[object, int]] = []
    fake_combined = _fake_combined({1: 0.5, 999: 0.3, 2: 0.2})  # miner 999 -> rank 2 of 3

    def fake_compute_combined(results, competition=None, cutoff_days=None):
        combined_calls.append((competition, len(results)))
        return fake_combined(results, competition=competition, cutoff_days=cutoff_days)

    monkeypatch.setattr(ev, "backtest", fake_backtest)
    monkeypatch.setattr(ev, "compute_combined_smoothed_scores", fake_compute_combined)

    result = ev.evaluate_candidate("cand", tmp_path, window_end=pd.Timestamp("2026-08-02T00:00:00Z"), window_days=1)

    assert len(backtest_calls) == 19  # 5 (crypto-24h) + 9 (com-equ-24h) + 5 (crypto-1h)
    assert ("BTC", 86400) in backtest_calls
    assert ("BTC", 3600) in backtest_calls

    assert len(combined_calls) == 3  # one call per competition
    comps_by_label = {comp.label: n for comp, n in combined_calls}
    assert set(comps_by_label) == {c.label for c in ev.COMPETITIONS}
    assert comps_by_label["Commodities/Equities 24h"] == 8  # 9 assets minus failed SPYX
    assert comps_by_label["Crypto 24h"] == 5
    assert comps_by_label["Crypto 1h"] == 5

    per_comp = result["per_competition"]
    assert set(per_comp) == {"crypto-24h", "com-equ-24h", "crypto-1h"}

    com_equ = per_comp["com-equ-24h"]
    assert com_equ["assets_failed"] == ["SPYX"]
    assert "error" in com_equ["per_asset"]["SPYX"]
    assert com_equ["per_asset"]["XAU"] == {"mean_crps": 1.0, "num_prompts": 24}
    assert com_equ["rank"] == 2
    assert com_equ["field_size"] == 3
    assert com_equ["percentile"] == 1.0 - (2 - 1) / 3

    crypto_24h = per_comp["crypto-24h"]
    assert crypto_24h["assets_failed"] == []
    assert crypto_24h["rank"] == 2
    assert crypto_24h["field_size"] == 3
    assert crypto_24h["percentile"] == 1.0 - (2 - 1) / 3

    crypto_1h = per_comp["crypto-1h"]
    assert crypto_1h["rank"] == 2
    assert crypto_1h["field_size"] == 3

    expected_mean = sum(per_comp[s]["percentile"] for s in per_comp) / 3
    assert result["mean_competition_percentile"] == expected_mean


def test_all_assets_failed_competition_yields_none(monkeypatch, tmp_path):
    # Every crypto-1h asset fails (time_length=3600 distinguishes it from crypto-24h, which
    # shares the same asset names) -> compute_combined_smoothed_scores is called with an empty
    # results list for that competition, and its rank/field_size/percentile must be None without
    # poisoning the other two competitions' entries or the cross-competition mean.
    import synth_lib.benchmark.verdict.evaluate as ev

    def fake_backtest(
        *, miner_name, asset, time_length, time_increment, n_backtest_days, predictions_dir, eval_end, competition
    ):
        if time_length == 3600:
            raise RuntimeError("crypto-1h store broken")
        return _FakeResult()

    fake_combined = _fake_combined({999: 0.6, 1: 0.4})  # miner 999 -> rank 1 of 2

    def fake_compute_combined(results, competition=None, cutoff_days=None):
        return fake_combined(results, competition=competition, cutoff_days=cutoff_days)

    monkeypatch.setattr(ev, "backtest", fake_backtest)
    monkeypatch.setattr(ev, "compute_combined_smoothed_scores", fake_compute_combined)

    result = ev.evaluate_candidate("cand", tmp_path, window_end=pd.Timestamp("2026-08-02T00:00:00Z"), window_days=1)

    crypto_1h = result["per_competition"]["crypto-1h"]
    assert crypto_1h["rank"] is None
    assert crypto_1h["field_size"] is None
    assert crypto_1h["percentile"] is None
    assert crypto_1h["rewards"] is None
    assert set(crypto_1h["assets_failed"]) == {"BTC", "ETH", "SOL", "XRP", "HYPE"}

    other_percentiles = [result["per_competition"][s]["percentile"] for s in ("crypto-24h", "com-equ-24h")]
    assert all(p is not None for p in other_percentiles)
    assert result["mean_competition_percentile"] == sum(other_percentiles) / 2
    # reward metrics ride along per surviving competition: 999 total 0.6 vs top other 0.4
    for slug in ("crypto-24h", "com-equ-24h"):
        assert result["per_competition"][slug]["rewards"]["reward_vs_top"] == pytest.approx(1.5)
        assert result["per_competition"][slug]["rewards"]["beats_field"] is True
    # the Score input is capped per competition (0-100 contract); raw 1.5 stays visible above
    assert result["mean_reward_vs_top"] == pytest.approx(1.0)


def test_write_verdict_ranks_by_mean_percentile(tmp_path):
    candidates = [
        {"name": "a", "mean_competition_percentile": 0.42, "per_competition": {"crypto-24h": {"rank": 5}}},
        {"name": "b", "mean_competition_percentile": 0.91, "per_competition": {"crypto-24h": {"rank": 1}}},
        {"name": "c", "mean_competition_percentile": None, "per_competition": {}},  # did-not-land
    ]
    out = tmp_path / "verdict.json"
    write_verdict(candidates, out)
    import json

    data = json.loads(out.read_text())
    assert data["ranking"] == ["b", "a", "c"]
    assert data["candidates"][0]["per_competition"] == {"crypto-24h": {"rank": 1}}
