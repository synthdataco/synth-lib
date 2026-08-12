"""Serving wrapper: live-contract adaptation, venue routing (never Pyth), and the unpacker."""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from synth.simulation_input import SimulationInput  # type: ignore[import-untyped]
from synth.validator import response_validation_v2  # type: ignore[import-untyped]

from synth_lib.benchmark.generate_predictions import load_simulate
from synth_lib.serving.serve import serve_request, servable_assets, venue_store, wrap_output
from synth_lib.serving.unpack_champion import unpack
from synth_lib.preparation.binance_client import BinanceClient
from synth_lib.preparation.minute_price_store import MinutePriceStore
from synth_lib.preparation.hyperliquid_client import HyperliquidClient

SCAFFOLD_MODELING = (
    Path(__file__).resolve().parents[2] / "synth_lib" / "benchmark" / "scaffold" / "workspace" / "agent" / "modeling.py"
)


def _series(days: int = 8) -> pd.Series:
    idx = pd.date_range("2026-01-01", periods=days * 24 * 60, freq="1min", tz="UTC")
    rng = np.random.default_rng(7)
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 1e-4, len(idx)))), index=idx, name="close")


def _store_with(tmp_path, series: pd.Series, asset: str = "BTC"):
    """A real MinutePriceStore over written day partitions — build_context reads the store's own
    daily parquet layout, so a stub would test nothing about the layout it depends on."""
    root = tmp_path / asset / "1m"
    root.mkdir(parents=True)
    for day, chunk in series.groupby(series.index.date):
        pd.DataFrame({"timestamp": chunk.index, "close": chunk.to_numpy()}).to_parquet(
            root / f"date={day.isoformat()}.parquet"
        )
    return MinutePriceStore(asset, root=root)


def test_wrap_output_passes_the_live_contract():
    series = _series()
    t = series.index[-1]
    simulate = load_simulate(SCAFFOLD_MODELING)
    raw = simulate(
        asset="BTC",
        start_time=t.isoformat(),
        time_increment=300,
        time_length=86_400,
        num_simulations=4,
        context_prices=series,
    )
    sim_input = SimulationInput(
        asset="BTC", start_time=t.isoformat(), time_increment=300, time_length=86_400, num_simulations=4
    )
    wrapped = wrap_output(raw, t.to_pydatetime(), 300)
    assert response_validation_v2.validate_responses(wrapped, sim_input, process_time_str="1.0") == "CORRECT"


def test_serve_request_end_to_end(tmp_path):
    series = _series()
    t = series.index[-1]
    store = _store_with(tmp_path, series)
    sim_input = SimulationInput(
        asset="BTC", start_time=t.isoformat(), time_increment=60, time_length=3_600, num_simulations=3
    )
    out = serve_request(load_simulate(SCAFFOLD_MODELING), store, sim_input)
    assert response_validation_v2.validate_responses(out, sim_input, process_time_str="1.0") == "CORRECT"


def test_venue_routing_never_pyth():
    assert isinstance(venue_store("BTC").client, BinanceClient)
    assert isinstance(venue_store("XAU").client, HyperliquidClient)
    with pytest.raises(ValueError, match="Pyth is retired"):
        venue_store("SPYX")  # Pyth-only legacy asset: refuse rather than route to a dead feed
    assert "SPYX" not in servable_assets()
    assert "BTC" in servable_assets()


# -- the unpacker, end to end on a synthetic bundle ------------------------------


def _make_bundle(tmp_path: Path) -> Path:
    """A minimal champion workspace: agent/modeling.py (the scaffold starter), bundled."""
    ws = tmp_path / "workspace"
    (ws / "agent").mkdir(parents=True)
    (ws / "agent" / "modeling.py").write_text(SCAFFOLD_MODELING.read_text())
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"], ["git", "commit", "-q", "-m", "champion"]):
        subprocess.run(cmd, cwd=ws, check=True, env=env, capture_output=True)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ws, check=True, capture_output=True, text=True
    ).stdout.strip()

    leg_dir = tmp_path / "campaign_results" / "test-camp" / "fake"
    leg_dir.mkdir(parents=True)
    subprocess.run(
        ["git", "bundle", "create", str(leg_dir / "workspace.bundle"), "--all"],
        cwd=ws,
        check=True,
        capture_output=True,
    )
    (leg_dir / "CHAMPION").write_text(
        f"sha: {sha}\nagent_dir: agent\nprofiles:\n  - low\n  - high\nnotes: test champion\n"
    )
    return tmp_path / "campaign_results"


def test_unpack_generates_a_servable_agent(tmp_path):
    results_dir = _make_bundle(tmp_path)
    dest_root = tmp_path / "champions"

    dest = unpack("test-camp", "fake", None, results_dir, dest_root)

    assert dest == dest_root / "test_camp_fake"
    assert (dest / "modeling.py").read_text() == SCAFFOLD_MODELING.read_text()
    miner = (dest / "miner.py").read_text()
    assert "from .modeling import simulate" in miner  # relative: the champion dir is a package
    assert "class TestCampFakeMiner(ChampionMiner):" in miner
    assert (dest / "entrypoint.sh").stat().st_mode & 0o111, "entrypoint must be executable"
    provenance = (dest / "PROVENANCE.md").read_text()
    assert "test-camp" in provenance and "sha" in provenance
    # unpack() gates through validate_responses before writing; existing dest must refuse
    with pytest.raises(FileExistsError):
        unpack("test-camp", "fake", None, results_dir, dest_root)


def test_unpack_refuses_a_champion_that_fails_the_gate(tmp_path):
    results_dir = _make_bundle(tmp_path)
    leg_dir = results_dir / "test-camp" / "fake"
    # Champion nominating a profile whose output the gate will actually exercise, but whose
    # modeling breaks the contract: NaN paths fail validate_responses.
    ws = tmp_path / "bad_ws"
    (ws / "agent").mkdir(parents=True)
    (ws / "agent" / "modeling.py").write_text(
        "def simulate(asset, start_time, time_increment, time_length, num_simulations, context_prices=None):\n"
        "    steps = time_length // time_increment + 1\n"
        "    return (start_time, time_increment, *[[float('nan')] * steps] * num_simulations)\n"
    )
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"], ["git", "commit", "-q", "-m", "bad"]):
        subprocess.run(cmd, cwd=ws, check=True, env=env, capture_output=True)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ws, check=True, capture_output=True, text=True
    ).stdout.strip()
    subprocess.run(
        ["git", "bundle", "create", str(leg_dir / "workspace.bundle"), "--all"],
        cwd=ws,
        check=True,
        capture_output=True,
    )
    (leg_dir / "CHAMPION").write_text(f"sha: {sha}\nagent_dir: agent\nprofiles:\n  - high\nnotes: bad\n")

    with pytest.raises(RuntimeError, match="contract gate failed"):
        unpack("test-camp", "fake", "bad_champ", results_dir, tmp_path / "champions")
    assert not (tmp_path / "champions" / "bad_champ").exists(), "gate failure must not leave a partial unpack"
