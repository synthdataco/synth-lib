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
from synth_lib.serving.unpack_champion import class_name_for, unpack
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


def _make_bundle(tmp_path: Path, *, extra: dict[str, str] | None = None) -> Path:
    """A minimal champion workspace: agent/modeling.py (the scaffold starter), bundled.

    `extra` adds files under agent/ (paths relative to it), for champions that carry data."""
    ws = tmp_path / "workspace"
    (ws / "agent").mkdir(parents=True)
    (ws / "agent" / "modeling.py").write_text(SCAFFOLD_MODELING.read_text())
    for rel, content in (extra or {}).items():
        path = ws / "agent" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
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
    # Absolute: entrypoint.sh runs miner.py as a script, so there is no package for a relative import.
    assert "from modeling import simulate" in miner
    assert not any(line.startswith("from .") for line in miner.splitlines())
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


def test_class_name_is_a_valid_identifier():
    """A free-text --name must not produce uncompilable code: `campaign-2` once emitted
    `class Campaign-2Miner`, which only failed when the miner was launched."""
    assert class_name_for("campaign-2") == "Campaign2Miner"
    assert class_name_for("test_camp_fake") == "TestCampFakeMiner"
    assert class_name_for("2nd-try") == "Champion2ndTryMiner"  # a class cannot start with a digit
    for name in ("campaign-2", "2nd-try", "a.b c"):
        compile(f"class {class_name_for(name)}: pass", "<gen>", "exec")


def test_unpack_copies_the_whole_champion_tree(tmp_path):
    """The unpacked champion is the agent tree at CHAMPION.sha plus the serving shell — the same
    content the results repo publishes, so what serves is what was scored."""
    results_dir = _make_bundle(
        tmp_path,
        extra={
            "artifacts/calib.npz": "not-really-an-npz",
            "artifacts/nested/sess.npz": "also-fake",
            "table.csv": "a,b\n1,2\n",
            "journal.md": "how the constants were fitted",
            "research/fit_calib.py": "# the fitting script\n",
        },
    )
    dest = unpack("test-camp", "fake", None, results_dir, tmp_path / "champions")

    assert (dest / "artifacts" / "calib.npz").read_text() == "not-really-an-npz"
    assert (dest / "artifacts" / "nested" / "sess.npz").exists()
    assert (dest / "table.csv").exists()
    assert (dest / "journal.md").exists() and (dest / "research" / "fit_calib.py").exists()
    for generated in ("miner.py", "entrypoint.sh", "PROVENANCE.md"):
        assert (dest / generated).exists()
    provenance = (dest / "PROVENANCE.md").read_text()
    assert "artifacts/calib.npz" in provenance and "table.csv" in provenance

    # The bundle travels too: without it a deployed champion cannot be checked back against its sha.
    bundled = dest / "workspace.bundle"
    assert bundled.exists()
    sha = (results_dir / "test-camp" / "fake" / "CHAMPION").read_text().split("sha:")[1].split()[0]
    clone = tmp_path / "verify"
    subprocess.run(["git", "clone", "-q", str(bundled), str(clone)], check=True, capture_output=True)
    assert (
        subprocess.run(
            ["git", "-C", str(clone), "rev-parse", sha], check=True, capture_output=True, text=True
        ).stdout.strip()
        == sha
    ), "the copied bundle must resolve the sha PROVENANCE.md claims"
    assert "workspace.bundle" in provenance and sha in provenance


def test_allow_missing_data_unpacks_a_dead_reference_and_discloses_it(tmp_path):
    """A named data file is not necessarily a needed one: a champion can keep an experiment it
    discarded behind an env-var-gated branch. The default refuses; the flag unpacks and records it,
    because the deployed copy must say what it is missing."""
    results_dir = _make_bundle(tmp_path, extra={"artifacts/kept.npz": "real"})
    modeling = SCAFFOLD_MODELING.read_text() + '\nWEIGHTS = "discarded.pt"  # only read when INNOV=="ml"\n'
    ws = tmp_path / "workspace"
    (ws / "agent" / "modeling.py").write_text(modeling)
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    for cmd in (["git", "add", "-A"], ["git", "commit", "-q", "-m", "dead reference"]):
        subprocess.run(cmd, cwd=ws, check=True, env=env, capture_output=True)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ws, check=True, capture_output=True, text=True
    ).stdout.strip()
    leg_dir = results_dir / "test-camp" / "fake"
    (leg_dir / "workspace.bundle").unlink()
    subprocess.run(
        ["git", "bundle", "create", str(leg_dir / "workspace.bundle"), "--all"],
        cwd=ws,
        check=True,
        capture_output=True,
    )
    (leg_dir / "CHAMPION").write_text(f"sha: {sha}\nagent_dir: agent\nprofiles:\n  - high\nnotes: x\n")

    with pytest.raises(RuntimeError, match="discarded.pt"):
        unpack("test-camp", "fake", "strict", results_dir, tmp_path / "champions")

    dest = unpack("test-camp", "fake", "lenient", results_dir, tmp_path / "champions", allow_missing_data=True)
    provenance = (dest / "PROVENANCE.md").read_text()
    assert "discarded.pt" in provenance and "--allow-missing-data" in provenance
    assert "artifacts/kept.npz" in provenance, "what IS present must still be listed"


def test_unpack_writes_the_archives_champion_and_keeps_the_agents(tmp_path):
    """An agent-committed CHAMPION names the commit BEFORE the one adding it — a commit cannot carry
    its own sha — and it lands exactly where a leg-level CHAMPION lives, which is where run_verdict
    reads the sha to score. The archive's must win; the agent's is kept for the record."""
    results_dir = _make_bundle(tmp_path, extra={"CHAMPION": "sha: 0000000\nagent_dir: agent\nprofiles: [high]\n"})
    leg_dir = results_dir / "test-camp" / "fake"
    real_sha = (leg_dir / "CHAMPION").read_text().split("sha:")[1].split()[0]

    dest = unpack("test-camp", "fake", "shadowed", results_dir, tmp_path / "champions")

    assert (dest / "CHAMPION").read_text() == (leg_dir / "CHAMPION").read_text()
    assert real_sha in (dest / "CHAMPION").read_text()
    assert "0000000" in (dest / "CHAMPION.agent").read_text(), "the agent's file must survive verbatim"
    provenance = (dest / "PROVENANCE.md").read_text()
    assert "CHAMPION.agent" in provenance and real_sha[:7] in provenance


def test_unpack_writes_a_champion_even_when_the_tree_had_none(tmp_path):
    """Two legs' agents never committed a nomination file; their unpacked trees were left with no
    CHAMPION at all, which run_verdict reads as 'no champion here' and skips."""
    results_dir = _make_bundle(tmp_path)
    dest = unpack("test-camp", "fake", "nofile", results_dir, tmp_path / "champions")
    assert (dest / "CHAMPION").exists()
    assert not (dest / "CHAMPION.agent").exists()
    assert "committed no nomination file" in (dest / "PROVENANCE.md").read_text()


def test_unpack_refuses_to_overwrite_a_champions_own_file(tmp_path):
    """The generated shell must never silently replace something the agent wrote."""
    results_dir = _make_bundle(tmp_path, extra={"miner.py": "# the agent's own miner\n"})
    with pytest.raises(RuntimeError, match="miner.py"):
        unpack("test-camp", "fake", "clash", results_dir, tmp_path / "champions")
    assert not (tmp_path / "champions" / "clash").exists()


def test_unpack_refuses_when_referenced_data_is_absent(tmp_path):
    """The champion's own loader falls back to an empty dict, so a missing artifact is silent —
    the unpacker must be the one to notice."""
    results_dir = _make_bundle(tmp_path)
    modeling = SCAFFOLD_MODELING.read_text() + '\n_ART_NAME = "calib.npz"  # loaded relative to __file__\n'
    ws = tmp_path / "workspace"
    (ws / "agent" / "modeling.py").write_text(modeling)
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    for cmd in (["git", "add", "-A"], ["git", "commit", "-q", "-m", "no artifacts"]):
        subprocess.run(cmd, cwd=ws, check=True, env=env, capture_output=True)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ws, check=True, capture_output=True, text=True
    ).stdout.strip()
    leg_dir = results_dir / "test-camp" / "fake"
    (leg_dir / "workspace.bundle").unlink()
    subprocess.run(
        ["git", "bundle", "create", str(leg_dir / "workspace.bundle"), "--all"],
        cwd=ws,
        check=True,
        capture_output=True,
    )
    (leg_dir / "CHAMPION").write_text(f"sha: {sha}\nagent_dir: agent\nprofiles:\n  - high\nnotes: x\n")

    with pytest.raises(RuntimeError, match="calib.npz"):
        unpack("test-camp", "fake", "hollow", results_dir, tmp_path / "champions")
    assert not (tmp_path / "champions" / "hollow").exists(), "a refused unpack must leave nothing"
