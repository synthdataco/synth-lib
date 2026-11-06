"""Serving wrapper: live-contract adaptation, venue routing (never Pyth), and the unpacker."""

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from synth.simulation_input import SimulationInput  # type: ignore[import-untyped]
from synth.validator import response_validation_v2  # type: ignore[import-untyped]

import synth_lib.serving.serve as serve
from synth_lib.benchmark.generate_predictions import load_simulate
from synth_lib.benchmark.verdict.run_verdict import DEFAULT_SEED, HASH_SEED_ENV, SEED_ENV, generation_env
from synth_lib.serving.serve import (
    ContextCache,
    build_context,
    serve_request,
    servable_assets,
    venue_store,
    wrap_output,
)
from synth_lib.serving.unpack_champion import MINER_TEMPLATE, class_name_for, contract_gate, unpack
from synth_lib.preparation.binance_client import BinanceClient
from synth_lib.preparation.config import OHLCV_COLUMNS
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
        closes = chunk.to_numpy()
        pd.DataFrame(
            {
                "timestamp": chunk.index,
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
                "volume": 1.0,
                "trade_count": 1.0,
            }
        ).to_parquet(root / f"date={day.isoformat()}.parquet")
    return MinutePriceStore(asset, root=root)


def _context(series: pd.Series) -> pd.DataFrame:
    """The OHLCV frame simulate() is handed, from a close series."""
    closes = series.to_numpy()
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1.0, "trade_count": 1.0},
        index=series.index,
    )


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
        context_prices=_context(series),
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


def test_missing_minutes_reach_the_champion_as_nan(tmp_path):
    """As in the verdict's context: a minute the store has no price for is not filled, and a day the
    store does not hold leaves the head of the window empty rather than repeating the first price."""
    series = _series()
    series.iloc[-100:-90] = np.nan
    t = series.index[-1]
    store = _store_with(tmp_path, series.loc[series.index >= t - pd.Timedelta(days=3)])
    context = build_context(store, t)
    assert context["close"].isna().sum() == 10 + 4 * 24 * 60
    assert context.index[0] == t - pd.Timedelta(days=7)
    assert context["close"].iloc[-1] == series.iloc[-1]


def test_serve_request_end_to_end_on_a_context_with_holes(tmp_path):
    series = _series()
    series.iloc[: 3 * 24 * 60] = np.nan
    series.iloc[-100:-90] = np.nan
    t = series.index[-1]
    store = _store_with(tmp_path, series)
    sim_input = SimulationInput(
        asset="BTC", start_time=t.isoformat(), time_increment=300, time_length=86_400, num_simulations=3
    )
    out = serve_request(load_simulate(SCAFFOLD_MODELING), store, sim_input)
    assert response_validation_v2.validate_responses(out, sim_input, process_time_str="1.0") == "CORRECT"


def _recording(seen: list):
    """A simulate() that records each context it is handed, then mutates it."""

    def simulate(asset, start_time, time_increment, time_length, num_simulations, context_prices):
        seen.append(context_prices.copy())
        context_prices["mine"] = 1.0
        last = float(context_prices["close"].iloc[-1])
        return (start_time, time_increment, *[[last] * (time_length // time_increment + 1)] * num_simulations)

    return simulate


def _input(t: pd.Timestamp, time_length: int = 86_400) -> SimulationInput:
    increment = 300 if time_length == 86_400 else 60
    return SimulationInput(
        asset="BTC", start_time=t.isoformat(), time_increment=increment, time_length=time_length, num_simulations=2
    )


@pytest.mark.parametrize("days", [14, 7])
def test_serve_request_hands_the_champion_its_context_days(tmp_path, days):
    series = _series(16)
    t = series.index[-1]
    seen: list = []
    serve_request(_recording(seen), _store_with(tmp_path, series), _input(t), context_days=days)
    assert seen[0].index[0] == t - pd.Timedelta(days=days)


def test_validators_asking_the_same_prompt_share_one_context(tmp_path, monkeypatch):
    """Built once per (asset, format, start_time), and each call gets its own copy to change."""
    series = _series()
    t = series.index[-1]
    store = _store_with(tmp_path, series)
    builds: list = []
    monkeypatch.setattr(serve, "build_context", lambda *a: builds.append(a) or build_context(*a))
    seen: list = []
    contexts = ContextCache()
    for _ in range(3):
        serve_request(_recording(seen), store, _input(t), contexts=contexts)
    assert len(builds) == 1 and "mine" not in seen[-1].columns
    serve_request(_recording(seen), store, _input(t, time_length=3_600), contexts=contexts)
    assert len(builds) == 2, "a crypto-1h prompt with the same start_time has its own context"


def test_a_shared_context_is_built_again_once_the_store_or_the_tail_moves_on(tmp_path):
    """A store refresh, or a tail holding a newer bar, between two validators asking the same prompt
    reaches the second one."""
    series = _series()
    t = series.index[-1] + pd.Timedelta(minutes=3)
    early = _store_with(tmp_path / "early", series.iloc[:-5])
    later = _store_with(tmp_path / "later", series)  # the same BTC store, refreshed
    seen: list = []
    contexts = ContextCache()
    serve_request(_recording(seen), early, _input(t), contexts=contexts)
    serve_request(_recording(seen), later, _input(t), contexts=contexts)
    contexts.store_refreshed()
    serve_request(_recording(seen), later, _input(t), contexts=contexts)
    assert [c.index[-1] for c in seen] == [t - pd.Timedelta(minutes=8)] * 2 + [t - pd.Timedelta(minutes=3)]

    seen.clear()
    for minutes in (3, 2, 2):
        index = pd.date_range(end=t - pd.Timedelta(minutes=minutes), periods=15, freq="1min")
        tail = pd.DataFrame({"timestamp": index, **{c: 1.0 for c in OHLCV_COLUMNS}})
        serve_request(_recording(seen), early, _input(t, time_length=3_600), tail=tail, contexts=contexts)
    assert [c.index[-1] for c in seen] == [t - pd.Timedelta(minutes=m) for m in (3, 2, 2)]


def test_the_context_cache_returns_the_same_frame_and_keeps_the_newest():
    t = pd.Timestamp("2026-09-30 12:00", tz="UTC").to_pydatetime()
    cache = ContextCache(size=2)
    frame = cache.get(("BTC", 3_600, t), pd.DataFrame)
    assert cache.get(("BTC", 3_600, t), lambda: pytest.fail("built twice")) is frame
    for minutes in (1, 2):
        cache.get(("BTC", 3_600, t + pd.Timedelta(minutes=minutes)), pd.DataFrame)
    assert cache.get(("BTC", 3_600, t), pd.DataFrame) is not frame, "the oldest key is evicted"


def _generated_miner(monkeypatch, **attributes) -> type:
    """The miner class unpack_champion generates, over a stand-in modeling module."""
    seen: list = []
    modeling = ModuleType("modeling")
    modeling.simulate = _recording(seen)
    modeling.seen = seen
    for name, value in attributes.items():
        setattr(modeling, name, value)
    monkeypatch.setitem(sys.modules, "modeling", modeling)
    source = MINER_TEMPLATE.substitute(name="test", campaign="c", leg="l", class_name="TestMiner")
    namespace: dict = {"__name__": "generated_miner"}
    exec(compile(source, "miner.py", "exec"), namespace)
    return namespace["TestMiner"]


@pytest.mark.parametrize("attributes, days", [({"CONTEXT_DAYS": 14}, 14), ({}, 7)])
def test_the_generated_miner_serves_the_champions_context_days(tmp_path, monkeypatch, attributes, days):
    miner_class = _generated_miner(monkeypatch, **attributes)
    assert miner_class.context_days == days
    series = _series(16)
    t = series.index[-1]
    miner = object.__new__(miner_class)  # Miner.__init__ needs a wallet, a subtensor and a chain
    miner._stores, miner._tails, miner._ready = {"BTC": _store_with(tmp_path, series)}, {}, {"BTC"}
    miner._contexts = ContextCache()
    asyncio.run(miner.forward_miner(SimpleNamespace(simulation_input=_input(t), simulation_output=None)))
    assert sys.modules["modeling"].seen[0].index[0] == t - pd.Timedelta(days=days)


@pytest.mark.parametrize("value", [2, 31, True, 14.0])
def test_the_generated_miner_refuses_context_days_outside_the_bounds(monkeypatch, value):
    with pytest.raises(ValueError, match="from 3 to 30"):
        _generated_miner(monkeypatch, CONTEXT_DAYS=value)


# Passes only on a synthetic context of exactly `days` days.
GATED_MODELING = """
{constant}

def simulate(asset, start_time, time_increment, time_length, num_simulations, context_prices=None):
    assert len(context_prices) == {days} * 24 * 60, len(context_prices)
    last = float(context_prices["close"].iloc[-1])
    return (start_time, time_increment, *[[last] * (time_length // time_increment + 1)] * num_simulations)
"""


@pytest.mark.parametrize("constant, days", [("CONTEXT_DAYS = 14", 14), ("", 7)])
def test_the_contract_gate_runs_on_context_days_plus_one(tmp_path, constant, days):
    (tmp_path / "modeling.py").write_text(GATED_MODELING.format(constant=constant, days=days + 1))
    assert contract_gate(tmp_path / "modeling.py", ("low", "high")) == days


@pytest.mark.parametrize("value", ["2", "31", "True", "14.0"])
def test_the_contract_gate_refuses_context_days_outside_the_bounds(tmp_path, value):
    (tmp_path / "modeling.py").write_text(GATED_MODELING.format(constant=f"CONTEXT_DAYS = {value}", days=8))
    with pytest.raises(ValueError, match="from 3 to 30"):
        contract_gate(tmp_path / "modeling.py", ("low", "high"))


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
    assert "\nimport modeling\n" in miner and "context_days = context_days_of(modeling)" in miner
    assert not any(line.startswith("from .") for line in miner.splitlines())
    assert "class TestCampFakeMiner(ChampionMiner):" in miner
    assert (dest / "entrypoint.sh").stat().st_mode & 0o111, "entrypoint must be executable"
    provenance = (dest / "PROVENANCE.md").read_text()
    assert "test-camp" in provenance and "sha" in provenance
    assert "- CONTEXT_DAYS: 7 " in provenance
    # unpack() gates through validate_responses before writing; existing dest must refuse
    with pytest.raises(FileExistsError):
        unpack("test-camp", "fake", None, results_dir, dest_root)


def test_the_entrypoint_pins_the_verdicts_seeds(tmp_path):
    """A live response replays offline only when both seeds are the ones the verdict generated with."""
    dest = unpack("test-camp", "fake", None, _make_bundle(tmp_path), tmp_path / "champions")
    exports = [line for line in (dest / "entrypoint.sh").read_text().splitlines() if line.startswith("export ")]
    script = "\n".join([*exports, f'echo "${SEED_ENV} ${HASH_SEED_ENV}"'])
    seeds = subprocess.run(
        ["bash", "-c", script], env={"PATH": os.environ["PATH"]}, capture_output=True, text=True, check=True
    ).stdout.split()
    expected = generation_env(DEFAULT_SEED)
    assert seeds == [expected[SEED_ENV], expected[HASH_SEED_ENV]]


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


def test_unpack_reads_the_published_champion_source_tree(tmp_path):
    """The archive publishes the workspace at CHAMPION.sha under champion_source/; when it is there,
    it is what gets unpacked, ahead of the bundle."""
    results_dir = _make_bundle(tmp_path)
    published = SCAFFOLD_MODELING.read_text() + "\n# from the published tree\n"
    tree = results_dir / "test-camp" / "fake" / "champion_source" / "agent"
    tree.mkdir(parents=True)
    (tree / "modeling.py").write_text(published)

    dest = unpack("test-camp", "fake", None, results_dir, tmp_path / "champions")

    assert (dest / "modeling.py").read_text() == published


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
