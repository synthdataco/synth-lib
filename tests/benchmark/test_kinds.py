"""Competition kinds: the built-in monte-carlo kind, and a kind another package registers through the
entry-point group, set up and run for one fake-CLI leg."""

import sys
from pathlib import Path

import pytest
import yaml

import synth_lib.benchmark.run_campaign as rc
from synth_lib.benchmark import monte_carlo
from synth_lib.benchmark.campaign import load_campaign
from synth_lib.benchmark.kinds import KINDS_GROUP, load_kind
from synth_lib.benchmark.budget import Phase
from synth_lib.benchmark.run_campaign import run_all, setup_campaign
from synth_lib.preparation.event_calendar.events import CALENDAR_DIR, CALENDAR_ENV, VIEW_FILE
from tests.benchmark.test_driver import HoldAdmin, _cfg, _exited_with, _run

TOY_KIND = """
from pathlib import Path

from synth_lib.benchmark.kinds import CompetitionKind

HERE = Path(__file__).parent


def _pin(ws):
    (ws / "PINNED").write_text("toy")


def _build_snapshot(data_root, snapshot, cfg):
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "toy.txt").write_text((data_root / "toy_source.txt").read_text())


def _probe(agent_dir, snapshot, python_exe):
    (agent_dir / "PROBED").write_text(str(snapshot))
    if (agent_dir / "BROKEN").exists():
        return False, "toy broken"
    return True, "toy interface ok"


KIND = CompetitionKind(
    primary_metric="toy_rank",
    template=HERE / "toy.tmpl",
    scaffold=HERE / "scaffold",
    pin_workspace=_pin,
    build_snapshot=_build_snapshot,
    render_data_md=lambda snapshot: "toy data: " + (snapshot / "toy.txt").read_text(),
    sandbox_env=lambda snapshot: {"TOY_DATA": "/workspace/market_data/toy.txt"},
    require_ready=lambda snapshot: None,
    probe=_probe,
    interface_prompt="toy check: {check}",
    landing_order="## LANDING ORDER — toy\\n\\nNominate the toy.\\n",
)
"""


def _register(plugins: Path, dist: str, kind: str, target: str) -> None:
    info = plugins / f"{dist}-0.1.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {dist}\nVersion: 0.1\n")
    (info / "entry_points.txt").write_text(f"[{KINDS_GROUP}]\n{kind} = {target}\n")


@pytest.fixture
def toy_kind(tmp_path, monkeypatch):
    """A package outside synth-lib that registers the `toy` kind."""
    plugins = tmp_path / "plugins"
    (plugins / "scaffold" / "agent").mkdir(parents=True)
    (plugins / "scaffold" / "agent" / "toy.py").write_text("ANSWER = 42\n")
    (plugins / "scaffold" / "pyproject.toml").write_text('[project]\nname = "toy"\ndependencies = []\n')
    (plugins / "toy.tmpl").write_text("# $campaign_name for $model_id\n\nData up to $data_cutoff.\n")
    (plugins / "toy_kind.py").write_text(TOY_KIND)
    _register(plugins, "toy_kind", "toy", "toy_kind:KIND")
    monkeypatch.syspath_prepend(str(plugins))
    yield plugins
    sys.modules.pop("toy_kind", None)


def _campaign(tmp_path: Path, **over) -> Path:
    raw = {
        "name": "toycamp",
        "objective": "t",
        "models": [{"id": "f1", "cli": "fake", "model": "none"}],
        "budget_usd_per_model": 10.0,
        "deadline_hours_per_model": 1.0,
        "data_cutoff": "2026-07-15",
        "forward_window_days": 7,
        "hardware": "h",
        "proxy_url": "http://x",
        "poll_seconds": 0.05,
        "landing_grace_seconds": 0.3,
        "gpu": False,
        **over,
    }
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_a_campaign_without_a_kind_is_monte_carlo(tmp_path):
    cfg = load_campaign(_campaign(tmp_path))
    assert cfg.kind == "monte-carlo"
    assert load_kind(cfg.kind) is monte_carlo.KIND


def test_an_unknown_kind_is_refused_at_load(tmp_path):
    with pytest.raises(ValueError, match="unknown competition kind 'nope'.*monte-carlo"):
        load_campaign(_campaign(tmp_path, kind="nope"))


def test_primary_metric_is_checked_against_the_kind(tmp_path, toy_kind):
    with pytest.raises(ValueError, match="pre-registered for kind toy: toy_rank"):
        load_campaign(_campaign(tmp_path, kind="toy"))
    with pytest.raises(ValueError, match="pre-registered for kind monte-carlo: synth_competition_rank"):
        load_campaign(_campaign(tmp_path, primary_metric="toy_rank"))
    assert load_campaign(_campaign(tmp_path, kind="toy", primary_metric="toy_rank")).kind == "toy"


def test_a_kind_registered_twice_is_refused(tmp_path, toy_kind):
    _register(toy_kind, "other_toy", "toy", "toy_kind:KIND")
    with pytest.raises(ValueError, match="registered more than once"):
        load_kind("toy")


def test_a_registered_kind_sets_up_and_runs_a_fake_leg(tmp_path, toy_kind):
    (tmp_path / "toy_source.txt").write_text("frozen\n")
    campaign = _campaign(tmp_path, kind="toy", primary_metric="toy_rank")
    state = setup_campaign(campaign, data_root=tmp_path, campaigns_root=tmp_path / "campaigns")

    assert (state.dir / "snapshot" / "toy.txt").read_text() == "frozen\n"
    ws = state.dir / "runs" / "f1"
    assert (ws / "agent" / "toy.py").exists() and (ws / "PINNED").read_text() == "toy"
    assert (ws / "CAMPAIGN.md").read_text() == "# toycamp for f1\n\nData up to 2026-07-15.\n"
    assert (ws / "agent" / "CAMPAIGN.md").read_text() == (ws / "CAMPAIGN.md").read_text()
    assert (ws / "DATA.md").read_text() == "toy data: frozen\n"

    results = run_all(
        state,
        admin=HoldAdmin(spend=8.6),
        sandboxed=False,
        fake_env={"FAKE_CLI_TICK_SECONDS": "0.02"},
        python_exe=sys.executable,
    )
    assert [r.landed for r in results] == [True]
    assert "## LANDING ORDER — toy" in (ws / "agent" / "BUDGET.md").read_text()
    assert "check interface: toy interface ok" in (ws / "agent" / "BUDGET.md").read_text()
    assert (ws / "agent" / "PROBED").read_text() == str(state.dir / "snapshot")


def test_a_registered_kind_sets_the_sandbox_env(tmp_path, toy_kind, monkeypatch):
    (tmp_path / "toy_source.txt").write_text("frozen\n")
    campaign = _campaign(tmp_path, kind="toy", primary_metric="toy_rank")
    state = setup_campaign(campaign, data_root=tmp_path, campaigns_root=tmp_path / "campaigns")
    # What would give a monte-carlo sandbox its offline bundle and calendar.
    (state.dir / "snapshot" / monte_carlo.OFFLINE_BUNDLE_DIR).mkdir()
    view = state.dir / "snapshot" / CALENDAR_DIR / VIEW_FILE
    view.parent.mkdir(parents=True)
    view.write_bytes(b"view")
    captured: dict = {}

    class DummyRun:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def run(self):
            return None

    monkeypatch.setattr(rc, "ModelRun", DummyRun)
    run_all(state, admin=None, sandboxed=True, python_exe="python")

    argv = captured["cmd_wrapper"](["echo", "hi"])
    assert "TOY_DATA=/workspace/market_data/toy.txt" in argv
    assert not any(arg.startswith((monte_carlo.OFFLINE_ROOT_ENV, CALENDAR_ENV)) for arg in argv)


def test_a_nomination_failing_a_registered_kind_s_probe_is_handed_back_with_its_prompt(tmp_path, toy_kind):
    run = _run(tmp_path, "lands", HoldAdmin(spend=1.0), cfg=_cfg(tmp_path / "campaigns", kind="toy"))
    (run.agent_dir / "CHAMPION").write_text("sha: abc\nagent_dir: agent\n")
    (run.agent_dir / "BROKEN").write_text("")
    run._probe_ok = False
    done, launched = _exited_with(run, Phase.NORMAL)
    assert not done and launched == ["toy check: toy broken"]
