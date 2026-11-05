"""The monte-carlo kind: a miner submits sampled price paths, scored on crypto-1h, crypto-24h and
com-equ-24h. Registered under `monte-carlo` in pyproject.toml's entry points."""

from __future__ import annotations

from pathlib import Path

from synth_lib.backtester.field_store import REWARDS_DIR, SCORES_DIR
from synth_lib.benchmark import workspace
from synth_lib.benchmark.campaign import PRIMARY_METRIC, CampaignConfig
from synth_lib.benchmark.kinds import CompetitionKind
from synth_lib.benchmark.nomination import probe_simulate_subprocess
from synth_lib.benchmark.sandbox.run_sandbox import SANDBOX_MARKET_DATA
from synth_lib.benchmark.snapshot import build_snapshot, render_data_md
from synth_lib.preparation.event_calendar.events import CALENDAR_DIR, CALENDAR_ENV, VIEW_FILE

TEMPLATE = Path(__file__).parent / "constitution" / "CAMPAIGN.md.tmpl"
SCAFFOLD = Path(__file__).parent / "scaffold" / "workspace"
LANDING_ORDER = (
    "## LANDING ORDER — NON-NEGOTIABLE\n\n"
    "Your envelope is nearly exhausted. STOP any new experiment NOW.\n"
    "1. Finalize `agent/journal.md` (including the last iteration).\n"
    "2. Write the complete `agent/report.md` (honest results, failures included).\n"
    "3. Nominate your champion: write the `agent/CHAMPION` file (sha, agent_dir, profiles, notes).\n"
    "   The path matters: the harness looks ONLY at `agent/CHAMPION`.\n"
    "4. Verify that the weights/artifacts required by `simulate()` are in the workspace.\n"
)
INTERFACE_PROMPT = (
    "Your nomination does not stand yet: the harness called agent/modeling.py's simulate() on both "
    "prompt formats and got: {check}. Fix it, run no new experiments, and nominate again. BUDGET.md "
    "shows the check."
)
# The bundle's subdirectory inside the snapshot.
OFFLINE_BUNDLE_DIR = "offline_data"
# Must match synth_lib.backtester.config._OFFLINE_ENV_VAR.
OFFLINE_ROOT_ENV = "SYNTH_BACKTESTER_OFFLINE_DATA_ROOT"


def offline_bundle_env(snapshot: Path) -> dict[str, str]:
    """Point synth-lib's offline mode at the bundle in <snapshot>/offline_data, when one is present:
    the agent then reads field scores from disk. The operator builds the bundle there, for the days
    the campaign may see; build_snapshot does not link it. `run` refuses to start without it
    (`require_offline_bundle`), and the verdict runs with --network none.

    Absent bundle => empty dict: only a direct run_all() caller with no bundle, such as the fake-CLI
    tests, gets there.
    """
    if not (snapshot / OFFLINE_BUNDLE_DIR).is_dir():
        return {}
    return {OFFLINE_ROOT_ENV: f"{SANDBOX_MARKET_DATA}/{OFFLINE_BUNDLE_DIR}"}


def calendar_env(snapshot: Path) -> dict[str, str]:
    """Point a champion at the calendar the snapshot froze, when it froze one."""
    if not (snapshot / CALENDAR_DIR / VIEW_FILE).exists():
        return {}
    return {CALENDAR_ENV: f"{SANDBOX_MARKET_DATA}/{CALENDAR_DIR}/{VIEW_FILE}"}


def require_offline_bundle(snapshot: Path) -> Path:
    """The bundle agents backtest against, or exit. Without it every agent backtest calls the live
    API, and the backtester's padding past the requested window reaches past the data cutoff."""
    bundle = snapshot / OFFLINE_BUNDLE_DIR
    # Day partitions, or the single files of a bundle built before them.
    has_scores = bundle.is_dir() and any(
        [*bundle.glob(f"{SCORES_DIR}/*/*/date=*.parquet"), *bundle.glob("miner_scores_*.parquet")]
    )
    has_rewards = bundle.is_dir() and any(
        [*bundle.glob(f"{REWARDS_DIR}/*/date=*.parquet"), *bundle.glob("rewards_history_*.parquet")]
    )
    if not (has_scores and has_rewards):
        raise SystemExit(
            f"no offline scores bundle in {bundle}: build it before `run`, one call per competition —\n"
            f"  python -m synth_lib.backtester.scripts.build_offline_bundle --competition <slug> "
            f"--days 32 --eval-end <data_cutoff> --out {bundle} --no-realized-paths"
        )
    return bundle


def _pin_workspace(ws: Path) -> None:
    workspace._pin_scaffold_to_host(ws / "pyproject.toml")


def _build_snapshot(data_root: Path, snapshot: Path, cfg: CampaignConfig) -> None:
    build_snapshot(data_root / "market_data", snapshot, cutoff=cfg.data_cutoff, start=cfg.data_start)


def _sandbox_env(snapshot: Path) -> dict[str, str]:
    return {**offline_bundle_env(snapshot), **calendar_env(snapshot)}


def _probe(agent_dir: Path, snapshot: Path, python_exe: str) -> tuple[bool, str]:
    """simulate() on both prompt formats, with the snapshot's calendar when it froze one, out of process
    with a timeout and without the host's secrets (synth_lib.benchmark.nomination.probe_simulate_subprocess)."""
    calendar = snapshot / CALENDAR_DIR / VIEW_FILE
    return probe_simulate_subprocess(
        agent_dir / "modeling.py",
        python_exe=python_exe,
        extra_env={CALENDAR_ENV: str(calendar.resolve())} if calendar.exists() else None,
    )


KIND = CompetitionKind(
    primary_metric=PRIMARY_METRIC,
    template=TEMPLATE,
    scaffold=SCAFFOLD,
    pin_workspace=_pin_workspace,
    build_snapshot=_build_snapshot,
    render_data_md=render_data_md,
    sandbox_env=_sandbox_env,
    require_ready=require_offline_bundle,
    probe=_probe,
    interface_prompt=INTERFACE_PROMPT,
    landing_order=LANDING_ORDER,
)
