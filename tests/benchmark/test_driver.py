"""The driver is tested end-to-end with FakeAdapter — zero LLM, zero Docker."""

import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

from synth_lib.benchmark.budget import BudgetTracker
from synth_lib.benchmark.campaign import BaselineSpec, CampaignConfig, ModelSpec
from synth_lib.benchmark.cli_adapters import FakeAdapter
from synth_lib.benchmark.clock import RunClock
from synth_lib.benchmark.driver import ModelRun


class RampAdmin:
    """Spend that rises on every poll — reaches exhaustion (100%)."""

    def __init__(self, step: float):
        self.spend = 0.0
        self.step = step

    def key_info(self, key: str) -> dict:
        self.spend += self.step
        return {"spend": self.spend, "max_budget": 10.0}


class HoldAdmin:
    """Fixed spend — allows staying durably in the LANDING phase (e.g. 8.6/10 = 86%)."""

    def __init__(self, spend: float):
        self.spend = spend

    def key_info(self, key: str) -> dict:
        return {"spend": self.spend, "max_budget": 10.0}


class FlakyHoldAdmin:
    """HoldAdmin variant with a `.fail` switch (pattern of FakeAdmin in tests/benchmark/test_budget.py):
    stays at `spend` (LANDING) for `fail_after` calls, then raises ConnectionError for the next
    `fail_calls` calls (simulated proxy outage), then recovers with spend pushed to `recovered_spend`."""

    def __init__(self, spend: float, fail_after: int, fail_calls: int, recovered_spend: float):
        self.spend = spend
        self.fail_after = fail_after
        self.fail_calls = fail_calls
        self.recovered_spend = recovered_spend
        self.fail = False
        self._n = 0

    def key_info(self, key: str) -> dict:
        self._n += 1
        self.fail = self.fail_after < self._n <= self.fail_after + self.fail_calls
        if self.fail:
            raise ConnectionError("proxy down")
        if self._n > self.fail_after + self.fail_calls:
            self.spend = self.recovered_spend
        return {"spend": self.spend, "max_budget": 10.0}


def _cfg(root, **over):
    base = dict(
        name="t",
        objective="o",
        models=(ModelSpec(id="f", cli="fake", model="none"),),
        budget_usd_per_model=10.0,
        deadline_hours_per_model=1.0,
        data_cutoff=date(2026, 7, 15),
        forward_window_days=7,
        baselines=(BaselineSpec(name="b", module="x"),),
        hardware="h",
        proxy_url="http://x",
        poll_seconds=0.05,
        landing_grace_seconds=0.3,
        max_crash_resumes=2,
        root=root,
    )
    base.update(over)
    return CampaignConfig(**base)


def _run(tmp_path, scenario: str, admin, cfg=None) -> "ModelRun":
    cfg = cfg or _cfg(tmp_path / "campaigns")
    ws = tmp_path / "ws"
    agent_dir = ws / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "journal.md").write_text("")
    model = cfg.models[0]
    adapter = FakeAdapter(model, cfg.proxy_url, "sk-v")
    tracker = BudgetTracker(
        admin=admin,
        virtual_key="sk-v",
        budget_usd=cfg.budget_usd_per_model,
        soft_landing_pct=cfg.soft_landing_pct,
        clock=RunClock(deadline_seconds=cfg.deadline_hours_per_model * 3600),
    )
    run = ModelRun(
        cfg,
        model,
        adapter,
        tracker,
        workspace=ws,
        artifacts_dir=tmp_path / "artifacts",
        extra_env={
            "FAKE_CLI_SCENARIO": scenario,
            "FAKE_CLI_AGENT_DIR": str(agent_dir),
            "FAKE_CLI_TICK_SECONDS": "0.02",
            "PYTHONUNBUFFERED": "1",
        },
        python_exe=sys.executable,
    )
    return run


def test_collect_bundles_the_workspace_repo(tmp_path):
    """The champion is a sha inside the workspace repo, not the markdown deliverables: the bundle
    is the only thing that survives deleting the campaign directory."""
    run = _run(tmp_path, "lands", HoldAdmin(spend=0.0))
    ws = run.workspace
    (ws / "agent" / "modeling.py").write_text("def simulate():\n    return 1\n")
    for cmd in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "champion"],
    ):
        subprocess.run(cmd, cwd=ws, check=True)
    sha = subprocess.run(
        ["git", "-C", str(ws), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()

    run._collect()

    bundle = tmp_path / "artifacts" / "workspace.bundle"
    assert bundle.exists()
    # The bundle is only useful if it restores to the exact commit CHAMPION names.
    restored = tmp_path / "restored"
    subprocess.run(["git", "clone", "-q", str(bundle), str(restored)], check=True)
    assert subprocess.run(["git", "-C", str(restored), "cat-file", "-e", sha], check=False).returncode == 0
    assert (restored / "agent" / "modeling.py").read_text().startswith("def simulate")


def test_collect_bundles_with_a_relative_artifacts_dir(tmp_path, monkeypatch):
    """The real campaign passes artifacts_dir RELATIVE to the repo root (state.dir is relative),
    and `git -C <workspace>` resolves a relative bundle path under the WORKSPACE — the smoke run
    failed with runs/claude/campaign_runs/... All-absolute tmp_path tests never caught it."""
    monkeypatch.chdir(tmp_path)
    run = _run(tmp_path, "lands", HoldAdmin(spend=0.0))
    run.artifacts_dir = Path("artifacts")  # relative, as run_campaign builds it
    for cmd in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "champion"],
    ):
        subprocess.run(cmd, cwd=run.workspace, check=True)

    run._collect()

    assert (tmp_path / "artifacts" / "workspace.bundle").exists()
    assert not (run.workspace / "artifacts").exists()  # and nothing leaked under the workspace


def test_collect_survives_a_workspace_that_is_not_a_repo(tmp_path):
    """Bundling is best-effort: a git failure must not cost us the markdown deliverables."""
    run = _run(tmp_path, "lands", HoldAdmin(spend=0.0))
    (run.agent_dir / "report.md").write_text("the report")

    run._collect()

    assert (tmp_path / "artifacts" / "report.md").read_text() == "the report"
    assert not (tmp_path / "artifacts" / "workspace.bundle").exists()


def test_normal_landing(tmp_path):
    result = _run(tmp_path, "lands", HoldAdmin(spend=8.6)).run()  # 86% => LANDING stable
    assert result.landed is True and result.champion_path is not None
    art = tmp_path / "artifacts"
    assert (art / "report.md").exists() and (art / "journal.md").exists()
    assert (art / "CHAMPION").exists() and (art / "transcript-0.log").exists()
    agent_dir = tmp_path / "ws" / "agent"
    assert "check interface" in (agent_dir / "BUDGET.md").read_text()


def test_ignores_landing_gets_killed_and_relaunched(tmp_path):
    # the fake journals but writes neither report.md nor CHAMPION => after landing_grace_seconds,
    # kill + relaunch with the LANDING ORDER prompt, which the fake obeys.
    result = _run(tmp_path, "ignores_landing", HoldAdmin(spend=8.6)).run()
    assert result.landing_relaunches == 1 and result.landed is True


def test_crash_is_resumed_up_to_cap(tmp_path):
    result = _run(tmp_path, "crashes", RampAdmin(step=0.01)).run()
    assert result.crash_resumes == 2 and result.landed is False  # cap reached => did-not-land


def test_silent_run_exhausts_and_did_not_land(tmp_path):
    result = _run(tmp_path, "silent", RampAdmin(step=3.0)).run()  # 4 polls => >100%
    assert result.landed is False and result.champion_path is None


def test_voluntary_stop_is_relaunched_not_crashed(tmp_path):
    # exit 0 with no champion = voluntary stop (the model thinks it's done); the constitution says
    # to never stop => it gets relaunched without consuming crash_resumes (capped at 2 by _cfg).
    # step=0.2 (not 1.0): LANDING (85%) at poll ~43 and EXHAUSTED (100%) at poll 50 => ~7 polls of
    # margin in LANDING to let at least one relaunch/land cycle complete before exhaustion — with
    # step=1.0 there's only 1 poll of margin, which makes `landed is True` intermittent (~1/8 across
    # repeated runs, the relaunch triggered by that single LANDING poll doesn't always have time
    # to land before EXHAUSTED kills the next process).
    result = _run(tmp_path, "stops_early", RampAdmin(step=0.2)).run()
    assert result.landed is True
    assert result.crash_resumes == 0
    assert result.voluntary_resumes >= 1


def test_probe_note_persists_across_polls(tmp_path):
    # CHAMPION pre-placed => the probe fires as early as the 1st iteration; the "crashes" scenario
    # forces several extra iterations (up to the crash_resumes cap) after the probe. Without the
    # fix (self._probe_note republished on EVERY write_budget_file), the next poll's notes-less
    # write_budget_file() would overwrite the "check interface" note in the final BUDGET.md.
    run = _run(tmp_path, "crashes", RampAdmin(step=0.01))
    (run.agent_dir / "CHAMPION").write_text("sha: pre\nagent_dir: campaign_f\nprofiles: [low]\nnotes: pre\n")
    run.run()
    assert "check interface" in (run.agent_dir / "BUDGET.md").read_text()


def test_initial_budget_file_written_before_first_launch(tmp_path):
    # The constitution promises the agent it can read BUDGET.md from its very first turn; without
    # this fix, the file only exists after the first poll_seconds sleep (1st loop turn).
    run = _run(tmp_path, "lands", HoldAdmin(spend=8.6))
    budget_path = run.agent_dir / "BUDGET.md"
    original_launch = run._launch
    seen: dict = {}

    def spy_launch(prompt, resume=False):
        seen["exists_at_first_launch"] = budget_path.exists()
        return original_launch(prompt, resume=resume)

    run._launch = spy_launch
    result = run.run()
    assert seen.get("exists_at_first_launch") is True
    assert result.landed is True


def test_handle_exit_voluntary_stop_during_landing_uses_landing_prompt(tmp_path):
    # Unit-level, deterministic (bypasses the fake_cli shortcut that reads the LANDING ORDER from
    # BUDGET.md rather than the received prompt, which would mask a regression in the full
    # integration below): _handle_exit must choose LANDING_PROMPT (not RESUME_PROMPT) for a
    # voluntary stop during Phase.LANDING.
    from synth_lib.benchmark.budget import BudgetStatus, Phase
    from synth_lib.benchmark.driver import LANDING_PROMPT

    run = _run(tmp_path, "lands", HoldAdmin(spend=8.6))
    captured: dict = {}

    def fake_launch(prompt, resume=False):
        captured["prompt"] = prompt
        captured["resume"] = resume
        return object()

    run._launch = fake_launch
    status = BudgetStatus(
        spend_usd=8.6,
        budget_usd=10.0,
        pct_usd=0.86,
        elapsed_s=0.0,
        remaining_s=100.0,
        pct_time=0.1,
        phase=Phase.LANDING,
    )
    done, _proc, crash_resumes, voluntary_resumes = run._handle_exit(object(), 0, status, 0, 0)
    assert done is False
    assert crash_resumes == 0
    assert voluntary_resumes == 1
    assert captured["prompt"].startswith(LANDING_PROMPT)
    assert "agent/CHAMPION MISSING" in captured["prompt"]  # the driver's view rides along
    assert captured["resume"] is True


def test_landing_prompt_reports_deliverable_state_and_misplaced_champion(tmp_path):
    # The landing relaunch prompt must carry the driver's ACTUAL view of the deliverables —
    # a static order loops forever against an agent that believes it is done (it answers "all set
    # for handoff" while its CHAMPION sits at the workspace root, where the harness never looks).
    run = _run(tmp_path, "lands", HoldAdmin(spend=8.6))

    prompt = run._landing_prompt()
    assert "agent/report.md MISSING" in prompt and "agent/CHAMPION MISSING" in prompt
    assert "workspace root" not in prompt  # no misplaced file, no move instruction

    (run.workspace / "CHAMPION").write_text("sha: abc\nagent_dir: agent\nprofiles: [low]\n")
    run.agent_dir.mkdir(parents=True, exist_ok=True)
    (run.agent_dir / "report.md").write_text("done")
    prompt = run._landing_prompt()
    assert "agent/report.md OK" in prompt
    assert "Found CHAMPION at the workspace root" in prompt and "move it to agent/CHAMPION" in prompt

    # once the champion is in the right place, the move instruction disappears
    (run.agent_dir / "CHAMPION").write_text("sha: abc\nagent_dir: agent\nprofiles: [low]\n")
    prompt = run._landing_prompt()
    assert "agent/CHAMPION OK" in prompt and "workspace root" not in prompt


def test_voluntary_stop_during_landing_gets_landing_prompt(tmp_path):
    # A voluntary stop (exit 0, no champion) during Phase.LANDING must be relaunched with
    # the LANDING ORDER, not the generic RESUME prompt — otherwise the agent in LANDING goes back
    # to "continue where you left off" instead of landing, wasting the already-critical budget.
    # HoldAdmin(8.6) => LANDING stable from the initial call to run(); with the M6 fix (BUDGET.md
    # written BEFORE the very first _launch), the agent sees the LANDING ORDER on its 1st launch
    # and lands without even needing a voluntary stop — so voluntary_resumes stays at 0 here.
    # The precise choice of relaunch prompt during LANDING (the behavior specific to this fix) is
    # verified unambiguously by test_handle_exit_voluntary_stop_during_landing_uses_landing_prompt
    # below (the "stops_early" fake also reads BUDGET.md, which would mask a regression in the
    # prompt choice in this end-to-end test). This test locks down the observable outcome:
    # no landing relaunch is needed (landing_relaunches == 0).
    result = _run(tmp_path, "stops_early", HoldAdmin(spend=8.6)).run()
    assert result.landed is True
    assert result.landing_relaunches == 0


def test_kill_terminates_grandchildren(tmp_path):
    # _kill must kill the whole process group (start_new_session=True on the _spawn side), not
    # just the direct process: a champion that launches a GPU training run as a child must not
    # survive a landing/exhaustion kill.
    run = _run(tmp_path, "silent", HoldAdmin(spend=0.0))
    pid_file = tmp_path / "grandchild.pid"
    script = (
        "import subprocess, sys, time\n"
        f"pid_file = {str(pid_file)!r}\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "open(pid_file, 'w').write(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    proc = run._spawn(["python", "-c", script])
    deadline = time.time() + 5
    while not pid_file.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert pid_file.exists(), "the grandchild did not start in time"
    grandchild_pid = int(pid_file.read_text())

    run._kill(proc)

    deadline = time.time() + 5
    dead = False
    while time.time() < deadline:
        try:
            os.kill(grandchild_pid, 0)
        except ProcessLookupError:
            dead = True
            break
        time.sleep(0.05)
    assert dead, "the grandchild survived the kill"


def test_landing_relaunch_waits_out_infra_outage(tmp_path):
    # LANDING (8.6/10 = 86%) then a proxy outage (.fail) for 20 polls (>= 2x landing_grace_seconds
    # at 0.3s over 0.05s/poll): the grace period elapses and no_progress is true (ignores_landing
    # scenario), so WITHOUT the `not status.infra_down` guard, the landing kill+relaunch would fire
    # during the outage. Deterministic variant chosen (documented in the report): .run() is
    # blocking, we can't interrupt the outage mid-course from the test; the outage therefore ends
    # with a spend jump to 10.1 (EXHAUSTED), which ends the run via exhaustion rather than a
    # post-outage landing relaunch (avoiding any race). We only verify that the guard held:
    # no landing relaunch happened even though grace+no_progress were both true for >=2x
    # landing_grace_seconds of outage.
    admin = FlakyHoldAdmin(spend=8.6, fail_after=2, fail_calls=20, recovered_spend=10.1)
    result = _run(tmp_path, "ignores_landing", admin).run()
    assert result.landing_relaunches == 0
    assert result.landed is False


def test_launch_applies_cmd_wrapper(tmp_path):
    # When a cmd_wrapper is provided (the sandboxed=True case), _launch must wrap the adapter's
    # raw command (e.g. via sandbox_cmd), not launch it as-is directly on the host.
    run = _run(tmp_path, "lands", HoldAdmin(spend=0.0))
    run.cmd_wrapper = lambda cmd: ["docker", "run", "--name", "synthbench-t-f", "img", " ".join(cmd)]

    spawned: dict = {}

    def fake_spawn(cmd):
        spawned["cmd"] = cmd

        class P:
            def poll(self):
                return None

        return P()

    run._spawn = fake_spawn
    run._launch("hello")
    assert spawned["cmd"][:2] == ["docker", "run"]
    assert "--name" in spawned["cmd"] and "synthbench-t-f" in spawned["cmd"]


def test_transcript_numbering_continues_after_existing_files(tmp_path):
    # A rerun used to restart the counter at 0 and OVERWRITE the previous attempt's transcripts
    # one by one (the stale-transcript trap). Numbering must continue after the highest existing
    # index so each attempt appends to the leg's audit trail.
    artifacts = tmp_path / "artifacts" / "m"
    artifacts.mkdir(parents=True)
    (artifacts / "transcript-0.log").write_text("attempt 1")
    (artifacts / "transcript-7.log").write_text("attempt 1, last")
    (artifacts / "transcript-junk.log").write_text("ignored")

    run = _run(tmp_path, "lands", HoldAdmin(spend=0.0))
    run.artifacts_dir.mkdir(parents=True, exist_ok=True)
    assert run._transcripts == 0  # fresh dir from the fixture

    run2 = ModelRun(
        run.cfg,
        run.model,
        run.adapter,
        run.tracker,
        workspace=run.workspace,
        artifacts_dir=artifacts,
        python_exe=run.python_exe,
    )
    assert run2._transcripts == 8
    (artifacts / "transcript-0.log").read_text() == "attempt 1"  # untouched


def test_launch_removes_leftover_container_first(tmp_path, monkeypatch):
    # A container can outlive its `docker run` client (e.g. the host OOM-killer takes the client;
    # every relaunch then dies on a name conflict until the crash-resume cap). _launch
    # must `docker rm -f <name>` before spawning whenever a container_name is set.
    run = _run(tmp_path, "lands", HoldAdmin(spend=0.0))
    run.container_name = "synthbench-t-f"

    calls: list = []
    monkeypatch.setattr("synth_lib.benchmark.driver.subprocess.run", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(run, "_spawn", lambda cmd: object())

    run._launch("go")
    assert calls and calls[0][0][0] == ["docker", "rm", "-f", "synthbench-t-f"]

    # without a container_name (unit tests, host runs), no docker call is made
    run.container_name = None
    calls.clear()
    run._launch("go")
    assert calls == []


def test_kill_uses_docker_stop_when_container_name_set(tmp_path, monkeypatch):
    # Killing the `docker run` client does not kill the container — _kill must go through
    # `docker stop <name>` when a container_name is provided, not through killpg.
    run = _run(tmp_path, "lands", HoldAdmin(spend=0.0))
    run.container_name = "synthbench-t-f"

    calls: list = []
    monkeypatch.setattr("synth_lib.benchmark.driver.subprocess.run", lambda *a, **k: calls.append((a, k)))

    class FakeProc:
        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    run._kill(FakeProc())
    assert calls, "docker stop was not invoked"
    assert calls[0][0][0] == ["docker", "stop", "-t", "10", "synthbench-t-f"]
