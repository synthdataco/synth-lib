"""Lifecycle of a run: launch -> poll (budget/landing/crash) -> kill -> collect."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from synth_lib.benchmark.budget import BudgetStatus, BudgetTracker, Phase
from synth_lib.benchmark.campaign import CampaignConfig, ModelSpec
from synth_lib.benchmark.cli_adapters import CLIAdapter
from synth_lib.benchmark.nomination import probe_simulate_subprocess

INITIAL_PROMPT = "Read the CAMPAIGN.md file in your agent folder and start your research campaign."
RESUME_PROMPT = (
    "You are resuming after an interruption. Re-read CAMPAIGN.md, BUDGET.md, and your journal.md, "
    "then continue where you left off."
)
LANDING_PROMPT = (
    "IMMEDIATE LANDING ORDER: run NO new experiments. Finalize agent/journal.md, "
    "write agent/report.md, write the agent/CHAMPION file, commit all your work, That's it."
)
DELIVERABLES = ("report.md", "journal.md", "CHAMPION", "suggestions.md")


@dataclass
class RunResult:
    model_id: str
    landed: bool
    champion_path: Path | None
    crash_resumes: int
    voluntary_resumes: int
    landing_relaunches: int
    final_spend_usd: float


class ModelRun:
    def __init__(
        self,
        cfg: CampaignConfig,
        model: ModelSpec,
        adapter: CLIAdapter,
        tracker: BudgetTracker,
        workspace: Path,
        artifacts_dir: Path,
        extra_env: dict[str, str] | None = None,
        python_exe: str = "python",
        container_name: str | None = None,
        cmd_wrapper: Callable[[list[str]], list[str]] | None = None,
    ):
        self.cfg = cfg
        self.model = model
        self.adapter = adapter
        self.tracker = tracker
        self.workspace = workspace
        self.agent_dir = workspace / "agent"
        self.artifacts_dir = artifacts_dir
        self.extra_env = extra_env or {}
        self.python_exe = python_exe
        # Sandbox wiring: container_name lets _kill go through `docker stop` (killing the
        # `docker run` client does not kill the container); cmd_wrapper wraps the adapter's raw
        # command (e.g. via sandbox_cmd) when sandboxed=True, identity otherwise.
        self.container_name = container_name
        self.cmd_wrapper = cmd_wrapper or (lambda cmd: cmd)
        # Continue transcript numbering after any existing files: a rerun used to restart at 0
        # and OVERWRITE the previous attempt's transcripts one by one (the stale-transcript trap,
        # OPERATIONS.md) — now each attempt appends to the leg's audit trail instead.
        existing = [
            int(m.group(1))
            for p in artifacts_dir.glob("transcript-*.log")
            if (m := re.fullmatch(r"transcript-(\d+)\.log", p.name))
        ]
        self._transcripts = max(existing, default=-1) + 1
        self._probe_note = ""

    def _log(self, msg: str) -> None:
        """Operator-facing progress line (lands in the nohup log). The driver used to print
        nothing while healthy, which made every incident start with 'is it even running?'."""
        print(f"[{datetime.now(timezone.utc):%m-%d %H:%M:%S}] [{self.model.id}] {msg}", flush=True)

    # -- process management -------------------------------------------------
    def _spawn(self, cmd: list[str]) -> subprocess.Popen:
        if cmd and cmd[0] == "python":
            cmd = [self.python_exe] + cmd[1:]
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        log = open(self.artifacts_dir / f"transcript-{self._transcripts}.log", "wb")
        self._transcripts += 1
        env = {**os.environ, **self.adapter.env(), **self.extra_env}
        try:
            proc = subprocess.Popen(
                cmd, cwd=self.workspace, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        finally:
            log.close()  # the parent no longer needs its fd; the child keeps its own via dup()
        return proc

    def _launch(self, prompt: str, resume: bool = False) -> subprocess.Popen:
        if self.container_name:
            # A container can outlive its `docker run` client (e.g. the host OOM-killer takes the
            # client: the driver sees a "crash" while the container keeps the name). Without this,
            # every relaunch dies instantly on a name conflict and one transient crash burns the
            # whole crash-resume budget in seconds.
            subprocess.run(["docker", "rm", "-f", self.container_name], capture_output=True)
        cmd = (self.adapter.resume_cmd(prompt) if resume else None) or self.adapter.launch_cmd(prompt)
        kind = "resume" if resume else "fresh"
        self._log(f"launch ({kind}) -> transcript-{self._transcripts}.log")
        return self._spawn(self.cmd_wrapper(cmd))

    def _kill(self, proc: subprocess.Popen) -> None:
        """Kills the current run. Inside a container (container_name set), killing `proc`'s
        `docker run` client does NOT kill the container: you must go through `docker stop`.
        Outside the sandbox, kill the whole process group (start_new_session=True on the
        _spawn side), not just `proc`: a champion that launches a training job as a child
        (e.g. GPU) must not survive a landing/exhaustion kill."""
        if proc.poll() is not None:
            return
        if self.container_name:
            subprocess.run(["docker", "stop", "-t", "10", self.container_name], capture_output=True)
            proc.wait()
            return
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()

    # -- observation ---------------------------------------------------------
    def _landing_progress_mtime(self) -> float:
        """Last write time of the landing deliverables (report.md/CHAMPION) — NOT journal.md:
        an agent that keeps experimenting also journals; only the report signals landing."""
        times = [p.stat().st_mtime for p in (self.agent_dir / n for n in ("report.md", "CHAMPION")) if p.exists()]
        return max(times, default=0.0)

    def _champion(self) -> Path | None:
        p = self.agent_dir / "CHAMPION"
        return p if p.exists() else None

    def _landing_prompt(self) -> str:
        """LANDING_PROMPT + the driver's actual view of the deliverables.

        A static landing order loops forever against an agent that believes it is done:
        an agent that wrote its CHAMPION to the workspace root will answer "all set for handoff"
        to every identical landing order, since the harness looks only at agent/CHAMPION. An agent
        can argue with an instruction it thinks it followed; it cannot argue with "no such
        file"."""
        states = ", ".join(
            f"agent/{name} {'OK' if (self.agent_dir / name).exists() else 'MISSING'}"
            for name in ("journal.md", "report.md", "CHAMPION")
        )
        prompt = f"{LANDING_PROMPT}\nHarness status: {states}."
        if self._champion() is None and (self.workspace / "CHAMPION").exists():
            prompt += " Found CHAMPION at the workspace root — wrong location: move it to agent/CHAMPION now."
        return prompt

    def _probe_interface(self, status: BudgetStatus) -> None:
        """Checks the nominated champion's simulate() contract once and logs the result.
        The result is memoized on the instance (self._probe_note): once known, it must stay visible
        in BUDGET.md on every subsequent poll, not just the one where the probe ran.
        Out-of-process with a timeout (see synth_lib.benchmark.nomination.probe_simulate_subprocess): the
        nominated champion is untrusted code — running it in-process would expose the host env, and
        a blocking simulate() would freeze run() entirely."""
        ok, msg = probe_simulate_subprocess(self.agent_dir / "modeling.py", python_exe=self.python_exe)
        # Surface msg on success too, not a bare "OK": probe_all_shapes reports WHICH formats
        # passed ("ok (24h + 1h)"), and that distinction is the whole point of the check — a
        # champion that ignores time_increment passes 24h and breaks on crypto-1h.
        self._probe_note = f"check interface: {msg}"
        self.tracker.write_budget_file(self.agent_dir, status, notes=self._probe_note)

    def _handle_exit(
        self, proc: subprocess.Popen, exit_code: int, status: BudgetStatus, crash_resumes: int, voluntary_resumes: int
    ) -> tuple[bool, subprocess.Popen, int, int]:
        """Decides what happens after the process exits: clean landing, infra outage
        (wait it out), voluntary stop with no champion (unlimited relaunch, never a crash), or
        crash (relaunch capped by max_crash_resumes). Returns (done, proc, crash_resumes, voluntary_resumes)."""
        if exit_code == 0 and self._champion() is not None:
            self._log("clean landing: process exited with agent/CHAMPION present")
            return True, proc, crash_resumes, voluntary_resumes  # clean landing
        if status.infra_down:
            self._log("proxy unreachable: clock paused, waiting for it to come back")
            return False, proc, crash_resumes, voluntary_resumes  # clock paused: waiting for the proxy to come back
        if exit_code == 0:
            # voluntary stop with no champion: the normal case in a real campaign (the model
            # believes it's done) — the constitution says never to stop; relaunch without ever
            # consuming crash_resumes (bounded only by budget/deadline EXHAUSTED). During
            # Phase.LANDING, the relaunch must carry the landing order, not the generic RESUME
            # prompt ("continue where you left off"), which would kick off a new experiment
            # instead of finalizing the report/CHAMPION.
            self._log(f"voluntary stop without champion (phase {status.phase.value}): relaunching")
            prompt = self._landing_prompt() if status.phase is Phase.LANDING else RESUME_PROMPT
            return False, self._launch(prompt, resume=True), crash_resumes, voluntary_resumes + 1
        if crash_resumes >= self.cfg.max_crash_resumes:
            self._log(
                f"CRASH (exit {exit_code}) with resume cap exhausted ({crash_resumes}): leg ends WITHOUT champion"
            )
            return True, proc, crash_resumes, voluntary_resumes
        self._log(f"crash (exit {exit_code}): resume {crash_resumes + 1}/{self.cfg.max_crash_resumes}")
        return False, self._launch(RESUME_PROMPT, resume=True), crash_resumes + 1, voluntary_resumes

    # -- main loop -----------------------------------------------------------
    def run(self) -> RunResult:
        # BUDGET.md must exist from the very first turn: the constitution promises the agent it
        # can read it from the start, not only after the first poll_seconds sleep.
        status = self.tracker.status()
        self.tracker.write_budget_file(self.agent_dir, status, notes=self._probe_note)
        self._log(
            f"leg start: model={self.model.model} budget=${self.cfg.budget_usd_per_model:.2f} "
            f"deadline={self.cfg.deadline_hours_per_model}h"
        )
        proc = self._launch(INITIAL_PROMPT)
        crash_resumes = 0
        voluntary_resumes = 0
        landing_relaunches = 0
        landing_ordered_at: float | None = None
        probed = False
        last_decile = -1
        while True:
            time.sleep(self.cfg.poll_seconds)
            status = self.tracker.status()
            self.tracker.write_budget_file(self.agent_dir, status, notes=self._probe_note)
            decile = int(max(status.pct_usd, status.pct_time) * 10)
            if decile > last_decile:  # one line per 10% of envelope, not one per poll
                last_decile = decile
                self._log(
                    f"spend ${status.spend_usd:.2f} ({status.pct_usd:.0%}) time {status.elapsed_s / 3600:.1f}h "
                    f"({status.pct_time:.0%}) phase={status.phase.value}"
                )
            if not probed and self._champion() is not None:
                probed = True
                self._probe_interface(status)
                self._log(f"champion nominated; {self._probe_note}")
            if status.phase is Phase.EXHAUSTED:
                self._log("envelope EXHAUSTED: killing the leg")
                self._kill(proc)
                break
            if status.phase is Phase.LANDING:
                if landing_ordered_at is None:
                    landing_ordered_at = time.time()  # wallclock: compared against the deliverables' mtimes
                    self._log("LANDING phase entered: landing order now in BUDGET.md")
                grace_elapsed = time.time() - landing_ordered_at > self.cfg.landing_grace_seconds
                no_progress = self._landing_progress_mtime() <= landing_ordered_at
                if (
                    landing_relaunches == 0
                    and self._champion() is None
                    and grace_elapsed
                    and no_progress
                    and not status.infra_down
                ):
                    self._log("no landing progress within grace: kill + one relaunch with the landing order")
                    self._kill(proc)
                    landing_relaunches += 1
                    # resume=True: a fresh session must spend its remaining slice of envelope
                    # re-orienting before it can finalize anything, and a leg can exhaust its
                    # envelope doing that (new session, re-read CAMPAIGN.md, EXHAUSTED before
                    # CHAMPION). The agent that did the work is the one that should land it;
                    # _handle_exit already resumes with the landing prompt for voluntary stops.
                    proc = self._launch(self._landing_prompt(), resume=True)
            exit_code = proc.poll()
            if exit_code is not None:
                done, proc, crash_resumes, voluntary_resumes = self._handle_exit(
                    proc, exit_code, status, crash_resumes, voluntary_resumes
                )
                if done:
                    break
        self._collect()
        champion = self._champion()
        self._log(
            f"leg END: landed={champion is not None} spend=${status.spend_usd:.2f} "
            f"crash_resumes={crash_resumes} voluntary={voluntary_resumes} landing_relaunches={landing_relaunches}"
        )
        return RunResult(
            model_id=self.model.id,
            landed=champion is not None,
            champion_path=champion,
            crash_resumes=crash_resumes,
            voluntary_resumes=voluntary_resumes,
            landing_relaunches=landing_relaunches,
            final_spend_usd=status.spend_usd,
        )

    def _collect(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        for name in DELIVERABLES:
            src = self.agent_dir / name
            if src.exists():
                (self.artifacts_dir / name).write_bytes(src.read_bytes())
        self._bundle_workspace()

    def _bundle_workspace(self) -> None:
        """Archives the workspace git repo alongside the markdown deliverables.

        The champion is a `sha` inside that repo (see the CHAMPION contract), NOT the markdown:
        without this bundle, deleting the campaign directory destroys the only copy of the code
        the verdict has to score. `--all` keeps the full history, so the experiments the agent
        tried and discarded stay readable — usually more informative than the endpoint alone.
        Only committed work is captured: the driver kills mid-turn, so anything the agent left
        unstaged is lost, which the constitution already accounts for by requiring CHAMPION.sha
        to be an existing commit.
        Best-effort by design: a bundle failure must never cost the deliverables above."""
        # resolve(): `git -C <workspace>` resolves a relative bundle path UNDER the workspace,
        # so the real campaign (artifacts_dir relative to the repo root) wrote into
        # runs/<model>/campaign_runs/... and failed. Absolute-tmp_path tests never caught it.
        bundle = (self.artifacts_dir / "workspace.bundle").resolve()
        result = subprocess.run(
            ["git", "-C", str(self.workspace), "bundle", "create", str(bundle), "--all"],
            capture_output=True,
        )
        if result.returncode != 0:
            detail = result.stderr.decode(errors="replace").strip()[-300:]
            print(f"[{self.model.id}] workspace bundle FAILED ({self.workspace}): {detail}", flush=True)
            return
        print(f"[{self.model.id}] workspace bundled: {bundle} ({bundle.stat().st_size / 1e6:.1f} MB)", flush=True)
