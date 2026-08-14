"""Campaign orchestrator: setup | run | verdict | status.

Usage:
  uv run python -m synth_lib.benchmark.run_campaign setup  --campaign campaigns/<n>/campaign.yaml
  uv run python -m synth_lib.benchmark.run_campaign run    --campaign campaigns/<n>/campaign.yaml
  uv run python -m synth_lib.benchmark.run_campaign verdict --campaign campaigns/<n>/campaign.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from synth_lib.benchmark.budget import BudgetTracker
from synth_lib.benchmark.campaign import CampaignConfig, load_campaign
from synth_lib.benchmark.cli_adapters import build_adapter
from synth_lib.benchmark.clock import RunClock
from synth_lib.benchmark.driver import ModelRun, RunResult
from synth_lib.benchmark.metering.keys import LiteLLMAdmin
from synth_lib.benchmark.sandbox.run_sandbox import DEFAULT_IMAGE, image_identity, sandbox_cmd
from synth_lib.benchmark.snapshot import build_snapshot, render_data_md
from synth_lib.benchmark.workspace import create_workspace

SANDBOX_IMAGE = DEFAULT_IMAGE
SANDBOX_NETWORK = "bridge"
# CLIs that reach the proxy via a passthrough route (URL forwarded verbatim, not /v1): their keys
# are minted uncapped.
PASSTHROUGH_CLIS = {"gemini-cli"}
# Where sandbox_cmd mounts the snapshot, and the bundle subdirectory inside it.
SANDBOX_MARKET_DATA = "/workspace/market_data"
OFFLINE_BUNDLE_DIR = "offline_data"
# synth_lib.backtester.config._OFFLINE_ENV_VAR. Duplicated rather than imported: the orchestrator
# deliberately does not pull the backtester (and its bittensor chain) in at startup.
OFFLINE_ROOT_ENV = "SYNTH_BACKTESTER_OFFLINE_DATA_ROOT"


def offline_bundle_env(snapshot: Path) -> dict[str, str]:
    """Point synth-lib's offline mode at a bundle inside the snapshot, when one is present.

    The bundle's scores/rewards/pool parquets are not date partitions, so build_snapshot does not
    link them — the operator builds them straight into <snapshot>/offline_data. With
    this set, the agent reads field scores from disk instead of paginating api.synthdata.co one day
    at a time (three concurrent agents would rate-limit each other); without it, synth-lib falls
    back to the live API, which works on the bridge network. It is NOT optional for the verdict,
    which runs with --network none.

    Absent bundle => empty dict, so a campaign without one behaves exactly as before.
    """
    if not (snapshot / OFFLINE_BUNDLE_DIR).is_dir():
        return {}
    return {OFFLINE_ROOT_ENV: f"{SANDBOX_MARKET_DATA}/{OFFLINE_BUNDLE_DIR}"}


def containerize_url(url: str) -> str:
    """Rewrites localhost/127.0.0.1 to host.docker.internal: for use INSIDE the container, where
    the host's proxy is not reachable via localhost."""
    return url.replace("//localhost", "//host.docker.internal").replace("//127.0.0.1", "//host.docker.internal")


def _sandbox_cmd_wrapper(
    *,
    workspace: Path,
    snapshot: Path,
    home: Path,
    env: dict[str, str],
    container_name: str,
    memory_gb: int,
    cpus: int,
    gpus: bool = True,
) -> Callable[[list[str]], list[str]]:
    """cmd_wrapper (spec ModelRun) that wraps the adapter's raw command (launch_cmd OR
    resume_cmd — both go through the same wrapper) in `docker run` via sandbox_cmd."""

    def wrapper(cmd: list[str]) -> list[str]:
        return sandbox_cmd(
            workspace=workspace,
            snapshot=snapshot,
            home=home,
            image=SANDBOX_IMAGE,
            memory_gb=memory_gb,
            cpus=cpus,
            network=SANDBOX_NETWORK,
            env=env,
            inner_cmd=shlex.join(cmd),
            name=container_name,
            gpus=gpus,
        )

    return wrapper


@dataclass
class CampaignState:
    cfg: CampaignConfig
    dir: Path

    @property
    def state_file(self) -> Path:
        return self.dir / "state.json"

    def load(self) -> dict:
        if not self.state_file.exists():
            return {"runs_done": []}
        try:
            return json.loads(self.state_file.read_text())
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"corrupt state ({self.state_file}): {exc}. " "Restore from a backup or rebuild runs_done by hand."
            ) from exc

    def save(self, data: dict) -> None:
        tmp = self.state_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, self.state_file)


def provision_home(adapter, home: Path) -> None:
    """Writes `adapter.provision_files()` into the run's mounted HOME, BEFORE launch:
    a codex without config.toml cannot reach the proxy. Keys start with `~/` — we resolve
    `~` to this `home`."""
    for rel, content in adapter.provision_files().items():
        assert rel.startswith("~/"), f"unexpected provisioning path (must start with ~/): {rel!r}"
        dest = home / rel[len("~/") :]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)


def setup_campaign(campaign_file: Path, data_root: Path, campaigns_root: Path | None = None) -> CampaignState:
    cfg = load_campaign(campaign_file)
    if campaigns_root is not None:
        cfg = replace(cfg, root=campaigns_root)
    state = CampaignState(cfg=cfg, dir=cfg.dir)
    state.dir.mkdir(parents=True, exist_ok=True)
    dest_yaml = state.dir / "campaign.yaml"
    if Path(campaign_file).resolve() != dest_yaml.resolve():
        shutil.copy(campaign_file, dest_yaml)
    # The market_data snapshot always comes from THIS repo (private source) — only the agent's
    # workspace becomes a standalone public repo (spec pivot "public workspace").
    build_snapshot(data_root / "market_data", state.dir / "snapshot", cutoff=cfg.data_cutoff, start=cfg.data_start)
    data_md = render_data_md(state.dir / "snapshot")
    for model in cfg.models:
        create_workspace(cfg, model, data_md=data_md)
    # The image is built from a Dockerfile in the deployment repo, so its CLI versions are versioned
    # apart from the engine. Recording its id is what lets a published result name what actually ran.
    state.save({"runs_done": [], "sandbox_image": image_identity(SANDBOX_IMAGE)})
    return state


def run_all(
    state: CampaignState,
    admin,
    sandboxed: bool = True,
    fake_env: dict[str, str] | None = None,
    python_exe: str = "python",
) -> list[RunResult]:
    cfg = state.cfg
    data = state.load()
    virtual_keys: dict[str, str] = data.setdefault("virtual_keys", {})
    results: list[RunResult] = []
    for model in cfg.models:
        if model.id in data["runs_done"]:
            print(f"== leg {model.id}: already in runs_done, skipping", flush=True)
            continue
        print(f"== leg {model.id} ({model.cli} / {model.model}): starting", flush=True)
        # Key persisted in state.json: a resume after a crash REUSES the same budget
        # (regenerating would hand out a fresh envelope, and LiteLLM aliases are unique => 400).
        if model.id in virtual_keys:
            key = virtual_keys[model.id]
            print(f"== leg {model.id}: reusing the persisted virtual key", flush=True)
        elif hasattr(admin, "generate_key"):
            # Passthrough-driven legs get an UNCAPPED proxy key: LiteLLM's budget reservation
            # leaks exactly max_budget on the /gemini passthrough, so any cap rejects everything
            # after the first call (litellm#27639 — a phantom budget 429). The
            # driver still enforces the real budget — BudgetTracker uses cfg.budget_usd_per_model
            # and true /key/info spend, never the key's own cap. Cost: no dead-driver backstop
            # for these legs, until the upstream fix lands.
            cap = None if model.cli in PASSTHROUGH_CLIS else cfg.budget_usd_per_model
            key = admin.generate_key(f"{cfg.name}-{model.id}", cap)
            virtual_keys[model.id] = key
            state.save(data)
            print(f"== leg {model.id}: minted key alias {cfg.name}-{model.id} (cap={cap})", flush=True)
        else:
            key = "sk-fake"
        # Inside the container, the host proxy is not reachable via localhost/127.0.0.1.
        proxy_url = containerize_url(cfg.proxy_url) if sandboxed else cfg.proxy_url
        adapter = build_adapter(model, proxy_url, key)
        # .resolve() once, here: a campaign root given as a relative path (the common case) is
        # otherwise re-interpreted by anything that changes directory — the champion interface
        # probe runs the nominated modeling.py out-of-process, and the fake CLI runs with
        # cwd=workspace. Absolute paths from this point down.
        workspace = (state.dir / "runs" / model.id).resolve()
        agent_dir = workspace / "agent"
        # Dedicated HOME per run (NOT under /tmp — codex refuses to create its helper
        # binaries there), provisioned BEFORE launch.
        home = state.dir / "runs" / f"{model.id}-home"
        home.mkdir(parents=True, exist_ok=True)
        provision_home(adapter, home)
        extra_env = dict(fake_env or {})
        extra_env["HOME"] = str(home)
        if model.cli == "codex":
            extra_env["CODEX_HOME"] = str(home / ".codex")
        if model.cli == "fake":
            # .resolve(): the fake CLI runs with cwd=workspace, so a campaign root given as a
            # relative path (the common case) would not resolve from inside the run.
            extra_env["FAKE_CLI_AGENT_DIR"] = str(agent_dir.resolve())
        container_name: str | None = None
        cmd_wrapper: Callable[[list[str]], list[str]] | None = None
        if sandboxed:
            container_name = f"synthbench-{cfg.name}-{model.id}"
            # HOME=/root INSIDE the container is implicit via mounting `home` onto /root
            # (sandbox_cmd) — the provisioning files above are already written there.
            cmd_wrapper = _sandbox_cmd_wrapper(
                workspace=workspace,
                snapshot=state.dir / "snapshot",
                home=home,
                # IS_SANDBOX=1: otherwise Claude Code refuses --dangerously-skip-permissions as root
                env={
                    **adapter.env(),
                    "IS_SANDBOX": "1",
                    **offline_bundle_env(state.dir / "snapshot"),
                },
                container_name=container_name,
                memory_gb=cfg.sandbox_memory_gb,
                cpus=cfg.sandbox_cpus,
                gpus=cfg.gpu,
            )
        tracker = BudgetTracker(
            admin=admin,
            virtual_key=key,
            budget_usd=cfg.budget_usd_per_model,
            soft_landing_pct=cfg.soft_landing_pct,
            clock=RunClock(deadline_seconds=cfg.deadline_hours_per_model * 3600),
        )
        run = ModelRun(
            cfg,
            model,
            adapter,
            tracker,
            workspace=workspace,
            artifacts_dir=state.dir / "artifacts" / model.id,
            extra_env=extra_env,
            python_exe=python_exe,
            container_name=container_name,
            cmd_wrapper=cmd_wrapper,
        )
        results.append(run.run())
        data["runs_done"].append(model.id)
        state.save(data)
    if "forward_window_start" not in data:
        data["forward_window_start"] = datetime.now(timezone.utc).isoformat()
        state.save(data)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["setup", "run", "verdict", "status"])
    ap.add_argument("--campaign", required=True, type=Path)
    ap.add_argument(
        "--data-root",
        "--repo-root",  # the old name: it never meant "the repo", only where market_data/ lives
        dest="data_root",
        type=Path,
        default=Path("."),
        help="directory containing market_data/ (default: cwd)",
    )
    ap.add_argument("--master-key-env", default="LITELLM_MASTER_KEY")
    args = ap.parse_args()
    cfg = load_campaign(args.campaign)
    state = CampaignState(cfg=cfg, dir=cfg.dir)
    if args.command == "setup":
        setup_campaign(args.campaign, args.data_root)
    elif args.command == "run":
        admin = LiteLLMAdmin(cfg.proxy_url, os.environ[args.master_key_env])
        run_all(state, admin=admin)
    elif args.command == "status":
        print(json.dumps(state.load(), indent=2))
    elif args.command == "verdict":
        raise SystemExit(
            "verdict: run after the forward window with "
            "`python -m synth_lib.benchmark.verdict.run_verdict --campaign <name> "
            "--window-start <YYYY-MM-DD> --window-end <YYYY-MM-DD>`."
        )


if __name__ == "__main__":
    main()
