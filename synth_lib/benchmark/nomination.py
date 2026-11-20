"""Champion nomination: parsing the CHAMPION file + probing the simulate() contract."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pandas as pd
import yaml

from synth_lib.benchmark.generate_predictions import context_days_of, context_slice, load_module
from synth_lib.benchmark.sandbox.run_sandbox import SANDBOX_VENV, LegSandbox, sandbox_cmd

REPO_ROOT = Path(__file__).resolve().parent.parent
# Top-level site-packages entries of a uv venv that no distribution records: uv's venv files and the bytecode cache.
UV_VENV_FILES = frozenset({"__pycache__", "_virtualenv.pth", "_virtualenv.py"})


@dataclass(frozen=True)
class Champion:
    sha: str
    agent_dir: str
    profiles: tuple[str, ...]
    notes: str = ""


def _parse_champion_lenient(text: str) -> dict:
    """Line-wise recovery of the four CHAMPION keys when strict YAML fails.

    CHAMPION files are written by LLM agents and archived verbatim; agents have produced
    one whose single-line free-text `notes` contains unquoted colons — invalid YAML, fine
    nomination. Everything from the `notes:` key onward is taken as raw text."""
    raw: dict = {}
    lines = text.splitlines()
    for i, line in enumerate(lines):
        key, _, value = line.partition(":")
        key = key.strip()
        if key in ("sha", "commit", "agent_dir"):
            raw[key] = value.strip()
        elif key == "profiles":
            flow = value.strip()
            if flow.startswith("["):
                raw["profiles"] = [p.strip() for p in flow.strip("[]").split(",")]
            else:
                block = []
                for item in lines[i + 1 :]:
                    if not item.strip().startswith("-"):
                        break
                    block.append(item.strip().lstrip("-").strip())
                raw["profiles"] = block
        elif key == "notes":
            raw["notes"] = "\n".join([value.strip(), *lines[i + 1 :]]).strip()
            break
    return raw


def _champion_fields(path: Path) -> dict:
    text = path.read_text()
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return _parse_champion_lenient(text)


def parse_champion(path: Path) -> Champion:
    raw = _champion_fields(path)
    # `commit` accepted as an alias for `sha`: agents have nominated with that key, and
    # archived CHAMPION files are the agent's verbatim deliverable — tolerate here, never edit them.
    sha = raw["sha"] if "sha" in raw else raw["commit"]
    return Champion(
        sha=str(sha),
        agent_dir=str(raw["agent_dir"]),
        profiles=tuple(raw["profiles"]),
        notes=str(raw.get("notes", "")),
    )


def nominated_sha(path: Path) -> str:
    """The commit a CHAMPION file of any kind names: its `sha`, or `commit`; "" when it names none."""
    raw = _champion_fields(path)
    if not isinstance(raw, dict):
        return ""
    sha = raw.get("sha", raw.get("commit"))
    return "" if sha is None else str(sha)


def checkout_champion(workspace: Path, sha: str, dest: Path) -> None:
    """The tree of the workspace repo at commit `sha`, in `dest`, cloned with only the objects its refs reach,
    as the verdict clones the workspace bundle. ValueError with git's message when it cannot."""
    for cmd in (
        ["git", "clone", "-q", "--no-local", "--no-checkout", str(workspace), str(dest)],
        ["git", "-C", str(dest), "checkout", "-q", sha],
    ):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise ValueError(f"cannot check out {sha!r}: {result.stderr.strip()[-500:]}")


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def venv_differences(venv: Path, lockfile: Path) -> list[str]:
    """How `venv` differs from the environment `lockfile` installs: distributions it does not list or lists at
    another version, and site-packages entries no distribution installed."""
    locked: dict[str, set[str]] = {}
    for package in tomllib.loads(lockfile.read_text()).get("package", []):
        locked.setdefault(_normalized(package["name"]), set()).add(package.get("version", ""))
    sites = sorted(venv.glob("lib/python3*/site-packages"))
    differences = []
    owned = set(UV_VENV_FILES)
    dists = importlib.metadata.distributions(path=[str(site) for site in sites])
    for dist in sorted(dists, key=lambda dist: dist.metadata["Name"] or ""):
        name = _normalized(dist.metadata["Name"] or "")
        versions = locked.get(name)
        if versions is None:
            differences.append(f"{name} is not in uv.lock")
        elif "" not in versions and dist.version not in versions:
            differences.append(f"{name} is {dist.version}, uv.lock has {' or '.join(sorted(versions))}")
        owned.update(file.parts[0] for file in dist.files or [])
    for site in sites:
        differences += [
            f"site-packages/{entry.name} is from no package"
            for entry in sorted(site.iterdir())
            if entry.name not in owned
        ]
    return differences


def load_simulate(modeling_path: Path):
    # A champion imports the modules beside it by name, as when its folder is the script's own.
    folder = str(modeling_path.resolve().parent)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    spec = importlib.util.spec_from_file_location(f"champion_{modeling_path.parent.name}", modeling_path)
    assert spec is not None and spec.loader is not None  # always true for a .py file path
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.simulate


def probe_simulate(
    module: ModuleType,
    *,
    asset: str = "BTC",
    time_increment: int = 300,
    time_length: int = 86_400,
    num_simulations: int = 1000,
) -> tuple[bool, str]:
    """Checks the shape of the contract's return value, for the champion's loaded modeling module.
    Synthetic context => no network needed here; real network isolation is provided by the calling
    sandbox (--network none).

    num_simulations defaults to the validator's real serving size
    (synth.validator.prompt_config.PromptConfig.num_simulations = 1000) rather than a smaller
    convenience value: a champion that breaks at 1000 paths (memory, a hardcoded shape) must
    fail here, at nomination time, instead of later at verdict/live time where
    synth.validator.response_validation_v2 rejects any response with a different path count.

    The context has the shape the verdict hands: the champion's CONTEXT_DAYS, cut by context_slice. On
    success the message is the call's wall time."""
    try:
        days = context_days_of(module)
        start = pd.Timestamp("2026-01-05T00:00:00Z")
        idx = pd.date_range(start - pd.Timedelta(days=days), start, freq="1min")
        frame = pd.DataFrame(
            {c: 100.0 for c in ("open", "high", "low", "close")} | {"volume": 1.0, "trade_count": 1.0},
            index=idx,
        )
        context = context_slice(frame, start, time_length, days)
        began = time.monotonic()
        out = module.simulate(
            asset=asset,
            start_time=start.isoformat(),
            time_increment=time_increment,
            time_length=time_length,
            num_simulations=num_simulations,
            context_prices=context,
        )
        seconds = time.monotonic() - began
        if not isinstance(out, tuple) or len(out) != 2 + num_simulations:
            got = f"tuple of length {len(out)}" if isinstance(out, tuple) else type(out).__name__
            return False, f"expected return: tuple(start, increment, *{num_simulations} paths) — got {got}"
        expected_steps = time_length // time_increment + 1
        for i, path in enumerate(out[2:]):
            if len(path) != expected_steps or not all(isinstance(v, (int, float)) for v in path[:3]):
                return False, f"path {i}: length {len(path)} != {expected_steps} or non-numeric values"
    except Exception as exc:  # any champion failure (including a malformed shape) is a probe failure
        return False, f"simulate() raised: {type(exc).__name__}: {exc}"
    return True, f"{seconds:.1f} s"


def probe_simulate_subprocess(
    modeling_path: Path,
    timeout_s: float = 120.0,
    python_exe: str = sys.executable,
    extra_env: dict[str, str] | None = None,
) -> tuple[bool, str]:
    """Like probe_simulate, but out-of-process with a hard timeout.

    The nominated champion is untrusted code: running it in-process exposes the entire host
    environment (secrets such as LITELLM_MASTER_KEY), and
    a simulate() that blocks (infinite loop, GPU deadlock) would freeze run() forever — since the
    budget/deadline polling would no longer be running. So we launch
    `python -m synth_lib.benchmark.nomination <modeling_path>` in a subprocess with a MINIMAL environment
    (PATH/HOME/VIRTUAL_ENV only, never the full os.environ: the champion's code must not see the
    host's secrets) and a hard timeout. `extra_env` adds what the champion is entitled to, such as
    the path of the snapshot's calendar.

    It does not cut off the network: probe_in_sandbox does.
    """
    env = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ.get("HOME", "/tmp"),
        "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV", ""),
        **(extra_env or {}),
    }
    try:
        result = subprocess.run(
            [python_exe, "-m", "synth_lib.benchmark.nomination", str(modeling_path)],
            capture_output=True,
            timeout=timeout_s,
            cwd=REPO_ROOT,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return False, f"probe timeout after {timeout_s}s"
    return _probe_outcome(result)


def probe_in_sandbox(
    tree: Path,
    venv: Path,
    snapshot: Path,
    sandbox: LegSandbox,
    env: dict[str, str],
    timeout_s: float = 120.0,
) -> tuple[bool, str]:
    """probe_simulate_subprocess in the leg's sandbox, without network: `tree` mounted as the workspace, `venv`
    read-only at its place in it, and the venv's interpreter running the champion. `env` holds sandbox paths.
    Fails without starting the container when the tree has no uv.lock or `venv` differs from what it installs.
    Files the container leaves in its home are removed where the host user can."""
    lockfile = tree / "uv.lock"
    if not lockfile.exists():
        return False, (
            "uv.lock is not in the nominated commit: the scoring builds the environment from pyproject.toml and "
            "uv.lock, so commit both"
        )
    differences = venv_differences(venv, lockfile)
    if differences:
        return False, (
            f".venv differs from the uv.lock of the nominated commit, from which the scoring builds the environment: "
            f"{'; '.join(differences)}. `uv add` what the champion needs and commit, `uv sync` to drop the other "
            "packages, and delete what no package installed"
        )
    home = Path(tempfile.mkdtemp(prefix="synthbench-probe-"))
    name = home.name
    try:
        cmd = sandbox_cmd(
            workspace=tree,
            snapshot=snapshot,
            home=home,
            image=sandbox.image,
            memory_gb=sandbox.memory_gb,
            cpus=sandbox.cpus,
            network="none",
            env={**env, "PYTHONDONTWRITEBYTECODE": "1"},
            inner_cmd=f"{SANDBOX_VENV}/bin/python -m synth_lib.benchmark.nomination /workspace/agent/modeling.py",
            name=name,
            gpus=sandbox.gpus,
            venv=venv,
        )
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            # The container outlives its `docker run` client.
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
            return False, f"probe timeout after {timeout_s}s"
    finally:
        shutil.rmtree(home, ignore_errors=True)
    return _probe_outcome(result)


def _probe_outcome(result: subprocess.CompletedProcess) -> tuple[bool, str]:
    """The JSON line `_main` prints, or the end of stderr when there is none."""
    if result.returncode != 0:
        return False, result.stderr.decode(errors="replace")[-2000:]
    for line in result.stdout.decode(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        return bool(payload["ok"]), str(payload["msg"])
    return False, result.stderr.decode(errors="replace")[-2000:]


def probe_all_shapes(modeling_path: Path) -> tuple[bool, str]:
    """Probes BOTH competition formats: 24h (86400/300) and 1h (3600/60), and reports the champion's
    CONTEXT_DAYS with the wall time of one call of each.

    A champion that ignores time_increment/time_length passes the 24h format and breaks on
    crypto-1h — the check must therefore cover both, otherwise the failure only shows up at
    verdict time, too late for the agent to fix it."""
    try:
        module = load_module(modeling_path)
        days = context_days_of(module)
    except Exception as exc:  # an import that fails, or a CONTEXT_DAYS out of bounds
        return False, f"{type(exc).__name__}: {exc}"
    seconds = {}
    for label, ti, tl in (("24h", 300, 86_400), ("1h", 60, 3_600)):
        ok, msg = probe_simulate(module, time_increment=ti, time_length=tl)
        if not ok:
            return False, f"[{label}] {msg}"
        seconds[label] = msg
    return True, f"ok (24h + 1h), CONTEXT_DAYS {days}, one call {seconds['24h']} (24h) and {seconds['1h']} (1h)"


def _main() -> None:
    ok, msg = probe_all_shapes(Path(sys.argv[1]))
    print(json.dumps({"ok": ok, "msg": msg}))


if __name__ == "__main__":
    _main()
