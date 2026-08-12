"""Champion nomination: parsing the CHAMPION file + probing the simulate() contract."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


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


def parse_champion(path: Path) -> Champion:
    text = path.read_text()
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError:
        raw = _parse_champion_lenient(text)
    # `commit` accepted as an alias for `sha`: agents have nominated with that key, and
    # archived CHAMPION files are the agent's verbatim deliverable — tolerate here, never edit them.
    sha = raw["sha"] if "sha" in raw else raw["commit"]
    return Champion(
        sha=str(sha),
        agent_dir=str(raw["agent_dir"]),
        profiles=tuple(raw["profiles"]),
        notes=str(raw.get("notes", "")),
    )


def load_simulate(modeling_path: Path):
    spec = importlib.util.spec_from_file_location(f"champion_{modeling_path.parent.name}", modeling_path)
    assert spec is not None and spec.loader is not None  # always true for a .py file path
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.simulate


def probe_simulate(
    modeling_path: Path,
    *,
    asset: str = "BTC",
    time_increment: int = 300,
    time_length: int = 86_400,
    num_simulations: int = 1000,
) -> tuple[bool, str]:
    """Checks the shape of the contract's return value. Synthetic context => no network needed here;
    real network isolation is provided by the calling sandbox (--network none).

    num_simulations defaults to the validator's real serving size
    (synth.validator.prompt_config.PromptConfig.num_simulations = 1000) rather than a smaller
    convenience value: a champion that breaks at 1000 paths (memory, a hardcoded shape) must
    fail here, at nomination time, instead of later at verdict/live time where
    synth.validator.response_validation_v2 rejects any response with a different path count."""
    try:
        simulate = load_simulate(modeling_path)
        start = pd.Timestamp("2026-01-05T00:00:00Z")
        idx = pd.date_range(end=start, periods=7 * 24 * 60, freq="1min", tz="UTC")
        context = pd.Series(100.0, index=idx)
        out = simulate(
            asset=asset,
            start_time=start.isoformat(),
            time_increment=time_increment,
            time_length=time_length,
            num_simulations=num_simulations,
            context_prices=context,
        )
        if not isinstance(out, tuple) or len(out) != 2 + num_simulations:
            got = f"tuple of length {len(out)}" if isinstance(out, tuple) else type(out).__name__
            return False, f"expected return: tuple(start, increment, *{num_simulations} paths) — got {got}"
        expected_steps = time_length // time_increment + 1
        for i, path in enumerate(out[2:]):
            if len(path) != expected_steps or not all(isinstance(v, (int, float)) for v in path[:3]):
                return False, f"path {i}: length {len(path)} != {expected_steps} or non-numeric values"
    except Exception as exc:  # any champion failure (including a malformed shape) is a probe failure
        return False, f"simulate() raised: {type(exc).__name__}: {exc}"
    return True, "ok"


def probe_simulate_subprocess(
    modeling_path: Path,
    timeout_s: float = 120.0,
    python_exe: str = sys.executable,
) -> tuple[bool, str]:
    """Like probe_simulate, but out-of-process with a hard timeout.

    The nominated champion is untrusted code: running it in-process exposes the entire host
    environment (secrets such as LITELLM_MASTER_KEY), and
    a simulate() that blocks (infinite loop, GPU deadlock) would freeze run() forever — since the
    budget/deadline polling would no longer be running. So we launch
    `python -m synth_lib.benchmark.nomination <modeling_path>` in a subprocess with a MINIMAL environment
    (PATH/HOME/VIRTUAL_ENV only, never the full os.environ: the champion's code must not see the
    host's secrets) and a hard timeout.

    Honest note: this does NOT cut off the network — real network isolation comes with the
    sandbox wiring (Task 14), which launches this same probe via `docker --network none`.
    """
    env = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ.get("HOME", "/tmp"),
        "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV", ""),
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
    """Probes BOTH competition formats: 24h (86400/300) and 1h (3600/60).

    A champion that ignores time_increment/time_length passes the 24h format and breaks on
    crypto-1h — the check must therefore cover both, otherwise the failure only shows up at
    verdict time, too late for the agent to fix it."""
    for label, ti, tl in (("24h", 300, 86_400), ("1h", 60, 3_600)):
        ok, msg = probe_simulate(modeling_path, time_increment=ti, time_length=tl)
        if not ok:
            return False, f"[{label}] {msg}"
    return True, "ok (24h + 1h)"


def _main() -> None:
    ok, msg = probe_all_shapes(Path(sys.argv[1]))
    print(json.dumps({"ok": ok, "msg": msg}))


if __name__ == "__main__":
    _main()
