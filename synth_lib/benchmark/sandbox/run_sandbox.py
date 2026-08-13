"""Builds the docker run command for a run (mounts, caps, network, GPU)."""

from __future__ import annotations

import os

from pathlib import Path

# One definition of the image name; override with SYNTH_BENCH_SANDBOX_IMAGE so an operator who
# builds their own image never has to edit code.
DEFAULT_IMAGE = os.environ.get("SYNTH_BENCH_SANDBOX_IMAGE", "synth-bench-sandbox")


def sandbox_cmd(
    *,
    workspace: Path,
    snapshot: Path,
    home: Path,
    image: str,
    memory_gb: int,
    cpus: int,
    network: str,
    env: dict[str, str],
    inner_cmd: str,
    name: str | None = None,
    gpus: bool = True,
) -> list[str]:
    # docker requires ABSOLUTE host paths (a relative path is read as a volume name)
    workspace, snapshot, home = workspace.resolve(), snapshot.resolve(), home.resolve()
    cmd = [
        "docker",
        "run",
        "--rm",
    ]
    if gpus:
        cmd += ["--gpus", "all"]
    cmd += [
        "--memory",
        f"{memory_gb}g",
        "--cpus",
        str(cpus),
        "--network",
        network,
        "--add-host",
        "host.docker.internal:host-gateway",
        "-v",
        f"{workspace}:/workspace:rw",
        "-v",
        f"{snapshot}:/workspace/market_data:ro",
        "-v",
        f"{home}:/root:rw",
        "-w",
        "/workspace",
    ]
    if name is not None:
        cmd += ["--name", name]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd += [image, inner_cmd]
    return cmd
