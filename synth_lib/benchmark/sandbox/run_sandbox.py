"""Builds the docker run command for a run (mounts, caps, network, GPU)."""

from __future__ import annotations

import os
import subprocess

from pathlib import Path

# One definition of the image name; override with SYNTH_BENCH_SANDBOX_IMAGE so an operator who
# builds their own image never has to edit code.
DEFAULT_IMAGE = os.environ.get("SYNTH_BENCH_SANDBOX_IMAGE", "synth-bench-sandbox")


def image_identity(image: str = DEFAULT_IMAGE) -> dict[str, str | None]:
    """What actually ran: the image's content id, and its registry digest when it has one.

    The Dockerfile lives in the deployment repo, not here, so the CLI versions inside a sandbox are
    versioned separately from the engine. Recording the id is what keeps a campaign auditable — a
    tag is a moving pointer, an id is the bytes. Returns None fields when docker cannot answer
    (no daemon, image not built yet): a campaign must not fail over provenance metadata.
    """
    result = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}|{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {"name": image, "id": None, "digest": None}
    image_id, _, digest = result.stdout.strip().partition("|")
    return {"name": image, "id": image_id or None, "digest": digest or None}


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
