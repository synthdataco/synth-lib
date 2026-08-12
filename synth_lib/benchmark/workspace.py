"""Scaffold of the campaign workspace: a standalone git repo, deliberately NOT a clone of the
operator's own repository — the agent starts from public sources and its own ideas only."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from synth_lib.benchmark.campaign import CampaignConfig, ModelSpec
from synth_lib.benchmark.constitution import render_constitution

SCAFFOLD_DIR = Path(__file__).parent / "scaffold" / "workspace"

# Anchored on the list brackets, not on the dep names, so reordering the scaffold's
# dependencies does not silently stop pre-seeding torch.
_DEPENDENCIES_RE = re.compile(r"(dependencies\s*=\s*\[)([^\]]*)(\])")
GPU_DEPENDENCY = '"torch>=2.0"'


def _add_gpu_dependency(pyproject: Path) -> None:
    """Pre-seeds torch for GPU campaigns.

    Left to the agent, every model spends budget discovering it must `uv add torch` and that
    CUDA works — three times over, under a deadline. Only for `gpu: true`: the wheel pulls
    ~3GB of bundled CUDA libs, which would eat a large slice of a short CPU campaign's
    deadline (smoke) for a dependency it never uses."""
    text = pyproject.read_text()
    patched, count = _DEPENDENCIES_RE.subn(
        lambda m: f"{m.group(1)}{m.group(2).rstrip()}, {GPU_DEPENDENCY}{m.group(3)}", text, count=1
    )
    if count != 1:
        raise RuntimeError(f"no `dependencies = [...]` found in {pyproject} — scaffold changed shape")
    pyproject.write_text(patched)


MAX_BLOB_BYTES = 50 * 1024 * 1024

# A guard rail, not a jail: `--no-verify` and CAMPAIGN_MAX_BLOB_BYTES both bypass it, deliberately
# — committed weights are how a champion runs offline. What it buys is that committing gigabytes
# becomes a visible, auditable act in the transcript instead of an accident. The bundle deliverable
# (`git bundle --all`) packs every committed blob, and a blob stays reachable from its commit even
# after a later commit deletes the file, so there is no undo.
_PRE_COMMIT_HOOK = """#!/bin/sh
# Installed by the campaign harness (synth_lib/benchmark/workspace.py).
limit="${CAMPAIGN_MAX_BLOB_BYTES:-__LIMIT__}"
oversized=$(git diff --cached --name-only --diff-filter=ACM | while read -r path; do
    blob=$(git rev-parse ":$path" 2>/dev/null) || continue
    size=$(git cat-file -s "$blob" 2>/dev/null) || continue
    if [ "$size" -gt "$limit" ]; then
        printf '  %s (%s bytes)\\n' "$path" "$size"
    fi
done)
if [ -n "$oversized" ]; then
    echo "pre-commit: refusing to commit files larger than $limit bytes:" >&2
    echo "$oversized" >&2
    echo "" >&2
    echo "Datasets, virtualenvs and generated predictions must stay UNTRACKED (see .gitignore):" >&2
    echo "only committed objects reach the workspace bundle, so untracked files cost nothing." >&2
    echo "If this file IS part of your champion (e.g. model weights it needs to run offline):" >&2
    echo "  git commit --no-verify" >&2
    exit 1
fi
exit 0
""".replace(
    "__LIMIT__", str(MAX_BLOB_BYTES)
)


def _install_pre_commit_hook(ws: Path) -> None:
    """Blocks accidental large commits. Must run AFTER `git init` (it writes into .git/hooks)
    and BEFORE the baseline commit, so the scaffold itself is checked by the same rule."""
    hook = ws / ".git" / "hooks" / "pre-commit"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(_PRE_COMMIT_HOOK)
    hook.chmod(0o755)


def _run_git(args: list[str], *, cwd: Path, action: str) -> None:
    try:
        subprocess.run(args, cwd=cwd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, (bytes, bytearray)) else str(exc.stderr)
        raise RuntimeError(f"{action} failed ({' '.join(args)}): {stderr}") from exc


def create_workspace(cfg: CampaignConfig, model: ModelSpec, data_md: str | None = None) -> Path:
    ws = cfg.dir / "runs" / model.id
    # Idempotence: detect an already-present workspace BEFORE any git command.
    if ws.exists() and any(ws.iterdir()):
        raise ValueError(
            f"workspace already present and non-empty: {ws} — clean it up with `rm -rf {ws}` before rerunning"
        )
    ws.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SCAFFOLD_DIR, ws)
    # Shipped as gitignore.tmpl because a packaged dotfile is easily lost by build backends;
    # written out under its real name here, the same post-processing the pyproject gets below.
    tmpl = ws / "gitignore.tmpl"
    if tmpl.exists():
        tmpl.rename(ws / ".gitignore")
    if cfg.gpu:
        _add_gpu_dependency(ws / "pyproject.toml")
    (ws / "agent" / "journal.md").write_text("")
    (ws / "agent" / "suggestions.md").write_text("")
    constitution = render_constitution(cfg, model)
    # Agents look at the workspace root first — the copy inside agent/ remains the reference
    # for the driver/deliverables.
    (ws / "CAMPAIGN.md").write_text(constitution)
    (ws / "agent" / "CAMPAIGN.md").write_text(constitution)
    if data_md is not None:
        # Written BEFORE the baseline commit so it travels in the workspace bundle: the
        # measured data facts (per-asset coverage, NaN semantics) generated from the snapshot.
        (ws / "DATA.md").write_text(data_md)
    _run_git(["git", "init", "-q"], cwd=ws, action="git init")
    _install_pre_commit_hook(ws)
    for args, action in (
        (["git", "add", "-A"], "git add"),
        (
            [
                "git",
                "-c",
                "user.email=agent@campaign.local",
                "-c",
                "user.name=campaign-harness",
                "commit",
                "-qm",
                "campaign baseline",
            ],
            "git commit",
        ),
    ):
        _run_git(args, cwd=ws, action=action)
    return ws
