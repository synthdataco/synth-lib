"""Verdict runner: score archived champions, in the sandbox.

For each leg under campaign_results/<campaign>/ that has a CHAMPION + workspace.bundle:
  1. `git clone` the bundle and check out CHAMPION.sha — the scored code is provably the
     nominated code, not whatever the workspace drifted to after nomination.
  2. Copy in generate_predictions.py (self-contained by design).
  3. Phase 1 (network=bridge): `uv sync` in the clone — bundles are git-only, the venv must be
     built, and that needs the network. It fetches only pinned packages, never prices.
  4. Phase 2 (network=none): one sandbox run per (competition, asset) generating predictions.
     The data root is mounted read-only at /workspace/market_data; the model itself only ever
     receives the pre-prompt context Series (see generate_predictions.py).
  5. On the host: evaluate_candidate() against the offline bundle -> per-competition rank +
     simulated emissions -> Score = 100 x mean over competitions of reward_vs_top (see
     evaluate.reward_metrics; the unweighted mean mirrors the subnet's 1/3-per-competition
     emission split, and the softmaxed reward_weight makes top positions worth more, which a
     rank percentile would flatten).
  6. Write campaign_results/<campaign>/<leg>/verdict.json.

The synth_default baseline runs on the HOST (trusted repo code; note its known caveat below).

Usage (on the box, after ingesting prices up to window_end + 1 day):
  export LITELLM_MASTER_KEY=...   # not needed here, but the offline bundle build needs network
  uv run python -m synth_lib.benchmark.verdict.run_verdict --campaign <name> \\
      --window-start <YYYY-MM-DD> --window-end <YYYY-MM-DD>
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
from synth.validator.competition_config import ALL_COMPETITIONS  # type: ignore[import-untyped]

from synth_lib.benchmark.nomination import parse_champion
from synth_lib.benchmark.campaign import PACKAGED_BASELINE, baseline_modeling_path
from synth_lib.benchmark.sandbox.run_sandbox import DEFAULT_IMAGE, image_identity, sandbox_cmd
from synth_lib.benchmark.verdict.evaluate import evaluate_candidate, prepare_offline_bundle

GENERATE_SCRIPT = Path(__file__).resolve().parents[1] / "generate_predictions.py"
SANDBOX_IMAGE = DEFAULT_IMAGE
SANDBOX_MEMORY_GB = 12
SANDBOX_CPUS = 12
# Prompt grid density per format: hourly for the 24h competitions, 10-minute for crypto-1h —
# roughly how the field itself is sampled by the scores API.
CADENCE_MINUTES = {86_400: 60, 3_600: 10}


def run(cmd: list[str], what: str) -> None:
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        tail = (result.stdout + result.stderr).decode(errors="replace")[-2000:]
        raise RuntimeError(f"{what} failed (rc={result.returncode}):\n{tail}")


def docker_limits() -> tuple[int, int]:
    """SANDBOX_CPUS/MEMORY capped to what the local daemon actually has — a laptop's Docker
    Desktop VM is often smaller than the host, and docker hard-errors on an over-ask."""
    result = subprocess.run(["docker", "info", "--format", "{{.NCPU}} {{.MemTotal}}"], capture_output=True, text=True)
    if result.returncode != 0:
        return SANDBOX_CPUS, SANDBOX_MEMORY_GB
    ncpu, mem_bytes = result.stdout.split()
    return min(SANDBOX_CPUS, int(ncpu)), min(SANDBOX_MEMORY_GB, int(mem_bytes) // 1024**3)


def sandbox(
    workspace: Path,
    data_root: Path,
    home: Path,
    inner: str,
    network: str,
    gpus: bool,
    cpus: int = SANDBOX_CPUS,
    memory_gb: int = SANDBOX_MEMORY_GB,
) -> list[str]:
    return sandbox_cmd(
        workspace=workspace,
        snapshot=data_root,  # mounted read-only at /workspace/market_data
        home=home,
        image=SANDBOX_IMAGE,
        memory_gb=memory_gb,
        cpus=cpus,
        network=network,
        env={},
        inner_cmd=inner,
        gpus=gpus,
    )


def clone_champion(bundle: Path, sha: str, dest: Path) -> None:
    run(["git", "clone", "-q", str(bundle), str(dest)], f"clone {bundle}")
    run(["git", "-C", str(dest), "checkout", "-q", sha], f"checkout {sha}")


def generate_all(
    workspace: Path,
    data_root: Path,
    home: Path,
    window: tuple[str, str],
    gpus: bool,
    limits: tuple[int, int],
) -> None:
    """One --network none sandbox run per (competition, asset)."""
    if not GENERATE_SCRIPT.exists():  # packaging regression: it must ship with the package
        raise FileNotFoundError(f"generation core missing at {GENERATE_SCRIPT}")
    shutil.copy(GENERATE_SCRIPT, workspace / "generate_predictions.py")
    for comp in ALL_COMPETITIONS:
        for asset in comp.asset_list:
            inner = (
                "uv run python generate_predictions.py"
                " --modeling agent/modeling.py"
                f" --asset {asset}"
                f" --window-start {window[0]} --window-end {window[1]}"
                " --data-root /workspace/market_data --out-dir predictions"
                f" --cadence-minutes {CADENCE_MINUTES[comp.time_length]}"
                f" --time-increment {comp.time_increment} --time-length {comp.time_length}"
            )
            try:
                run(
                    sandbox(
                        workspace,
                        data_root,
                        home,
                        inner,
                        network="none",
                        gpus=gpus,
                        cpus=limits[0],
                        memory_gb=limits[1],
                    ),
                    f"generate {asset} tl={comp.time_length}",
                )
            except RuntimeError as exc:
                # Per-asset fault isolation, mirroring evaluate_candidate: a dead asset (real
                # case: SPYX — feed retired mid-July, no prices in any post-cutoff window) must
                # not sink the leg. No predictions -> the asset lands in assets_failed at scoring.
                print(f"  SKIP {asset} tl={comp.time_length}: {str(exc).splitlines()[-1]}", flush=True)
                continue
            print(f"  generated {asset} tl={comp.time_length}", flush=True)


def generate_baseline(modeling: Path, data_root: Path, out_dir: Path, window: tuple[str, str]) -> None:
    """Host-side, trusted repo code. Known caveat: the subnet's default generator ignores
    context_prices and anchors on a price it fetches ITSELF — for past prompts that anchor may be
    wrong, so read the baseline's verdict with suspicion and drop it from the article if broken."""
    for comp in ALL_COMPETITIONS:
        for asset in comp.asset_list:
            try:
                run(
                    [
                        "uv",
                        "run",
                        "python",
                        str(GENERATE_SCRIPT),
                        *("--modeling", str(modeling)),
                        *("--asset", asset),
                        *("--window-start", window[0], "--window-end", window[1]),
                        *("--data-root", str(data_root), "--out-dir", str(out_dir)),
                        *("--cadence-minutes", str(CADENCE_MINUTES[comp.time_length])),
                        *("--time-increment", str(comp.time_increment), "--time-length", str(comp.time_length)),
                    ],
                    f"baseline {asset} tl={comp.time_length}",
                )
            except RuntimeError as exc:
                print(f"  SKIP baseline {asset} tl={comp.time_length}: {str(exc).splitlines()[-1]}", flush=True)
                continue
            print(f"  baseline {asset} tl={comp.time_length}", flush=True)


def score(name: str, predictions: Path, window_end: pd.Timestamp, window_days: int) -> dict:
    result = evaluate_candidate(name, predictions, window_end, window_days)
    mrt = result["mean_reward_vs_top"]
    result["score"] = round(100 * mrt, 1) if mrt is not None else None
    ranks = [c["rank"] for c in result["per_competition"].values() if c["rank"] is not None]
    # Crude by design (field sizes differ per competition) but requested for cross-window
    # comparisons: the Score is softmax-top-weighted and near-zero for everyone off the podium,
    # while the mean rank still moves in the mid-field.
    result["mean_competition_rank"] = round(sum(ranks) / len(ranks), 1) if ranks else None
    return result


def verdict_payload(result: dict, sha: str | None, window: tuple[str, str]) -> dict:
    return {
        "score": result["score"],
        "mean_competition_rank": result["mean_competition_rank"],
        "score_definition": "100 x mean over competitions of min(1, candidate_total_reward /"
        " top_other_miner_total_reward) (validator reward_weight summed over the window; bounded"
        " 0-100: 0 = earned nothing, 100 = matched or beat the field's best in every competition;"
        " per-competition beats_field flags when the cap binds)",
        "mean_reward_vs_top": result["mean_reward_vs_top"],
        "mean_competition_percentile": result["mean_competition_percentile"],
        "per_competition": result["per_competition"],
        "window": {
            "start": window[0],
            "end": window[1],
            "note": "pseudo-out-of-sample: after the "
            "data cutoff, but agents had live field-API access during their runs",
        },
        "champion_sha": sha,
        "sandbox_image": image_identity(SANDBOX_IMAGE),
        "cadence_minutes": CADENCE_MINUTES,
        "num_simulations": 1000,
    }


def main() -> None:  # noqa: C901 — a linear operator script; splitting it would obscure the order
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--window-start", required=True)
    ap.add_argument("--window-end", required=True)
    ap.add_argument("--results-dir", type=Path, default=Path("campaign_results"))
    ap.add_argument("--data-root", type=Path, default=Path("market_data"))
    ap.add_argument("--legs", nargs="*", default=None, help="default: every leg with a CHAMPION")
    ap.add_argument(
        "--tag",
        default=None,
        help="write verdict-<tag>.json instead of verdict.json — use for extra windows (in-sample "
        "diagnostics) so they never overwrite the pre-registered window's canonical verdict",
    )
    ap.add_argument("--force", action="store_true", help="rescore legs whose verdict.json already exists")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument(
        "--baseline-module",
        default=PACKAGED_BASELINE,
        help="dotted module path or file path exposing simulate() (default: the packaged control)",
    )
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--keep-work", action="store_true", help="keep clones + predictions for inspection")
    args = ap.parse_args()

    campaign_dir = args.results_dir / args.campaign
    window = (args.window_start, args.window_end)
    window_end = pd.Timestamp(args.window_end, tz="UTC")
    window_days = (date.fromisoformat(args.window_end) - date.fromisoformat(args.window_start)).days
    data_root = args.data_root.resolve()

    print("building the offline scores bundle for the scoring window (network, resumable)...", flush=True)
    prepare_offline_bundle(window_end, window_days)  # also exports SYNTH_BACKTESTER_OFFLINE_DATA_ROOT

    legs = args.legs or sorted(p.name for p in campaign_dir.iterdir() if p.is_dir() and (p / "CHAMPION").exists())
    work = Path(tempfile.mkdtemp(prefix=f"verdict-{args.campaign}-"))
    limits = docker_limits()
    if limits != (SANDBOX_CPUS, SANDBOX_MEMORY_GB):
        print(f"docker daemon smaller than the reference box: sandboxes capped at {limits[0]} cpus / {limits[1]} GB")
    print(f"legs: {legs}; work dir: {work}", flush=True)

    verdict_name = f"verdict-{args.tag}.json" if args.tag else "verdict.json"
    for leg in legs:
        out = campaign_dir / leg / verdict_name
        if out.exists() and not args.force:
            print(f"[{leg}] {verdict_name} already exists — SKIPPING (pass --force to rescore)", flush=True)
            continue
        champion = parse_champion(campaign_dir / leg / "CHAMPION")
        clone = work / leg
        home = work / f"{leg}-home"
        home.mkdir(parents=True, exist_ok=True)
        print(f"[{leg}] clone @ {champion.sha}", flush=True)
        clone_champion(campaign_dir / leg / "workspace.bundle", champion.sha, clone)
        print(f"[{leg}] phase 1: uv sync (network)", flush=True)
        run(
            sandbox(
                clone, data_root, home, "uv sync", network="bridge", gpus=False, cpus=limits[0], memory_gb=limits[1]
            ),
            f"{leg} uv sync",
        )
        print(f"[{leg}] phase 2: generation (--network none)", flush=True)
        generate_all(clone, data_root, home, window, gpus=not args.no_gpu, limits=limits)
        print(f"[{leg}] scoring", flush=True)
        result = score(leg, clone / "predictions", window_end, window_days)
        out.write_text(json.dumps(verdict_payload(result, champion.sha, window), indent=2) + "\n")
        print(f"[{leg}] score={result['score']} mean_rank={result['mean_competition_rank']} -> {out}", flush=True)
        if not args.keep_work:
            shutil.rmtree(clone, ignore_errors=True)

    baseline_out = campaign_dir / "baselines" / "synth_default" / verdict_name
    if not args.skip_baseline and baseline_out.exists() and not args.force:
        print(f"[baseline] {verdict_name} already exists — SKIPPING (pass --force to rescore)", flush=True)
    elif not args.skip_baseline:
        baseline_modeling = baseline_modeling_path(args.baseline_module)
        predictions = work / "baseline-predictions"
        print("[baseline] generating (host)", flush=True)
        generate_baseline(baseline_modeling, data_root, predictions, window)
        result = score("synth_default", predictions, window_end, window_days)
        baseline_out.parent.mkdir(parents=True, exist_ok=True)
        baseline_out.write_text(json.dumps(verdict_payload(result, None, window), indent=2) + "\n")
        print(
            f"[baseline] score={result['score']} mean_rank={result['mean_competition_rank']} -> {baseline_out}",
            flush=True,
        )
        if not args.keep_work:
            shutil.rmtree(predictions, ignore_errors=True)

    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
