# synth_lib.benchmark — an autonomous-agent benchmark harness

This subpackage runs a coding CLI (Claude Code, Codex, Gemini CLI, Kimi Code, …) as an **autonomous
research agent** whose job is to improve a Synth SN50 miner, and then scores what it produced
against the real field of miners on the live subnet.

One *campaign* runs one or more *legs*, one leg per model, sequentially on the same hardware. A leg
is: an agent alone in a Docker sandbox with a frozen price snapshot, a fixed credit budget and a
fixed wall clock, told to do research and nominate a champion. When the campaign ends, the *verdict*
replays the validator's own scoring over a later window to find out whether the champion would
actually have earned anything.

The published results and the campaigns that produced them live in a separate repository; this
package is the engine.

## The six mechanisms

**Constitution.** Each leg gets a rendered `CAMPAIGN.md` (from `constitution/CAMPAIGN.md.tmpl`):
the objective, the competitions and their scoring, what data it has, the hard rules (no-lookahead,
prices from the snapshot only, scoring through synth-lib, no dissecting individual miners' served
predictions), the deliverables, and the scoring formula it will be judged by — pre-registered, so
the agent can design for the metric instead of guessing it.

**Envelope.** A leg has a credit budget and a deadline. `budget.py` maintains `BUDGET.md` inside the
workspace on every poll — credits, elapsed time, phase — which is the agent's only view of its own
runway. Spend comes from the proxy's ledger, never from a CLI's self-report.

**Landing.** At 85% of budget *or* time, the harness writes a non-negotiable LANDING ORDER into
`BUDGET.md`: stop experimenting, finalize the journal and report, nominate. If nothing lands within
a grace period the leg is killed and relaunched once with the landing order carried in the prompt
(plus the harness's own view of which deliverables are missing, so an agent that believes it has
already finished cannot argue with "no such file"). At 100% it is killed. Nominating is final: a leg
that stops with `agent/CHAMPION` present is done and forfeits the rest of its envelope.

**Metering.** All model traffic goes through a LiteLLM proxy (`metering/`), one budget-capped
virtual key per leg, so every model is measured with the same instrument and the ledger is the
single source of truth for spend. Probe any new provider or CLI before it carries a real leg (the
deployment repo has the runbook) and keep the answers: the adapters are written from them.

**Sandbox.** `sandbox/` builds one image with the pinned CLIs; each leg runs in a container with its
workspace, a read-only price snapshot at `/workspace/market_data`, a per-run HOME, capped CPU/RAM,
and optionally the GPU (one job at a time via `gpu_lock.py`). The agent's workspace is a standalone
git repo — no access to the operator's own code.

**Verdict.** `verdict/` clones the leg's workspace bundle, checks out `CHAMPION.sha` (so the scored
code is provably the nominated code), regenerates predictions for a scoring window inside a
`--network none` sandbox, and scores them with `synth_lib.backtester` — the validator's own CRPS and
aggregation — against the archived field.

## Scoring

```
Score = 100 × mean over the 3 competitions of min(1, champion_total_reward_weight
                                                    ÷ best_other_miner_total_reward_weight)
```

The candidate is injected into the archived field as one more miner over the scoring window and run
through the validator's own aggregation (per-asset coefficients → per-miner normalization →
softmax). Summing its `reward_weight` over the window's scoring rounds simulates what it would have
**earned**; the Score is that as a fraction of the best other miner's earnings, capped per
competition so one runaway competition cannot mask a weak one, and averaged unweighted because the
subnet pays a third of emissions per competition. 0 means it earned nothing; 100 means it matched or
beat the field's best everywhere. `beats_field` records where the cap bound.

The Score is deliberately top-heavy, because the subnet's payout is: the softmax concentrates
emissions near the top of the field, so climbing from 2nd to 1st is worth far more than 50th to
49th, and a mid-pack rank earns approximately nothing.

**Ranks are reported alongside it, and they are what you read to understand a result.** A Score of
zero says "earned nothing" but not *how far* from earning; ranks do. Every verdict carries:

- **average rank per competition** over the window's scoring rounds (`rank_over_rounds`: mean and
  best), plus the final-round rank the validator would actually have paid on, and the field size;
- **average rank per asset** (`per_asset[*].mean_rank` / `best_rank`), which is the only
  cross-window-comparable rank: per-asset scores are keyed by (asset, horizon, increment), so they
  remain meaningful across changes in how competitions are aggregated, whereas competition-level
  ranks and reward weights are only comparable within one reward regime.

Read Score for "would this have made money", per-competition mean rank for "how close", per-asset
mean rank for "where the strength actually is".

## Layout

| Path | What |
| --- | --- |
| `campaign.py` | campaign config (`CampaignConfig`, `ModelSpec`) + validation |
| `run_campaign.py` | `setup` (snapshot + workspaces) and `run` (sequential legs) entrypoints |
| `driver.py` | one leg's lifecycle: launch → poll → land → kill → collect |
| `budget.py`, `clock.py` | envelope tracking and the agent-visible `BUDGET.md` |
| `workspace.py`, `scaffold/` | the standalone agent workspace: starter model, evaluator, pins |
| `constitution.py`, `constitution/` | the rendered `CAMPAIGN.md` |
| `snapshot.py` | freezes `market_data` at the cutoff (hard links + manifest) and renders `DATA.md` |
| `nomination.py` | the `CHAMPION` contract and the `simulate()` interface probe |
| `cli_adapters.py` | per-CLI argv/env, including a `fake` CLI for zero-cost dry runs |
| `sandbox/` | `docker run` construction (mounts, caps, network, GPU) |
| `metering/` | the LiteLLM admin client: one budget-capped virtual key per leg |
| `verdict/` | prediction generation and scoring of an archived champion |
| `generate_predictions.py` | the generation core; copied into champion clones, so numpy/pandas/stdlib only |

## What this package does NOT ship

Everything you deploy or edit per box and per campaign — the sandbox `Dockerfile`, the metering
proxy's compose file and model list, `.env.proxy`, campaign yamls, and the operator runbooks
(provision → install → run → operate) — lives in the reference deployment, [synth-bench](https://github.com/synthdataco/synth-bench),
which is also where results are published. Nothing here reads those files; shipping them in a wheel
only meant copying them back out of site-packages. What stays is what the engine imports or reads:
the constitution template, the workspace scaffold, `generate_predictions.py`, the baseline, and the
`fake` CLI.

The image *name* is still the engine's business — `SYNTH_BENCH_SANDBOX_IMAGE`, default
`synth-bench-sandbox` — because the runner passes it to `docker run`; the recipe that builds it is
not.

## Not a general-purpose agent runner

The objective, the workspace scaffold and the whole verdict are specific to the Synth SN50 miner
problem. What generalizes is the pattern — pre-registered metric, metered envelope, forced landing,
sandboxed workspace, adversarial scoring against a live field — not the code paths.
