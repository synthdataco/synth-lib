# Backtester update: 2-profile → 3-competition model

**Date:** 2026-07-11
**Status:** Approved design, pending implementation plan
**Author:** Nicolas (with Claude)

## Context

The backtester (`synth_lib/backtester/`) evaluates a local miner against historical
Synth (Bittensor SN50) subnet data. Its competition-structure assumptions are almost
entirely inherited from the pinned `synth-subnet` dependency, currently pinned in
`uv.lock` to **v1.0.0 @ commit `1c3e7fc`** (dated 2026-04-14).

Since that pin, the subnet was refactored (`main` HEAD `7714c5b8`, 2026-07-10, ~36
commits ahead). The defining change is **PR #283 "split in 3 competitions"** (~2026-06-18).
The backtester is now stale: it reads `PromptConfig` fields that no longer exist and
models 2 profiles where the subnet now scores 3 competitions.

### The reframing: prompt cycle ≠ scoring competition

The refactor **decoupled** two concepts that used to be one object:

- **`synth/validator/prompt_config.py`** — `LOW_FREQUENCY` (13 assets, 24h) and
  `HIGH_FREQUENCY` (5 crypto, 1h) now drive **only the prompt/query cycle** (which
  synapses get sent to miners, data retention, density tapering). The scoring fields
  (`scoring_intervals`, `window_days`, `softmax_beta`, `smoothed_score_coefficient`)
  were **removed** from `PromptConfig`.
- **`synth/validator/competition_config.py`** (new) — drives **scoring & rewards** via
  `ALL_COMPETITIONS = [COM_EQU_24H, CRYPTO_24H, CRYPTO_1H]`. Module constant
  `SMOOTHED_SCORE_COEFFICIENT = 1/3`.

| Competition | `label` | API slug | Assets | time_length / incr | window_days | softmax_beta |
|---|---|---|---|---|---|---|
| `CRYPTO_1H`  | `Crypto 1h`  | `crypto-1h`  | BTC, ETH, SOL, XRP, HYPE | 3600 / 60 | 5 | −0.3 |
| `CRYPTO_24H` | `Crypto 24h` | `crypto-24h` | BTC, ETH, SOL, XRP, HYPE | 86400 / 300 | 10 | −0.15 |
| `COM_EQU_24H`| `Commodities/Equities 24h` | `com-equ-24h` | XAU, SPYX, NVDAX, GOOGLX, TSLAX, AAPLX, WTIOIL, SPCX | 86400 / 300 | 10 | −0.15 |

The single 24h prompt cycle (`LOW_FREQUENCY`, 13 assets) writes scores for all 13
assets; the two 24h competitions each read back their disjoint asset subset. Routing
to a competition is purely by `(time_length, asset ∈ comp.asset_list)` — there is no
competition column on prompts/predictions.

**Aggregation:** each competition's `reward_weight` sums to `1/3` (softmax with
`comp.softmax_beta`, × `SMOOTHED_SCORE_COEFFICIENT`). `combine_moving_averages` sums a
miner's `reward_weight` across the competitions it appears in — a miner active on all
assets reaches up to `3 × 1/3 = 1.0`.

## Goal & scope (decided)

1. **Full re-levelling** — bump the pin and refactor the backtester end-to-end.
2. **Current era only** — support only the 3-competition structure; **warn** if a
   backtest window extends before the split (mirroring the existing 2026-03-11 CRPS
   warning). No dual-mode / pre-split model.
3. **Highest-fidelity bar** — structural correctness + green tests + **empirical live
   validation** + faithful treatment of the new mechanisms (density tapering, Pyth Pro
   / settled-witness, new cadence).

**Chosen architecture: Approach A — CompetitionConfig-native.** Replace `PromptConfig`
with `CompetitionConfig` throughout, mirroring the validator's own decoupling. The only
local "adapter" is a ~12-line slug↔label↔config helper. (Approaches B "local adapter"
and C "minimal patch" were rejected — B adds marginal value because
`compute_smoothed_score` requires the real `CompetitionConfig` anyway; C fights the
architecture since the `low` prompt now maps to two competitions.)

## Key facts established from live research (main HEAD)

- **CRPS did not change again.** The pinned commit (2026-04-14) already post-dates the
  2026-03-11 fix (#227). The only CRPS-touching commit since (#273, 2026-06-08) is
  reporting-only (`detailed_crps_data` breakdown); the scalar total CRPS fed into
  rewards is byte-for-byte unchanged. → the 2026-03-11 warning stays valid; no new
  formula cutoff.
- **The public API already exposes the competition dimension:**
  - `/validation/scores/historical` (per-asset CRPS): **no** competition field —
    derivable from `(time_length, asset)`. Backtester usage unchanged.
  - `/rewards/scores` (smoothed score / reward weight): `prompt_name` now accepts the
    slugs `crypto-1h` / `crypto-24h` / `com-equ-24h`. **Legacy `low` → `crypto-24h`
    only; legacy `high` → `crypto-1h` only; `com-equ-24h` has no legacy alias.** No
    `prompt_name` → all 3 merged. Each label's weights sum to `1/3`.
  - `/rewards/historical` (daily pool USD): **no** competition dimension, subnet-wide.
    Unchanged.
- **Owner miner:** before setting weights the validator appends an owner miner
  (miner_id=0, uid 248 mainnet / 23 testnet) whose `reward_weight = Σ(all others)`,
  so after on-chain normalization real miners' weights are effectively **halved**.

## Component design

### A. Dependency & the local helper

- Bump `synth-subnet` in `pyproject.toml` / `uv.lock`: `1c3e7fc` → `main` HEAD
  (`7714c5b8`), pinned to that commit for reproducibility.
- Import `ALL_COMPETITIONS`, `SMOOTHED_SCORE_COEFFICIENT` from
  `synth.validator.competition_config`. Drop `PromptConfig` from the backtester.
- New local helper in `backtest.py` (synth does not define API slugs):

```python
COMPETITION_SLUGS = {                       # comp.label -> API slug
    "Crypto 24h": "crypto-24h",
    "Crypto 1h": "crypto-1h",
    "Commodities/Equities 24h": "com-equ-24h",
}
def competition_for(asset, time_length): ...  # (asset ∈ asset_list, time_length) -> CompetitionConfig
def slug_for(comp): return COMPETITION_SLUGS[comp.label]
```

### B. `backtest.py` plumbing

| Current (broken) | New |
|---|---|
| `prompt_config: PromptConfig` (throughout) | `competition: CompetitionConfig` |
| `prompt_config.window_days / scoring_intervals / softmax_beta` | `competition.window_days / scoring_intervals / softmax_beta` |
| `prompt_config.smoothed_score_coefficient` (=0.5) | constant `SMOOTHED_SCORE_COEFFICIENT` (=1/3) |
| `_PROFILE_LABELS[(tl, ti)]` → low/high (`backtest.py:107`) | `competition_for(asset, tl)` → slug — the `(tl, ti)` key is **ambiguous** for the two 24h competitions |
| `prompt_label = "low" if tl == 86400 …` (`backtest.py:949`) | `slug_for(competition)` |
| offline `miner_scores_{asset}_{low\|high}.parquet` | `miner_scores_{asset}_{slug}.parquet` |
| offline `rewards_history_{low\|high}.parquet` | `rewards_history_{slug}.parquet` (3 files) |

### C. Fetch layer

- **`get_rewards_history`** (`backtest.py:375`): `prompt_name` moves from legacy
  `low`/`high` to the **slugs** `crypto-24h` / `com-equ-24h` / `crypto-1h`. **Critical
  bug fix** — today `low` returns only `crypto-24h`, silently dropping the 8
  commodity/equity assets, and each label sums to `1/3` not the intended full profile.
- **`get_miner_scores`** (`backtest.py:299`): unchanged — already per
  `(asset, time_length, time_increment)`; competition derived from asset membership.
  Only fetches assets in `competition.asset_list`.
- **`get_daily_miner_pool_usd`** (`backtest.py:437`): unchanged (subnet-wide pool).

### D. Smoothing & aggregation

- **`compute_combined_smoothed_scores`** (`backtest.py:696`): takes per-asset results
  of **one** competition + its `CompetitionConfig`; passes `competition` to
  `compute_smoothed_score` (reads `comp.softmax_beta`, applies `ASSET_COEFFICIENTS` on
  the asset subset, × `SMOOTHED_SCORE_COEFFICIENT`). `reward_weight` sums to `1/3`.
- **`_compute_grand_total_weights`** (`backtest.py:1860`): from "≥2 profiles" to
  **sum of the 3 competitions** → up to `1.0` (crypto miner in crypto-24h + crypto-1h;
  com/equ miner in com-equ-24h; a miner on every asset → 1.0).

### E. Earnings formula

- `_compute_earnings_df` (`backtest.py:1937`): `smoothed_score_coefficient` (0.5) →
  `SMOOTHED_SCORE_COEFFICIENT` (1/3). `share_of_round = reward_weight / (1/3)`.
- **Emission normalization (`EMISSION_NORMALIZATION_FACTOR`):** `reward_weight × pool /
  rounds` overestimates the USD a competitive miner actually earns, because realized
  emission is not exactly proportional to the set reward_weight. Make the factor
  **explicit** in `_compute_earnings_df` and **calibrate it empirically** against on-chain
  `/leaderboard` emissions. **Empirical result (window 2026-07-04→2026-07-06, all 3
  competitions, top-15 by actual USD):** raw bt/actual SUM = 1414.19/980.97 = **1.442**
  (a consistent +44–47% overestimate across top miners) → factor = actual/bt = **0.6935**.
  The naive "owner miner halves weights → ~2× overestimate" hypothesis was **not** borne
  out: uid 248 does not appear in `/leaderboard` for these windows, and the measured
  overestimate is ~1.44×, attributable to Yuma-consensus flattening rather than an owner
  take. NOTE: the calibration only works once `compute_backtester_usd` divides
  `rounds_per_day` **per competition** (crypto-1h's higher cadence would otherwise average,
  not sum, across competitions — an earlier bug that produced a spurious 0.357/2.804).
- `validate_earnings_formula.py`: `--prompt-name low|high` → `--competition
  crypto-24h|com-equ-24h|crypto-1h|all`. The actual side (`/leaderboard`) is aggregated
  per neuron (competition-agnostic), so the backtester side must **sum the 3
  competitions** per miner to compare like-for-like.
- `plot_earnings_comparison.py`: sum `LOW+HIGH` → sum the **3 competitions** per miner.

### F. Warnings (honours "current era only")

- **2026-03-11 CRPS warning:** kept. Gate `label != "high"` → `slug != "crypto-1h"`.
  `window_days` for the 1h competition is now **5** (was 3) → `safe_sim_reg =
  cutoff + 5d = 2026-03-16`. Rename `HF`→`crypto_1h` in identifiers.
- **New "3-competition split" warning:** constant `COMPETITION_SPLIT_DATE` (~2026-06-18).
  Warn if a backtest window starts before the split, for **any** competition (pre-split
  the structure was 2-profile ÷2 and `com-equ-24h` did not exist). **Implementation:**
  determine the true mainnet activation date by probing `/rewards/scores` for the first
  timestamp `com-equ-24h` appears, rather than hard-coding the commit date.

### G. Fidelity of the new mechanisms

| Mechanism | Real impact on a historical backtest | Plan |
|---|---|---|
| New cadence (5min / 2min) | None — `rounds_per_day` is computed from real API timestamps, not hard-coded | Verify; no code change |
| Density tapering (`thin_after_minutes`) | Likely small — tapering is uniform across miners; changes the density of prompts in the smoothing window | Measure via the empirical lever below; model only if the gap exceeds tolerance |
| Pyth Pro + settled-witness | Settled-witness = none (historical candles are closed). Pyth Pro prices may differ slightly from the free Pyth benchmarks endpoint the backtester uses | Measure local-CRPS vs API-stored-CRPS gap; keep the free endpoint + document if within tolerance, else consider Pyth Pro (API key required) |

**Empirical validation lever (free, underpins the whole "+validation" + "+fidelity"
bar):** `/validation/scores/historical` returns the **real per-prompt CRPS of all
miners**, and `/rewards/scores` the **real per-competition reward weights**. Feed the
API's real CRPS through our `compute_combined_smoothed_scores` and **compare the
reconstructed reward weights to the API's real weights**. This validates the entire
smoothing / aggregation / normalization pipeline against live, independent of the price
source. If it matches within tolerance, tapering + price source do not bias our
recompute, and that is documented.

### H. CLI (`run_backtest.py`)

- `PROFILES_BY_NAME` (low/high/all) → `COMPETITIONS_BY_NAME` keyed by slug
  (`crypto-1h` / `crypto-24h` / `com-equ-24h` / `all`) → subsets of `ALL_COMPETITIONS`.
- `--profile` → `--competition` (same slugs). `_build_filtered_profiles` →
  `_build_filtered_competitions` (intersect asset selection with `competition.asset_list`).
- `_run` iterates competitions (thread pool up to 3); grand-total across the 3.
- `_populate_random_dir` iterates `competition.asset_list / time_length / time_increment`.
- **Assumed subtlety:** BTC/ETH/SOL/XRP/HYPE are in both crypto-24h and crypto-1h;
  `--asset BTC --competition all` runs BTC twice (horizons 86400 vs 3600, distinguished
  by the `_86400` / `_3600` filename suffix). Correct — two distinct prediction horizons.

### I. Tests (`tests/backtester/test_backtest.py`)

- Imports `HIGH_FREQUENCY / LOW_FREQUENCY` → `ALL_COMPETITIONS` + `SMOOTHED_SCORE_COEFFICIENT`.
- Value fixes: `:856` sum `0.5` → `1/3`; `:1143` competition `asset_list`; `:1151`
  `window_days` 3 → **5**, `SAFE_SIM_REG` → 2026-03-16; `TestComputeEarningsDf` folds in
  the owner-miner factor.
- **Test to rework:** `test_xau_dominance_over_btc` (`:822`) mixes BTC + XAU in one
  `compute_combined` call — but BTC ∈ crypto-24h and XAU ∈ com-equ-24h are now
  **different competitions**. Rewrite using two assets in the **same** competition
  (e.g. XAU vs SPYX in com-equ-24h) to test asset-coefficient dominance.
- **New tests:** split-date warning; `competition_for` / `slug_for` helper; empirical
  reconstruction test (API-shaped CRPS → reward weights sum to 1/3 per competition,
  1.0 grand total).

### J. README

Update `--profile` → `--competition`, the HF/LF sections → 3 competitions, earnings
semantics (1/3 + owner-miner), the 2026-03-11 caveat (now crypto-1h, window 5d), and add
the split caveat.

## Delivery order

1. Bump the pin → `competition_config` imports resolve.
2. `backtest.py` core: helper, competition plumbing, slug fetch fix, smoothing, grand-total.
3. Earnings + owner-miner factor.
4. Warnings (crypto-1h rename + split warning; probe the activation date).
5. CLI `run_backtest.py`.
6. Tests reworked + new, green.
7. **Empirical validation** (reconstructed reward weights vs API; earnings vs on-chain)
   → calibrate the owner-miner factor + fidelity spikes (tapering, price source).
8. README.

## Open items / risks

- **Emission factor — RESOLVED:** calibrated to **0.6935** (see the Earnings section);
  it is an approximation with ~±a few % per-miner spread and may drift, so absolute USD is
  an estimate, not a guarantee. Recalibrate periodically with `validate_earnings_formula.py`.
- **`COMPETITION_SPLIT_DATE` — RESOLVED:** = **2026-06-23** (first `com-equ-24h` appearance
  in `/rewards/scores`; earlier ranges 404), not the 2026-06-18 commit date.
- **Pipeline correctness — VALIDATED:** `validate_reward_reconstruction.py` (crypto-24h,
  2026-07-05) shows reconstructed per-round reward_weight sums to exactly 1/3 and aligns
  to on-chain weights (mean abs error 1.45e-3); residual drift is attributable to the
  1-day CRPS window truncating the 10-day smoothing lookback, not a pipeline defect.
- **Pyth Pro price divergence** is measured, not assumed; switching the price source is
  out of scope unless the CRPS gap exceeds tolerance (would need a Pyth Pro API key).
