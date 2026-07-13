# synth-lib

Tools for downloading minute-level market data and backtesting miners against
[Synth Subnet](https://github.com/synthdataco/synth-subnet) scoring.

The library is organised into two pieces:

- [synth_lib/preparation/market_data.py](synth_lib/preparation/market_data.py) — downloads
  and stores minute closes as daily parquet partitions.
- [synth_lib/backtester/backtest.py](synth_lib/backtester/backtest.py) — scores a miner's
  predictions against real price paths and computes smoothed scores / reward weights
  using the same logic the live validator runs.

## Requirements

- Python ≥ 3.12
- [`uv`](https://docs.astral.sh/uv/) for dependency management

Install dependencies once:

```bash
uv sync
```

## 1. Download market data

Data is fetched from Pyth (spot equities, commodities, majors) or Hyperliquid
(perps) depending on the asset, and cached under
`market_data/pyth/{ASSET}/1m/date=YYYY-MM-DD.parquet`. Existing finalised
partitions are skipped unless `--force-refresh` is passed.

### All supported assets, default 15-month window

```bash
uv run synth_lib/preparation/market_data.py
```

This downloads 15 months of history ending yesterday. The asset list comes
from `synth.validator.price_data_provider.PriceDataProvider` and includes
majors such as `BTC`, `ETH`, `SOL`, plus tokenised equities/commodities
(`XAU`, `SPYX`, `NVDAX`, `TSLAX`, `AAPLX`, `GOOGLX`).

### A single asset

```bash
uv run synth_lib/preparation/market_data.py --asset BTC
```

### Recent N days (includes today)

```bash
# Last 3 days for BTC only, including today
uv run synth_lib/preparation/market_data.py --asset BTC --days 3

# Last 7 days across every asset
uv run synth_lib/preparation/market_data.py --days 7
```

`--days N` anchors the window at today (inclusive). Today's partition is
marked `is_final=False` and is re-downloaded on every subsequent run.

### Re-download existing partitions

```bash
uv run synth_lib/preparation/market_data.py --asset BTC --force-refresh
```

### CLI flags

| Flag | Default | Description |
| --- | --- | --- |
| `--asset` | all assets | Single asset symbol; omit to download every supported asset |
| `--days` | (15-month default window) | Download N days ending today (inclusive); overrides the default window |
| `--force-refresh` | off | Re-download partitions that already exist on disk |

### From Python

```python
from synth_lib.preparation.market_data import download_market_data, download_all_assets

# Single asset, default window
download_market_data("BTC")

# Recent N days, every asset
download_all_assets(days=30)
```

### Output layout

```
market_data/
└── pyth/
    └── BTC/
        └── 1m/
            ├── date=2025-01-15.parquet
            ├── date=2025-01-16.parquet
            └── ...
```

Each parquet contains `timestamp`, `close`, `source`, `ingested_at`, and
`is_final` columns. Rows are minute-aligned UTC; gaps are stored as NaN.

## 2. Run a backtest

A backtest compares a miner's prediction files to real prices, computes CRPS
per prompt, and then replays the validator's smoothed-score / reward-weight
calculation to produce the miner's rank over time.

### Prediction files

The backtester reads prediction files from
`miner_outputs/{miner_name}/predictions/**/*.json`. Filenames must follow:

```
YYYY-MM-DD_HH:MM:SSZ_{ASSET}_{time_length}.json
```

For example: `2026-03-28_00:00:00Z_BTC_86400.json`. Two JSON layouts are
accepted (see [load_prediction](synth_lib/backtester/backtest.py)):

- Flat notebook format: `{"start_timestamp", "asset", "time_increment", "time_length", "paths", ...}`
- ArtifactManager format: `{"simulation_input": {...}, "prediction": [meta, meta, path, ...]}`

`paths` is a `num_simulations × (num_steps + 1)` array of simulated price paths
starting from the current price.

### Running the backtester

The subnet scores three **competitions**, defined in
`synth.validator.competition_config`. The backtester replays each one
independently:

| Competition | `--competition` slug | Assets | Horizon | `window_days` | `softmax_beta` |
| --- | --- | --- | --- | --- | --- |
| Crypto 1h | `crypto-1h` | BTC, ETH, SOL, XRP, HYPE | 1h | 5 | −0.3 |
| Crypto 24h | `crypto-24h` | BTC, ETH, SOL, XRP, HYPE | 24h | 10 | −0.15 |
| Commodities/Equities 24h | `com-equ-24h` | XAU, SPYX, NVDAX, GOOGLX, TSLAX, AAPLX, WTIOIL, SPCX | 24h | 10 | −0.15 |

Each competition's `reward_weight` sums to `SMOOTHED_SCORE_COEFFICIENT = 1/3`;
a miner's on-chain weight is the sum across the competitions it competes in
(grand-total up to 1.0).

Targets are expressed as a pair: which competition(s) to run (`--competition`)
and which asset(s) to run within them (`--asset`). Defaults are
`--competition all --asset ALL`, i.e. every asset in every competition. If no
prediction files exist under `miner_outputs/{miner_name}/predictions/`, the
script auto-generates random-walk predictions into a temp directory so the
pipeline can be verified end-to-end.

BTC, ETH, SOL, XRP, and HYPE belong to **both** `crypto-24h` and `crypto-1h`,
so `--asset BTC --competition all` runs BTC twice — once per horizon (86400 vs
3600 seconds). The two runs stay distinct via the `_86400` / `_3600` suffix in
their per-asset filenames.

```bash
# Default: every competition × every asset, last 2 days
uv run synth_lib/backtester/scripts/run_backtest.py --miner-name gbm_agent

# BTC across every competition (runs it in both crypto-24h and crypto-1h)
uv run synth_lib/backtester/scripts/run_backtest.py --miner-name gbm_agent --asset BTC

# Multiple assets in crypto-24h (space-, comma-, or quoted list all work)
uv run synth_lib/backtester/scripts/run_backtest.py --miner-name gbm_agent --competition crypto-24h --asset BTC ETH SOL
uv run synth_lib/backtester/scripts/run_backtest.py --miner-name gbm_agent --competition crypto-24h --asset BTC,ETH,SOL
uv run synth_lib/backtester/scripts/run_backtest.py --miner-name gbm_agent --competition crypto-24h --asset "BTC ETH SOL"

# Every asset in com-equ-24h only
uv run synth_lib/backtester/scripts/run_backtest.py --miner-name gbm_agent --competition com-equ-24h --asset ALL

# Longer window or custom predictions directory
uv run synth_lib/backtester/scripts/run_backtest.py \
    --miner-name gbm_agent \
    --days 7 \
    --competition crypto-24h --asset BTC \
    --predictions-dir /path/to/predictions
```

Assets not in a given competition's `asset_list` are filtered out for that
competition; if nothing matches anywhere the script exits with a clear error
listing the supported assets per competition.

CLI flags (see [run_backtest.py](synth_lib/backtester/scripts/run_backtest.py)):

| Flag | Default | Description |
| --- | --- | --- |
| `--miner-name` | `btc_research` | Subdirectory under `miner_outputs/` for predictions and chart output |
| `--days` | `2` | Length of the backtest window (ending now) |
| `--competition` | `all` | Competition to backtest: `crypto-1h`, `crypto-24h`, `com-equ-24h`, or `all` |
| `--asset` | `ALL` | One or more asset symbols (space-, comma-, or space-inside-quotes separated), or `ALL` for every asset in the selected competition(s) |
| `--predictions-dir` | `miner_outputs/{miner_name}/predictions` | Where to read predictions from |

### Parallelism

The runner uses three nested pools, so a full sweep finishes much faster than
the asset count would suggest:

- Competitions (`crypto-1h`, `crypto-24h`, `com-equ-24h`) run in a `ThreadPoolExecutor` (one thread per competition).
- Inside each competition, assets run in their own `ThreadPoolExecutor` (up to 6 at a time, I/O-bound on the Synth API).
- Inside each asset, per-prompt CRPS scoring is dispatched to a shared `ProcessPoolExecutor` sized to `cpu_count() - 2`.

### From Python

```python
from synth.validator.competition_config import CRYPTO_24H
from synth_lib.backtester.backtest import run_backtest, backtest

# Single asset. If `competition` is omitted, backtest() auto-resolves it from
# (asset, time_length) — here (BTC, 86400) → CRYPTO_24H.
single = backtest(
    miner_name="gbm_agent",
    asset="BTC",
    time_length=86_400,     # 24h prompts (crypto-24h) — use 3_600 for crypto-1h
    time_increment=300,     # 5-minute steps
    n_backtest_days=7,
    competition=CRYPTO_24H, # optional; inferred from (asset, time_length) if omitted
)

# Whole competition (parallel across assets + emits per-competition TOTAL charts when ≥2 succeed)
results, combined = run_backtest(
    miner_name="gbm_agent",
    competition=CRYPTO_24H,
    n_backtest_days=7,
)

print(single.summary)       # {num_prompts, mean_crps, final_smoothed_score, ...}
single.prompt_df            # per-prompt CRPS and scores (incl. every other miner)
single.smoothed_scores      # per-round smoothed score + reward_weight
```

### Output

Charts are written to `miner_outputs/{miner_name}/charts/`:

Per-asset (the `{time_length}` suffix is `86400` for the 24h competitions or
`3600` for `crypto-1h`, so an asset scored in two competitions produces two
sets of files):

- `rank_evolution_{asset}_{time_length}.png` — rank over time (1 = best)
- `crps_over_time_…png`, `crps_by_hour_…png`, `crps_by_day_…png`
- `crps_ratio_dist_…png` — distribution of your CRPS relative to median
- `weekly_percentile_…png` — percentile rank per calendar week

Per competition (emitted when ≥2 assets in that competition produce results),
keyed by the competition slug:

- `rank_evolution_TOTAL_{competition}.png` — combined rank across the competition's assets
- `estimated_earnings_{competition}.png` — per-round USD + cumulative earnings estimate

For example `estimated_earnings_crypto-24h.png` and
`rank_evolution_TOTAL_crypto-24h.png`.

Grand total (emitted when ≥2 competitions produced data):

- `rank_evolution_GRAND_TOTAL.png`
- `estimated_earnings_GRAND_TOTAL.png`

The console prints per-asset rank, reward weight, smoothed score, prompt
count, mean CRPS, and the paths of every saved chart.

### Earnings estimate

The `estimated_earnings_*` charts translate reward weights into USD. Each
competition's `reward_weight` sums to `SMOOTHED_SCORE_COEFFICIENT = 1/3`, and
the grand total sums the three competitions to ~1.0 of the miner emission pool.

USD figures apply an empirically-calibrated
`EMISSION_NORMALIZATION_FACTOR = 0.6935`. Raw `reward_weight × pool`
**overestimates** a competitive miner's realized emission by ~1.44× — measured
live over 2026-07-04→2026-07-06 across the top-15 miners, a consistent +44–47%
— because realized emission is flatter than the set weight (Yuma consensus).
The absolute USD number is therefore an **estimate**; recalibrate periodically
with [`scripts/validate_earnings_formula.py`](synth_lib/backtester/scripts/validate_earnings_formula.py),
which compares the backtester's USD against the on-chain `/leaderboard`.

## Known caveats

### crypto-1h CRPS formula change on 2026-03-11

On 2026-03-11 the Synth validator changed how it computes CRPS for the
`crypto-1h` competition (formerly the HIGH_FREQUENCY profile). CRPS values that
the API still returns for prompts scored before that date were produced by the
previous formula and are not directly comparable to the values the validator
computes today. Practically, that means a `crypto-1h` backtest whose window
includes any pre-cutoff prompts will mix two different scoring regimes, so the
resulting ranks and reward weights won't match what the same predictions would
receive on the live network now.

The smoothing window also looks back `competition.window_days` (5 days for
`crypto-1h`), so the smoothed score remains contaminated by pre-cutoff CRPS
until 2026-03-16.

If you see the corresponding `UserWarning` from `run_backtest`, you have two
ways to get current-formula ranks:

```bash
# 1. Restrict evaluation to dates on or after the formula change.
uv run synth_lib/backtester/scripts/run_backtest.py \
    --miner-name gbm_agent --competition crypto-1h \
    --eval-end 2026-04-15 --days 30

# 2. Simulate registering the miner on 2026-03-16 (= cutoff + window_days),
#    so the smoothing window is fully post-change.
uv run synth_lib/backtester/scripts/run_backtest.py \
    --miner-name gbm_agent --competition crypto-1h \
    --simulate-registration 2026-03-16
```

The other competitions (`crypto-24h`, `com-equ-24h`) are unaffected.

### 3-competition split on 2026-06-23

The subnet split from the old 2-profile model (LOW/HIGH_FREQUENCY) into the
current three competitions on 2026-06-23. The backtester models **only** the
current 3-competition era, so a backtest window that starts before 2026-06-23
mixes in the incompatible pre-split reward structure and produces misleading
ranks and reward weights. `run_backtest` emits a `UserWarning` in that case;
restrict backtest windows to dates on or after **2026-06-23**.

### Fidelity vs the live validator

A few live-validator behaviours differ from what the backtester replays. Each
was checked; none currently requires a code change.

**Reward-round cadence (crypto-24h ~5-min, crypto-1h ~2-min rounds).** No code
change is needed: the earnings `rounds_per_day` divisor is computed from the
real API round timestamps, so it auto-adapts to whatever cadence the validator
is running.

**Density tapering (`thin_after_minutes`).** Validated as non-biasing.
[`scripts/validate_reward_reconstruction.py`](synth_lib/backtester/scripts/validate_reward_reconstruction.py)
feeds the API's real per-prompt CRPS through the backtester's own
`compute_combined_smoothed_scores` and reproduces the on-chain reward weights:
the reconstructed per-round weights sum to exactly 1/3 and align to on-chain
values (mean absolute error 1.45e-3; crypto-24h, 2026-07-05). The small residual
is attributable to a 1-day CRPS window truncating the 10-day smoothing lookback,
not to tapering.

**Pyth Pro feed + settled-witness.** The settled-witness (waiting for the candle
to close) is moot for historical backtests, since past candles are already
settled. The backtester scores our miner against the free Pyth benchmarks
endpoint rather than the validator's Pyth Pro feed; a small absolute-CRPS
divergence is possible but was **not** directly measured (we lack other miners'
raw predictions to recompute their CRPS). It does not affect rank / relative
performance or the reward-weight reconstruction (which uses the API's own CRPS),
so keeping the free endpoint is the recommendation — revisit only if
absolute-CRPS calibration is ever required.

Two scripts back these checks:
[`scripts/validate_earnings_formula.py`](synth_lib/backtester/scripts/validate_earnings_formula.py)
(backtester USD vs the on-chain `/leaderboard`, per competition or `all`) and
[`scripts/validate_reward_reconstruction.py`](synth_lib/backtester/scripts/validate_reward_reconstruction.py)
(reward-weight reconstruction vs the live API).

## Tests

Unit and integration tests live in [tests/backtester/](tests/backtester/).

```bash
uv run pytest tests/backtester/
```

## Notes

- The backtester pulls scored prompts and rewards history from
  `https://api.synthdata.co`. The Synth API rate-limits; requests are retried
  with exponential backoff. Long multi-asset runs take a while.
- `download_price_data` reads exclusively from local parquet partitions.
  Make sure the relevant `market_data/pyth/{ASSET}/1m/` directory is populated
  (see section 1) — the `crypto-1h` competition needs coverage up to today
  since its prompt window is only 1h.
