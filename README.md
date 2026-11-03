# synth-lib

Tools for downloading minute-level market data and backtesting miners against
[Synth Subnet](https://github.com/synthdataco/synth-subnet) scoring.

New to mining on SN50? Start with the [miner tutorial](https://github.com/synthdataco/synth-subnet/blob/main/docs/miner_tutorial.md) — backtesting is step 2.3 of that journey.

The library is organised into four packages:

- [synth_lib/preparation/](synth_lib/preparation/) — downloads minute OHLCV candles from each
  asset's venue and stores them as daily parquet partitions
  ([market_data.py](synth_lib/preparation/market_data.py) is the CLI).
- [synth_lib/backtester/](synth_lib/backtester/) — scores a miner's predictions against real price
  paths and computes smoothed scores / reward weights using the same logic the live validator runs
  ([orchestration.py](synth_lib/backtester/orchestration.py) holds `backtest`, `run_backtest` and
  `score_crps`).
- [synth_lib/benchmark/](synth_lib/benchmark/README.md) — runs a coding agent as an autonomous
  researcher to improve a miner, then scores the champion it nominates against the live field.
- [synth_lib/serving/](synth_lib/serving/) — serves a benchmark champion as an SN50 miner.

This README covers preparation and the backtester.

## Requirements

- Python ≥ 3.12
- [`uv`](https://docs.astral.sh/uv/) for dependency management

Install dependencies once:

```bash
uv sync
```

## 1. Download market data

Data is fetched from Binance spot (`BTC`, `ETH`, `SOL`, `XRP`) or Hyperliquid
(`HYPE` spot and the tokenised equity/commodity perps) depending on the asset,
mirroring the validator's routing, and cached under
`market_data/prices/{ASSET}/1m/date=YYYY-MM-DD.parquet`. Existing finalised
partitions are skipped unless `--force-refresh` is passed. If `market_data/prices/`
does not exist but a store laid out under the legacy `market_data/pyth/` name does,
that store keeps being read and written in place.

### All supported assets, default 15-month window

```bash
uv run synth_lib/preparation/market_data.py
```

This downloads 15 months of history ending yesterday. The asset list comes
from `synth.validator.price_data_provider.PriceDataProvider` and includes
majors such as `BTC`, `ETH`, `SOL`, plus tokenised equities/commodities
(`XAU`, `SP500`, `NVDAX`, `TSLAX`, `AAPLX`, `GOOGLX`).

### A single asset

```bash
uv run synth_lib/preparation/market_data.py --asset BTC
```

### Recent N days (includes today)

```bash
# BTC only: today and the 3 days before it
uv run synth_lib/preparation/market_data.py --asset BTC --days 3

# Every asset: today and the 7 days before it
uv run synth_lib/preparation/market_data.py --days 7
```

`--days N` downloads today and the N days before it (N + 1 daily partitions).
Today's partition is marked `is_final=False` and is re-downloaded on every
subsequent run.

### Re-download existing partitions

```bash
uv run synth_lib/preparation/market_data.py --asset BTC --force-refresh
```

### CLI flags

| Flag              | Default                   | Description                                                                                                               |
| ----------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `--asset`         | all assets                | Single asset symbol; omit to download every supported asset                                                               |
| `--days`          | (15-month default window) | Download today and the N days before it; overrides the default window                                                     |
| `--start`         | none                      | First day of an explicit inclusive `YYYY-MM-DD` window, given with `--end`; overrides `--days`                            |
| `--end`           | none                      | Last day of that window, e.g. to resume a crashed run from the day it stopped                                             |
| `--force-refresh` | off                       | Re-download partitions that already exist on disk; a re-download with fewer prices than the stored partition is discarded |

### From Python

```python
from synth_lib.preparation.market_data import download_market_data, download_all_assets

# Single asset, default window
download_market_data("BTC")

# Today and the 30 days before it, every asset
download_all_assets(days=30)
```

### Output layout

```
market_data/
└── prices/
    └── BTC/
        └── 1m/
            ├── date=2025-01-15.parquet
            ├── date=2025-01-16.parquet
            └── ...
```

Each parquet is a full 1440-row UTC minute grid with `timestamp`, `open`, `high`,
`low`, `close`, `volume`, `trade_count`, `source`, `ingested_at`, and `is_final`
columns. `close` is the series miners are scored against and the other market
columns describe the same minute; `source`, `ingested_at` and `is_final` record
provenance. A minute without a trade is the venue's flat candle at the previous
close with `trade_count` 0, which the validator scores against like any other. A
minute the venue returned no candle for is NaN in the six market columns.

### Assets with no deep history (Hyperliquid)

Hyperliquid serves only ~5000 candles per interval, so **1-minute history is
unreachable beyond ~3.5 days** for every Hyperliquid-routed asset: `HYPE` and
all of `com-equ-24h` (`XAU`, `NVDAX`, `TSLAX`, `AAPLX`, `GOOGLX`, `SP500`,
`SPCX`, `WTIOIL`). Older days ingest as NaN, and CRPS drops missing prices, so a
holed reference scores better than the real one would (an empty one scores 0).

`backtest()` and `run_backtest()` repair this automatically: for every prompt
whose local price slice has holes they fetch `/validation/realized-path` — the
array the validator's own CRPS consumed — and score against that, so the result
matches the live network exactly. Paths are cached per prompt under

```
market_data/realized/{ASSET}/{time_length}_{time_increment}/date=YYYY-MM-DD.parquet
```

and skipped on re-run, so an interrupted backfill resumes. Nothing needs
pre-downloading. Binance-routed assets (`BTC`, `ETH`, `SOL`, `XRP`) are
unaffected — they have full 1-minute history since listing.

Caveat: such a window is scoring-grade but not context-grade.
`get_context_window` and `get_real_price_path` raise on NaN, so model training
over an old Hyperliquid window is still not possible.

## 2. Run a backtest

A backtest compares a miner's prediction files to real prices, computes CRPS
per prompt, and then replays the validator's smoothed-score / reward-weight
calculation to produce the miner's rank over time.

### Prediction files

The backtester reads predictions from `miner_outputs/{miner_name}/predictions/`
in either of two layouts. One file per day, as written by
[generate_predictions.py](synth_lib/benchmark/generate_predictions.py):
`{ASSET}/{time_length}_{time_increment}/date=YYYY-MM-DD.npy`, a float32
`(prompts, simulations, steps)` array beside a `.json` index naming the prompt
`start_times` it holds. Or one JSON file per prompt, anywhere under that
directory, named:

```
YYYY-MM-DD_HH:MM:SSZ_{ASSET}_{time_length}.json
```

For example: `2026-03-28_00:00:00Z_BTC_86400.json`. A per-prompt file can use
either of two JSON layouts (see [load_prediction](synth_lib/backtester/loading.py)):

- Flat notebook format: `{"start_timestamp", "asset", "time_increment", "time_length", "paths", ...}`
- ArtifactManager format: `{"simulation_input": {...}, "prediction": [meta, meta, path, ...]}`

`paths` is a `num_simulations × (num_steps + 1)` array of simulated price paths
starting from the current price.

A scored prompt is answered by the latest prediction that starts at or before it,
up to an hour earlier (`backtest(match_tolerance_minutes=...)`). A prediction is
never matched to a prompt that started before it: its context would hold part of
the path that prompt is scored on. Predictions made at the scored prompts' own
start times (`get_miner_scores(...)["start_time"]`) answer every prompt exactly.

### Running the backtester

The subnet scores three **competitions**, defined in
`synth.validator.competition_config`. The backtester replays each one
independently:

| Competition              | `--competition` slug | Assets                                                | Horizon | `window_days` | `softmax_beta` |
| ------------------------ | -------------------- | ----------------------------------------------------- | ------- | ------------- | -------------- |
| Crypto 1h                | `crypto-1h`          | BTC, ETH, SOL, XRP, HYPE                              | 1h      | 5             | −0.3           |
| Crypto 24h               | `crypto-24h`         | BTC, ETH, SOL, XRP, HYPE                              | 24h     | 10            | −0.15          |
| Commodities/Equities 24h | `com-equ-24h`        | XAU, NVDAX, GOOGLX, TSLAX, AAPLX, WTIOIL, SPCX, SP500 | 24h     | 10            | −0.15          |

Each competition's `reward_weight` sums to `SMOOTHED_SCORE_COEFFICIENT = 1/3`;
a miner's weight is the sum across the competitions it competes in (up to 1.0
over these three). The validator also blends in Synth Ultra
(`VHFT_COMPETITION`, scored off-subnet) with the same coefficient, so once
`set_weights` normalizes, the four competitions split emissions equally. The
backtester does not replay it.

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

| Flag                        | Default                                  | Description                                                                                                                                                               |
| --------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--miner-name`              | `btc_research`                           | Subdirectory under `miner_outputs/` for predictions and chart output                                                                                                      |
| `--days`                    | `2`                                      | Length of the backtest window, ending at `--eval-end`                                                                                                                     |
| `--competition`             | `all`                                    | Competition to backtest: `crypto-1h`, `crypto-24h`, `com-equ-24h`, or `all`                                                                                               |
| `--asset`                   | `ALL`                                    | One or more asset symbols (space-, comma-, or space-inside-quotes separated), or `ALL` for every asset in the selected competition(s)                                     |
| `--predictions-dir`         | `miner_outputs/{miner_name}/predictions` | Where to read predictions from                                                                                                                                            |
| `--eval-end`                | now                                      | Window end, `YYYY-MM-DD`, read as 00:00 UTC of that date: prompts scored later that day are left out                                                                      |
| `--simulate-registration`   | none                                     | Registration date `YYYY-MM-DD`: drops the miner's rows for prompts starting before it, so the late-joiner backfill applies, and starts the window `window_days` before it |
| `--simulate-deregistration` | none                                     | Deregistration date `YYYY-MM-DD`: drops the miner's rows scored after it and ends the window there                                                                        |
| `--no-auto-bundle`          | off                                      | Fetch live for `--days` above 3 instead of building an offline bundle (see [Long backtests / offline mode](#long-backtests--offline-mode))                                |

### Parallelism

The runner uses three nested pools, so a full sweep finishes much faster than
the asset count would suggest:

- Competitions (`crypto-1h`, `crypto-24h`, `com-equ-24h`) run in a `ThreadPoolExecutor` (one thread per competition).
- Inside each competition, assets run in their own `ThreadPoolExecutor` (up to 6 at a time, I/O-bound on the Synth API).
- Inside each asset, per-prompt CRPS scoring is dispatched to a shared `ProcessPoolExecutor` sized to `cpu_count() - 2`.

### From Python

```python
from synth.validator.competition_config import CRYPTO_24H
from synth_lib.backtester.orchestration import backtest, run_backtest, score_crps

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

# CRPS alone, each prediction scored at its own start time against the local store: no field,
# no API call, so any window the store covers. crypto-1h uses the latest formula.
crps = score_crps(miner_name="gbm_agent", asset="BTC", time_length=86_400)
print(crps.summary)         # {num_prompts, mean_crps, realized_coverage, ...}
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
- `estimated_earnings_{competition}.png` — share of each round, daily USD and cumulative earnings estimate

For example `estimated_earnings_crypto-24h.png` and
`rank_evolution_TOTAL_crypto-24h.png`.

Grand total (emitted by `run_backtest.py` when ≥2 competitions ran and each of them had ≥2
assets with results):

- `rank_evolution_GRAND_TOTAL.png`
- `estimated_earnings_GRAND_TOTAL.png`

The console prints per-asset rank, reward weight, smoothed score, prompt
count, mean CRPS, realized coverage, and the paths of every saved chart.

### Earnings estimate

The `estimated_earnings_*` charts translate reward weights into USD. Each
competition's `reward_weight` sums to `SMOOTHED_SCORE_COEFFICIENT = 1/3`, and
the grand total sums the three competitions to ~1.0 of the miner emission pool.
The validator also rewards Synth Ultra (VHFT) as a fourth competition with the
same coefficient, which the estimate does not model.

USD figures apply an empirically-calibrated
`EMISSION_NORMALIZATION_FACTOR = 0.6935`. Raw `reward_weight × pool`
**overestimates** a competitive miner's realized emission by ~1.44× — measured
live over 2026-07-04→2026-07-06 across the top-15 miners, a consistent +44–47%
— because realized emission is flatter than the set weight (Yuma consensus).
The absolute USD number is therefore an **estimate**; recalibrate periodically
with [`scripts/validate_earnings_formula.py`](synth_lib/backtester/scripts/validate_earnings_formula.py),
which compares the backtester's USD against on-chain emissions from `/leaderboard/historical`.

### Long backtests / offline mode

The scores endpoints reject ranges over a few days, so a long backtest (e.g.
30 days) cannot fetch its data live in one call. **`run_backtest.py` handles
this automatically**: for `--days` above 3 it builds (or resumes) an offline
bundle under `offline_data/auto_{end}_{days}d/` and runs against it — no extra
steps. Pass `--no-auto-bundle` or set `SYNTH_BACKTESTER_OFFLINE_DATA_ROOT`
yourself to override.

To build a bundle manually (CI images, network-restricted sandboxes, or reuse
across many runs), use the same machinery directly:

```bash
# 1. Download 30 days of scores/rewards/pool in small chunks (resumable)
uv run synth_lib/backtester/scripts/build_offline_bundle.py \
    --competition crypto-24h --days 30 --eval-end 2026-09-01

# 2. Pre-download the price parquets covering the window (+ one horizon and the 7-day context)
uv run synth_lib/preparation/market_data.py --start 2026-07-25 --end 2026-09-01

# 3. Run the backtest against the bundle instead of the live API, on the bundle's end date
SYNTH_BACKTESTER_OFFLINE_DATA_ROOT=offline_data/crypto-24h \
    uv run synth_lib/backtester/scripts/run_backtest.py \
    --miner-name my_agent --competition crypto-24h --days 30 --eval-end 2026-09-01
```

The builder also caches the validator's realized path of every bundled prompt,
for every asset, into `market_data/realized/` (pass `--no-realized-paths` to
skip). A prompt whose local minute prices have gaps, as Hyperliquid-routed
assets do beyond the venue's few days of retention, is scored against it.

With `SYNTH_BACKTESTER_OFFLINE_DATA_ROOT` set, `get_miner_scores`,
`get_rewards_history`, and `get_daily_miner_pool_usd` read the bundled
parquets instead of calling the Synth API, and the realized-path fallback
reads the warmed cache without fetching. This also makes runs reproducible and
rate-limit-proof (prices still come from the local `market_data/` parquets
either way). Bundle layout under the root, one partition per UTC day:

```
miner_scores/{competition}/{ASSET}/date=YYYY-MM-DD.parquet
rewards_history/{competition}/date=YYYY-MM-DD.parquet
miner_pool_usd/date=YYYY-MM-DD.parquet
```

A rerun fetches only the days of its window that have no complete partition, so a wider window
adds only its new days (`--force-refresh` fetches every day again). A day fetched before it settled,
only up to the window's end, or that came back empty, is kept as `date=YYYY-MM-DD.partial.parquet`
and fetched again by the next run. Bundles built before day partitions
(`miner_scores_{ASSET}_{competition}.parquet` and so on) are still read.

Note the competition's smoothed-score window (`window_days`, 10 for the 24h
competitions) is trimmed from the start of the backtest as warmup, so a
30-day bundle yields ~20 days of ranked results.

## Known caveats

### crypto-1h CRPS formula change on 2026-03-11

On 2026-03-11 the validator changed how it computes `crypto-1h` CRPS (then the
HIGH_FREQUENCY profile), so the API's CRPS for earlier prompts comes from the
previous formula. `run_backtest` warns when a `crypto-1h` window starts before
that date. Such a window also predates the 3-competition split below, so the same
rule covers it: keep windows on or after 2026-06-23.

### crypto-1h volatility term

A `crypto-1h` prompt score is `price CRPS + λ × vol CRPS` (λ = 5.25): on top of
the CRPS on price changes, the validator scores the realized volatility of each
consecutive block of the hour — one 60-minute, four 15-minute and twelve
5-minute blocks — comparing the standard deviation of the 1-minute price
changes inside each block to the realized one.

The validator added this term on 2026-09-22 at 13:16 UTC. The backtester scores
each prompt with the formula live at its `scored_time`: your predictions on
prompts scored before the switch go without the term, like the field's CRPS from
the API, and later ones with it, so a window that spans the switch compares like
with like. `backtest(..., latest_formula=True)` scores every prompt with the term
instead, as `score_crps` does by default: CRPS then compares across the switch,
but ranks against the field before it do not, and `backtest` warns.

The other competitions (`crypto-24h`, `com-equ-24h`) have no volatility term.

### Outlier cap on 2026-09-04

Since 2026-09-04 09:45 UTC the validator clips any raw CRPS above 10× the
prompt's median CRPS, in every competition, and fills missed responses with the
95th percentile of the scores it did not clip. The backtester replays this by
`scored_time` too: prompts scored before then are scored without the clip.

Until about 2026-07-16 the validator also capped every score at the prompt's
90th percentile and filled missed responses with that value. The backtester does
not replay that cap, so prompts scored before then, and the rounds that average
them, are not scored as live scored them.

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

**Reward-round cadence.** No code change is needed: the earnings
`rounds_per_day` divisor counts each day's real API round timestamps, so it
follows whatever cadence the validator runs.

**Density tapering (`thin_after_minutes`).** Validated as non-biasing.
[`scripts/validate_reward_reconstruction.py`](synth_lib/backtester/scripts/validate_reward_reconstruction.py)
feeds the API's real per-prompt CRPS through the backtester's own
`compute_combined_smoothed_scores` and reproduces the on-chain reward weights:
the reconstructed per-round weights sum to exactly 1/3 and align to on-chain
values (mean absolute error 1.45e-3; crypto-24h, 2026-07-05, measured with the
90th-percentile cap the validator ran that day, which the backtester no longer
replays; see "Outlier cap"). The small residual
is attributable to a 1-day CRPS window truncating the 10-day smoothing lookback,
not to tapering.

**Price source + settled-witness.** The local store fetches the same 1-minute
venue candles the validator reads, routed the same way: Binance spot for
`BTC`/`ETH`/`SOL`/`XRP`, Hyperliquid for `HYPE` and `com-equ-24h`. A fetch keeps
a minute only once a later candle shows it closed, so the settled-witness holds
for every stored candle, and a minute without a trade is the venue's flat candle
in both. Where the store has no price, `backtest()` scores the prompt from the
validator's realized path. Prompts scored before 2026-07-23, when the validator
still read Pyth, can differ in absolute CRPS. The reward-weight reconstruction
uses the API's own CRPS and is unaffected.

Two scripts back these checks:
[`scripts/validate_earnings_formula.py`](synth_lib/backtester/scripts/validate_earnings_formula.py)
(backtester USD vs on-chain emissions from `/leaderboard/historical`, per competition or `all`) and
[`scripts/validate_reward_reconstruction.py`](synth_lib/backtester/scripts/validate_reward_reconstruction.py)
(reward-weight reconstruction vs the live API).

## Tests

Tests live under [tests/](tests/), one directory per area: `backtester`,
`preparation`, `serving`, `benchmark` and `evaluation`.

```bash
uv run pytest                    # everything
uv run pytest tests/backtester/  # the backtester only
```

## Notes

- The backtester pulls scored prompts and rewards history from
  `https://api.synthdata.co`. The Synth API rate-limits; requests are retried
  with exponential backoff. Long multi-asset runs take a while.
- `download_price_data` reads the local parquet partitions under
  `market_data/prices/{ASSET}/1m/` and fetches any missing day from the venue
  first (a read-only store raises instead). A day already on disk is read as it
  is, so refresh today's partition (`market_data.py --days 1`, see section 1)
  before a run that ends now; otherwise `backtest()` scores the prompts past its
  last fetch from the validator's realized paths.
- In `backtest()`, prompts older than Hyperliquid's ~3.5-day retention are scored
  from the validator's realized paths instead (see "Assets with no deep
  history"). The no-auth API endpoints are rate-limited to a couple of requests
  per second, so the first long Hyperliquid backtest spends a while warming that
  cache; later runs reuse it.
- `score_crps` uses the store alone. It leaves out a prediction with no price
  after its start, and scores one with gaps on the minutes it has, which makes
  its CRPS read better than it should: compare runs only at equal
  `realized_coverage`.
