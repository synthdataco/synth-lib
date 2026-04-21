# synth-lib

Tools for downloading minute-level market data and backtesting miners against
[Synth Subnet](https://github.com/synthdataco/synth-subnet) scoring.

The library is organised into two pieces:

- [app/lib/preparation/market_data.py](app/lib/preparation/market_data.py) — downloads
  and stores minute closes as daily parquet partitions.
- [app/lib/backtester/backtest.py](app/lib/backtester/backtest.py) — scores a miner's
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
uv run app/lib/preparation/market_data.py
```

This downloads 15 months of history ending yesterday. The asset list comes
from `synth.validator.price_data_provider.PriceDataProvider` and includes
majors such as `BTC`, `ETH`, `SOL`, plus tokenised equities/commodities
(`XAU`, `SPYX`, `NVDAX`, `TSLAX`, `AAPLX`, `GOOGLX`).

### A single asset

```bash
uv run app/lib/preparation/market_data.py --asset BTC
```

### Recent N days (includes today)

```bash
# Last 3 days for BTC only, including today
uv run app/lib/preparation/market_data.py --asset BTC --days 3

# Last 7 days across every asset
uv run app/lib/preparation/market_data.py --days 7
```

`--days N` anchors the window at today (inclusive). Today's partition is
marked `is_final=False` and is re-downloaded on every subsequent run.

### Re-download existing partitions

```bash
uv run app/lib/preparation/market_data.py --asset BTC --force-refresh
```

### CLI flags

| Flag | Default | Description |
| --- | --- | --- |
| `--asset` | all assets | Single asset symbol; omit to download every supported asset |
| `--days` | (15-month default window) | Download N days ending today (inclusive); overrides the default window |
| `--force-refresh` | off | Re-download partitions that already exist on disk |

### From Python

```python
from app.lib.preparation.market_data import download_market_data, download_all_assets

# Custom historical window with the programmatic API
download_market_data("BTC", total_months=6, heldout_months=0)

# Recent N days, every asset
download_all_assets(days=30)
```

The Python helpers still accept `total_months` / `heldout_months` kwargs for
finer control; the CLI has been simplified to just `--asset` / `--days`.

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
accepted (see [load_prediction](app/lib/backtester/backtest.py)):

- Flat notebook format: `{"start_timestamp", "asset", "time_increment", "time_length", "paths", ...}`
- ArtifactManager format: `{"simulation_input": {...}, "prediction": [meta, meta, path, ...]}`

`paths` is a `num_simulations × (num_steps + 1)` array of simulated price paths
starting from the current price.

### Running the backtester

Targets are expressed as a pair: which frequency profile to run (`--profile`)
and which asset(s) to run within it (`--asset`). If no prediction files exist
under `miner_outputs/{miner_name}/predictions/`, the script auto-generates
random-walk predictions into a temp directory so the pipeline can be verified
end-to-end.

```bash
# Default: LOW_FREQUENCY, BTC, last 2 days
uv run app/lib/backtester/scripts/run_backtest.py --miner-name gbm_agent

# BTC across both profiles
uv run app/lib/backtester/scripts/run_backtest.py --miner-name gbm_agent --profile all --asset BTC

# Multiple assets in LOW_FREQUENCY
uv run app/lib/backtester/scripts/run_backtest.py --miner-name gbm_agent --profile low --asset BTC ETH TSLAX

# Every asset in HIGH_FREQUENCY
uv run app/lib/backtester/scripts/run_backtest.py --miner-name gbm_agent --profile high --asset ALL

# Full sweep: every profile × every asset
uv run app/lib/backtester/scripts/run_backtest.py --miner-name gbm_agent --profile all --asset ALL

# Longer window or custom predictions directory
uv run app/lib/backtester/scripts/run_backtest.py \
    --miner-name gbm_agent \
    --days 7 \
    --profile low --asset BTC \
    --predictions-dir /path/to/predictions
```

Assets not in a given profile's `asset_list` are silently skipped for that
profile (e.g. `--profile high --asset TSLAX` produces no runs, since TSLAX
isn't in HIGH_FREQUENCY).

CLI flags (see [run_backtest.py](app/lib/backtester/scripts/run_backtest.py)):

| Flag | Default | Description |
| --- | --- | --- |
| `--miner-name` | `btc_research` | Subdirectory under `miner_outputs/` for predictions and chart output |
| `--days` | `2` | Length of the backtest window (ending now) |
| `--profile` | `low` | Frequency profile: `low`, `high`, or `all` |
| `--asset` | `BTC` | One or more asset symbols, or `ALL` to expand to every asset in the selected profile(s) |
| `--predictions-dir` | `miner_outputs/{miner_name}/predictions` | Where to read predictions from |

### From Python

```python
from app.lib.backtester.backtest import backtest

result = backtest(
    miner_name="gbm_agent",
    asset="BTC",
    time_length=86_400,     # 24h prompts (LOW_FREQUENCY) — use 3_600 for HIGH_FREQUENCY
    time_increment=300,     # 5-minute steps
    n_backtest_days=7,
)

print(result.summary)       # {num_prompts, mean_crps, final_smoothed_score, ...}
result.prompt_df            # per-prompt CRPS and scores (incl. every other miner)
result.smoothed_scores      # per-round smoothed score + reward_weight
```

### Output

Charts are written to `miner_outputs/{miner_name}/charts/`:

- `rank_evolution_{asset}_{time_length}.png` — rank over time (1 = best)
- `crps_over_time_…png`, `crps_by_hour_…png`, `crps_by_day_…png`
- `crps_ratio_dist_…png` — distribution of your CRPS relative to median
- `weekly_percentile_…png` — percentile rank per calendar week

The console prints `num_prompts`, mean CRPS, and final smoothed score per
asset/profile combination.

## Tests

Unit and integration tests live in [tests/lib/backtester/](tests/lib/backtester/).

```bash
uv run pytest tests/lib/backtester/
```

## Notes

- The backtester pulls scored prompts and rewards history from
  `https://api.synthdata.co`. The Synth API rate-limits; requests are retried
  with exponential backoff. Long multi-asset runs take a while.
- `download_price_data` reads exclusively from local parquet partitions.
  Make sure the relevant `market_data/pyth/{ASSET}/1m/` directory is populated
  (see section 1) — the HIGH_FREQUENCY profile needs coverage up to today
  since its prompt window is only 1h.
