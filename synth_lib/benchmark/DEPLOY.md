# DEPLOY — install, get data, smoke, run, score

Assumes a box provisioned per [PROVISIONING.md](PROVISIONING.md). Commands are shown from a
benchmark repository that depends on `synth-lib[benchmark]`; paths under `synth_lib/benchmark/` refer
to files inside this package (`python -c "import synth_lib.benchmark as b, os;
print(os.path.dirname(b.__file__))"` prints where they landed).

## 1. Install

```bash
uv sync --extra benchmark

# Proxy: one master key + your provider keys. NEVER commit this file.
cp <pkg>/metering/.env.proxy.example metering/.env.proxy
$EDITOR metering/.env.proxy          # LITELLM_MASTER_KEY=$(openssl rand -hex 32), provider keys

docker compose -f <pkg>/metering/docker-compose.yaml up -d
curl -s localhost:4000/health/liveliness           # -> I'm alive!

docker build -t synth-bench-sandbox <pkg>/sandbox/
```

The proxy's model list is `metering/litellm_config.yaml`. Every alias a campaign names must exist
there; the file ships with examples for several providers, and each one you add needs explicit
prices under `model_info` (a model with no price meters as **zero spend**, which silently disables
the budget cap). Read `metering/PROXY_COMPAT.md` before adding a provider or a CLI.

## 2. Data

A campaign needs two different things, from two different places.

| What                                                       | Purpose                                                                                                             | Source                                                                               |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| 1-minute closes per asset, in a local `market_data/` store | The **model's input**: the 7-day context each prompt hands to `simulate()`, and the price series agents backtest on | Venue clients in `synth_lib.preparation` (Binance, Hyperliquid), or your own archive |
| Realized paths + field scores/rewards                      | The **scoring ground truth**: the arrays the validator's own CRPS consumed, and the field to rank against           | The public Synth API, via `synth_lib.backtester.scripts.build_offline_bundle`        |

### Minute prices

`synth_lib.preparation` routes each asset to the venue it is scored against and writes daily parquet
partitions (`market_data/prices/{ASSET}/1m/date=YYYY-MM-DD.parquet`, a full 1440-row minute grid with
NaN where no trade occurred):

```bash
uv run python -m synth_lib.preparation.market_data --assets BTC ETH SOL XRP --start-days-ago 400
uv run python -m synth_lib.preparation.market_data --assets HYPE XAU NVDAX --recent-days 3
```

**Retention asymmetry, and it is severe.** Binance serves deep minute history for the cryptos.
Hyperliquid's candle endpoint returns at most ~5000 minutes — **about 3.5 days** — so the tokenised
equity/commodity perps and HYPE cannot be backfilled from the venue at all. Three options:

1. **Accumulate forward.** Run the recent-days ingest on a daily timer; the store grows day by day.
   Start it well before you need a campaign with a long commodity/equity window.
2. **An S3 archive of Hyperliquid history.** Deeper history than the candle endpoint serves exists
   in S3 archives of the venue's own data — Hyperliquid publishes raw archives, and some data
   providers republish them at coarser resolutions. These are requester-pays: you need AWS
   credentials that can be billed, and you pay request + egress. What you can reach depends on the
   provider, so check access before planning a window around it.

   **No ingester ships with this package** — you write the fetch and the store write yourself.
   Four things decide whether the result is usable:

   - **Listing dates bound everything.** The HIP-3 markets these tokenised assets trade on appear
     only from the day their market launched. There is no history before that, at any price; it is a
     fact about the market, not a gap in the archive.
   - **Aggregate to the last close in each minute.** Archives are finer-grained than a minute;
     `last(close)` per minute reproduces the venue's own 1-minute candle close. Aggregate a
     trade/candle close series — an order-book mid is a different series and will not match.
   - **Watch the numeric type.** Archives commonly store prices as fixed-point decimals. Decimal →
     float64 conversion is not always correctly rounded (a price like 7413.9 can land on
     7413.900000000001), which leaves an archive-derived minute one ULP from the same price fetched
     as JSON from the API. Route the cast through a string if the two must agree exactly.
   - **Mirror the partition contract**: one `date=YYYY-MM-DD.parquet` per asset-day, a full 1440-row
     minute grid, NaN where no trade occurred. That is what `MinutePriceStore.ingest_day` produces,
     and what the snapshot builder and `generate_predictions.py` expect.

   One caveat if you validate archive-derived minutes against minutes you swept live from the API:
   a sweep that requests data up to `now` receives the current, still-forming candle, whose close is
   only the last trade so far. Disagreement on those minutes means the live-swept side is
   provisional, not that the aggregation is wrong. Compare against a bulk-fetched window instead.

3. **Restrict the campaign** to the crypto competitions (`crypto-24h`, `crypto-1h`) until the store
   has depth. Scoring still works for all three competitions — scoring truth comes from the API, not
   from your store — but an agent cannot fit a model on data it does not have.

Never source prices from Pyth: the feed is retired, and `build_price_client` refuses the assets it
used to serve rather than hand back a client that fetches nothing.

### Realized paths (before `setup`)

We cache the realized paths so the agent can backtest offline instead of fetching one request per
scored prompt from the live API, on its own clock. They are also what makes scoring work where your
store is thin: the backtester slices realized prices out of the same local store, and any prompt whose
slice contains NaN — every HL-routed asset older than ~3.5 days — is scored from the validator's own
array instead, verbatim, so local CRPS matches the live network. They must land in
`market_data/realized/` **before `setup`**, because that is what `build_snapshot` hard-links into the
snapshot the sandbox mounts.

```bash
for SLUG in crypto-24h com-equ-24h crypto-1h; do
  uv run python -m synth_lib.backtester.scripts.build_offline_bundle \
    --competition "$SLUG" --days 32 --eval-end <data_cutoff>
done
```

### Field scores (after `setup`, before `run`)

The same script also writes scores/rewards/pool parquets, which land **outside** `market_data/` and
therefore never reach the sandbox. Without them, agents fetch field scores from the live API — which
works, but is paginated one day per request, so a month-long backtest is a lot of traffic and
concurrent agents rate-limit each other. Build that half directly into the snapshot instead:

```bash
BUNDLE=campaign_runs/<name>/snapshot/offline_data
for SLUG in crypto-24h com-equ-24h crypto-1h; do
  uv run python -m synth_lib.backtester.scripts.build_offline_bundle \
    --competition "$SLUG" --days 32 --eval-end <data_cutoff> --out "$BUNDLE" --no-realized-paths
done
```

`run_campaign` then exports `SYNTH_BACKTESTER_OFFLINE_DATA_ROOT` into every sandbox — but only if
that directory exists. The slug is in the filenames, so one directory serves all three competitions.
Note the bundle is not covered by `manifest.json` (which tracks only `date=*.parquet`), so
`verify_snapshot` cannot detect tampering with it.

### Before `setup`

```bash
ls market_data/prices/BTC/1m | tail -3      # does coverage reach your data_cutoff?
```

`build_snapshot` errors only when _no_ partition matches the window; it cannot detect a `data_cutoff`
that overshoots your real coverage, so an optimistic cutoff produces a quietly incomplete snapshot
that fails later, inside an agent's budget. Keep `data_cutoff` at least `horizon + 1d` behind today
or the backtester rejects the window as unsettled. Hard links are made **at setup**: data ingested
afterwards is invisible to a campaign already set up.

## 3. Smoke first, always

Two smokes before a real budget. The first costs nothing:

```bash
# 1. fake CLI: exercises setup, workspace, budget file, landing, nomination probe, collection
#    (a campaign yaml with `- {id: dry, cli: fake, model: fake-model}`)
uv run python -m synth_lib.benchmark.run_campaign setup --campaign campaigns/dry-run.yaml
uv run python -m synth_lib.benchmark.run_campaign run   --campaign campaigns/dry-run.yaml

# 2. real CLIs, tiny envelope (a few dollars, minutes): every adapter, auth, and the sandbox
export LITELLM_MASTER_KEY=$(grep '^LITELLM_MASTER_KEY=' metering/.env.proxy | cut -d= -f2-)
uv run python -m synth_lib.benchmark.run_campaign setup --campaign campaigns/smoke.yaml
uv run python -m synth_lib.benchmark.run_campaign run   --campaign campaigns/smoke.yaml
```

Export **only** the master key. With provider keys also in your shell, some CLIs send their own
credential alongside the proxy's and bypass metering.

What the real smoke must prove, per leg:

1. The container starts and the agent finds `CAMPAIGN.md` at the workspace root.
2. `uv sync` succeeds **inside** the sandbox (it needs the `bridge` network for pinned packages).
3. `uv run agent/predict.py --asset BTC --days 2 --eval-end <date>` prints a `summary:` — proof the
   read-only snapshot is readable and synth-lib scoring works in-container.
4. `BUDGET.md` is rewritten with climbing spend and the landing order arrives.
5. When `CHAMPION` appears, `check interface:` in `BUDGET.md` reads `ok (24h + 1h)`.
6. Isolation: inside the container, `ls /workspace` shows no operator code, and
   `ls /workspace/market_data/prices/BTC/1m | tail` stops at `data_cutoff`.

## 4. Run a campaign

```bash
export LITELLM_MASTER_KEY=$(grep '^LITELLM_MASTER_KEY=' metering/.env.proxy | cut -d= -f2-)
uv run python -m synth_lib.benchmark.run_campaign setup --campaign campaigns/<name>.yaml
nohup uv run python -m synth_lib.benchmark.run_campaign run --campaign campaigns/<name>.yaml \
    >> ~/<name>.log 2>&1 &
tail -f ~/<name>.log
```

`setup` is not idempotent (it refuses a non-empty workspace). Legs run sequentially in yaml order.
The run log narrates every launch, phase change, spend decile and exit decision; the agents' own
output goes to `campaign_runs/<name>/artifacts/<leg>/transcript-N.log`.

Campaign yaml essentials: `models` (each `{id, cli, model}` where `model` **must** match a served
alias), `budget_usd_per_model`, `deadline_hours_per_model`, `data_start`/`data_cutoff`,
`forward_window_days`, `hardware` (free text, shown to the agent), `gpu`, `sandbox_cpus`,
`sandbox_memory_gb`, `poll_seconds`. Per-turn CLIs wait up to one poll interval between turns, so
keep `poll_seconds` small (30–60) when a leg's CLI exits after each turn.

## 5. Verdict

After the scoring window has passed (and settled — the backtester refuses a window younger than
`horizon + 1d`):

```bash
uv run python -m synth_lib.benchmark.verdict.run_verdict --campaign <name> \
    --window-start <YYYY-MM-DD> --window-end <YYYY-MM-DD>
```

Per leg it clones the champion at `CHAMPION.sha`, builds its venv in one bridge-network sandbox pass,
generates every (competition, asset) in `--network none` sandboxes against the read-only store, and
scores on the host. Results land in `campaign_results/<campaign>/<leg>/verdict.json`. Flags worth
knowing: `--legs` (subset), `--tag` (an extra window without overwriting the canonical verdict),
`--force` (rescore), `--skip-baseline`, `--no-gpu`, `--keep-work`.

**Disk**: predictions are generated at the validator's 1000 paths, so one leg's full set is on the
order of 10 GB. The runner deletes each leg's set after scoring unless `--keep-work`. Do not lower
`--num-simulations` to save space — the field's CRPS was computed at 1000 paths, and a smaller
sample biases the candidate's CRPS upward.
