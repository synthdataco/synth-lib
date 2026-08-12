# OPERATIONS — what you run while a campaign is live

Companion to [DEPLOY.md](DEPLOY.md) (install and first run). Assumed once per shell:

```bash
export LITELLM_MASTER_KEY=$(grep '^LITELLM_MASTER_KEY=' metering/.env.proxy | cut -d= -f2-)
[ -n "$LITELLM_MASTER_KEY" ] && echo "key loaded (${#LITELLM_MASTER_KEY} chars)"
C=<campaign-name>
```

`run` needs that variable in the host env to mint per-leg virtual keys; `setup` does not.

## Preflight — before spending a cent

```bash
df -h / | tail -1                       # >40 GB free: each leg builds a torch venv
docker ps -a --filter name=synthbench --format '{{.Names}}' | grep . && echo "LEFTOVERS — remove them"

# provider keys present in the CONTAINER (lengths only, never values)
docker compose -f <pkg>/metering/docker-compose.yaml exec litellm sh -c \
  'for v in LITELLM_MASTER_KEY ANTHROPIC_API_KEY OPENAI_API_KEY GEMINI_API_KEY; do eval "n=\${#$v}"; echo "$v $n"; done'

curl -s localhost:4000/health/liveliness
docker images synth-bench-sandbox --format '{{.Repository}} {{.Size}}'

# every campaign model resolves to a served alias
uv run python -c "
import yaml
from pathlib import Path
from synth_lib.benchmark.campaign import load_campaign
served = {m['model_name'] for m in yaml.safe_load(Path('metering/litellm_config.yaml').read_text())['model_list']}
for f in sorted(Path('campaigns').glob('*.yaml')):
    c = load_campaign(f)
    for m in c.models:
        print(('OK  ' if m.model in served else 'MISS'), c.name, m.id, '->', m.model)
"

# one real call per NEW alias before it ever carries a leg (cents). 200 + non-zero cost = go.
curl -sS -D /dev/stderr -o /dev/null -X POST http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"<alias>","max_tokens":16,"messages":[{"role":"user","content":"Reply only: OK"}]}' \
  2>&1 | grep -iE "http/|response-cost|error"
```

A model with no `model_info` price meters as **zero spend**, which disables the budget cap without
erroring — that is why the probe checks for a non-zero cost, not just a 200.

## Launch and watch

```bash
uv run python -m synth_lib.benchmark.run_campaign setup  --campaign campaigns/$C.yaml
nohup uv run python -m synth_lib.benchmark.run_campaign run --campaign campaigns/$C.yaml >> ~/$C.log 2>&1 &
uv run python -m synth_lib.benchmark.run_campaign status --campaign campaigns/$C.yaml

tail -f ~/$C.log                                         # launches, phases, spend deciles, exits
watch -n 30 'grep -E "Credits|Time|Phase|interface" campaign_runs/'$C'/runs/*/agent/BUDGET.md'
docker ps --filter name=synthbench --format '{{.Names}}\t{{.Status}}'   # one container per ACTIVE leg
tail -f campaign_runs/$C/artifacts/<leg>/transcript-0.log               # the agent's own stream
```

Proof a leg is alive, in order of reliability: the `run` process exists (`pgrep -af run_campaign`);
`BUDGET.md`'s mtime is fresher than one poll interval (the driver rewrites it every poll); the newest
transcript is still growing. A container that disappears for less than a poll interval is a per-turn
CLI between turns, not a stopped leg.

## Inspect a finished campaign

```bash
cat campaign_runs/$C/state.json          # runs_done, virtual_keys, forward_window_start
# PASS per leg = non-empty report.md AND CHAMPION. runs_done lists a leg even if it never landed,
# so file sizes are the truth, not the state.
wc -c campaign_runs/$C/artifacts/*/{report.md,CHAMPION,journal.md} 2>/dev/null
cat campaign_runs/$C/artifacts/*/CHAMPION
grep "check interface:" campaign_runs/$C/runs/*/agent/BUDGET.md      # want: ok (24h + 1h)

# restore a leg's full workspace history from its bundle
git clone campaign_runs/$C/artifacts/<leg>/workspace.bundle /tmp/restored && git -C /tmp/restored log --oneline
```

## Spend

The proxy ledger is the single source of truth — every CLI self-reports a different number.
Passthrough routes post their cost asynchronously, so `sleep 5` before reading or a working setup
looks unmetered. The `x-litellm-key-spend` response header is the spend at request *start*, not the
cost of that call.

```bash
# per leg: spend vs cap, and WHICH model it was attributed to (catches a wrong-model fallback)
uv run python -c "
import json, os, requests
s = json.load(open('campaign_runs/$C/state.json'))
mk = os.environ['LITELLM_MASTER_KEY']
for m, k in s['virtual_keys'].items():
    i = requests.get('http://localhost:4000/key/info', headers={'Authorization': f'Bearer {mk}'}, params={'key': k}).json()['info']
    rows = requests.get('http://localhost:4000/spend/logs', headers={'Authorization': f'Bearer {mk}'}, params={'api_key': k}).json()
    per = {}
    for r in rows: per[r.get('model')] = per.get(r.get('model'), 0) + float(r.get('spend') or 0)
    print(f\"{m:8} spend={float(i.get('spend') or 0):.4f}/{i.get('max_budget')}  models={ {x: round(v,4) for x,v in per.items()} }\")
"

# token anatomy of one leg: many-stepped vs verbose vs uncached
LEG=<leg> uv run python -c "
import json, os, requests
s = json.load(open('campaign_runs/$C/state.json'))
mk = os.environ['LITELLM_MASTER_KEY']
rows = requests.get('http://localhost:4000/spend/logs', headers={'Authorization': f'Bearer {mk}'}, params={'api_key': s['virtual_keys'][os.environ['LEG']]}).json()
p = c = 0
for r in rows: p += int(r.get('prompt_tokens') or 0); c += int(r.get('completion_tokens') or 0)
print(f'{len(rows)} calls, prompt={p:,} completion={c:,} (avg {p//max(len(rows),1):,} prompt/call)')
"
```

### Warm-cache probe — after ANY proxy upgrade or pricing change

Custom prices belong in `model_info`, not `litellm_params` (the latter silently ignores
`cache_read_input_token_cost`). Verify per custom-priced alias, streaming **and** non-streaming —
CLIs that stream exercise a separate usage path:

```bash
LONG=$(python3 -c "print('The quick brown fox. ' * 400)")
for STREAM in false true; do for i in 1 2; do
  curl -sS http://localhost:4000/v1/chat/completions \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H 'Content-Type: application/json' \
    -d "{\"model\":\"<alias>\",\"stream\":$STREAM,\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"$LONG Reply OK\"}]}" > /dev/null
done; done
sleep 8
curl -s "http://localhost:4000/spend/logs" -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  | python3 -c "
import sys, json
for r in json.load(sys.stdin)[-4:]: print(r.get('spend'), r.get('prompt_tokens'), r.get('completion_tokens'))
"
# calls 2 and 4 hit a warm cache: their spend must be a fraction of calls 1 and 3, in BOTH pairs.
```

## Archive a finished campaign

Campaign working directories are disposable; what survives is a curated copy in the results
repository. Two rules: **`state.json` must be redacted** (it holds the virtual keys), and champion
code is published as an unpacked source tree rather than a git bundle.

```bash
D=<results-repo>/campaign_results/$C
mkdir -p "$D"
cp -r campaign_runs/$C/artifacts/* "$D/"                 # reports, journals, CHAMPION, transcripts, bundles
for leg in "$D"/*/; do
  R="campaign_runs/$C/runs/$(basename "$leg")/agent"
  cp "$R/CAMPAIGN.md" "$leg" 2>/dev/null || true         # the constitution the agent actually read
  cp "$R/BUDGET.md"   "$leg" 2>/dev/null || true         # last-poll spend/time
done
cp campaign_runs/$C/campaign.yaml "$D/campaign.yaml"
cp campaign_runs/$C/snapshot/manifest.json "$D/snapshot-manifest.json"
uv run python -c "
import json
s = json.load(open('campaign_runs/$C/state.json'))
s.pop('virtual_keys', None)                              # NEVER publish the keys
json.dump(s, open('$D/state.json', 'w'), indent=2)
"

# champion source in the clear, one tree per leg (drop the bundles before publishing)
for leg in "$D"/*/; do
  L=$(basename "$leg"); SHA=$(awk '/^(sha|commit):/ {print $2; exit}' "$leg/CHAMPION" 2>/dev/null) || continue
  [ -n "$SHA" ] && git clone -q "$leg/workspace.bundle" /tmp/ch-$L && git -C /tmp/ch-$L checkout -q "$SHA" \
    && rm -rf /tmp/ch-$L/.git && mkdir -p "$leg/champion" && cp -r /tmp/ch-$L/* "$leg/champion/" && rm -rf /tmp/ch-$L
done
```

Container-written files are root-owned on the host; a later `rm -rf` of the campaign directory needs
`sudo`. Publishing checklist before committing anything: no provider keys, no virtual keys, no
absolute paths from your box, and `state.json` redacted.

## Verdict — score archived champions on a later window

```bash
uv run python -m synth_lib.benchmark.verdict.run_verdict --campaign $C \
    --window-start <YYYY-MM-DD> --window-end <YYYY-MM-DD>
```

Two data sources, two roles — do not conflate them:

| Data | Source | Role |
| --- | --- | --- |
| 1-minute prices | your local store | the champion's INPUT: the 7-day context per prompt. synth-lib's loader also requires a partition to exist for every day it touches. |
| Realized paths + field scores | the public API, fetched by `prepare_offline_bundle` (automatic, first step) | the scoring GROUND TRUTH: the arrays the validator's own CRPS consumed, and the field to rank against. |

No-lookahead chain, per leg: clone the workspace bundle and check out `CHAMPION.sha` (the scored code
is provably the nominated code) → one `uv sync` sandbox pass on `bridge` (pinned packages, never
prices) → generation in `--network none` sandboxes with the data root mounted read-only → the
generator hands each prompt only `series.loc[t − 7d : t]`, which a dedicated test enforces.

Gotchas: the backtester refuses an unsettled window (`eval_end ≤ now − horizon − 1d`) and scoring
reads realized prices past `window_end`, so wait for the calendar. Use `--no-gpu` when something else
holds the GPU. Extra diagnostic windows go under `--tag <name>` so they never overwrite the canonical
`verdict.json`. The `synth_default` baseline runs host-side and anchors on a price it fetches itself,
so treat its verdict on past windows with suspicion.
