# First-contact probe for a new CLI or provider

Run this before a new CLI or a new model alias ever carries a real campaign leg. It costs cents and
answers the questions that are version-dependent, each of them enough to break a leg on its own:
does the CLI accept the proxy as its endpoint, does headless mode work, does it resume, which wire
protocol does it speak, is the spend actually metered, and what does the proxy return when the budget
runs out. Keep the answers next to your campaign configs — the CLI adapters are written from them.

Prerequisite: `cp .env.proxy.example .env.proxy`, fill in the master key and the provider keys you
need, never commit it.

## 1. Proxy up, and a throwaway capped key

```bash
docker compose -f docker-compose.yaml up -d && sleep 10
curl -s http://localhost:4000/health/liveliness        # expected: "I'm alive!"
source .env.proxy
VKEY=$(curl -s -X POST http://localhost:4000/key/generate \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H "Content-Type: application/json" \
  -d '{"key_alias": "probe", "max_budget": 1.0}' | python3 -c "import sys,json;print(json.load(sys.stdin)['key'])")
```

## 2. Does the alias serve at all, and is it priced?

```bash
curl -sS -D /dev/stderr -o /dev/null -X POST http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $VKEY" -H 'Content-Type: application/json' \
  -d '{"model":"<alias>","max_tokens":16,"messages":[{"role":"user","content":"Reply only: OK"}]}' \
  2>&1 | grep -iE "http/|response-cost|error"
```

A 200 is not enough: `x-litellm-response-cost` must be **non-zero**. A model with no price in
`model_info` meters as $0.00, and a budget cap on a zero-cost model never binds.

## 3. The CLI, headless, through the proxy

Each CLI takes a different route. Verify the call lands, then that `/key/info` moved.

```bash
# Claude Code — Anthropic-compatible route
ANTHROPIC_BASE_URL=http://localhost:4000 ANTHROPIC_AUTH_TOKEN=$VKEY \
  claude -p "Reply only: OK" --model <alias> --output-format json

# Codex — needs a provider block; `wire_api` is version-dependent (newer releases: "responses")
mkdir -p ~/.codex && cat > ~/.codex/config.toml <<'EOF'
model = "<alias>"
model_provider = "litellm"
[model_providers.litellm]
name = "LiteLLM"
base_url = "http://localhost:4000/v1"
env_key = "LITELLM_KEY_CODEX"
wire_api = "responses"
EOF
LITELLM_KEY_CODEX=$VKEY codex exec --skip-git-repo-check "Reply only: OK"

# Gemini CLI — the /gemini passthrough; the model id goes in the URL path, so it must be a real one
GOOGLE_GEMINI_BASE_URL=http://localhost:4000/gemini GEMINI_API_KEY=$VKEY GEMINI_SANDBOX=false \
  gemini -p "Reply only: OK" --output-format json

# Kimi Code — OpenAI-compatible via its KIMI_MODEL_* env family
KIMI_MODEL_NAME=<alias> KIMI_MODEL_API_KEY=$VKEY KIMI_MODEL_PROVIDER_TYPE=openai \
  KIMI_MODEL_BASE_URL=http://localhost:4000/v1 kimi -p "Reply only: OK"

sleep 5   # passthrough routes post cost asynchronously
curl -s "http://localhost:4000/key/info?key=$VKEY" -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  | python3 -m json.tool | grep spend
```

Then test **resume** the same way the driver will use it (`--resume <session>`, `exec resume --last`,
…): a CLI that cannot resume headless cannot be landed after a crash, and that changes its adapter.

## 4. What happens at the budget wall

```bash
BURN=$(curl -s -X POST http://localhost:4000/key/generate -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" -d '{"key_alias":"burn","max_budget":0.000001}' \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['key'])")
# make two calls with $BURN: note the exact HTTP status of the second (400 and 429 both occur,
# by version) and the body — `budget_exceeded` means the proxy, not the provider.
```

## 5. Cached-token pricing

If the alias has custom prices, run the warm-cache probe from `OPERATIONS.md` — streaming _and_
non-streaming. Cache pricing that silently falls back to the full input rate overstates an agentic
leg's cost several-fold.

## 6. The sandbox image

```bash
docker build -t synth-bench-sandbox ../sandbox/
docker run --rm synth-bench-sandbox "uv --version && node --version && claude --version"
docker run --rm --gpus all synth-bench-sandbox "nvidia-smi -L"     # gpu: true campaigns only
```
