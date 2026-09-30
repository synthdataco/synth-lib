import json
import tomllib

from synth_lib.benchmark.campaign import ModelSpec
from synth_lib.benchmark.cli_adapters import GOAL_CONDITION, build_adapter


def test_claude_adapter_cmd_and_env():
    a = build_adapter(
        ModelSpec(id="c", cli="claude-code", model="claude-model"),
        proxy_url="http://localhost:4000",
        virtual_key="sk-v",
    )
    cmd = a.launch_cmd("Read CAMPAIGN.md and start.")
    assert cmd[:2] == ["claude", "-p"]
    assert "--dangerously-skip-permissions" in cmd and "--model" in cmd
    env = a.env()
    assert env["ANTHROPIC_BASE_URL"] == "http://localhost:4000"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-v"
    assert a.resume_cmd("continue") is not None  # claude can resume via a generic session id (-c)
    assert "--effort" not in cmd and "--settings" not in cmd  # defaults: the CLI's own effort, no hook
    # every subagent and workflow agent on the leg's model; background workflows waited for
    assert env["CLAUDE_CODE_SUBAGENT_MODEL_FORCE"] == "1"
    assert env["CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS"] == "0"


def test_claude_adapter_carries_effort_ultracode_and_goal_into_every_launch():
    a = build_adapter(
        ModelSpec(id="c", cli="claude-code", model="claude-opus-5-model", effort="max", ultracode=True, goal=True),
        proxy_url="http://localhost:4000",
        virtual_key="sk-v",
    )
    # a resumed session gets its settings from the command line again, or it runs without them
    for cmd in (a.launch_cmd("Read CAMPAIGN.md and start."), a.resume_cmd("continue")):
        assert cmd[cmd.index("--effort") + 1] == "max"
        settings = json.loads(cmd[cmd.index("--settings") + 1])
        assert settings["ultracode"] is True
        (hook,) = settings["hooks"]["Stop"][0]["hooks"]
        assert hook["type"] == "prompt" and hook["prompt"] == GOAL_CONDITION
        assert hook["model"] == "claude-opus-5-model"  # the evaluator runs on the leg's own served alias
    # the goal rides in the settings: the prompt is the driver's, unchanged
    assert a.launch_cmd("Read CAMPAIGN.md and start.")[2] == "Read CAMPAIGN.md and start."


def test_codex_adapter_writes_config_and_env():
    a = build_adapter(
        ModelSpec(id="x", cli="codex", model="codex-model", wire_api="responses"),
        proxy_url="http://localhost:4000",
        virtual_key="sk-v",
    )
    cmd = a.launch_cmd("go")
    assert cmd[:2] == ["codex", "exec"]
    assert "--skip-git-repo-check" in cmd
    assert a.env()["LITELLM_KEY_CODEX"] == "sk-v"
    cfg = a.provision_files()["~/.codex/config.toml"]
    assert 'wire_api = "responses"' in cfg and "http://localhost:4000/v1" in cfg
    resume = a.resume_cmd("go")
    # Flag ORDER is the contract: codex 0.146 rejects exec-level flags placed after the `resume`
    # subcommand, or it dies at argv parsing. Everything must sit between `exec` and `resume`.
    assert resume[:2] == ["codex", "exec"]
    assert resume.index("--sandbox") < resume.index("resume") < resume.index("--last")
    assert resume[-1] == "go"
    parsed = tomllib.loads(cfg)
    assert "model_reasoning_effort" not in parsed  # default: the model catalog's own level
    # sub-agents stay on the leg's model: the spawn tool cannot pick another one
    assert parsed["features"]["multi_agent_v2"] == {"expose_spawn_agent_model_overrides": False}


def test_codex_adapter_sets_the_reasoning_effort_at_top_level():
    a = build_adapter(
        ModelSpec(id="x", cli="codex", model="gpt-6-sol", effort="ultra"),
        proxy_url="http://localhost:4000",
        virtual_key="sk-v",
    )
    parsed = tomllib.loads(a.provision_files()["~/.codex/config.toml"])
    # a key written after a [table] header would belong to that table instead
    assert parsed["model_reasoning_effort"] == "ultra"
    assert parsed["model"] == "gpt-6-sol" and parsed["model_providers"]["litellm"]["wire_api"] == "responses"


def test_gemini_adapter_env_disables_sandbox():
    a = build_adapter(
        ModelSpec(id="g", cli="gemini-cli", model="gemini/gemini-2.5-pro"),
        proxy_url="http://localhost:4000",
        virtual_key="sk-v",
    )
    env = a.env()
    assert env["GOOGLE_GEMINI_BASE_URL"] == "http://localhost:4000/gemini"
    assert env["GEMINI_API_KEY"] == "sk-v" and env["GEMINI_SANDBOX"] == "false"
    assert a.resume_cmd("go") is None  # generic fallback: fresh relaunch


def test_kimi_adapter_routes_through_the_proxy_env_family():
    a = build_adapter(
        ModelSpec(id="k", cli="kimi-code", model="kimi-k3"),
        proxy_url="http://localhost:4000",
        virtual_key="sk-v",
    )
    cmd = a.launch_cmd("Read CAMPAIGN.md and start.")
    assert cmd[:2] == ["kimi", "-p"]
    assert "--output-format" in cmd and "stream-json" in cmd
    # kimi 0.32.0 rejects --prompt combined with ANY permission flag (smoke-2 argv deaths)
    assert "--yolo" not in cmd and "--auto" not in cmd
    assert "-m" not in cmd  # the model comes from KIMI_MODEL_NAME, not a flag
    env = a.env()
    assert env["KIMI_MODEL_NAME"] == "kimi-k3"
    assert env["KIMI_MODEL_API_KEY"] == "sk-v"  # the virtual key — never a provider key
    assert env["KIMI_MODEL_PROVIDER_TYPE"] == "openai"
    assert env["KIMI_MODEL_BASE_URL"] == "http://localhost:4000/v1"
    # a CLI that self-updates mid-campaign changes the subject of the experiment
    assert env["KIMI_CODE_NO_AUTO_UPDATE"] == "1"
    resume = a.resume_cmd("continue")
    assert resume[:2] == ["kimi", "-c"]
    assert "--yolo" not in resume and "--auto" not in resume


def test_fake_adapter_runs_fake_cli(tmp_path):
    a = build_adapter(ModelSpec(id="f", cli="fake", model="none"), proxy_url="http://x", virtual_key="sk-v")
    cmd = a.launch_cmd("go")
    assert cmd[0] == "python" and cmd[1].endswith("fake_cli.py")


def test_gemini_adapter_pins_model_and_provisions_auth():
    a = build_adapter(
        ModelSpec(id="g", cli="gemini-cli", model="gemini-2.5-pro"), proxy_url="http://x", virtual_key="sk-v"
    )
    cmd = a.launch_cmd("go")
    assert "-m" in cmd and cmd[cmd.index("-m") + 1] == "gemini-2.5-pro"
    files = a.provision_files()
    assert "~/.gemini/settings.json" in files and "gemini-api-key" in files["~/.gemini/settings.json"]
