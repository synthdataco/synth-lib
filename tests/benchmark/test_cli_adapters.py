from synth_lib.benchmark.campaign import ModelSpec
from synth_lib.benchmark.cli_adapters import build_adapter


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
