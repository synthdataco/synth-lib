"""One adapter per CLI: launch command, resume, proxy env, files to provision."""

from __future__ import annotations

from pathlib import Path

from synth_lib.benchmark.campaign import ModelSpec

FAKE_CLI_PATH = Path(__file__).parent / "fake_cli.py"


class CLIAdapter:
    def __init__(self, spec: ModelSpec, proxy_url: str, virtual_key: str):
        self.spec = spec
        self.proxy_url = proxy_url.rstrip("/")
        self.virtual_key = virtual_key

    def launch_cmd(self, prompt: str) -> list[str]:
        raise NotImplementedError

    def resume_cmd(self, prompt: str) -> list[str] | None:
        """None => no native resume: the driver relaunches launch_cmd with the resume prompt."""
        return None

    def env(self) -> dict[str, str]:
        return {}

    def provision_files(self) -> dict[str, str]:
        """{path (~ = the run's HOME): content} to write before launch."""
        return {}


class ClaudeCodeAdapter(CLIAdapter):
    def launch_cmd(self, prompt: str) -> list[str]:
        return [
            "claude",
            "-p",
            prompt,
            "--model",
            self.spec.model,
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
        ]

    def resume_cmd(self, prompt: str) -> list[str] | None:
        # -c: resumes the most recent session in the current directory (the workspace)
        return [
            "claude",
            "-c",
            "-p",
            prompt,
            "--model",
            self.spec.model,
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
        ]

    def env(self) -> dict[str, str]:
        return {"ANTHROPIC_BASE_URL": self.proxy_url, "ANTHROPIC_AUTH_TOKEN": self.virtual_key}


class CodexAdapter(CLIAdapter):
    def launch_cmd(self, prompt: str) -> list[str]:
        return ["codex", "exec", "--json", "--skip-git-repo-check", "--sandbox", "danger-full-access", prompt]

    def resume_cmd(self, prompt: str) -> list[str] | None:
        # exec-level flags MUST precede the `resume` subcommand: codex 0.146.0 rejects them after
        # it ("error: unexpected argument '--sandbox' found"), and every relaunch then dies at
        # argument parsing rather than reaching the model. Verified
        # against the image: flags-before-subcommand parses and proceeds to auth.
        return [
            "codex",
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--sandbox",
            "danger-full-access",
            "resume",
            "--last",
            prompt,
        ]

    def env(self) -> dict[str, str]:
        return {"LITELLM_KEY_CODEX": self.virtual_key}

    def provision_files(self) -> dict[str, str]:
        return {
            "~/.codex/config.toml": (
                f'model = "{self.spec.model}"\n'
                'model_provider = "litellm"\n\n'
                "[model_providers.litellm]\n"
                'name = "LiteLLM"\n'
                f'base_url = "{self.proxy_url}/v1"\n'
                'env_key = "LITELLM_KEY_CODEX"\n'
                f'wire_api = "{self.spec.wire_api}"\n'
            )
        }


class GeminiAdapter(CLIAdapter):
    def launch_cmd(self, prompt: str) -> list[str]:
        # -m: without it the CLI picks its own default models (observed during the spike: 3.1-flash-lite)
        return [
            "gemini",
            "-p",
            prompt,
            "-m",
            self.spec.model,
            "--output-format",
            "stream-json",
            "--approval-mode",
            "yolo",
        ]

    def env(self) -> dict[str, str]:
        return {
            "GOOGLE_GEMINI_BASE_URL": f"{self.proxy_url}/gemini",
            "GEMINI_API_KEY": self.virtual_key,
            "GEMINI_SANDBOX": "false",
            "GEMINI_CLI_TRUST_WORKSPACE": "true",
        }

    def provision_files(self) -> dict[str, str]:
        # Without selectedAuthType the CLI refuses to start headless ("Invalid auth method", spike 07-24).
        return {
            "~/.gemini/settings.json": (
                '{"selectedAuthType": "gemini-api-key", ' '"security": {"auth": {"selectedType": "gemini-api-key"}}}\n'
            )
        }


class KimiCodeAdapter(CLIAdapter):
    """Moonshot's Kimi Code CLI (kimi.com/code). Not an npm package: installed in the sandbox
    image via its install.sh with KIMI_INSTALL_DIR=/usr/local — the default is $HOME/.kimi-code,
    which the per-run HOME mount over /root would hide at runtime."""

    def launch_cmd(self, prompt: str) -> list[str]:
        # NO permission flag: kimi 0.32.0 rejects combining --prompt with --yolo AND with --auto
        # ("error: Cannot combine ..."), so the leg dies at argv parsing before reaching a model
        # call. Print mode is non-interactive by construction and handles approvals itself.
        return ["kimi", "-p", prompt, "--output-format", "stream-json"]

    def resume_cmd(self, prompt: str) -> list[str] | None:
        # -c: continue the most recent session in the current working directory (the workspace) —
        # same cwd-keyed semantics the claude adapter relies on. Same no-permission-flag rule.
        return ["kimi", "-c", "-p", prompt, "--output-format", "stream-json"]

    def env(self) -> dict[str, str]:
        # The KIMI_MODEL_* family overrides the CLI's configured model for this launch, which is
        # what routes it through the metering proxy: model name = the LiteLLM alias, key = the
        # run's virtual key. No -m flag — KIMI_MODEL_NAME defines the model outright.
        return {
            "KIMI_MODEL_NAME": self.spec.model,
            "KIMI_MODEL_API_KEY": self.virtual_key,
            "KIMI_MODEL_PROVIDER_TYPE": "openai",
            "KIMI_MODEL_BASE_URL": f"{self.proxy_url}/v1",
            # A CLI that self-updates mid-campaign changes the subject of the experiment.
            "KIMI_CODE_NO_AUTO_UPDATE": "1",
            "KIMI_DISABLE_TELEMETRY": "1",
        }


class FakeAdapter(CLIAdapter):
    """Drives tests/benchmark/fake_cli.py — no LLM, no cost."""

    def launch_cmd(self, prompt: str) -> list[str]:
        return ["python", str(FAKE_CLI_PATH), prompt]


_ADAPTERS = {
    "claude-code": ClaudeCodeAdapter,
    "codex": CodexAdapter,
    "gemini-cli": GeminiAdapter,
    "kimi-code": KimiCodeAdapter,
    "fake": FakeAdapter,
}


def build_adapter(spec: ModelSpec, proxy_url: str, virtual_key: str) -> CLIAdapter:
    return _ADAPTERS[spec.cli](spec, proxy_url, virtual_key)
