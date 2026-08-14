"""Tests for loading/validating campaign.yaml."""

from pathlib import Path

import pytest

from synth_lib.benchmark.campaign import CampaignConfig, load_campaign

VALID_YAML = """
name: smoke
objective: Improve the Synth miner.
models:
  - {id: claude, cli: claude-code, model: claude-model}
  - {id: codex, cli: codex, model: codex-model, wire_api: responses}
  - {id: gemini, cli: gemini-cli, model: gemini/gemini-2.5-pro}
budget_usd_per_model: 200.0
deadline_hours_per_model: 72
data_cutoff: 2026-07-15
forward_window_days: 7
baselines:
  - {name: synth_default, module: synth_lib/benchmark/verdict/baseline_default.py}
  - {name: another_baseline, module: baselines/another/modeling.py}
hardware: "Ryzen 16c / 32GB / RTX 3070 8GB / WSL2"
proxy_url: http://localhost:4000
"""


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "campaign.yaml"
    p.write_text(text)
    return p


def test_load_valid_campaign(tmp_path):
    cfg = load_campaign(_write(tmp_path, VALID_YAML))
    assert isinstance(cfg, CampaignConfig)
    assert [m.id for m in cfg.models] == ["claude", "codex", "gemini"]
    assert cfg.models[1].wire_api == "responses"
    assert cfg.soft_landing_pct == 0.85  # default
    assert cfg.poll_seconds == 300 and cfg.landing_grace_seconds == 900  # defaults
    assert cfg.max_crash_resumes == 5  # default
    assert cfg.primary_metric == "synth_competition_rank"  # default, fixed literal
    assert cfg.sandbox_memory_gb == 12 and cfg.sandbox_cpus == 12  # defaults
    assert cfg.data_cutoff.isoformat() == "2026-07-15"


def test_sandbox_limits_overridable(tmp_path):
    cfg = load_campaign(_write(tmp_path, VALID_YAML + "\nsandbox_memory_gb: 48\nsandbox_cpus: 14\n"))
    assert cfg.sandbox_memory_gb == 48 and cfg.sandbox_cpus == 14


@pytest.mark.parametrize("field", ["sandbox_memory_gb", "sandbox_cpus"])
def test_non_positive_sandbox_limits_rejected(tmp_path, field):
    with pytest.raises(ValueError, match=field):
        load_campaign(_write(tmp_path, VALID_YAML + f"\n{field}: 0\n"))


def test_duplicate_model_ids_rejected(tmp_path):
    bad = VALID_YAML.replace("id: codex", "id: claude", 1)
    with pytest.raises(ValueError, match="unique"):
        load_campaign(_write(tmp_path, bad))


def test_unknown_cli_rejected(tmp_path):
    bad = VALID_YAML.replace("cli: codex", "cli: cursor")
    with pytest.raises(ValueError, match="cli"):
        load_campaign(_write(tmp_path, bad))


def test_bad_budget_rejected(tmp_path):
    bad = VALID_YAML.replace("budget_usd_per_model: 200.0", "budget_usd_per_model: 0")
    with pytest.raises(ValueError, match="budget"):
        load_campaign(_write(tmp_path, bad))


def test_soft_landing_must_be_fraction(tmp_path):
    bad = VALID_YAML + "\nsoft_landing_pct: 1.5\n"
    with pytest.raises(ValueError, match="soft_landing"):
        load_campaign(_write(tmp_path, bad))


def test_empty_models_rejected(tmp_path):
    bad = VALID_YAML.replace(
        "models:\n"
        "  - {id: claude, cli: claude-code, model: claude-model}\n"
        "  - {id: codex, cli: codex, model: codex-model, wire_api: responses}\n"
        "  - {id: gemini, cli: gemini-cli, model: gemini/gemini-2.5-pro}\n",
        "models: []\n",
    )
    with pytest.raises(ValueError, match="models"):
        load_campaign(_write(tmp_path, bad))


def test_baseline_module_defaults_to_the_packaged_control(tmp_path):
    """`module` used to be repo-relative, which had no valid value once the engine became an
    installed package: the default baseline ships inside it."""
    from synth_lib.benchmark.campaign import PACKAGED_BASELINE, BaselineSpec, baseline_modeling_path

    assert BaselineSpec(name="synth_default").module == PACKAGED_BASELINE
    resolved = baseline_modeling_path(PACKAGED_BASELINE)
    assert resolved.name == "baseline_default.py" and resolved.exists()
    assert baseline_modeling_path("agent/mine.py") == Path("agent/mine.py")  # a path still works
    with pytest.raises(ValueError, match="not importable"):
        baseline_modeling_path("no.such.module")


def test_campaign_without_a_baselines_key_loads(tmp_path):
    """A campaign that names no control is legal — the key was required, so `baselines: []` was the
    only honest value a public yaml could give."""
    from synth_lib.benchmark.campaign import load_campaign

    text = tmp_path / "c.yaml"
    text.write_text(
        "name: nobase\nobjective: x\nmodels:\n  - {id: a, cli: fake, model: m}\n"
        "budget_usd_per_model: 1\ndeadline_hours_per_model: 1\ndata_cutoff: 2026-08-01\n"
        "forward_window_days: 1\nhardware: h\nproxy_url: http://localhost:4000\n"
    )
    assert load_campaign(text).baselines == ()
