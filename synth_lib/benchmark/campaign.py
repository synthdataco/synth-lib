"""Loading and validation of campaign.yaml — the single source of truth for published numbers."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

KNOWN_CLIS = ("claude-code", "codex", "gemini-cli", "kimi-code", "fake")
PRIMARY_METRIC = "synth_competition_rank"
# CampaignConfig fields that must be strictly positive (checked in _validate).
POSITIVE_FIELDS = (
    "budget_usd_per_model",
    "deadline_hours_per_model",
    "forward_window_days",
    "sandbox_memory_gb",
    "sandbox_cpus",
)


@dataclass(frozen=True)
class ModelSpec:
    id: str
    cli: str
    model: str
    wire_api: str = "responses"  # codex only — "chat" was removed in 0.145.0


PACKAGED_BASELINE = "synth_lib.benchmark.verdict.baseline_default"


@dataclass(frozen=True)
class BaselineSpec:
    """A control to score alongside the champions.

    `module` is a dotted import path OR a filesystem path to a file exposing `simulate()`. It used to
    be repo-relative only, which had no valid value once the engine became an installed package: the
    default baseline ships inside it, so a campaign could either name a path into site-packages or
    declare no baseline at all. Resolve it with `baseline_modeling_path`."""

    name: str
    module: str = PACKAGED_BASELINE


def baseline_modeling_path(module: str) -> Path:
    """Filesystem path of a baseline's `simulate()` file, from a dotted module or a path."""
    if "/" in module or module.endswith(".py"):
        return Path(module)
    try:
        spec = importlib.util.find_spec(module)
    except ModuleNotFoundError as exc:  # a missing PARENT package raises instead of returning None
        raise ValueError(f"baseline module {module!r} is not importable and is not a path") from exc
    if spec is None or spec.origin is None:
        raise ValueError(f"baseline module {module!r} is not importable and is not a path")
    return Path(spec.origin)


@dataclass(frozen=True)
class CampaignConfig:
    name: str
    objective: str
    models: tuple[ModelSpec, ...]
    budget_usd_per_model: float
    deadline_hours_per_model: float
    data_cutoff: date
    forward_window_days: int
    baselines: tuple[BaselineSpec, ...]
    hardware: str
    proxy_url: str
    soft_landing_pct: float = 0.85
    max_crash_resumes: int = 5
    poll_seconds: float = 300.0
    landing_grace_seconds: float = 900.0
    # Lower bound of the snapshot (inclusive). Decouples the disk from the campaign: keep months of
    # ingested market_data and hand a campaign only the window it should train on. None = no lower
    # bound. Structural breaks make this a real choice, not just a size knob — see CAMPAIGN.md.tmpl.
    data_start: date | None = None
    gpu: bool = True  # --gpus all in the sandbox; false = campaign without GPU (e.g. smoke test)
    # Sandbox envelope per run (docker --memory / --cpus). Here rather than hardcoded in
    # run_campaign so a campaign declares the machine it was sized for: the host must have at
    # least this much, and the constitution tells the agent what it got (`hardware:`).
    sandbox_memory_gb: int = 12
    sandbox_cpus: int = 12
    primary_metric: str = PRIMARY_METRIC
    root: Path = field(default=Path("campaign_runs"), compare=False)

    @property
    def dir(self) -> Path:
        return self.root / self.name


def _as_date(value: object) -> date | None:
    if value is None or isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def load_campaign(path: Path | str) -> CampaignConfig:
    raw = yaml.safe_load(Path(path).read_text())
    models = tuple(ModelSpec(**m) for m in raw.pop("models"))
    baselines = tuple(BaselineSpec(**b) for b in raw.pop("baselines", []) or [])
    # yaml already gives a date for an unquoted YYYY-MM-DD; normalise the quoted form too, so a
    # stray string cannot reach the date comparisons in build_snapshot.
    cutoff = _as_date(raw.pop("data_cutoff"))
    start = _as_date(raw.pop("data_start", None))
    cfg = CampaignConfig(models=models, baselines=baselines, data_cutoff=cutoff, data_start=start, **raw)
    _validate(cfg)
    return cfg


def _validate(cfg: CampaignConfig) -> None:
    if not cfg.models:
        raise ValueError("models cannot be empty")
    ids = [m.id for m in cfg.models]
    if len(ids) != len(set(ids)):
        raise ValueError("model ids must be unique")
    for m in cfg.models:
        if m.cli not in KNOWN_CLIS:
            raise ValueError(f"unknown cli: {m.cli!r} (expected: {KNOWN_CLIS})")
    # New way: add a strictly-positive field to POSITIVE_FIELDS rather than another `if` here —
    # one branch each was pushing _validate past flake8's max-complexity.
    for name in POSITIVE_FIELDS:
        if getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be > 0")
    if not 0 < cfg.soft_landing_pct < 1:
        raise ValueError("soft_landing_pct must be in (0, 1)")
    if cfg.data_start is not None and cfg.data_start > cfg.data_cutoff:
        raise ValueError(f"data_start ({cfg.data_start}) must be <= data_cutoff ({cfg.data_cutoff})")
    if cfg.primary_metric != PRIMARY_METRIC:
        raise ValueError(f"primary_metric is pre-registered: {PRIMARY_METRIC}")
