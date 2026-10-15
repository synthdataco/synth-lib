"""Rendering of CAMPAIGN.md from the template (string.Template, no jinja2)."""

from __future__ import annotations

from pathlib import Path
from string import Template

from synth_lib.benchmark.campaign import CampaignConfig, ModelSpec

TEMPLATE_PATH = Path(__file__).parent / "constitution" / "CAMPAIGN.md.tmpl"
# Appended to the hardware line when the campaign sets min_gpu_minutes.
GPU_REQUIREMENT = (
    " The harness meters GPU activity at every poll and `BUDGET.md` shows the count: a nomination "
    "stands only once {minutes:g} minutes of GPU work are metered. Stopping with `agent/CHAMPION` "
    "before that does not end your campaign; you are resumed to keep working."
)


def render_constitution(cfg: CampaignConfig, model: ModelSpec) -> str:
    return Template(TEMPLATE_PATH.read_text()).substitute(
        campaign_name=cfg.name,
        model_id=model.id,
        objective=cfg.objective.strip(),
        budget_usd=cfg.budget_usd_per_model,
        deadline_hours=cfg.deadline_hours_per_model,
        agent_dir="agent",
        data_cutoff=cfg.data_cutoff.isoformat(),
        data_start=cfg.data_start.isoformat() if cfg.data_start else "the start of the store",
        hardware=cfg.hardware,
        gpu_requirement=GPU_REQUIREMENT.format(minutes=cfg.min_gpu_minutes) if cfg.min_gpu_minutes else "",
    )
