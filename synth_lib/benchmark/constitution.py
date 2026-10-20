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


# Inserted when the snapshot froze an event calendar: a data bullet, a clause of hard rule 2 and a
# method bullet, each appended to the line before its placeholder.
CALENDAR_DATA = """
- **`market_data/calendar/events.parquet`** (its path is in `$SYNTH_BENCHMARK_CALENDAR`): scheduled
  events — US macro releases (CPI, NFP, PPI, PCE, retail sales, GDP), FOMC statements, NYSE closures
  and early closes, the EIA weekly petroleum report, and the earnings of the companies behind the
  tokenised stocks — with every date published before your cutoff, upcoming ones included. DATA.md
  gives its columns, its span and the volatility each event type brought. Your miner gets the same
  file in production, refreshed daily: read it inside `simulate()` through the env var, never from
  a copy or a hardcoded path, and expect an event type or a date to be missing."""
CALENDAR_RULE = """
   Event dates come only from that calendar: a table typed from memory or fetched from the web goes
   stale and wrong where your miner serves."""
CALENDAR_METHOD = """
- **Validate an event effect leave-one-event-out.** A type occurs 8 to 12 times a year, so a
  multiplier fitted and tested on the same releases is in-sample."""


def render_constitution(cfg: CampaignConfig, model: ModelSpec, has_calendar: bool = False) -> str:
    return Template(TEMPLATE_PATH.read_text()).substitute(
        calendar_data=CALENDAR_DATA if has_calendar else "",
        calendar_rule=CALENDAR_RULE if has_calendar else "",
        calendar_method=CALENDAR_METHOD if has_calendar else "",
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
