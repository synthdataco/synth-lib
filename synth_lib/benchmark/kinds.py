"""Competition kinds: what a campaign's agents build, and how the engine prepares, briefs and checks it.

A campaign names its kind in campaign.yaml (`kind:`, default `monte-carlo`). Kinds are looked up by
name in the `synth_lib.benchmark.kinds` entry-point group: synth-lib registers `monte-carlo`
(monte_carlo.py), and another package registers its own kinds in its pyproject.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from synth_lib.benchmark.campaign import CampaignConfig

KINDS_GROUP = "synth_lib.benchmark.kinds"
DEFAULT_KIND = "monte-carlo"


@dataclass(frozen=True)
class CompetitionKind:
    # The metric a campaign.yaml of this kind pre-registers as `primary_metric`.
    primary_metric: str
    # CAMPAIGN.md template; constitution.render_constitution fills its $ variables.
    template: Path
    # Copied as the agent workspace. Must hold `agent/`, and a pyproject.toml with
    # `dependencies = [...]` for a `gpu: true` campaign.
    scaffold: Path
    # (workspace) rewrites the copied scaffold's pins.
    pin_workspace: Callable[[Path], None]
    # (data_root, snapshot, cfg) freezes the data the agents get.
    build_snapshot: Callable[[Path, Path, CampaignConfig], None]
    # (snapshot) -> DATA.md.
    render_data_md: Callable[[Path], str]
    # (snapshot) -> environment of the agent's sandbox.
    sandbox_env: Callable[[Path], dict[str, str]]
    # (snapshot) raises SystemExit when `run` must not start.
    require_ready: Callable[[Path], object]
    # (agent_dir, snapshot, python_exe) -> (ok, message): the nominated champion's interface check. Runs
    # the champion out of process, with none of the host's secrets in its environment, and returns within
    # a bounded time: the driver calls it in its poll loop. An exception counts as a failed check.
    probe: Callable[[Path, Path, str], tuple[bool, str]]
    # Hands back a nomination that fails the probe. Must contain `{check}`, replaced by the probe's message.
    interface_prompt: str
    # BUDGET.md's landing section. Must contain "LANDING ORDER": the CLI goal condition and the fake
    # CLI look for it.
    landing_order: str


def load_kind(name: str) -> CompetitionKind:
    """The kind registered under `name` in KINDS_GROUP."""
    found = entry_points(group=KINDS_GROUP, name=name)
    if not found:
        known = sorted(ep.name for ep in entry_points(group=KINDS_GROUP))
        raise ValueError(f"unknown competition kind {name!r} (registered: {known})")
    if len(found) > 1:
        raise ValueError(f"competition kind {name!r} is registered more than once: {sorted(ep.value for ep in found)}")
    kind = next(iter(found)).load()
    if not isinstance(kind, CompetitionKind):
        raise TypeError(f"competition kind {name!r} is a {type(kind).__name__}, not a CompetitionKind")
    if "{check}" not in kind.interface_prompt:
        raise ValueError(f"competition kind {name!r}: interface_prompt has no {{check}}")
    if "LANDING ORDER" not in kind.landing_order:
        raise ValueError(f"competition kind {name!r}: landing_order has no LANDING ORDER")
    return kind
