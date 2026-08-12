"""CI contract gate for every unpacked champion under champions/.

Discovery is by PROVENANCE.md — the marker unpack_champion.py writes — so a hand-written miner that
happens to sit alongside them is not swept in. Each champion is re-gated through
the validator's OWN validate_responses on both live formats, on every CI run: the unpack-time
gate proves the champion was valid when unpacked, this one proves it still is after any later
edit to modeling.py, the wrapper, or the pinned synth-subnet contract itself.
"""

from pathlib import Path

import pytest

from synth_lib.serving.unpack_champion import contract_gate

CHAMPIONS_DIR = Path(__file__).resolve().parents[2] / "champions"
CHAMPIONS = sorted(marker.parent for marker in CHAMPIONS_DIR.glob("*/PROVENANCE.md"))


@pytest.mark.parametrize("champion_dir", CHAMPIONS, ids=lambda p: p.name)
def test_unpacked_champion_passes_the_live_contract(champion_dir):
    # Both formats regardless of nominated profiles: a deployed miner receives every prompt.
    contract_gate(champion_dir / "modeling.py", ("low", "high"))
