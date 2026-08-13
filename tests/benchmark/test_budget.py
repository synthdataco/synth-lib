from synth_lib.benchmark.budget import BudgetTracker, Phase
from synth_lib.benchmark.clock import RunClock


class FakeAdmin:
    """Stand-in for LiteLLMAdmin: spend driven by the test."""

    def __init__(self):
        self.spend = 0.0
        self.fail = False

    def key_info(self, key: str) -> dict:
        if self.fail:
            raise ConnectionError("proxy down")
        return {"spend": self.spend, "max_budget": 100.0}


def _tracker(admin, t):
    clock = RunClock(deadline_seconds=1000.0, now_fn=lambda: t[0])
    return BudgetTracker(admin=admin, virtual_key="sk-x", budget_usd=100.0, soft_landing_pct=0.85, clock=clock)


def test_phases_follow_max_of_usd_and_time():
    t = [0.0]
    admin = FakeAdmin()
    tr = _tracker(admin, t)
    assert tr.status().phase == Phase.NORMAL
    admin.spend = 86.0  # 86% $ > 85%
    assert tr.status().phase == Phase.LANDING
    admin.spend = 10.0
    t[0] = 870.0  # 87% time
    assert tr.status().phase == Phase.LANDING
    t[0] = 1001.0  # 100% time
    assert tr.status().phase == Phase.EXHAUSTED


def test_proxy_outage_pauses_clock_and_reports_unknown():
    t = [0.0]
    admin = FakeAdmin()
    tr = _tracker(admin, t)
    admin.fail = True
    st = tr.status()
    assert st.infra_down is True and st.phase == Phase.NORMAL  # we never kill on an outage
    t[0] = 500.0  # the outage lasts 500 s
    admin.fail = False
    st = tr.status()
    assert st.infra_down is False
    assert tr.clock.elapsed() == 0.0  # outage excluded from the wall-clock


def test_outage_at_90pct_keeps_landing():
    t = [0.0]
    admin = FakeAdmin()
    admin.spend = 90.0
    tr = _tracker(admin, t)
    assert tr.status().phase == Phase.LANDING
    admin.fail = True
    st = tr.status()
    assert st.phase == Phase.LANDING  # must never fall back to NORMAL during an outage
    assert st.infra_down is True


def test_exhausted_by_usd():
    t = [0.0]
    admin = FakeAdmin()
    admin.spend = 101.0
    tr = _tracker(admin, t)
    assert tr.status().phase == Phase.EXHAUSTED


def test_budget_md_rendering(tmp_path):
    t = [0.0]
    admin = FakeAdmin()
    admin.spend = 90.0
    tr = _tracker(admin, t)
    st = tr.status()
    tr.write_budget_file(tmp_path, st, notes="check interface: OK")
    text = (tmp_path / "BUDGET.md").read_text()
    assert "90.00 / 100.00 USD" in text
    assert "LANDING ORDER" in text  # phase LANDING => order visible
    assert "STOP any new experiment NOW" in text
    assert "check interface: OK" in text
