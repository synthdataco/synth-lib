from synth_lib.benchmark.clock import RunClock


def test_elapsed_excludes_infra_pauses():
    t = [0.0]
    clock = RunClock(deadline_seconds=100.0, now_fn=lambda: t[0])
    t[0] = 10.0
    assert clock.elapsed() == 10.0
    clock.pause()
    t[0] = 30.0  # 20 s of infra outage
    clock.resume()
    t[0] = 40.0
    assert clock.elapsed() == 20.0  # 10 before pause + 10 after
    assert clock.remaining() == 80.0
    assert clock.pct_elapsed() == 0.2
    assert clock.paused_total() == 20.0


def test_pause_is_idempotent():
    t = [0.0]
    clock = RunClock(deadline_seconds=10.0, now_fn=lambda: t[0])
    clock.pause()
    clock.pause()  # no-op
    t[0] = 5.0
    clock.resume()
    clock.resume()  # no-op
    assert clock.elapsed() == 0.0
