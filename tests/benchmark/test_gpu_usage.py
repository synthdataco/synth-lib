"""GPU-busy minutes and peak memory, from one nvidia-smi read per driver poll."""

import subprocess

from synth_lib.benchmark import gpu_usage
from synth_lib.benchmark.gpu_usage import GpuUsage


def _smi(monkeypatch, *outputs):
    """nvidia-smi answering each read with the next output: a CSV string, or an exception to raise."""
    answers = iter(outputs)

    def run(cmd, **kw):
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return subprocess.CompletedProcess(cmd, 0, stdout=answer, stderr="")

    monkeypatch.setattr(gpu_usage.subprocess, "run", run)


def test_a_poll_counts_when_any_device_is_busy(monkeypatch):
    _smi(monkeypatch, "0, 300\n", "85, 18000\n", "3, 18000\n0, 120\n", "0, 50\n12, 900\n")
    usage = GpuUsage()
    for _ in range(4):
        usage.sample(30)

    assert usage.busy_seconds == 60  # polls 2 and 4
    assert usage.busy_minutes == 1
    assert usage.peak_memory_mib == 18000


def test_no_nvidia_smi_turns_the_meter_off(monkeypatch):
    _smi(monkeypatch, FileNotFoundError("nvidia-smi"))
    usage = GpuUsage()
    usage.sample(30)
    usage.sample(30)  # never asked again

    assert not usage.available and "nvidia-smi" in usage.error
    assert usage.line(60) == ""


def test_an_unreadable_value_turns_the_meter_off(monkeypatch):
    _smi(monkeypatch, "[N/A], 300\n")
    usage = GpuUsage()
    usage.sample(30)
    assert not usage.available


def test_the_budget_line_states_the_requirement(monkeypatch):
    _smi(monkeypatch, "90, 2048\n")
    usage = GpuUsage()
    usage.sample(120)

    assert usage.line() == "- GPU: 2 min busy so far, peak memory 2.0 GiB"
    assert usage.line(60).endswith(". A nomination stands from 60 min")
