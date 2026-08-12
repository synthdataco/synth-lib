import pytest

from synth_lib.benchmark.gpu_lock import GpuLock, GpuLockBusyError


def test_lock_is_exclusive(tmp_path):
    lock_file = tmp_path / "gpu.lock"
    with GpuLock(lock_file):
        with pytest.raises(GpuLockBusyError):
            GpuLock(lock_file).acquire(blocking=False)
    # released => re-acquirable
    with GpuLock(lock_file):
        pass
