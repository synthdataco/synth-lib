from pathlib import Path

from synth_lib.benchmark.sandbox.run_sandbox import sandbox_cmd


def test_sandbox_cmd_mounts_and_caps():
    cmd = sandbox_cmd(
        workspace=Path("/c/runs/claude"),
        snapshot=Path("/c/snapshot"),
        home=Path("/c/runs/claude-home"),
        image="synth-bench-sandbox",
        memory_gb=12,
        cpus=12,
        network="bridge",
        env={"ANTHROPIC_BASE_URL": "http://host.docker.internal:4000"},
        inner_cmd="claude -p 'go'",
        name="synthbench-t-claude",
    )
    joined = " ".join(cmd)
    assert "--gpus all" in joined and "--memory 12g" in joined and "--cpus 12" in joined
    assert "-v /c/runs/claude:/workspace:rw" in joined
    assert "-v /c/snapshot:/workspace/market_data:ro" in joined
    # The workspace is standalone: only the three mounts above, and nothing to hide with a tmpfs.
    assert "--tmpfs" not in joined
    assert "--add-host host.docker.internal:host-gateway" in joined
    assert "-e ANTHROPIC_BASE_URL=http://host.docker.internal:4000" in joined
    assert "--name synthbench-t-claude" in joined


def test_sandbox_cmd_without_name_omits_flag():
    cmd = sandbox_cmd(
        workspace=Path("/c/runs/claude"),
        snapshot=Path("/c/snapshot"),
        home=Path("/c/runs/claude-home"),
        image="synth-bench-sandbox",
        memory_gb=12,
        cpus=12,
        network="bridge",
        env={},
        inner_cmd="claude -p 'go'",
    )
    assert "--name" not in cmd


def test_network_none_for_verdict():
    cmd = sandbox_cmd(
        workspace=Path("/w"),
        snapshot=Path("/s"),
        home=Path("/h"),
        image="i",
        memory_gb=8,
        cpus=8,
        network="none",
        env={},
        inner_cmd="x",
    )
    assert "--network none" in " ".join(cmd)


def test_relative_paths_are_absolutized(tmp_path, monkeypatch):
    """Smoke regression 07-24: docker reads a relative path as a volume name."""
    monkeypatch.chdir(tmp_path)
    from pathlib import Path

    cmd = sandbox_cmd(
        workspace=Path("rel/ws"),
        snapshot=Path("rel/snap"),
        home=Path("rel/home"),
        image="i",
        memory_gb=8,
        cpus=8,
        network="none",
        env={},
        inner_cmd="x",
    )
    joined = " ".join(cmd)
    assert f"-v {tmp_path}/rel/ws:/workspace:rw" in joined
    assert f"-v {tmp_path}/rel/snap:/workspace/market_data:ro" in joined
