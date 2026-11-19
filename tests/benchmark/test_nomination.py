import json
import subprocess
import sys
from pathlib import Path

import pytest

from synth_lib.benchmark.fake_cli import commit
from synth_lib.benchmark.generate_predictions import load_module
from synth_lib.benchmark.nomination import (
    Champion,
    checkout_champion,
    nominated_sha,
    parse_champion,
    probe_all_shapes,
    probe_in_sandbox,
    probe_simulate,
    probe_simulate_subprocess,
    venv_differences,
)
from synth_lib.benchmark.sandbox.run_sandbox import LegSandbox

TOY_MODELING = """
def simulate(asset, start_time, time_increment=300, time_length=86400, num_simulations=100,
             context_prices=None):
    steps = time_length // time_increment + 1
    return (start_time, time_increment, *[[100.0] * steps for _ in range(num_simulations)])
"""

# Writes one "x" per import.
COUNTING_MODELING = """
from pathlib import Path

with open(Path(__file__).with_name("imports.log"), "a") as log:
    log.write("x")
""" + TOY_MODELING

BROKEN_MODELING = "def simulate(*a, **k):\n    return (0,)\n"

UNSIZED_MODELING = "def simulate(*a, **k):\n    return (0, 300, *[1] * 5)\n"

CALENDAR_MODELING = """
import os
from pathlib import Path

def simulate(asset, start_time, time_increment=300, time_length=86400, num_simulations=100,
             context_prices=None):
    assert Path(os.environ["SYNTH_BENCHMARK_CALENDAR"]).exists()
    steps = time_length // time_increment + 1
    return (start_time, time_increment, *[[100.0] * steps for _ in range(num_simulations)])
"""

SIBLING_MODELING = """
def simulate(asset, start_time, time_increment=300, time_length=86400, num_simulations=100,
             context_prices=None):
    from probe_sibling import flat_paths

    return (start_time, time_increment, *flat_paths(time_length // time_increment + 1, num_simulations))
"""

SIBLING_MODULE = "def flat_paths(steps, n):\n    return [[100.0] * steps for _ in range(n)]\n"

# Passes only on the context the verdict hands at `days` days: up to the bar labelled start_time - 3 min
# for 24h, - 2 min for 1h.
LENGTH_MODELING = """
{constant}

def simulate(asset, start_time, time_increment=300, time_length=86400, num_simulations=100,
             context_prices=None):
    assert len(context_prices) == {days} * 24 * 60 - (2 if time_length == 86400 else 1), len(context_prices)
    steps = time_length // time_increment + 1
    return (start_time, time_increment, *[[100.0] * steps for _ in range(num_simulations)])
"""

SLEEPY_MODELING = """
import time

def simulate(asset, start_time, time_increment=300, time_length=86400, num_simulations=100,
             context_prices=None):
    time.sleep(30)
    steps = time_length // time_increment + 1
    return (start_time, time_increment, *[[100.0] * steps for _ in range(num_simulations)])
"""


def test_parse_champion(tmp_path):
    (tmp_path / "CHAMPION").write_text("sha: abc123\nagent_dir: agent\n" "profiles: [low]\nnotes: v1\n")
    ch = parse_champion(tmp_path / "CHAMPION")
    assert ch == Champion(sha="abc123", agent_dir="agent", profiles=("low",), notes="v1")


def test_parse_champion_accepts_commit_as_sha_alias(tmp_path):
    # agents have nominated with `commit:` instead of `sha:`; archived CHAMPION files are
    # the agent's verbatim deliverable, so the parser tolerates the alias rather than editing them.
    (tmp_path / "CHAMPION").write_text("commit: 412b3b4\nagent_dir: agent\nprofiles: [low, high]\n")
    ch = parse_champion(tmp_path / "CHAMPION")
    assert ch.sha == "412b3b4" and ch.profiles == ("low", "high")


def test_parse_champion_survives_invalid_yaml_notes(tmp_path):
    # A real agent-authored CHAMPION: `commit:` alias AND a single-line free-text notes value
    # with unquoted colons, which is invalid YAML. The lenient fallback must recover all keys.
    (tmp_path / "CHAMPION").write_text(
        "commit: 412b3b4\n"
        "agent_dir: agent\n"
        "profiles: [low, high]\n"
        "notes: per-asset defaults: SP500 0.85/0.05 (base=6, vov=0.05); XAU 0.94/0.02\n"
    )
    ch = parse_champion(tmp_path / "CHAMPION")
    assert ch.sha == "412b3b4"
    assert ch.agent_dir == "agent"
    assert ch.profiles == ("low", "high")
    assert ch.notes.startswith("per-asset defaults: SP500")


def test_parse_champion_lenient_handles_block_profiles(tmp_path):
    (tmp_path / "CHAMPION").write_text(
        "sha: abc123\n"
        "agent_dir: agent\n"
        "profiles:\n  - low\n  - high\n"
        "notes: fan width: state-dependent (k=1.30)\n"  # the colon makes strict YAML fail
    )
    ch = parse_champion(tmp_path / "CHAMPION")
    assert ch.profiles == ("low", "high")
    assert "state-dependent" in ch.notes


def test_probe_accepts_conforming_simulate(tmp_path):
    (tmp_path / "modeling.py").write_text(TOY_MODELING)
    ok, msg = probe_simulate(load_module(tmp_path / "modeling.py"), num_simulations=5)
    assert ok is True, msg


def test_probe_rejects_bad_shape(tmp_path):
    (tmp_path / "modeling.py").write_text(BROKEN_MODELING)
    ok, msg = probe_simulate(load_module(tmp_path / "modeling.py"), num_simulations=5)
    assert ok is False and "tuple" in msg


def test_probe_rejects_unsized_paths(tmp_path):
    # tuple of the right length (2 + num_simulations) but whose "paths" are ints:
    # len(path) must raise a TypeError that stays internal to the probe (no driver crash).
    (tmp_path / "modeling.py").write_text(UNSIZED_MODELING)
    ok, msg = probe_simulate(load_module(tmp_path / "modeling.py"), num_simulations=5)
    assert ok is False, msg


def test_starter_modeling_passes_probe():
    starter = Path(__file__).resolve().parents[2] / "synth_lib" / "benchmark" / "scaffold" / "modeling_starter.py"
    ok, msg = probe_simulate(load_module(starter), num_simulations=5)
    assert ok is True, msg


def test_probe_subprocess_ok(tmp_path):
    (tmp_path / "modeling.py").write_text(TOY_MODELING)
    ok, msg = probe_simulate_subprocess(tmp_path / "modeling.py")
    assert ok is True, msg
    assert msg.startswith("ok (24h + 1h), CONTEXT_DAYS 7, one call ")  # both competition formats
    assert msg.endswith(" s (1h)")


def test_probe_subprocess_imports_a_sibling_module(tmp_path):
    """A champion that imports a module beside it by name passes, as it does when served."""
    (tmp_path / "modeling.py").write_text(SIBLING_MODELING)
    (tmp_path / "probe_sibling.py").write_text(SIBLING_MODULE)
    ok, msg = probe_simulate_subprocess(tmp_path / "modeling.py")
    assert ok is True, msg


def test_probe_subprocess_times_out(tmp_path):
    (tmp_path / "modeling.py").write_text(SLEEPY_MODELING)
    ok, msg = probe_simulate_subprocess(tmp_path / "modeling.py", timeout_s=2.0)
    assert ok is False
    assert "timeout" in msg


def test_probe_all_shapes_rejects_hardcoded_24h_model(tmp_path):
    """A simulate() that hardcodes 289 points passes the 24h probe but must fail on the 1h one."""
    (tmp_path / "modeling.py").write_text(
        "def simulate(asset, start_time, time_increment=300, time_length=86400,\n"
        "             num_simulations=100, context_prices=None):\n"
        "    return (start_time, time_increment, *[[100.0] * 289 for _ in range(num_simulations)])\n"
    )
    ok, msg = probe_all_shapes(tmp_path / "modeling.py")
    assert ok is False and "[1h]" in msg


def test_probe_all_shapes_accepts_generic_model(tmp_path):
    (tmp_path / "modeling.py").write_text(
        "def simulate(asset, start_time, time_increment=300, time_length=86400,\n"
        "             num_simulations=100, context_prices=None):\n"
        "    steps = time_length // time_increment + 1\n"
        "    return (start_time, time_increment, *[[100.0] * steps for _ in range(num_simulations)])\n"
    )
    ok, msg = probe_all_shapes(tmp_path / "modeling.py")
    assert ok is True, msg


def test_probe_all_shapes_rejects_model_hardcoded_to_100_paths(tmp_path):
    """A simulate() that hardcodes 100 paths (ignoring num_simulations) is exactly the bug this
    fix targets: the validator (synth.validator.prompt_config.PromptConfig.num_simulations) asks
    for 1000 paths and response_validation_v2 rejects anything else, so probing at the real
    serving size (1000, the new default) must catch this — probing at 100 would have let it
    through, only to break at verdict/live time."""
    (tmp_path / "modeling.py").write_text(
        "def simulate(asset, start_time, time_increment=300, time_length=86400,\n"
        "             num_simulations=100, context_prices=None):\n"
        "    steps = time_length // time_increment + 1\n"
        "    return (start_time, time_increment, *[[100.0] * steps for _ in range(100)])\n"
    )
    ok, msg = probe_all_shapes(tmp_path / "modeling.py")
    assert ok is False
    assert "1000" in msg and "102" in msg, msg  # names both the expected and actual path counts


@pytest.mark.parametrize("constant, days", [("CONTEXT_DAYS = 14", 14), ("", 7)])
def test_the_probe_hands_the_champion_its_context_days(tmp_path, constant, days):
    (tmp_path / "modeling.py").write_text(LENGTH_MODELING.format(constant=constant, days=days))
    ok, msg = probe_all_shapes(tmp_path / "modeling.py")
    assert ok is True, msg
    assert f"CONTEXT_DAYS {days}," in msg and "(24h) and" in msg


def test_the_probe_imports_the_champion_once(tmp_path):
    (tmp_path / "modeling.py").write_text(COUNTING_MODELING)
    ok, msg = probe_all_shapes(tmp_path / "modeling.py")
    assert ok is True, msg
    assert (tmp_path / "imports.log").read_text() == "x"


@pytest.mark.parametrize("value", ["2", "31", "True", "14.0"])
def test_the_probe_fails_context_days_outside_the_bounds(tmp_path, value):
    (tmp_path / "modeling.py").write_text(LENGTH_MODELING.format(constant=f"CONTEXT_DAYS = {value}", days=7))
    ok, msg = probe_all_shapes(tmp_path / "modeling.py")
    assert ok is False
    assert msg == f"ValueError: CONTEXT_DAYS must be a plain int from 3 to 30 in modeling.py, got {value}"


def test_probe_subprocess_hands_the_champion_its_calendar(tmp_path):
    """A champion that reads its calendar must pass the probe: the probe gets the env it is entitled to."""
    (tmp_path / "modeling.py").write_text(CALENDAR_MODELING)
    (tmp_path / "events.parquet").write_bytes(b"view")
    ok, _ = probe_simulate_subprocess(tmp_path / "modeling.py")
    assert ok is False
    ok, msg = probe_simulate_subprocess(
        tmp_path / "modeling.py", extra_env={"SYNTH_BENCHMARK_CALENDAR": str(tmp_path / "events.parquet")}
    )
    assert ok is True, msg


SANDBOX = LegSandbox(image="bench-image", memory_gb=16, cpus=4, gpus=False)

VENV_ONLY_MODELING = "import only_in_venv\n" + TOY_MODELING


def site_packages(venv: Path) -> Path:
    return venv / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"


def install_distribution(venv: Path, name: str, files: dict[str, str], version: str = "1.0") -> None:
    """Records in `venv` an installed distribution `name` at `version` that wrote `files` into site-packages."""
    site = site_packages(venv)
    info = site / f"{name}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    for path, text in files.items():
        (site / path).parent.mkdir(parents=True, exist_ok=True)
        (site / path).write_text(text)
    record = [*files, f"{info.name}/METADATA", f"{info.name}/RECORD"]
    (info / "RECORD").write_text("".join(f"{path},,\n" for path in record))


def fake_venv(venv: Path, modules: dict[str, str]) -> Path:
    """A uv venv whose interpreter is the test's, with `modules` importable only through it, each installed as a
    distribution of its name at version 1.0."""
    site = site_packages(venv)
    (site / "__pycache__").mkdir(parents=True)
    (site / "_virtualenv.py").write_text("")
    (site / "_virtualenv.pth").write_text("import _virtualenv\n")
    for name, source in modules.items():
        install_distribution(venv, name, {f"{name}.py": source})
    (venv / "bin").mkdir()
    python = venv / "bin" / "python"
    python.write_text(f'#!/bin/sh\nPYTHONPATH="{site}" exec "{sys.executable}" "$@"\n')
    python.chmod(0o755)
    return venv


def lockfile(packages: list[str], version: str = "1.0") -> str:
    """A uv.lock listing `packages` at `version`."""
    entries = "".join(f'\n[[package]]\nname = "{name}"\nversion = "{version}"\n' for name in packages)
    return f'version = 1\nrevision = 3\nrequires-python = ">=3.12"\n{entries}'


def _tree(tmp_path: Path, modeling: str, locked: list[str] | None = None) -> Path:
    tree = tmp_path / "tree"
    (tree / "agent").mkdir(parents=True)
    (tree / "agent" / "modeling.py").write_text(modeling)
    if locked is not None:
        (tree / "uv.lock").write_text(lockfile(locked))
    return tree


def _calls(log: Path) -> list[list[str]]:
    return [json.loads(line) for line in log.read_text().splitlines()]


@pytest.mark.parametrize(
    ("text", "sha"),
    [
        ("sha: abc123\nagent_dir: agent\nprofiles: [low]\nnotes: v1\n", "abc123"),
        ("commit: 412b3b4\nagent_dir: agent\n", "412b3b4"),
        ("commit: 412b3b4\nprofiles: [low]\nnotes: fan width: k=1.30\n", "412b3b4"),  # not YAML
        ("sha: 5fbf857\nagent_dir: agent\nentrypoint: champion.main:serve\n", "5fbf857"),
        ("agent_dir: agent\nprofiles: [low]\n", ""),
        ("sha:\n", ""),
        ("", ""),
    ],
)
def test_nominated_sha_reads_the_commit_of_any_kind_s_champion(tmp_path, text, sha):
    (tmp_path / "CHAMPION").write_text(text)
    assert nominated_sha(tmp_path / "CHAMPION") == sha


def test_checkout_champion_gives_the_tree_of_the_commit(tmp_path):
    ws = tmp_path / "ws"
    (ws / "agent").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(ws)], check=True)
    (ws / "agent" / "modeling.py").write_text("committed\n")
    sha = commit(ws)
    (ws / "agent" / "modeling.py").write_text("edited after the commit\n")
    (ws / "agent" / "scratch.py").write_text("never committed\n")

    checkout_champion(ws, sha, tmp_path / "tree")
    assert (tmp_path / "tree" / "agent" / "modeling.py").read_text() == "committed\n"
    assert not (tmp_path / "tree" / "agent" / "scratch.py").exists()

    with pytest.raises(ValueError, match="cannot check out 'deadbeef'"):
        checkout_champion(ws, "deadbeef", tmp_path / "other")


def test_checkout_champion_refuses_a_commit_no_ref_reaches(tmp_path):
    ws = tmp_path / "ws"
    (ws / "agent").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(ws)], check=True)
    (ws / "agent" / "modeling.py").write_text("v1\n")
    amended = commit(ws)
    git = ["git", "-C", str(ws), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "commit", "-q", "--amend", "-m", "amended"], check=True)

    with pytest.raises(ValueError, match=f"cannot check out '{amended}'"):
        checkout_champion(ws, amended, tmp_path / "tree")


def test_the_sandboxed_probe_runs_the_workspace_venv_without_network(tmp_path, fake_docker):
    tree = _tree(tmp_path, TOY_MODELING, locked=[])
    venv = fake_venv(tmp_path / "venv", {})
    ok, msg = probe_in_sandbox(tree, venv, tmp_path / "snapshot", SANDBOX, {"SYNTH_BENCHMARK_CALENDAR": "/x"})
    assert ok is True, msg

    [run] = _calls(fake_docker)
    argv = " ".join(run)
    assert "--network none" in argv and "--gpus" not in argv
    assert f"-v {tree}:/workspace:rw" in argv and f"-v {venv}:/workspace/.venv:ro" in argv
    assert "--memory 16g --cpus 4" in argv
    assert [run[i + 1] for i, arg in enumerate(run) if arg == "-e"] == [
        "SYNTH_BENCHMARK_CALENDAR=/x",
        "PYTHONDONTWRITEBYTECODE=1",
    ]
    assert run[-2:] == [
        "bench-image",
        "/workspace/.venv/bin/python -m synth_lib.benchmark.nomination /workspace/agent/modeling.py",
    ]


def test_a_package_in_the_workspace_venv_and_its_lock_passes_in_the_sandbox(tmp_path, fake_docker):
    tree = _tree(tmp_path, VENV_ONLY_MODELING, locked=["only-in-venv"])
    venv = fake_venv(tmp_path / "venv", {"only_in_venv": ""})

    ok, msg = probe_in_sandbox(tree, venv, tmp_path / "snapshot", SANDBOX, {})
    assert ok is True, msg
    ok, msg = probe_simulate_subprocess(tree / "agent" / "modeling.py")
    assert ok is False and "No module named 'only_in_venv'" in msg


@pytest.mark.parametrize(
    ("locked", "version", "difference"),
    [([], "1.0", "only-in-venv is not in uv.lock"), (["only-in-venv"], "2.0", "only-in-venv is 1.0, uv.lock has 2.0")],
)
def test_a_venv_that_differs_from_uv_lock_fails_the_check_without_a_container(
    tmp_path, fake_docker, locked, version, difference
):
    venv = fake_venv(tmp_path / "venv", {"only_in_venv": ""})
    tree = _tree(tmp_path, VENV_ONLY_MODELING)
    (tree / "uv.lock").write_text(lockfile(locked, version))

    ok, msg = probe_in_sandbox(tree, venv, tmp_path / "snapshot", SANDBOX, {})
    assert ok is False and msg.startswith(".venv differs from the uv.lock of the nominated commit")
    assert f": {difference}. `uv add`" in msg
    assert fake_docker.read_text() == ""


def test_a_nominated_commit_without_uv_lock_fails_the_check(tmp_path, fake_docker):
    venv = fake_venv(tmp_path / "venv", {})
    ok, msg = probe_in_sandbox(_tree(tmp_path, TOY_MODELING), venv, tmp_path / "snapshot", SANDBOX, {})
    assert ok is False and msg.startswith("uv.lock is not in the nominated commit")
    assert fake_docker.read_text() == ""


def test_venv_differences_names_what_uv_lock_would_not_install(tmp_path):
    venv = fake_venv(tmp_path / "venv", {})
    install_distribution(venv, "Foo_Bar", {"foo_bar.py": ""})
    install_distribution(venv, "zope.interface", {"zope/interface/__init__.py": ""})
    install_distribution(venv, "forked", {"forked.py": ""}, version="2.0")
    install_distribution(venv, "workspace", {"_workspace.pth": "/workspace/src\n"}, version="0.3.0")
    install_distribution(venv, "extra", {"extra.py": ""})
    (site_packages(venv) / "vendor.pth").write_text("/workspace/vendor\n")
    (site_packages(venv) / "copied").mkdir()
    lock = lockfile(["foo-bar", "zope-interface", "forked", "locked-not-installed"])
    lock += '\n[[package]]\nname = "forked"\nversion = "2.0"\n'
    lock += '\n[[package]]\nname = "workspace"\nsource = { editable = "." }\n'
    (tmp_path / "uv.lock").write_text(lock)

    assert venv_differences(venv, tmp_path / "uv.lock") == [
        "extra is not in uv.lock",
        "site-packages/copied is from no package",
        "site-packages/vendor.pth is from no package",
    ]


def test_a_package_in_neither_venv_fails_in_the_sandbox_with_its_import_error(tmp_path, fake_docker):
    tree = _tree(tmp_path, VENV_ONLY_MODELING, locked=[])
    ok, msg = probe_in_sandbox(tree, fake_venv(tmp_path / "venv", {}), tmp_path / "snapshot", SANDBOX, {})
    assert ok is False and msg == "ModuleNotFoundError: No module named 'only_in_venv'"


def test_the_sandboxed_probe_times_out_and_removes_its_container(tmp_path, fake_docker):
    tree = _tree(tmp_path, SLEEPY_MODELING, locked=[])
    venv = fake_venv(tmp_path / "venv", {})
    ok, msg = probe_in_sandbox(tree, venv, tmp_path / "snapshot", SANDBOX, {}, timeout_s=2.0)
    assert ok is False and msg == "probe timeout after 2.0s"

    run, remove = _calls(fake_docker)
    assert remove == ["rm", "-f", run[run.index("--name") + 1]]
