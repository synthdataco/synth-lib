from pathlib import Path

from synth_lib.benchmark.nomination import Champion, parse_champion, probe_simulate, probe_simulate_subprocess

TOY_MODELING = """
def simulate(asset, start_time, time_increment=300, time_length=86400, num_simulations=100,
             context_prices=None):
    steps = time_length // time_increment + 1
    return (start_time, time_increment, *[[100.0] * steps for _ in range(num_simulations)])
"""

BROKEN_MODELING = "def simulate(*a, **k):\n    return (0,)\n"

UNSIZED_MODELING = "def simulate(*a, **k):\n    return (0, 300, *[1] * 5)\n"

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
    ok, msg = probe_simulate(tmp_path / "modeling.py", num_simulations=5)
    assert ok is True, msg


def test_probe_rejects_bad_shape(tmp_path):
    (tmp_path / "modeling.py").write_text(BROKEN_MODELING)
    ok, msg = probe_simulate(tmp_path / "modeling.py", num_simulations=5)
    assert ok is False and "tuple" in msg


def test_probe_rejects_unsized_paths(tmp_path):
    # tuple of the right length (2 + num_simulations) but whose "paths" are ints:
    # len(path) must raise a TypeError that stays internal to the probe (no driver crash).
    (tmp_path / "modeling.py").write_text(UNSIZED_MODELING)
    ok, msg = probe_simulate(tmp_path / "modeling.py", num_simulations=5)
    assert ok is False, msg


def test_starter_modeling_passes_probe():
    starter = Path(__file__).resolve().parents[2] / "synth_lib" / "benchmark" / "scaffold" / "modeling_starter.py"
    ok, msg = probe_simulate(starter, num_simulations=5)
    assert ok is True, msg


def test_probe_subprocess_ok(tmp_path):
    (tmp_path / "modeling.py").write_text(TOY_MODELING)
    ok, msg = probe_simulate_subprocess(tmp_path / "modeling.py")
    assert ok is True, msg
    assert msg == "ok (24h + 1h)"  # the probe covers both competition formats


def test_probe_subprocess_times_out(tmp_path):
    (tmp_path / "modeling.py").write_text(SLEEPY_MODELING)
    ok, msg = probe_simulate_subprocess(tmp_path / "modeling.py", timeout_s=2.0)
    assert ok is False
    assert "timeout" in msg


def test_probe_all_shapes_rejects_hardcoded_24h_model(tmp_path):
    """A simulate() that hardcodes 289 points passes the 24h probe but must fail on the 1h one."""
    from synth_lib.benchmark.nomination import probe_all_shapes

    (tmp_path / "modeling.py").write_text(
        "def simulate(asset, start_time, time_increment=300, time_length=86400,\n"
        "             num_simulations=100, context_prices=None):\n"
        "    return (start_time, time_increment, *[[100.0] * 289 for _ in range(num_simulations)])\n"
    )
    ok, msg = probe_all_shapes(tmp_path / "modeling.py")
    assert ok is False and "[1h]" in msg


def test_probe_all_shapes_accepts_generic_model(tmp_path):
    from synth_lib.benchmark.nomination import probe_all_shapes

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
    from synth_lib.benchmark.nomination import probe_all_shapes

    (tmp_path / "modeling.py").write_text(
        "def simulate(asset, start_time, time_increment=300, time_length=86400,\n"
        "             num_simulations=100, context_prices=None):\n"
        "    steps = time_length // time_increment + 1\n"
        "    return (start_time, time_increment, *[[100.0] * steps for _ in range(100)])\n"
    )
    ok, msg = probe_all_shapes(tmp_path / "modeling.py")
    assert ok is False
    assert "1000" in msg and "102" in msg, msg  # names both the expected and actual path counts
