"""synth-lib is installed editable in its own venv, so it has no commit to name. Workspace tests
stub only the environment lookup — the substitution and its placeholder guard still run.

A test that exercises the lookup itself opts out with `@pytest.mark.real_host_rev`.

`fake_docker` stands in for the sandbox where there is no Docker daemon.
"""

import os
import sys
from pathlib import Path

import pytest

from synth_lib.benchmark import workspace

STUB_REV = "0" * 40

# Logs its arguments; for `run`, runs the command on the host with each mount's container path replaced by
# its host path, in the mounted workspace, with only PATH and the `-e` variables.
FAKE_DOCKER = """#!{python}
import json
import os
import re
import sys

with open({log!r}, "a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1] != "run":
    sys.exit(0)
args, mounts, env = sys.argv[2:], {{}}, {{}}
while args[0].startswith("-"):
    flag = args.pop(0)
    if flag == "--rm":
        continue
    value = args.pop(0)
    if flag == "-v":
        source, target = value.split(":")[:2]
        mounts[target] = source
    elif flag == "-e":
        key, _, val = value.partition("=")
        env[key] = val
image, inner = args
targets = re.compile("|".join(re.escape(t) for t in sorted(mounts, key=len, reverse=True)))


def host(text):
    return targets.sub(lambda m: mounts[m.group()], text)


os.chdir(host("/workspace"))
inner_env = {{"PATH": os.environ["PATH"], **{{k: host(v) for k, v in env.items()}}}}
os.execve("/bin/bash", ["bash", "-c", host(inner)], inner_env)
"""


@pytest.fixture(autouse=True)
def stub_host_rev(request, monkeypatch):
    if request.node.get_closest_marker("real_host_rev"):
        return
    monkeypatch.setattr(workspace, "_host_synth_lib_rev", lambda: STUB_REV)


@pytest.fixture
def fake_docker(tmp_path, monkeypatch) -> Path:
    """A `docker` first on PATH; returns the file it logs each call to, one JSON list per line."""
    bin_dir = tmp_path / "fake-docker-bin"
    bin_dir.mkdir()
    log = bin_dir / "calls.jsonl"
    log.touch()
    docker = bin_dir / "docker"
    docker.write_text(FAKE_DOCKER.format(python=sys.executable, log=str(log)))
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return log
