"""synth-lib is installed editable in its own venv, so it has no commit to name. Workspace tests
stub only the environment lookup — the substitution and its placeholder guard still run.

A test that exercises the lookup itself opts out with `@pytest.mark.real_host_rev`.
"""

import pytest

from synth_lib.benchmark import workspace

STUB_REV = "0" * 40


@pytest.fixture(autouse=True)
def stub_host_rev(request, monkeypatch):
    if request.node.get_closest_marker("real_host_rev"):
        return
    monkeypatch.setattr(workspace, "_host_synth_lib_rev", lambda: STUB_REV)
