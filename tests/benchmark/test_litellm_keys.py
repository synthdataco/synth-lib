"""setup must be able to run twice.

Virtual-key aliases are unique proxy-wide and a key's plaintext exists only in the response that
created it. setup is not atomic — it can die partway through a panel — so a rerun meets aliases
that are taken by keys nobody can read. Before delete-then-create that needed manual curl against
the proxy before the campaign could start at all.
"""

import pytest
import requests

from synth_lib.benchmark.metering.keys import LiteLLMAdmin


class _Resp:
    def __init__(self, status: int, payload: dict | None = None, text: str = ""):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


@pytest.fixture
def calls(monkeypatch):
    seen: list[tuple[str, dict]] = []
    taken = {"campaign-4-smoke-fable-5-1"}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.append((url, json))
        if url.endswith("/key/delete"):
            alias = json["key_aliases"][0]
            if alias not in taken:
                return _Resp(400, text="not found")
            taken.discard(alias)
            return _Resp(200, {"deleted_keys": [alias]})
        if json["key_alias"] in taken:
            return _Resp(400, text="already exists. Unique key aliases across all keys are required.")
        taken.add(json["key_alias"])
        return _Resp(200, {"key": "sk-new"})

    monkeypatch.setattr(requests, "post", fake_post)
    return seen


def test_an_alias_left_by_a_failed_setup_no_longer_blocks_the_next_one(calls):
    admin = LiteLLMAdmin("http://proxy", "sk-master")
    assert admin.generate_key("campaign-4-smoke-fable-5-1", 10.0) == "sk-new"
    assert [u.rsplit("/", 1)[-1] for u, _ in calls] == ["delete", "generate"]


def test_a_free_alias_costs_one_harmless_delete(calls):
    admin = LiteLLMAdmin("http://proxy", "sk-master")
    assert admin.generate_key("campaign-4-smoke-opus-5", 10.0) == "sk-new"
    assert [u.rsplit("/", 1)[-1] for u, _ in calls] == ["delete", "generate"]


def test_deleting_an_absent_alias_is_not_an_error(calls):
    assert LiteLLMAdmin("http://proxy", "sk-master").delete_key_alias("never-existed") is False


def test_a_real_delete_failure_still_raises(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(500, text="proxy on fire"))
    with pytest.raises(requests.HTTPError, match="/key/delete"):
        LiteLLMAdmin("http://proxy", "sk-master").delete_key_alias("x")
