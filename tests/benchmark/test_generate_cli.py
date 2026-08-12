"""Tests du helper de coercion tz de generate_cli.py (M3)."""

import pandas as pd

from synth_lib.benchmark.verdict.generate_cli import _to_utc


def test_to_utc_localizes_naive_string():
    ts = _to_utc("2026-08-01T00:00:00")
    assert ts.tzinfo is not None
    assert ts == pd.Timestamp("2026-08-01T00:00:00Z")


def test_to_utc_leaves_aware_string_as_is():
    ts = _to_utc("2026-08-01T00:00:00+02:00")
    assert ts.tzinfo is not None
    assert ts == pd.Timestamp("2026-07-31T22:00:00Z")  # same instant
