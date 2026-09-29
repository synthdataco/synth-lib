"""A leg must not silently measure a different model than the one on the panel.

A safety classifier can decline a turn, and the API can serve it from a fallback model instead.
Routing is then sticky for about an hour, so one declined turn can hand the rest of a leg to a
model the campaign never chose. The only record is a system event in the transcript.
"""

import json

from synth_lib.benchmark.driver import ModelRun


def _run(tmp_path) -> ModelRun:
    run = object.__new__(ModelRun)  # __init__ wants a container, an adapter and a workspace
    run.artifacts_dir = tmp_path
    return run


def _event(fallback: str) -> str:
    return json.dumps(
        {
            "type": "system",
            "subtype": "model_refusal_fallback",
            "original_model": "claude-fable-5-1-model",
            "fallback_model": fallback,
            "api_refusal_category": "cyber",
        }
    )


def test_a_substituted_model_is_found_in_the_transcript(tmp_path):
    (tmp_path / "transcript-0.log").write_text(
        '{"type":"assistant","message":"working"}\n' + _event("claude-opus-4-8") + "\n"
    )
    assert _run(tmp_path).substituted_models() == {"claude-opus-4-8"}


def test_a_clean_leg_reports_nothing(tmp_path):
    (tmp_path / "transcript-0.log").write_text('{"type":"assistant","message":"working"}\n')
    assert _run(tmp_path).substituted_models() == set()


def test_every_transcript_is_searched_and_substitutes_are_deduped(tmp_path):
    (tmp_path / "transcript-0.log").write_text(_event("claude-opus-4-8") + "\n")
    (tmp_path / "transcript-1.log").write_text(_event("claude-opus-4-8") + "\n" + _event("claude-sonnet-5") + "\n")
    assert _run(tmp_path).substituted_models() == {"claude-opus-4-8", "claude-sonnet-5"}


def test_a_truncated_transcript_line_does_not_break_the_check(tmp_path):
    """Transcripts are written live; the last line of a killed leg can be half a JSON object."""
    (tmp_path / "transcript-0.log").write_text(_event("claude-opus-4-8") + "\n" + '{"subtype":"model_refusal_fallb')
    assert _run(tmp_path).substituted_models() == {"claude-opus-4-8"}
