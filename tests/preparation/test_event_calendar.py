"""The event calendar: sources, the ledger's point-in-time dating, and the builder."""

import logging
from datetime import date, timedelta

import pandas as pd
import pytest

from synth_lib.preparation.event_calendar import build, earnings, eia, fomc, fred, nyse
from synth_lib.preparation.event_calendar.events import (
    CALENDAR_DIR,
    LEDGER_FILE,
    VIEW_FILE,
    TruncatedFetch,
    concerns,
    empty_ledger,
    event_frame,
    known_at,
    merge,
    new_york,
    read_ledger,
)

T = pd.Timestamp
# A span wide enough to hold every event the tests fetch.
SPAN = (T("2026-01-01", tz="UTC"), T("2028-01-01", tz="UTC"))


def _rows(*events: tuple[str, str], source: str = "test:src") -> pd.DataFrame:
    """(time, event) pairs as fetched rows of one source."""
    return event_frame(
        [{"time_utc": T(t, tz="UTC"), "event": e, "scope": "all", "detail": "", "source": source} for t, e in events]
    )


class FakeResponse:
    def __init__(self, payload=None, text="", status_code=200):
        self._payload, self.text, self.status_code = payload, text, status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        return self.answer(url, params or {})


class TestNewYork:
    def test_converts_with_daylight_saving(self):
        assert new_york(date(2026, 7, 15), 8, 30) == T("2026-07-15 12:30", tz="UTC")
        assert new_york(date(2026, 1, 15), 8, 30) == T("2026-01-15 13:30", tz="UTC")


class TestMerge:
    FETCH = T("2026-10-01", tz="UTC")

    def test_a_first_fetch_dates_rows_their_publication_lead_before_the_event(self):
        rows = _rows(("2026-10-14 12:30", "CPI"), ("2027-06-16 18:00", "FOMC"))
        ledger, added, superseded = merge(empty_ledger(), rows, "test:src", self.FETCH, SPAN)
        assert (added, superseded) == (2, 0)
        as_of = dict(zip(ledger["event"], ledger["as_of"]))
        assert as_of["CPI"] == T("2026-09-23 12:30", tz="UTC")  # 21 days before
        assert as_of["FOMC"] == self.FETCH  # 90 days before is after the fetch: capped at it

    def test_a_later_fetch_dates_a_new_row_at_the_fetch(self):
        ledger, _, _ = merge(empty_ledger(), _rows(("2026-10-14 12:30", "CPI")), "test:src", self.FETCH, SPAN)
        later = T("2026-10-05", tz="UTC")
        ledger, added, _ = merge(
            ledger, _rows(("2026-10-14 12:30", "CPI"), ("2026-11-10 13:30", "CPI")), "test:src", later, SPAN
        )
        assert added == 1
        assert ledger.loc[ledger["time_utc"] == T("2026-11-10 13:30", tz="UTC"), "as_of"].item() == later
        # the row it already held keeps its date
        assert ledger.loc[ledger["time_utc"] == T("2026-10-14 12:30", tz="UTC"), "as_of"].item() < self.FETCH

    def test_an_upcoming_row_the_fetch_drops_is_superseded_and_a_past_one_kept(self):
        rows = _rows(("2026-09-10 12:30", "CPI"), ("2026-10-14 12:30", "CPI"))
        ledger, _, _ = merge(empty_ledger(), rows, "test:src", self.FETCH, SPAN)
        moved = _rows(("2026-10-24 12:30", "CPI"))  # the past row is absent too, the upcoming one moved
        ledger, added, superseded = merge(ledger, moved, "test:src", T("2026-10-08", tz="UTC"), SPAN)
        assert (added, superseded) == (1, 1)
        dropped = ledger.loc[ledger["time_utc"] == T("2026-10-14 12:30", tz="UTC"), "superseded_at"].item()
        assert dropped == T("2026-10-08", tz="UTC")
        assert pd.isna(ledger.loc[ledger["time_utc"] == T("2026-09-10 12:30", tz="UTC"), "superseded_at"].item())

    def test_a_fetch_dropping_several_upcoming_rows_is_refused(self):
        rows = _rows(("2026-10-14 12:30", "CPI"), ("2026-11-10 13:30", "CPI"), ("2026-12-10 13:30", "CPI"))
        ledger, _, _ = merge(empty_ledger(), rows, "test:src", self.FETCH, SPAN)
        with pytest.raises(TruncatedFetch, match="drops 2 upcoming"):
            merge(ledger, _rows(("2026-10-14 12:30", "CPI")), "test:src", T("2026-10-02", tz="UTC"), SPAN)
        _, _, superseded = merge(
            ledger, _rows(("2026-10-14 12:30", "CPI")), "test:src", T("2026-10-02", tz="UTC"), SPAN, allow_losses=True
        )
        assert superseded == 2

    def test_an_empty_fetch_is_refused(self):
        with pytest.raises(TruncatedFetch, match="no events"):
            merge(empty_ledger(), _rows(), "test:src", self.FETCH, SPAN)

    def test_sources_are_judged_apart(self):
        ledger, _, _ = merge(empty_ledger(), _rows(("2026-10-14 12:30", "CPI"), source="a"), "a", self.FETCH, SPAN)
        # a first fetch of another source does not supersede the first one's rows
        ledger, _, superseded = merge(ledger, _rows(("2026-10-02 12:30", "NFP"), source="b"), "b", self.FETCH, SPAN)
        assert superseded == 0 and len(ledger) == 2


class TestKnownAt:
    def test_a_cutoff_before_a_move_sees_the_original_date_and_one_after_the_new(self):
        first = T("2026-09-01", tz="UTC")
        ledger, _, _ = merge(empty_ledger(), _rows(("2026-10-14 12:30", "CPI")), "test:src", first, SPAN)
        ledger, _, _ = merge(ledger, _rows(("2026-10-24 12:30", "CPI")), "test:src", T("2026-10-08", tz="UTC"), SPAN)
        before = known_at(ledger, T("2026-10-05", tz="UTC"))
        after = known_at(ledger, T("2026-10-09", tz="UTC"))
        assert list(before["time_utc"]) == [T("2026-10-14 12:30", tz="UTC")]
        assert list(after["time_utc"]) == [T("2026-10-24 12:30", tz="UTC")]

    def test_a_row_published_after_the_time_is_absent(self):
        ledger, _, _ = merge(
            empty_ledger(), _rows(("2026-10-14 12:30", "CPI")), "test:src", T("2026-09-01", tz="UTC"), SPAN
        )
        ledger, _, _ = merge(
            ledger,
            _rows(("2026-10-14 12:30", "CPI"), ("2026-11-10 13:30", "CPI")),
            "test:src",
            T("2026-10-20", tz="UTC"),
            SPAN,
        )
        assert len(known_at(ledger, T("2026-10-11", tz="UTC"))) == 1
        assert len(known_at(ledger, T("2026-10-20", tz="UTC"))) == 2
        assert list(known_at(ledger, T("2026-10-20", tz="UTC")).columns) == [
            "time_utc",
            "event",
            "scope",
            "detail",
            "source",
        ]


class TestConcerns:
    def test_scope_all_or_listed(self):
        view = event_frame(
            [
                {"time_utc": T("2026-10-14", tz="UTC"), "event": "CPI", "scope": "all"},
                {"time_utc": T("2026-10-21", tz="UTC"), "event": "EARNINGS", "scope": "TSLAX"},
                {"time_utc": T("2026-11-26", tz="UTC"), "event": "NYSE_CLOSED", "scope": "TSLAX,SP500"},
            ]
        )
        assert list(concerns(view, "SP500")) == [True, False, True]
        assert list(concerns(view, "BTC")) == [True, False, False]

    def test_an_empty_view_gives_a_boolean_mask(self):
        view = event_frame([])
        mask = concerns(view, "BTC")
        assert mask.dtype == bool and list(view[mask].columns) == list(view.columns)


class TestNyse:
    def test_holidays_match_the_exchange_calendar(self):
        assert sorted(nyse.holidays(2025)) == [
            date(2025, 1, 1),
            date(2025, 1, 9),  # day of mourning
            date(2025, 1, 20),
            date(2025, 2, 17),
            date(2025, 4, 18),  # Good Friday
            date(2025, 5, 26),
            date(2025, 6, 19),
            date(2025, 7, 4),
            date(2025, 9, 1),
            date(2025, 11, 27),
            date(2025, 12, 25),
        ]
        assert date(2026, 7, 3) in nyse.holidays(2026)  # July 4 on a Saturday
        assert date(2027, 12, 24) in nyse.holidays(2027)  # Christmas on a Saturday
        assert date(2027, 6, 18) in nyse.holidays(2027)  # Juneteenth on a Saturday
        assert not any(day.month == 12 and day.day == 31 for day in nyse.holidays(2021))  # Jan 1 2022 a Saturday

    def test_early_closes(self):
        assert sorted(nyse.early_closes(2025)) == [date(2025, 7, 3), date(2025, 11, 28), date(2025, 12, 24)]
        # July 3 2026 is the Independence Day holiday and Christmas Eve 2027 the Christmas one
        assert sorted(nyse.early_closes(2026)) == [date(2026, 11, 27), date(2026, 12, 24)]
        assert sorted(nyse.early_closes(2027)) == [date(2027, 11, 26)]

    def test_fetch_keeps_the_range_and_anchors_the_session(self):
        frame = nyse.fetch(date(2026, 11, 1), date(2026, 11, 30))[nyse.SOURCE]
        assert list(frame["event"]) == ["NYSE_CLOSED", "NYSE_EARLY_CLOSE"]
        assert list(frame["time_utc"]) == [T("2026-11-26 14:30", tz="UTC"), T("2026-11-27 18:00", tz="UTC")]
        assert set(frame["scope"]) == {nyse.SCOPE}


FOMC_PAGE = """
<a id="1">2026 FOMC Meetings</a></h4></div>
<div class="row fomc-meeting"><div class="fomc-meeting__month col-xs-5"><strong>January</strong></div>
<div class="fomc-meeting__date col-xs-4">27-28</div></div>
<div class="row fomc-meeting"><div class="fomc-meeting__month col-xs-5"><strong>April/May</strong></div>
<div class="fomc-meeting__date col-xs-4">28-1*</div></div>
<div class="row fomc-meeting"><div class="fomc-meeting__month col-xs-5"><strong>August</strong></div>
<div class="fomc-meeting__date col-xs-4">4 (unscheduled)</div></div>
<a id="2">2027 FOMC Meetings</a></h4></div>
<div class="row fomc-meeting"><div class="fomc-meeting__month col-xs-5"><strong>January</strong></div>
<div class="fomc-meeting__date col-xs-4">26-27</div></div>
"""


class TestFomc:
    def test_parses_the_decision_day_across_months_and_skips_unscheduled(self):
        assert fomc.parse_meetings(FOMC_PAGE) == [
            (date(2026, 1, 28), False),
            (date(2026, 5, 1), True),
            (date(2027, 1, 27), False),
        ]

    def test_a_changed_layout_raises(self):
        with pytest.raises(ValueError, match="layout"):
            fomc.parse_meetings("<html>nothing here</html>")

    def test_fetch_anchors_the_statement_at_14_new_york(self):
        session = FakeSession(lambda url, params: FakeResponse(text=FOMC_PAGE))
        frame = fomc.fetch(date(2026, 1, 1), date(2026, 12, 31), session=session)[fomc.SOURCE]
        assert list(frame["time_utc"]) == [T("2026-01-28 19:00", tz="UTC"), T("2026-05-01 18:00", tz="UTC")]
        assert list(frame["detail"]) == ["", "with projections"]


class TestFred:
    @staticmethod
    def _answer(name_of=None, status=200):
        def answer(url, params):
            if status != 200:
                return FakeResponse(status_code=status)
            release_id = params["release_id"]
            if url.endswith("/release"):
                name = (name_of or {}).get(release_id) or next(
                    expected for rid, expected in fred.RELEASES.values() if rid == release_id
                )
                return FakeResponse({"releases": [{"id": release_id, "name": name}]})
            return FakeResponse({"release_dates": [{"date": "2026-10-14"}, {"date": "2026-11-10"}]})

        return answer

    def test_one_source_per_release_at_0830_new_york(self):
        session = FakeSession(self._answer())
        out = fred.fetch(date(2026, 10, 1), date(2026, 10, 31), "the-key", session=session)
        assert set(out) == {f"fred:{event}" for event in fred.RELEASES}
        assert list(out["fred:CPI"]["time_utc"]) == [T("2026-10-14 12:30", tz="UTC")]
        dates_call = next(params for url, params in session.calls if url.endswith("/release/dates"))
        assert dates_call["include_release_dates_with_no_data"] == "true"

    def test_a_release_naming_another_one_is_left_out_and_the_others_kept(self, caplog):
        session = FakeSession(self._answer(name_of={10: "Industrial Production"}))
        with caplog.at_level(logging.WARNING, logger=fred.__name__):
            out = fred.fetch(date(2026, 10, 1), date(2026, 10, 31), "the-key", session=session)
        assert set(out) == {f"fred:{event}" for event in fred.RELEASES if event != "CPI"}
        assert "fred:CPI: fetch failed (FRED release 10 is 'Industrial Production'" in caplog.text

    def test_an_http_error_never_carries_the_key(self, caplog):
        session = FakeSession(self._answer(status=400))
        with caplog.at_level(logging.WARNING, logger=fred.__name__):
            out = fred.fetch(date(2026, 10, 1), date(2026, 10, 31), "the-secret-key", session=session)
        assert out == {} and "HTTP 400" in caplog.text
        assert "the-secret-key" not in caplog.text


EIA_PAGE = """
<table><thead><tr><th>Data for the week ending</th><th>Alternate release date</th><th>Release day</th>
<th>Release time</th><th>Holiday</th></tr></thead><tbody>
<tr class="past-holiday"><th scope="row">September 4, 2026</th><td>September 10, 2026</td><td>Thursday</td>
<td>12:00 p.m.</td><td>Labor Day</td></tr>
<tr><th scope="row">December 19, 2025</th><td>December 29, 2025</td><td>Monday</td><td>5:00 p.m.</td>
<td>Christmas</td></tr>
</tbody></table>
"""


class TestEia:
    def test_parses_the_exception_rows(self):
        assert eia.parse_exceptions(EIA_PAGE) == {
            date(2026, 9, 4): (date(2026, 9, 10), 12, 0, "Labor Day"),
            date(2025, 12, 19): (date(2025, 12, 29), 17, 0, "Christmas"),
        }

    def test_wednesday_1030_unless_listed(self):
        frame = eia.releases(date(2026, 9, 1), date(2026, 9, 17), eia.parse_exceptions(EIA_PAGE))
        assert list(frame["time_utc"]) == [
            T("2026-09-02 14:30", tz="UTC"),
            T("2026-09-10 16:00", tz="UTC"),
            T("2026-09-16 14:30", tz="UTC"),
        ]
        assert list(frame["detail"]) == ["", "Labor Day", ""]

    def test_a_changed_layout_raises(self):
        with pytest.raises(ValueError, match="layout"):
            eia.parse_exceptions("<table></table>")

    def test_only_the_years_the_page_lists_are_generated(self):
        frame = eia.releases(date(2024, 12, 1), date(2027, 2, 28), eia.parse_exceptions(EIA_PAGE))
        days = frame["time_utc"].dt.tz_convert("America/New_York").dt.date
        assert (days.min(), days.max()) == (date(2025, 1, 1), date(2026, 12, 30))  # the page lists 2025 and 2026


def _nasdaq_day(*rows):
    return {"data": {"rows": list(rows)}}


class TestEarnings:
    def test_parse_day_keeps_the_tickers_and_anchors_the_session(self):
        payload = _nasdaq_day(
            {"symbol": "TSLA", "time": "time-after-hours", "fiscalQuarterEnding": "Sep/2026"},
            {"symbol": "IBM", "time": "time-after-hours", "fiscalQuarterEnding": "Sep/2026"},
            {"symbol": "NVDA", "time": "time-pre-market", "fiscalQuarterEnding": ""},
            {"symbol": "AAPL", "time": "time-not-supplied", "fiscalQuarterEnding": "Sep/2026"},
        )
        rows = earnings.parse_day(payload, date(2026, 10, 21))
        assert [(r["scope"], r["time_utc"]) for r in rows] == [
            ("TSLAX", T("2026-10-21 20:00", tz="UTC")),
            ("NVDAX", T("2026-10-21 12:00", tz="UTC")),
            ("AAPLX", T("2026-10-21 20:00", tz="UTC")),
        ]
        assert rows[0]["detail"] == "after the close, fiscal quarter Sep/2026"
        assert rows[2]["detail"].startswith("session not given")

    def test_fetch_asks_once_per_weekday_and_splits_by_company(self):
        def answer(url, params):
            day = params["date"]
            return FakeResponse(
                _nasdaq_day({"symbol": "TSLA", "time": "time-after-hours"}) if day == "2026-10-21" else {}
            )

        session = FakeSession(answer)
        out, _ = earnings.fetch(date(2026, 10, 17), date(2026, 10, 23), session=session, pause_seconds=0)
        assert [params["date"] for _, params in session.calls] == [f"2026-10-{d}" for d in (19, 20, 21, 22, 23)]
        assert set(out) == {"nasdaq:TSLA"} and len(out["nasdaq:TSLA"]) == 1  # no date, no source

    def test_a_settled_day_is_not_read_again(self, tmp_path):
        settled = tmp_path / earnings.SETTLED_FILE
        today = date.today()
        past = (today - timedelta(days=40), today - timedelta(days=30))
        first = FakeSession(lambda url, params: FakeResponse({}))
        _, commit = earnings.fetch(*past, session=first, pause_seconds=0, settled_file=settled)
        assert first.calls and not settled.exists()  # nothing is settled before the ledger is written
        commit(set())
        again = FakeSession(lambda url, params: FakeResponse({}))
        assert earnings.fetch(*past, session=again, pause_seconds=0, settled_file=settled)[0] == {}
        assert again.calls == []
        # a day stays unsettled until SETTLE_DAYS after it: the last week and the future are read again
        later = FakeSession(lambda url, params: FakeResponse({}))
        earnings.fetch(past[0], today + timedelta(days=7), session=later, pause_seconds=0, settled_file=settled)
        read = [date.fromisoformat(params["date"]) for _, params in later.calls]
        assert read and min(read) > past[1]

    def test_a_refused_company_keeps_its_days_readable(self, tmp_path):
        settled = tmp_path / earnings.SETTLED_FILE
        past = (date.today() - timedelta(days=40), date.today() - timedelta(days=30))
        report = next(past[0] + timedelta(days=n) for n in range(7) if (past[0] + timedelta(days=n)).weekday() < 5)

        def answer(url, params):
            tsla = _nasdaq_day({"symbol": "TSLA", "time": "time-after-hours"})
            return FakeResponse(tsla if params["date"] == report.isoformat() else {})

        out, commit = earnings.fetch(*past, session=FakeSession(answer), pause_seconds=0, settled_file=settled)
        commit({"nasdaq:TSLA"})
        again = FakeSession(answer)
        out, _ = earnings.fetch(*past, session=again, pause_seconds=0, settled_file=settled)
        assert [params["date"] for _, params in again.calls] == [report.isoformat()]
        assert set(out) == {"nasdaq:TSLA"}

    def test_the_read_logs_its_progress(self, caplog):
        session = FakeSession(lambda url, params: FakeResponse({}))
        with caplog.at_level(logging.INFO, logger=earnings.__name__):
            earnings.fetch(date(2026, 1, 1), date(2026, 3, 31), session=session, pause_seconds=0)
        assert "nasdaq: 64 of 64 weekdays in 2026-01-01 .. 2026-03-31 to read" in caplog.text
        assert "nasdaq: 50/64 days read" in caplog.text


class TestBuild:
    NOW = T("2026-10-20 12:00", tz="UTC")

    def test_update_writes_the_ledger_and_the_view_and_survives_a_failed_source(self, tmp_path, monkeypatch, caplog):
        def broken(start, end, calendar_dir):
            raise RuntimeError("down")

        monkeypatch.setattr(
            build,
            "SOURCES",
            {
                "nyse": build.SOURCES["nyse"],
                "broken": broken,
                "empty": lambda start, end, calendar_dir: build.Fetched({"e:src": event_frame([])}),
            },
        )
        with caplog.at_level(logging.INFO, logger=build.__name__):
            view = build.update(tmp_path, date(2026, 11, 1), date(2026, 11, 30), self.NOW)
        assert list(view["event"]) == ["NYSE_CLOSED", "NYSE_EARLY_CLOSE"]
        assert (tmp_path / CALENDAR_DIR / LEDGER_FILE).exists()
        assert pd.read_parquet(tmp_path / CALENDAR_DIR / VIEW_FILE).shape[0] == 2
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("broken: fetch failed (down)" in line for line in warnings)
        assert any("e:src: the fetch listed no events" in line for line in warnings)

    def test_a_truncated_fetch_keeps_the_ledger(self, tmp_path, monkeypatch, caplog):
        three = _rows(("2026-10-21 12:30", "CPI"), ("2026-11-10 13:30", "CPI"), ("2026-12-10 13:30", "CPI"))
        monkeypatch.setattr(
            build, "SOURCES", {"src": lambda start, end, calendar_dir: build.Fetched({"test:src": three})}
        )
        build.update(tmp_path, date(2026, 10, 1), date(2026, 12, 31), self.NOW)
        monkeypatch.setattr(
            build, "SOURCES", {"src": lambda start, end, calendar_dir: build.Fetched({"test:src": three.iloc[:1]})}
        )
        with caplog.at_level(logging.WARNING, logger=build.__name__):
            view = build.update(tmp_path, date(2026, 10, 1), date(2026, 12, 31), self.NOW)
        assert len(view) == 3 and any("refusing" in r.getMessage() for r in caplog.records)
        assert read_ledger(tmp_path / CALENDAR_DIR / LEDGER_FILE)["superseded_at"].isna().all()

    def test_the_view_is_on_disk_before_the_next_source_runs(self, tmp_path, monkeypatch):
        assert next(iter(build.SOURCES)) == "nyse"  # needs no network
        seen = []

        def slow(start, end, calendar_dir):
            seen.append((tmp_path / CALENDAR_DIR / VIEW_FILE).exists())
            return build.Fetched({})

        monkeypatch.setattr(build, "SOURCES", {"nyse": build.SOURCES["nyse"], "slow": slow})
        build.update(tmp_path, date(2026, 11, 1), date(2026, 11, 30), self.NOW)
        assert seen == [True]

    def test_a_narrower_fetch_leaves_the_events_past_its_end(self, tmp_path, monkeypatch):
        two = _rows(("2026-10-21 12:30", "CPI"), ("2026-12-10 13:30", "CPI"))

        def source(start, end, calendar_dir):
            days = two["time_utc"].dt.date
            return build.Fetched({"test:src": two[(days >= start) & (days <= end)]})

        monkeypatch.setattr(build, "SOURCES", {"src": source})
        build.update(tmp_path, date(2026, 10, 1), date(2026, 12, 31), self.NOW)
        view = build.update(tmp_path, date(2026, 10, 1), date(2026, 11, 30), self.NOW)
        assert len(view) == 2

    def test_a_source_settles_once_the_ledger_holding_its_rows_is_written(self, tmp_path, monkeypatch):
        rows = _rows(("2026-10-21 12:30", "CPI"))
        seen = []

        def settle(refused):
            ledger = read_ledger(tmp_path / CALENDAR_DIR / LEDGER_FILE)
            seen.append((refused, list(ledger["source"])))

        fetched = build.Fetched({"test:src": rows, "test:empty": event_frame([])}, settle)
        monkeypatch.setattr(build, "SOURCES", {"src": lambda start, end, calendar_dir: fetched})
        build.update(tmp_path, date(2026, 10, 1), date(2026, 12, 31), self.NOW)
        assert seen == [({"test:empty"}, ["test:src"])]

    def test_a_failed_ledger_write_settles_nothing(self, tmp_path, monkeypatch):
        settled = []
        fetched = build.Fetched({"test:src": _rows(("2026-10-21 12:30", "CPI"))}, settled.append)
        monkeypatch.setattr(build, "SOURCES", {"src": lambda start, end, calendar_dir: fetched})

        def disk_full(frame, path):
            raise OSError("disk full")

        monkeypatch.setattr(build, "write_atomic", disk_full)
        with pytest.raises(OSError):
            build.update(tmp_path, date(2026, 10, 1), date(2026, 12, 31), self.NOW)
        assert settled == []
