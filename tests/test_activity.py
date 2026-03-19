"""Tests for activity log: append, read, rotation, and transitions."""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# _append_activity
# ---------------------------------------------------------------------------

class TestAppendActivity:
    def test_appends_single_entry_with_auto_ts(self, tmp_path):
        log = tmp_path / "activity.jsonl"
        with patch("aw.core.ACTIVITY_LOG_PATH", log):
            from aw.core import _append_activity

            _append_activity([{"event": "hello"}])

        lines = log.read_text().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["event"] == "hello"
        assert "ts" in entry
        # ts must be a valid ISO-8601 string
        datetime.fromisoformat(entry["ts"])

    def test_appends_multiple_entries(self, tmp_path):
        log = tmp_path / "activity.jsonl"
        with patch("aw.core.ACTIVITY_LOG_PATH", log):
            from aw.core import _append_activity

            _append_activity([
                {"event": "a"},
                {"event": "b"},
                {"event": "c"},
            ])

        lines = log.read_text().splitlines()
        assert len(lines) == 3
        assert json.loads(lines[0])["event"] == "a"
        assert json.loads(lines[2])["event"] == "c"

    def test_preserves_existing_ts(self, tmp_path):
        log = tmp_path / "activity.jsonl"
        custom_ts = "2026-01-01T00:00:00+00:00"
        with patch("aw.core.ACTIVITY_LOG_PATH", log):
            from aw.core import _append_activity

            _append_activity([{"event": "custom", "ts": custom_ts}])

        entry = json.loads(log.read_text().splitlines()[0])
        assert entry["ts"] == custom_ts

    def test_empty_list_writes_no_lines(self, tmp_path):
        log = tmp_path / "activity.jsonl"
        with patch("aw.core.ACTIVITY_LOG_PATH", log):
            from aw.core import _append_activity

            _append_activity([])

        # The file may be touched (open "a") but no JSONL lines are written
        content = log.read_text() if log.exists() else ""
        assert content.strip() == ""

    def test_rotation_at_max_bytes(self, tmp_path):
        log = tmp_path / "activity.jsonl"
        old = tmp_path / "activity.jsonl.old"
        # Write content exceeding the threshold
        log.write_text("x" * 100 + "\n")
        with (
            patch("aw.core.ACTIVITY_LOG_PATH", log),
            patch("aw.core.ACTIVITY_MAX_BYTES", 50),
        ):
            from aw.core import _append_activity

            _append_activity([{"event": "after_rotation"}])

        # Old file should contain the original content
        assert old.exists()
        assert old.read_text().startswith("x")
        # New file should only have the new entry
        lines = log.read_text().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["event"] == "after_rotation"


# ---------------------------------------------------------------------------
# _read_activity
# ---------------------------------------------------------------------------

class TestReadActivity:
    def test_missing_file_returns_empty(self, tmp_path):
        log = tmp_path / "does_not_exist.jsonl"
        with patch("aw.core.ACTIVITY_LOG_PATH", log):
            from aw.core import _read_activity

            result = _read_activity()

        assert result == []

    def test_reads_all_entries(self, tmp_path):
        log = tmp_path / "activity.jsonl"
        now = datetime.now(timezone.utc).isoformat()
        lines = [
            json.dumps({"event": "a", "ts": now}),
            json.dumps({"event": "b", "ts": now}),
        ]
        log.write_text("\n".join(lines) + "\n")
        with patch("aw.core.ACTIVITY_LOG_PATH", log):
            from aw.core import _read_activity

            result = _read_activity()

        assert len(result) == 2
        assert result[0]["event"] == "a"
        assert result[1]["event"] == "b"

    def test_since_minutes_filters_old_entries(self, tmp_path):
        log = tmp_path / "activity.jsonl"
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        recent_ts = datetime.now(timezone.utc).isoformat()
        lines = [
            json.dumps({"event": "old", "ts": old_ts}),
            json.dumps({"event": "recent", "ts": recent_ts}),
        ]
        log.write_text("\n".join(lines) + "\n")
        with patch("aw.core.ACTIVITY_LOG_PATH", log):
            from aw.core import _read_activity

            result = _read_activity(since_minutes=30)

        assert len(result) == 1
        assert result[0]["event"] == "recent"

    def test_limit_truncates_to_most_recent(self, tmp_path):
        log = tmp_path / "activity.jsonl"
        now = datetime.now(timezone.utc).isoformat()
        lines = [json.dumps({"event": f"e{i}", "ts": now}) for i in range(10)]
        log.write_text("\n".join(lines) + "\n")
        with patch("aw.core.ACTIVITY_LOG_PATH", log):
            from aw.core import _read_activity

            result = _read_activity(limit=3)

        assert len(result) == 3
        # Should be the last 3 entries
        assert result[0]["event"] == "e7"
        assert result[2]["event"] == "e9"


# ---------------------------------------------------------------------------
# _log_transitions
# ---------------------------------------------------------------------------

class TestLogTransitions:
    def test_first_seen_event(self, tmp_path):
        log = tmp_path / "activity.jsonl"
        with (
            patch("aw.core.ACTIVITY_LOG_PATH", log),
            patch("aw.core._prev_activity_state", {}),
        ):
            from aw.core import _log_transitions

            _log_transitions([
                {"project_id": "webapp", "health": "green", "status": "active", "agent": "claude"},
            ])

        lines = log.read_text().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["event"] == "first_seen"
        assert entry["project"] == "webapp"
        assert entry["health"] == "green"

    def test_health_transition(self, tmp_path):
        log = tmp_path / "activity.jsonl"
        prev = {"webapp": {"health": "green", "status": "active"}}
        with (
            patch("aw.core.ACTIVITY_LOG_PATH", log),
            patch("aw.core._prev_activity_state", prev),
        ):
            from aw.core import _log_transitions

            _log_transitions([
                {"project_id": "webapp", "health": "red", "status": "active", "agent": "claude", "reason": "error: fatal"},
            ])

        entry = json.loads(log.read_text().splitlines()[0])
        assert entry["event"] == "health_green_to_red"
        assert entry["reason"] == "error: fatal"

    def test_status_transition(self, tmp_path):
        log = tmp_path / "activity.jsonl"
        prev = {"webapp": {"health": "green", "status": "active"}}
        with (
            patch("aw.core.ACTIVITY_LOG_PATH", log),
            patch("aw.core._prev_activity_state", prev),
        ):
            from aw.core import _log_transitions

            _log_transitions([
                {"project_id": "webapp", "health": "green", "status": "waiting", "agent": "claude"},
            ])

        entry = json.loads(log.read_text().splitlines()[0])
        assert entry["event"] == "status_active_to_waiting"

    def test_no_change_no_log(self, tmp_path):
        log = tmp_path / "activity.jsonl"
        prev = {"webapp": {"health": "green", "status": "active"}}
        with (
            patch("aw.core.ACTIVITY_LOG_PATH", log),
            patch("aw.core._prev_activity_state", prev),
        ):
            from aw.core import _log_transitions

            _log_transitions([
                {"project_id": "webapp", "health": "green", "status": "active", "agent": "claude"},
            ])

        assert not log.exists()

    def test_skips_dash_project_id(self, tmp_path):
        log = tmp_path / "activity.jsonl"
        with (
            patch("aw.core.ACTIVITY_LOG_PATH", log),
            patch("aw.core._prev_activity_state", {}),
        ):
            from aw.core import _log_transitions

            _log_transitions([
                {"project_id": "-", "health": "green", "status": "idle", "agent": "-"},
            ])

        assert not log.exists()
