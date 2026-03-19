# Test Suite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a pytest test suite covering the core pure functions that don't require a live tmux session, plus integration tests for the activity log.

**Architecture:** Tests live in `tests/` at the repo root. Focus on pure functions first (no tmux, no subprocess) for fast, reliable tests. Use `tmp_path` fixtures for file-based tests. Mock tmux calls only where necessary. No external dependencies beyond pytest.

**Tech Stack:** Python 3.11, pytest (installed via pip)

---

### Task 1: Set up test infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `pytest.ini`

**Step 1: Install pytest**

```bash
pip3 install pytest
```

**Step 2: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
pythonpath = scripts
```

**Step 3: Create `tests/__init__.py`**

Empty file.

**Step 4: Create `tests/conftest.py`**

```python
"""Shared fixtures for agent-wrangler tests."""
import json
import pytest
from pathlib import Path


@pytest.fixture
def sample_projects_config(tmp_path):
    """Create a temporary projects.json with test data."""
    config = {
        "projects": [
            {"id": "webapp", "name": "webapp", "path": "/home/user/webapp", "default_branch": "main"},
            {"id": "api", "name": "api", "path": "/home/user/api", "default_branch": "main", "group": "business"},
            {"id": "docs", "name": "docs", "path": "/home/user/docs", "default_branch": "main", "barn": True},
        ]
    }
    config_file = tmp_path / "projects.json"
    config_file.write_text(json.dumps(config))
    return config_file, config


@pytest.fixture
def sample_store(tmp_path):
    """Create a temporary team_grid.json with test data."""
    store = {
        "default_session": "test-grid",
        "default_layout": "tiled",
        "default_projects": ["webapp", "api"],
        "persistence": {"enabled": False, "autosave_minutes": 15, "last_snapshot": ""},
        "profiles": {
            "current": "default",
            "items": {"default": {"managed_sessions": [], "max_panes": 10}},
        },
    }
    store_file = tmp_path / "team_grid.json"
    store_file.write_text(json.dumps(store))
    return store_file, store
```

**Step 5: Verify pytest runs**

```bash
python3 -m pytest --co -q
```

Expected: `no tests ran` (no test files yet)

**Step 6: Commit**

```bash
git add tests/ pytest.ini
git commit -m "test: set up pytest infrastructure with fixtures"
```

---

### Task 2: Test detection functions (pure, no tmux)

These functions take strings and return results — perfect unit tests.

**Files:**
- Create: `tests/test_detection.py`

**Step 1: Write tests**

```python
"""Tests for error/prompt/status detection functions."""
from aw.core import (
    detect_error_marker,
    detect_missing_command,
    detect_port_in_use,
    detect_prompt_waiting,
    _parse_claude_status_from_text,
)


class TestDetectErrorMarker:
    def test_finds_traceback(self):
        assert detect_error_marker("some output\ntraceback\nmore") == "traceback"

    def test_finds_fatal(self):
        assert detect_error_marker("fatal error occurred") == "fatal"

    def test_finds_npm_err(self):
        assert detect_error_marker("npm err! something failed") == "npm err!"

    def test_finds_command_not_found(self):
        assert detect_error_marker("zsh: command not found: foo") == "zsh: command not found"

    def test_returns_none_for_clean_output(self):
        assert detect_error_marker("all good here\nno problems") is None

    def test_returns_none_for_empty(self):
        assert detect_error_marker("") is None

    def test_returns_none_for_none_input(self):
        assert detect_error_marker(None) is None


class TestDetectMissingCommand:
    def test_finds_missing_command(self):
        assert detect_missing_command("zsh: command not found: pnpm") == "pnpm"

    def test_finds_missing_claude(self):
        assert detect_missing_command("command not found: claude") == "claude"

    def test_returns_none_for_clean(self):
        assert detect_missing_command("everything is fine") is None

    def test_returns_none_for_empty(self):
        assert detect_missing_command("") is None


class TestDetectPortInUse:
    def test_finds_port(self):
        assert detect_port_in_use("error: port 3000 is in use") == "3000"

    def test_finds_port_8080(self):
        assert detect_port_in_use("port 8080 is in use by another process") == "8080"

    def test_returns_none_for_clean(self):
        assert detect_port_in_use("server started on port 3000") is None

    def test_returns_none_for_empty(self):
        assert detect_port_in_use("") is None


class TestDetectPromptWaiting:
    def test_claude_prompt_detected(self):
        raw = "some output\nmore output\n❯\n"
        assert detect_prompt_waiting(raw, "claude") is True

    def test_claude_angle_bracket_prompt(self):
        raw = "some output\nmore output\n>\n"
        assert detect_prompt_waiting(raw, "claude") is True

    def test_claude_active_output(self):
        raw = "Generating code...\nWriting file foo.py\nDone.\n"
        assert detect_prompt_waiting(raw, "claude") is False

    def test_codex_prompt(self):
        raw = "output\n>\n"
        assert detect_prompt_waiting(raw, "codex") is True

    def test_skips_status_bar_lines(self):
        raw = "real output here\n❯\nOpus 4.6 | ●●●●○○○○○○ 86k/200k (42%) | ~$2.93\n"
        # The ❯ should be found before the status bar line
        assert detect_prompt_waiting(raw, "claude") is True

    def test_empty_returns_false(self):
        assert detect_prompt_waiting("", "claude") is False

    def test_no_agent_returns_false(self):
        assert detect_prompt_waiting("❯\n", "") is False


class TestParseClaudeStatus:
    def test_parses_model_line(self):
        raw = (
            "some output\n"
            "more output\n"
            "Opus 4.6 | ●●●●○○○○○○ 86k/200k (42%) | ~$2.93\n"
            "5hr ○○○○○○○○○○ 2% in 3h 33m | 7d ●●●○○○○○○○ 39% in 1d 15h | extra $0.00/$50\n"
        )
        result = _parse_claude_status_from_text(raw)
        assert result is not None
        assert result["model"] == "Opus 4.6"
        assert result["tokens_k"] == 86
        assert result["tokens_max_k"] == 200
        assert result["context_pct"] == 42
        assert result["cost"] == 2.93

    def test_parses_rate_limits(self):
        raw = (
            "output\n"
            "Opus 4.6 | ●●●●○○○○○○ 86k/200k (42%) | ~$2.93\n"
            "5hr ○○○○○○○○○○ 2% in 3h 33m | 7d ●●●○○○○○○○ 39% in 1d 15h | extra $0.00/$50\n"
        )
        result = _parse_claude_status_from_text(raw)
        assert result is not None
        assert result.get("rate_5hr") == 2
        assert result.get("rate_7d") == 39

    def test_returns_none_for_empty(self):
        assert _parse_claude_status_from_text("") is None

    def test_returns_none_for_no_status(self):
        assert _parse_claude_status_from_text("just regular output\nnothing special\n") is None

    def test_returns_none_for_single_line(self):
        assert _parse_claude_status_from_text("one line") is None
```

**Step 2: Run tests**

```bash
python3 -m pytest tests/test_detection.py -v
```

Expected: All pass.

**Step 3: Commit**

```bash
git add tests/test_detection.py
git commit -m "test: detection functions - error markers, prompts, status parsing"
```

---

### Task 3: Test health level logic

**Files:**
- Create: `tests/test_health.py`

**Step 1: Write tests**

```python
"""Tests for health level classification."""
from aw.core import pane_health_level, style_for_level


class TestPaneHealthLevel:
    def test_error_marker_returns_red(self):
        level, attention, reason = pane_health_level(
            monitor={"status": "active", "agent": "claude"},
            error_marker="traceback",
            wait_attention_min=5,
        )
        assert level == "red"
        assert attention is True
        assert "traceback" in reason

    def test_active_agent_tty_match_no_prompt(self):
        """Agent running in pane, not at prompt = green."""
        level, attention, reason = pane_health_level(
            monitor={"status": "active", "agent": "claude"},
            error_marker=None,
            wait_attention_min=5,
            prompt_waiting=False,
        )
        assert level == "green"
        assert attention is False

    def test_active_agent_tty_match_at_prompt(self):
        """Agent running in pane, at prompt = yellow."""
        level, attention, reason = pane_health_level(
            monitor={"status": "waiting", "agent": "claude"},
            error_marker=None,
            wait_attention_min=5,
            prompt_waiting=True,
        )
        assert level == "yellow"

    def test_path_match_active(self):
        """Agent matched by directory, sentinel says active = green."""
        level, _, _ = pane_health_level(
            monitor={"status": "active", "agent": "claude", "path_match": True},
            error_marker=None,
            wait_attention_min=5,
        )
        assert level == "green"

    def test_path_match_waiting(self):
        """Agent matched by directory, sentinel says waiting = yellow."""
        level, _, _ = pane_health_level(
            monitor={"status": "waiting", "agent": "claude", "path_match": True},
            error_marker=None,
            wait_attention_min=5,
        )
        assert level == "yellow"

    def test_no_agent_idle(self):
        """No agent running = yellow with 'no agent' reason."""
        level, _, reason = pane_health_level(
            monitor={"status": "idle", "agent": ""},
            error_marker=None,
            wait_attention_min=5,
        )
        assert level == "yellow"
        assert reason == "no agent"

    def test_background_status(self):
        level, _, reason = pane_health_level(
            monitor={"status": "background", "agent": ""},
            error_marker=None,
            wait_attention_min=5,
        )
        assert level == "yellow"
        assert reason == "background"

    def test_error_takes_precedence_over_agent(self):
        """Error marker should override even active agent."""
        level, attention, _ = pane_health_level(
            monitor={"status": "active", "agent": "claude"},
            error_marker="fatal",
            wait_attention_min=5,
        )
        assert level == "red"
        assert attention is True


class TestStyleForLevel:
    def test_green(self):
        inactive, active = style_for_level("green")
        assert "colour22" in inactive
        assert "colour214" in active

    def test_yellow(self):
        inactive, active = style_for_level("yellow")
        assert "colour130" in inactive

    def test_red(self):
        inactive, active = style_for_level("red")
        assert "colour88" in inactive
```

**Step 2: Run tests**

```bash
python3 -m pytest tests/test_health.py -v
```

**Step 3: Commit**

```bash
git add tests/test_health.py
git commit -m "test: health level classification and style mapping"
```

---

### Task 4: Test config and store functions

**Files:**
- Create: `tests/test_config.py`

**Step 1: Write tests**

```python
"""Tests for config loading, store normalization, and project resolution."""
import json
from pathlib import Path
from unittest.mock import patch

from aw.core import (
    _normalize_store,
    default_store,
    sanitize_snapshot_name,
    split_csv,
    choose_layout,
    infer_project_id_from_path,
    infer_project_id_from_command,
)


class TestNormalizeStore:
    def test_none_returns_defaults(self):
        result = _normalize_store(None)
        assert result["default_session"] == "amir-grid"
        assert result["profiles"]["current"] == "default"

    def test_empty_dict_returns_defaults(self):
        result = _normalize_store({})
        assert "profiles" in result
        assert "persistence" in result

    def test_preserves_custom_session(self):
        result = _normalize_store({"default_session": "my-grid"})
        assert result["default_session"] == "my-grid"

    def test_normalizes_persistence(self):
        result = _normalize_store({"persistence": {"enabled": True, "autosave_minutes": 30}})
        assert result["persistence"]["enabled"] is True
        assert result["persistence"]["autosave_minutes"] == 30

    def test_handles_corrupted_persistence(self):
        result = _normalize_store({"persistence": "not a dict"})
        assert result["persistence"]["enabled"] is False

    def test_normalizes_profiles(self):
        result = _normalize_store({
            "profiles": {
                "current": "work",
                "items": {
                    "work": {"managed_sessions": ["sess1"], "max_panes": 6},
                },
            },
        })
        assert result["profiles"]["current"] == "work"
        assert result["profiles"]["items"]["work"]["max_panes"] == 6

    def test_adds_default_profile_if_missing(self):
        result = _normalize_store({"profiles": {"items": {}}})
        assert "default" in result["profiles"]["items"]

    def test_resets_current_if_not_in_items(self):
        result = _normalize_store({
            "profiles": {"current": "nonexistent", "items": {}},
        })
        assert result["profiles"]["current"] == "default"


class TestSanitizeSnapshotName:
    def test_simple_name(self):
        assert sanitize_snapshot_name("my-session") == "my-session.json"

    def test_already_has_extension(self):
        assert sanitize_snapshot_name("backup.json") == "backup.json"

    def test_strips_special_chars(self):
        result = sanitize_snapshot_name("my session!@#")
        assert " " not in result
        assert "!" not in result
        assert result.endswith(".json")

    def test_empty_returns_snapshot(self):
        assert sanitize_snapshot_name("") == "snapshot.json"

    def test_none_returns_snapshot(self):
        assert sanitize_snapshot_name(None) == "snapshot.json"


class TestSplitCsv:
    def test_splits_values(self):
        assert split_csv("a,b,c") == ["a", "b", "c"]

    def test_strips_whitespace(self):
        assert split_csv(" a , b , c ") == ["a", "b", "c"]

    def test_empty_returns_empty(self):
        assert split_csv("") == []

    def test_none_returns_empty(self):
        assert split_csv(None) == []

    def test_filters_empty_items(self):
        assert split_csv("a,,b,") == ["a", "b"]


class TestChooseLayout:
    def test_auto_2_panes(self):
        assert choose_layout("auto", 2) == "even-horizontal"

    def test_auto_4_panes(self):
        assert choose_layout("auto", 4) == "tiled"

    def test_auto_6_panes(self):
        assert choose_layout("auto", 6) == "main-vertical"

    def test_auto_9_panes(self):
        assert choose_layout("auto", 9) == "tiled"

    def test_auto_12_panes(self):
        assert choose_layout("auto", 12) == "even-vertical"

    def test_explicit_layout(self):
        assert choose_layout("tiled", 3) == "tiled"

    def test_invalid_layout_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Invalid layout"):
            choose_layout("bad-layout", 3)


class TestInferProjectIdFromPath:
    def test_exact_match(self):
        proj_map = {"webapp": {"path": "/home/user/webapp"}}
        assert infer_project_id_from_path("/home/user/webapp", proj_map) == "webapp"

    def test_subdirectory_match(self):
        proj_map = {"webapp": {"path": "/home/user/webapp"}}
        assert infer_project_id_from_path("/home/user/webapp/src", proj_map) == "webapp"

    def test_no_match(self):
        proj_map = {"webapp": {"path": "/home/user/webapp"}}
        assert infer_project_id_from_path("/home/user/other", proj_map) is None

    def test_longest_match_wins(self):
        proj_map = {
            "parent": {"path": "/home/user"},
            "child": {"path": "/home/user/webapp"},
        }
        assert infer_project_id_from_path("/home/user/webapp/src", proj_map) == "child"

    def test_empty_path(self):
        assert infer_project_id_from_path("", {"a": {"path": "/foo"}}) is None


class TestInferProjectIdFromCommand:
    def test_path_in_command(self):
        proj_map = {"webapp": {"path": "/home/user/webapp"}}
        assert infer_project_id_from_command("cd /home/user/webapp && npm start", proj_map) == "webapp"

    def test_no_match(self):
        proj_map = {"webapp": {"path": "/home/user/webapp"}}
        assert infer_project_id_from_command("npm start", proj_map) is None
```

**Step 2: Run tests**

```bash
python3 -m pytest tests/test_config.py -v
```

**Step 3: Commit**

```bash
git add tests/test_config.py
git commit -m "test: config normalization, snapshot names, layout selection, project inference"
```

---

### Task 5: Test activity log

**Files:**
- Create: `tests/test_activity.py`

**Step 1: Write tests**

```python
"""Tests for the activity log (JSONL append/read/transitions)."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from aw.core import _append_activity, _read_activity, _log_transitions


class TestAppendActivity:
    def test_creates_file_and_appends(self, tmp_path):
        log_path = tmp_path / "activity.jsonl"
        with patch("aw.core.ACTIVITY_LOG_PATH", log_path):
            _append_activity([{"event": "test", "project": "foo"}])

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["event"] == "test"
        assert entry["project"] == "foo"
        assert "ts" in entry  # auto-added

    def test_appends_multiple_entries(self, tmp_path):
        log_path = tmp_path / "activity.jsonl"
        with patch("aw.core.ACTIVITY_LOG_PATH", log_path):
            _append_activity([
                {"event": "a"},
                {"event": "b"},
            ])

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_preserves_existing_entries(self, tmp_path):
        log_path = tmp_path / "activity.jsonl"
        log_path.write_text('{"event":"old"}\n')
        with patch("aw.core.ACTIVITY_LOG_PATH", log_path):
            _append_activity([{"event": "new"}])

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["event"] == "old"
        assert json.loads(lines[1])["event"] == "new"

    def test_empty_list_is_noop(self, tmp_path):
        log_path = tmp_path / "activity.jsonl"
        with patch("aw.core.ACTIVITY_LOG_PATH", log_path):
            _append_activity([])
        assert not log_path.exists()

    def test_rotates_at_max_size(self, tmp_path):
        log_path = tmp_path / "activity.jsonl"
        # Write a file that exceeds the max size
        log_path.write_text("x" * 100)
        with patch("aw.core.ACTIVITY_LOG_PATH", log_path), \
             patch("aw.core.ACTIVITY_MAX_BYTES", 50):
            _append_activity([{"event": "after_rotate"}])

        # Original should be rotated
        old_path = log_path.with_suffix(".jsonl.old")
        assert old_path.exists()
        # New file should have just the new entry
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1


class TestReadActivity:
    def test_reads_all_entries(self, tmp_path):
        log_path = tmp_path / "activity.jsonl"
        now = datetime.now(timezone.utc).isoformat()
        log_path.write_text(
            json.dumps({"ts": now, "event": "a"}) + "\n"
            + json.dumps({"ts": now, "event": "b"}) + "\n"
        )
        with patch("aw.core.ACTIVITY_LOG_PATH", log_path):
            entries = _read_activity()
        assert len(entries) == 2

    def test_filters_by_time(self, tmp_path):
        log_path = tmp_path / "activity.jsonl"
        old_ts = "2020-01-01T00:00:00+00:00"
        now = datetime.now(timezone.utc).isoformat()
        log_path.write_text(
            json.dumps({"ts": old_ts, "event": "old"}) + "\n"
            + json.dumps({"ts": now, "event": "new"}) + "\n"
        )
        with patch("aw.core.ACTIVITY_LOG_PATH", log_path):
            entries = _read_activity(since_minutes=5)
        assert len(entries) == 1
        assert entries[0]["event"] == "new"

    def test_respects_limit(self, tmp_path):
        log_path = tmp_path / "activity.jsonl"
        now = datetime.now(timezone.utc).isoformat()
        lines = [json.dumps({"ts": now, "event": f"e{i}"}) for i in range(10)]
        log_path.write_text("\n".join(lines) + "\n")
        with patch("aw.core.ACTIVITY_LOG_PATH", log_path):
            entries = _read_activity(limit=3)
        assert len(entries) == 3
        # Should be the last 3
        assert entries[0]["event"] == "e7"

    def test_returns_empty_for_missing_file(self, tmp_path):
        log_path = tmp_path / "nonexistent.jsonl"
        with patch("aw.core.ACTIVITY_LOG_PATH", log_path):
            entries = _read_activity()
        assert entries == []


class TestLogTransitions:
    def test_logs_first_seen(self, tmp_path):
        log_path = tmp_path / "activity.jsonl"
        with patch("aw.core.ACTIVITY_LOG_PATH", log_path), \
             patch("aw.core._prev_activity_state", {}):
            _log_transitions([
                {"project_id": "webapp", "health": "green", "status": "active", "agent": "claude", "reason": "", "cc_stats": None},
            ])

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["event"] == "first_seen"
        assert entry["project"] == "webapp"

    def test_logs_health_change(self, tmp_path):
        log_path = tmp_path / "activity.jsonl"
        prev_state = {"webapp": {"health": "green", "status": "active", "agent": "claude"}}
        with patch("aw.core.ACTIVITY_LOG_PATH", log_path), \
             patch("aw.core._prev_activity_state", prev_state):
            _log_transitions([
                {"project_id": "webapp", "health": "red", "status": "active", "agent": "claude", "reason": "error: fatal", "cc_stats": None},
            ])

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert "health_green_to_red" in entry["event"]

    def test_no_log_when_unchanged(self, tmp_path):
        log_path = tmp_path / "activity.jsonl"
        prev_state = {"webapp": {"health": "green", "status": "active", "agent": "claude"}}
        with patch("aw.core.ACTIVITY_LOG_PATH", log_path), \
             patch("aw.core._prev_activity_state", prev_state):
            _log_transitions([
                {"project_id": "webapp", "health": "green", "status": "active", "agent": "claude", "reason": "", "cc_stats": None},
            ])

        assert not log_path.exists()  # No changes = no writes
```

**Step 2: Run tests**

```bash
python3 -m pytest tests/test_activity.py -v
```

**Step 3: Commit**

```bash
git add tests/test_activity.py
git commit -m "test: activity log - append, read, rotation, transitions"
```

---

### Task 6: Test rail rendering functions (pure)

**Files:**
- Create: `tests/test_rail.py`

**Step 1: Write tests**

```python
"""Tests for rail rendering functions (pure, no tmux)."""
from aw.rail import _sparkline, _context_bar, _campfire_header


class TestSparkline:
    def test_empty_history(self):
        assert _sparkline([]) == ""

    def test_all_green(self):
        result = _sparkline([2, 2, 2])
        assert "\033[32m" in result  # green color

    def test_all_red(self):
        result = _sparkline([0, 0, 0])
        assert "\033[31m" in result  # red color

    def test_mixed(self):
        result = _sparkline([0, 1, 2])
        assert "\033[31m" in result
        assert "\033[33m" in result
        assert "\033[32m" in result

    def test_truncates_to_10(self):
        history = list(range(20))
        result = _sparkline(history)
        # Should only render last 10 entries
        assert result.count("▁") + result.count("▅") + result.count("█") <= 10


class TestContextBar:
    def test_zero_percent(self):
        result = _context_bar(0)
        assert "░" in result

    def test_100_percent(self):
        result = _context_bar(100)
        assert "█" in result

    def test_50_percent_yellow(self):
        result = _context_bar(50)
        assert "\033[33m" in result  # yellow threshold

    def test_80_percent_red(self):
        result = _context_bar(80)
        assert "\033[31m" in result  # red threshold

    def test_custom_width(self):
        result = _context_bar(50, width=10)
        # 50% of 10 = 5 filled + 5 empty
        assert result.count("█") == 5


class TestCampfireHeader:
    def test_returns_lines(self):
        result = _campfire_header(0, {"green": 3, "yellow": 1, "red": 0})
        assert isinstance(result, list)
        assert len(result) > 0

    def test_includes_herd_count(self):
        result = _campfire_header(0, {"green": 2, "yellow": 1, "red": 1})
        text = "\n".join(result)
        assert "4 head" in text

    def test_includes_ranch_board(self):
        result = _campfire_header(0, {"green": 1, "yellow": 0, "red": 0})
        text = "\n".join(result)
        assert "RANCH BOARD" in text
```

**Step 2: Run tests**

```bash
python3 -m pytest tests/test_rail.py -v
```

**Step 3: Commit**

```bash
git add tests/test_rail.py
git commit -m "test: rail rendering - sparklines, context bar, campfire header"
```

---

### Task 7: Test terminal sentinel (pure functions)

**Files:**
- Create: `tests/test_sentinel.py`

**Step 1: Write tests**

```python
"""Tests for terminal_sentinel pure functions."""
from terminal_sentinel import (
    parse_ps_time,
    fmt_seconds,
    command_bin,
    is_wrapper,
    detect_ai_tool,
    is_ai_command,
)


class TestParsePsTime:
    def test_minutes_seconds(self):
        assert parse_ps_time("1:30.00") == 90.0

    def test_hours_minutes_seconds(self):
        assert parse_ps_time("1:30:00") == 5400.0

    def test_with_days(self):
        assert parse_ps_time("2-0:00:00") == 172800.0

    def test_empty(self):
        assert parse_ps_time("") == 0.0

    def test_just_seconds(self):
        assert parse_ps_time("45") == 45.0


class TestFmtSeconds:
    def test_seconds(self):
        assert fmt_seconds(30) == "30s"

    def test_minutes(self):
        assert fmt_seconds(120) == "2m"

    def test_hours(self):
        assert fmt_seconds(7200) == "2h"

    def test_days(self):
        assert fmt_seconds(172800) == "2d"


class TestCommandBin:
    def test_full_path(self):
        assert command_bin("/usr/bin/python3") == "python3"

    def test_with_args(self):
        assert command_bin("node server.js") == "node"

    def test_dash_prefix(self):
        assert command_bin("-zsh") == "zsh"


class TestIsWrapper:
    def test_zsh_is_wrapper(self):
        assert is_wrapper("zsh") is True

    def test_bash_is_wrapper(self):
        assert is_wrapper("/bin/bash") is True

    def test_tmux_is_wrapper(self):
        assert is_wrapper("tmux") is True

    def test_python_is_not_wrapper(self):
        assert is_wrapper("python3 app.py") is False

    def test_claude_is_not_wrapper(self):
        assert is_wrapper("claude --help") is False


class TestDetectAiTool:
    def test_claude(self):
        assert detect_ai_tool("claude --dangerously-skip-permissions") == "claude"

    def test_codex(self):
        assert detect_ai_tool("codex run") == "codex"

    def test_gemini(self):
        assert detect_ai_tool("gemini chat") == "gemini"

    def test_anthropic(self):
        assert detect_ai_tool("/path/to/anthropic/thing") == "claude"

    def test_no_match(self):
        assert detect_ai_tool("python3 app.py") is None

    def test_empty(self):
        assert detect_ai_tool("") is None


class TestIsAiCommand:
    def test_claude_is_ai(self):
        assert is_ai_command("claude") is True

    def test_python_is_not_ai(self):
        assert is_ai_command("python3") is False
```

**Step 2: Run tests**

```bash
python3 -m pytest tests/test_sentinel.py -v
```

**Step 3: Commit**

```bash
git add tests/test_sentinel.py
git commit -m "test: terminal sentinel - time parsing, command classification, AI detection"
```

---

### Task 8: Run full suite and verify coverage

**Step 1: Run all tests**

```bash
python3 -m pytest tests/ -v --tb=short
```

Expected: All tests pass.

**Step 2: Count coverage**

```bash
python3 -m pytest tests/ -v --tb=short 2>&1 | tail -5
```

Report the total count. Should be ~60+ tests across 6 test files.

**Step 3: Update CLAUDE.md**

In `CLAUDE.md`, change the line that says "No build step, no test suite" to:

```
**No build step.** Pure Python 3.10+ and Bash. Tests: `python3 -m pytest tests/ -v`. All dependencies are stdlib (pytest for testing).
```

**Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md to reference test suite"
```

---

## Verification

After all tasks:

```bash
python3 -m pytest tests/ -v --tb=short
```

All tests should pass. The suite covers:
- Error/prompt/status detection (string parsing)
- Health level classification (decision logic)
- Config normalization (data validation)
- Activity log (file I/O with rotation)
- Rail rendering (display functions)
- Terminal sentinel (process classification)

## What's NOT tested (by design)

- Functions requiring a live tmux session (create_grid_session, refresh_pane_health, etc.)
- Functions that shell out to `ps`, `lsof`, or `osascript`
- The argparse registration and CLI routing
- The bash router script

These would need integration tests with a tmux fixture, which is a separate effort.
