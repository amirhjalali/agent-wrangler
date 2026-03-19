"""Tests for config normalization, snapshot names, layout selection, and project inference."""

import pytest

from aw.core import (
    _normalize_store,
    choose_layout,
    infer_project_id_from_command,
    infer_project_id_from_path,
    sanitize_snapshot_name,
    split_csv,
)


# ---------------------------------------------------------------------------
# _normalize_store
# ---------------------------------------------------------------------------

class TestNormalizeStore:
    """Tests for _normalize_store(data)."""

    def test_none_returns_defaults(self):
        result = _normalize_store(None)
        assert result["default_session"] == "amir-grid"
        assert result["profiles"]["current"] == "default"
        assert "default" in result["profiles"]["items"]

    def test_empty_dict_returns_defaults(self):
        result = _normalize_store({})
        assert result["default_session"] == "amir-grid"
        assert result["default_layout"] == "auto"
        assert result["persistence"]["enabled"] is False

    def test_preserves_custom_session(self):
        result = _normalize_store({"default_session": "my-session"})
        assert result["default_session"] == "my-session"

    def test_preserves_custom_persistence(self):
        result = _normalize_store({
            "persistence": {"enabled": True, "autosave_minutes": 30, "last_snapshot": "snap.json"},
        })
        assert result["persistence"]["enabled"] is True
        assert result["persistence"]["autosave_minutes"] == 30
        assert result["persistence"]["last_snapshot"] == "snap.json"

    def test_corrupted_persistence_uses_defaults(self):
        result = _normalize_store({"persistence": "broken"})
        assert result["persistence"]["enabled"] is False
        assert result["persistence"]["autosave_minutes"] == 15

    def test_corrupted_profiles_creates_default(self):
        result = _normalize_store({"profiles": "not-a-dict"})
        assert "default" in result["profiles"]["items"]
        assert result["profiles"]["current"] == "default"

    def test_ensures_default_profile_exists(self):
        result = _normalize_store({
            "profiles": {"current": "custom", "items": {"custom": {"managed_sessions": [], "max_panes": 5}}},
        })
        assert "default" in result["profiles"]["items"]
        assert "custom" in result["profiles"]["items"]
        assert result["profiles"]["items"]["custom"]["max_panes"] == 5

    def test_resets_current_profile_if_missing(self):
        result = _normalize_store({
            "profiles": {"current": "ghost", "items": {"work": {"managed_sessions": [], "max_panes": 8}}},
        })
        assert result["profiles"]["current"] == "default"


# ---------------------------------------------------------------------------
# sanitize_snapshot_name
# ---------------------------------------------------------------------------

class TestSanitizeSnapshotName:
    """Tests for sanitize_snapshot_name(name)."""

    def test_none_returns_snapshot_json(self):
        assert sanitize_snapshot_name(None) == "snapshot.json"

    def test_empty_returns_snapshot_json(self):
        assert sanitize_snapshot_name("") == "snapshot.json"

    def test_whitespace_returns_snapshot_json(self):
        assert sanitize_snapshot_name("   ") == "snapshot.json"

    def test_strips_special_characters(self):
        result = sanitize_snapshot_name("my save!@#$%^&*()")
        assert result == "my-save.json"

    def test_preserves_json_extension(self):
        assert sanitize_snapshot_name("backup.json") == "backup.json"

    def test_appends_json_when_missing(self):
        assert sanitize_snapshot_name("backup") == "backup.json"


# ---------------------------------------------------------------------------
# split_csv
# ---------------------------------------------------------------------------

class TestSplitCsv:
    """Tests for split_csv(value)."""

    def test_none_returns_empty(self):
        assert split_csv(None) == []

    def test_empty_string_returns_empty(self):
        assert split_csv("") == []

    def test_splits_and_strips(self):
        assert split_csv(" a , b , c ") == ["a", "b", "c"]

    def test_filters_empties(self):
        assert split_csv("a,,b,  ,c") == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# choose_layout
# ---------------------------------------------------------------------------

class TestChooseLayout:
    """Tests for choose_layout(layout, pane_count)."""

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

    def test_explicit_layout_passes_through(self):
        assert choose_layout("tiled", 3) == "tiled"

    def test_invalid_layout_raises(self):
        with pytest.raises(ValueError, match="Invalid layout"):
            choose_layout("waterfall", 3)

    def test_none_defaults_to_auto(self):
        assert choose_layout(None, 1) == "even-horizontal"


# ---------------------------------------------------------------------------
# infer_project_id_from_path
# ---------------------------------------------------------------------------

class TestInferProjectIdFromPath:
    """Tests for infer_project_id_from_path(path, proj_map)."""

    _proj_map = {
        "webapp": {"path": "/home/user/webapp"},
        "api": {"path": "/home/user/api"},
        "nested": {"path": "/home/user/webapp/packages/core"},
    }

    def test_exact_match(self):
        assert infer_project_id_from_path("/home/user/webapp", self._proj_map) == "webapp"

    def test_subdirectory_match(self):
        assert infer_project_id_from_path("/home/user/api/src/main.py", self._proj_map) == "api"

    def test_longest_match_wins(self):
        result = infer_project_id_from_path("/home/user/webapp/packages/core/index.ts", self._proj_map)
        assert result == "nested"

    def test_no_match_returns_none(self):
        assert infer_project_id_from_path("/tmp/other", self._proj_map) is None

    def test_empty_path_returns_none(self):
        assert infer_project_id_from_path("", self._proj_map) is None


# ---------------------------------------------------------------------------
# infer_project_id_from_command
# ---------------------------------------------------------------------------

class TestInferProjectIdFromCommand:
    """Tests for infer_project_id_from_command(command, proj_map)."""

    _proj_map = {
        "webapp": {"path": "/home/user/webapp"},
        "api": {"path": "/home/user/api"},
    }

    def test_matches_path_in_command(self):
        result = infer_project_id_from_command("cd /home/user/webapp && npm start", self._proj_map)
        assert result == "webapp"

    def test_no_match_returns_none(self):
        assert infer_project_id_from_command("echo hello", self._proj_map) is None
