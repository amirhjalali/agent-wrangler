"""Tests for pane_health_level and style_for_level from aw.core."""

from aw.core import pane_health_level, style_for_level


class TestPaneHealthLevel:
    """Tests for pane_health_level(monitor, error_marker, wait_attention_min, prompt_waiting)."""

    def test_error_marker_returns_red(self):
        monitor = {"status": "active", "agent": "claude", "path_match": True}
        level, needs_attention, reason = pane_health_level(monitor, "segfault", 5)
        assert level == "red"
        assert needs_attention is True
        assert "segfault" in reason

    def test_agent_path_match_active_returns_green(self):
        monitor = {"status": "active", "agent": "claude", "path_match": True}
        level, needs_attention, reason = pane_health_level(monitor, None, 5)
        assert level == "green"
        assert needs_attention is False
        assert reason == ""

    def test_agent_path_match_waiting_returns_yellow(self):
        monitor = {"status": "waiting", "agent": "claude", "path_match": True}
        level, needs_attention, reason = pane_health_level(monitor, None, 5)
        assert level == "yellow"
        assert needs_attention is False
        assert reason == ""

    def test_agent_tty_not_prompt_waiting_returns_green(self):
        monitor = {"status": "active", "agent": "claude", "path_match": False}
        level, needs_attention, reason = pane_health_level(monitor, None, 5, prompt_waiting=False)
        assert level == "green"
        assert needs_attention is False
        assert reason == ""

    def test_agent_tty_prompt_waiting_returns_yellow(self):
        monitor = {"status": "active", "agent": "claude", "path_match": False}
        level, needs_attention, reason = pane_health_level(monitor, None, 5, prompt_waiting=True)
        assert level == "yellow"
        assert needs_attention is False
        assert reason == ""

    def test_no_agent_idle_returns_yellow_no_agent(self):
        monitor = {"status": "idle", "agent": "", "path_match": False}
        level, needs_attention, reason = pane_health_level(monitor, None, 5)
        assert level == "yellow"
        assert needs_attention is False
        assert reason == "no agent"

    def test_no_agent_active_returns_yellow_no_agent(self):
        monitor = {"status": "active", "agent": "", "path_match": False}
        level, needs_attention, reason = pane_health_level(monitor, None, 5)
        assert level == "yellow"
        assert needs_attention is False
        assert reason == "no agent"

    def test_background_returns_yellow_background(self):
        monitor = {"status": "background", "agent": "", "path_match": False}
        level, needs_attention, reason = pane_health_level(monitor, None, 5)
        assert level == "yellow"
        assert needs_attention is False
        assert reason == "background"


class TestStyleForLevel:
    """Tests for style_for_level(level) returning border style tuples."""

    def test_red_level(self):
        inactive, active = style_for_level("red")
        assert "colour88" in inactive
        assert "colour214" in active

    def test_yellow_level(self):
        inactive, active = style_for_level("yellow")
        assert "colour130" in inactive
        assert "colour214" in active

    def test_green_level(self):
        inactive, active = style_for_level("green")
        assert "colour22" in inactive
        assert "colour214" in active
