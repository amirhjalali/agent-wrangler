"""Tests for terminal_sentinel pure functions: time parsing, command classification, AI detection."""

from terminal_sentinel import (
    command_bin,
    detect_ai_tool,
    fmt_seconds,
    is_ai_command,
    is_wrapper,
    parse_ps_time,
)


class TestParseTime:
    """Tests for parse_ps_time(raw) — converts ps time strings to seconds."""

    def test_minutes_seconds(self):
        assert parse_ps_time("12:34.56") == 12 * 60 + 34.56

    def test_hours_minutes_seconds(self):
        assert parse_ps_time("02:15:30") == 2 * 3600 + 15 * 60 + 30

    def test_days_hours_minutes_seconds(self):
        assert parse_ps_time("3-04:30:00") == 3 * 86400 + 4 * 3600 + 30 * 60

    def test_empty_string(self):
        assert parse_ps_time("") == 0.0

    def test_whitespace_only(self):
        assert parse_ps_time("   ") == 0.0


class TestFmtSeconds:
    """Tests for fmt_seconds(seconds) — human-readable duration."""

    def test_under_minute(self):
        assert fmt_seconds(45) == "45s"

    def test_minutes(self):
        assert fmt_seconds(300) == "5m"

    def test_hours(self):
        assert fmt_seconds(7200) == "2h"

    def test_days(self):
        assert fmt_seconds(172800) == "2d"

    def test_zero(self):
        assert fmt_seconds(0) == "0s"


class TestCommandBin:
    """Tests for command_bin(command) — extracts binary basename."""

    def test_absolute_path(self):
        assert command_bin("/usr/bin/python3") == "python3"

    def test_command_with_args(self):
        assert command_bin("node server.js") == "node"

    def test_dash_prefix_shell(self):
        assert command_bin("-zsh") == "zsh"


class TestIsWrapper:
    """Tests for is_wrapper(command) — identifies shell/wrapper processes."""

    def test_zsh_is_wrapper(self):
        assert is_wrapper("zsh") is True

    def test_bash_is_wrapper(self):
        assert is_wrapper("/bin/bash") is True

    def test_python_not_wrapper(self):
        assert is_wrapper("python3 app.py") is False

    def test_login_shell_is_wrapper(self):
        assert is_wrapper("-zsh") is True


class TestDetectAiTool:
    """Tests for detect_ai_tool(command) and is_ai_command(command)."""

    def test_claude(self):
        assert detect_ai_tool("claude chat") == "claude"

    def test_codex(self):
        assert detect_ai_tool("codex --model o4-mini") == "codex"

    def test_gemini(self):
        assert detect_ai_tool("gemini generate") == "gemini"

    def test_chatgpt(self):
        assert detect_ai_tool("chatgpt ask") == "chatgpt"

    def test_non_ai(self):
        assert detect_ai_tool("vim main.py") is None

    def test_is_ai_command_true(self):
        assert is_ai_command("claude code") is True

    def test_is_ai_command_false(self):
        assert is_ai_command("node index.js") is False
