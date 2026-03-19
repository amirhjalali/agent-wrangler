"""Tests for detection functions in aw.core — error markers, prompts, status parsing."""

from aw.core import (
    detect_error_marker,
    detect_missing_command,
    detect_port_in_use,
    detect_prompt_waiting,
    _parse_claude_status_from_text,
)


class TestDetectErrorMarker:
    def test_returns_traceback(self):
        assert detect_error_marker("File line 1\ntraceback (most recent call last)") == "traceback"

    def test_returns_fatal(self):
        assert detect_error_marker("fatal: not a git repository") == "fatal"

    def test_returns_npm_err(self):
        assert detect_error_marker("npm err! code ELIFECYCLE") == "npm err!"

    def test_returns_command_not_found(self):
        assert detect_error_marker("zsh: command not found: foo") == "zsh: command not found"

    def test_returns_exception(self):
        assert detect_error_marker("KeyError exception raised") == "exception"

    def test_returns_elifecycle(self):
        assert detect_error_marker("elifecycle error") == "elifecycle"

    def test_returns_command_failed(self):
        assert detect_error_marker("command failed with exit code 1") == "command failed"

    def test_returns_exited_1(self):
        assert detect_error_marker("process exited (1)") == "exited (1)"

    def test_no_match_returns_none(self):
        assert detect_error_marker("all good, no errors here") is None

    def test_empty_string_returns_none(self):
        assert detect_error_marker("") is None

    def test_none_returns_none(self):
        assert detect_error_marker(None) is None


class TestDetectMissingCommand:
    def test_extracts_command_name(self):
        assert detect_missing_command("command not found: docker") == "docker"

    def test_dotted_command(self):
        assert detect_missing_command("command not found: node.js") == "node.js"

    def test_path_command(self):
        assert detect_missing_command("command not found: /usr/local/bin/foo") == "/usr/local/bin/foo"

    def test_no_match_returns_none(self):
        assert detect_missing_command("everything is fine") is None

    def test_empty_string_returns_none(self):
        assert detect_missing_command("") is None

    def test_none_returns_none(self):
        assert detect_missing_command(None) is None


class TestDetectPortInUse:
    def test_extracts_port_number(self):
        assert detect_port_in_use("port 3000 is in use") == "3000"

    def test_extracts_high_port(self):
        assert detect_port_in_use("Error: port 8080 is in use, try another") == "8080"

    def test_no_match_returns_none(self):
        assert detect_port_in_use("server started on port 3000") is None

    def test_empty_string_returns_none(self):
        assert detect_port_in_use("") is None

    def test_none_returns_none(self):
        assert detect_port_in_use(None) is None


class TestDetectPromptWaiting:
    def test_claude_prompt_detected(self):
        text = "some output\nmore output\n❯"
        assert detect_prompt_waiting(text, "claude") is True

    def test_claude_greater_than_prompt(self):
        text = "some output\n>"
        assert detect_prompt_waiting(text, "claude") is True

    def test_codex_prompt_detected(self):
        text = "thinking...\ndone\n>"
        assert detect_prompt_waiting(text, "codex") is True

    def test_gemini_prompt_detected(self):
        text = "response done\n❯"
        assert detect_prompt_waiting(text, "gemini") is True

    def test_active_output_not_waiting(self):
        text = "Generating code...\nWriting files...\nAlmost done"
        assert detect_prompt_waiting(text, "claude") is False

    def test_skips_status_bar_lines(self):
        text = "some output\n❯\nOpus 4.6 | ●●●●○○ 86k/200k (42%) | ~$2.93"
        assert detect_prompt_waiting(text, "claude") is True

    def test_skips_separator_lines(self):
        text = "some output\n❯\n─────────────"
        assert detect_prompt_waiting(text, "claude") is True

    def test_empty_string_returns_false(self):
        assert detect_prompt_waiting("", "claude") is False

    def test_none_text_returns_false(self):
        assert detect_prompt_waiting(None, "claude") is False

    def test_none_agent_returns_false(self):
        assert detect_prompt_waiting("❯", None) is False

    def test_empty_agent_returns_false(self):
        assert detect_prompt_waiting("❯", "") is False


class TestParseClaudeStatusFromText:
    def test_parses_full_status_line(self):
        raw = (
            "Some output above\n"
            "More output\n"
            "Opus 4.6 | ●●●●○○○○○○ 86k/200k (42%) | ~$2.93\n"
        )
        result = _parse_claude_status_from_text(raw)
        assert result is not None
        assert result["model"] == "Opus 4.6"
        assert result["tokens_k"] == 86
        assert result["tokens_max_k"] == 200
        assert result["context_pct"] == 42
        assert result["cost"] == 2.93

    def test_parses_sonnet_model(self):
        raw = (
            "output\n"
            "Sonnet 4.0 | ●●○○○○○○○○ 12k/200k (6%) | ~$0.45\n"
        )
        result = _parse_claude_status_from_text(raw)
        assert result is not None
        assert result["model"] == "Sonnet 4.0"
        assert result["tokens_k"] == 12
        assert result["cost"] == 0.45

    def test_parses_rate_limit_line(self):
        raw = (
            "output\n"
            "more output\n"
            "Opus 4.6 | ●●●●○○○○○○ 86k/200k (42%) | ~$2.93\n"
            "5hr ○○○○○○○○○○ 2% in 3h 33m | 7d ●●●○○○○○○○ 39% in 1d 15h | extra $0.00/$50\n"
        )
        result = _parse_claude_status_from_text(raw)
        assert result is not None
        assert result["model"] == "Opus 4.6"
        assert result["rate_5hr"] == 2
        assert result["rate_7d"] == 39

    def test_empty_string_returns_none(self):
        assert _parse_claude_status_from_text("") is None

    def test_none_returns_none(self):
        assert _parse_claude_status_from_text(None) is None

    def test_no_status_returns_none(self):
        assert _parse_claude_status_from_text("just some random text\nno status here") is None

    def test_single_line_returns_none(self):
        assert _parse_claude_status_from_text("only one line") is None
