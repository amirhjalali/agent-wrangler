"""Tests for rail rendering functions from aw.rail."""

from aw.rail import _sparkline, _context_bar, _campfire_header


RST = "\033[0m"
RED = "\033[31m"
YEL = "\033[33m"
GRN = "\033[32m"
DIM = "\033[2m"


class TestSparkline:
    """Tests for _sparkline(history) — colored bar sparkline."""

    def test_empty_history_returns_empty_string(self):
        assert _sparkline([]) == ""

    def test_single_green_entry(self):
        result = _sparkline([2])
        assert GRN in result
        assert result.endswith(RST)

    def test_single_red_entry(self):
        result = _sparkline([0])
        assert RED in result

    def test_single_yellow_entry(self):
        result = _sparkline([1])
        assert YEL in result

    def test_truncates_to_last_10_entries(self):
        history = [2] * 15
        result = _sparkline(history)
        # Each entry produces one colored bar char; count the block chars
        bars = "".join(c for c in result if c in "▁▂▃▄▅▆▇█")
        assert len(bars) == 10

    def test_mixed_colors_in_order(self):
        result = _sparkline([0, 1, 2])
        # Red should appear before yellow, yellow before green in the string
        red_pos = result.index(RED)
        yel_pos = result.index(YEL)
        grn_pos = result.index(GRN)
        assert red_pos < yel_pos < grn_pos


class TestContextBar:
    """Tests for _context_bar(pct, width) — usage bar with threshold coloring."""

    def test_zero_percent_all_empty(self):
        result = _context_bar(0, width=20)
        assert "█" not in result
        assert "░" * 20 in result

    def test_100_percent_all_filled(self):
        result = _context_bar(100, width=20)
        assert "█" * 20 in result
        assert "░" not in result

    def test_below_50_uses_green(self):
        result = _context_bar(30, width=20)
        assert GRN in result
        assert RED not in result
        assert YEL not in result

    def test_between_50_and_79_uses_yellow(self):
        result = _context_bar(60, width=20)
        assert YEL in result
        assert RED not in result

    def test_80_and_above_uses_red(self):
        result = _context_bar(80, width=20)
        assert RED in result

    def test_custom_width(self):
        result = _context_bar(50, width=10)
        filled = result.count("█")
        empty = result.count("░")
        assert filled == 5
        assert empty == 5


class TestCampfireHeader:
    """Tests for _campfire_header(frame, counts) — ranch board header."""

    def test_returns_list_of_strings(self):
        counts = {"green": 3, "yellow": 1, "red": 0}
        result = _campfire_header(0, counts)
        assert isinstance(result, list)
        assert all(isinstance(line, str) for line in result)

    def test_contains_ranch_board_text(self):
        counts = {"green": 2, "yellow": 1, "red": 1}
        result = _campfire_header(0, counts)
        joined = "".join(result)
        assert "RANCH BOARD" in joined

    def test_total_head_count(self):
        counts = {"green": 5, "yellow": 2, "red": 1}
        result = _campfire_header(0, counts)
        joined = "".join(result)
        assert "8 head" in joined

    def test_zero_counts(self):
        counts = {"green": 0, "yellow": 0, "red": 0}
        result = _campfire_header(0, counts)
        joined = "".join(result)
        assert "0 head" in joined
