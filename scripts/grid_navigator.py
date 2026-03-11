#!/usr/bin/env python3
"""Grid Navigator - tmux pane browser with health coloring and direct actions."""
from __future__ import annotations

import argparse
import curses
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from tmux_teams import (
    ensure_tmux,
    hide_pane,
    list_hidden_panes,
    list_panes,
    load_store,
    pane_send,
    project_map,
    refresh_pane_health,
    session_exists,
    show_pane,
    tmux,
)
from session_stats import (
    cheap_stats_for_pane,
    format_stats_summary,
    load_stats,
    poll_usage_for_pane,
    save_stats,
    should_poll,
    update_session_stats,
)

# ---------------------------------------------------------------------------
# Color pair IDs
# ---------------------------------------------------------------------------
PAIR_TITLE = 1
PAIR_FOOTER = 2
PAIR_GREEN = 3
PAIR_YELLOW = 4
PAIR_RED = 5
PAIR_DEFAULT = 6


def _init_colors() -> None:
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(PAIR_TITLE, curses.COLOR_CYAN, -1)
    curses.init_pair(PAIR_FOOTER, curses.COLOR_MAGENTA, -1)
    curses.init_pair(PAIR_GREEN, curses.COLOR_GREEN, -1)
    curses.init_pair(PAIR_YELLOW, curses.COLOR_YELLOW, -1)
    curses.init_pair(PAIR_RED, curses.COLOR_RED, -1)
    curses.init_pair(PAIR_DEFAULT, -1, -1)
    curses.init_pair(PAIR_HIDDEN, curses.COLOR_WHITE, -1)


def _health_color(health: str) -> int:
    """Return the curses color pair for a health level string."""
    h = health.lower()
    if h == "green":
        return curses.color_pair(PAIR_GREEN)
    if h == "yellow":
        return curses.color_pair(PAIR_YELLOW)
    if h == "red":
        return curses.color_pair(PAIR_RED)
    return curses.color_pair(PAIR_DEFAULT)


def _fetch_rows(session: str, stats_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Refresh pane health, collect stats, return row dicts including hidden panes."""
    try:
        rows = refresh_pane_health(
            session=session,
            capture_lines=40,
            wait_attention_min=1,
            apply_colors=False,
        )
    except Exception:
        rows = []

    # Mark visible rows
    for row in rows:
        row["hidden"] = False

    # Cheap stats for every pane
    for row in rows:
        pid = str(row.get("pane_id") or "")
        proj = str(row.get("project_id") or "")
        if pid and proj:
            cheap = cheap_stats_for_pane(pid, tmux)
            update_session_stats(pid, proj, cheap, None, stats_data)

    # Periodic /usage poll: one idle claude session per cycle
    if should_poll(stats_data):
        for row in rows:
            is_waiting = str(row.get("status") or "") == "waiting"
            is_claude = str(row.get("ai_tool") or row.get("agent") or "") == "claude"
            if is_waiting and is_claude:
                pid = str(row.get("pane_id") or "")
                proj = str(row.get("project_id") or "")
                if pid and proj:
                    usage = poll_usage_for_pane(pid, tmux)
                    if usage:
                        update_session_stats(pid, proj, None, usage, stats_data)
                break  # Only one per cycle

    # Append hidden panes at the bottom
    for h in list_hidden_panes(session):
        rows.append({
            "pane_id": h["pane_id"],
            "index": -1,
            "project_id": h["project_id"],
            "title": h["project_id"],
            "tty": "",
            "agent": h["agent"],
            "status": h["status"],
            "wait": None,
            "health": "hidden",
            "needs_attention": False,
            "reason": "hidden",
            "hidden": True,
            "hidden_window": h["window_name"],
        })

    save_stats(stats_data)
    return rows


PAIR_HIDDEN = 7


def _format_row(row: dict[str, Any], selected: bool, width: int) -> tuple[str, int]:
    """Return (line_text, curses_attr) for a single pane row."""
    is_hidden = row.get("hidden", False)
    prefix = " > " if selected else "   "
    project = str(row.get("project_id") or row.get("pane_title") or "-")[:20].ljust(20)
    agent = str(row.get("agent") or row.get("ai_tool") or "-")[:8].ljust(8)
    status = str(row.get("status") or "-")[:10].ljust(10)
    tag = "[hidden] " if is_hidden else ""
    pane_id = str(row.get("pane_id") or "")
    line = f"{prefix}{tag}{project} {agent} {status} {pane_id}"
    if len(line) > width:
        line = line[:width]

    if is_hidden:
        color = curses.color_pair(PAIR_HIDDEN) | curses.A_DIM
    else:
        color = _health_color(str(row.get("health") or ""))
    attr = color | curses.A_BOLD if selected else color
    return line, attr


def grid_main(stdscr: Any, session: str, interval: int, manager_window: str) -> None:
    """Main curses loop for the grid navigator."""
    _init_colors()
    curses.curs_set(0)
    stdscr.nodelay(False)
    stdscr.timeout(interval * 1000)

    stats_data = load_stats()
    rows: list[dict[str, Any]] = _fetch_rows(session, stats_data)
    selected = 0
    last_refresh = time.monotonic()

    while True:
        stdscr.erase()
        max_y, max_x = stdscr.getmaxyx()

        # --- Title bar (row 0) ---
        title = f" GRID NAVIGATOR  |  {session}  |  {len(rows)} panes "
        if len(title) > max_x:
            title = title[:max_x]
        try:
            stdscr.addnstr(0, 0, title, max_x, curses.color_pair(PAIR_TITLE) | curses.A_BOLD)
        except curses.error:
            pass

        # --- Footer (last row) ---
        footer = " j/k:nav  Enter:jump  h:hide/show  c:claude  x:codex  s:send  K:ctrl-c  r:refresh  q:quit "
        if len(footer) > max_x:
            footer = footer[:max_x]
        try:
            stdscr.addnstr(max_y - 1, 0, footer, max_x, curses.color_pair(PAIR_FOOTER) | curses.A_BOLD)
        except curses.error:
            pass

        # --- Stats header (row 1) ---
        stats_line = " " + format_stats_summary(rows, stats_data)
        if len(stats_line) > max_x:
            stats_line = stats_line[:max_x]
        try:
            stdscr.addnstr(1, 0, stats_line, max_x, curses.color_pair(PAIR_TITLE))
        except curses.error:
            pass

        # --- Pane list (rows 3 .. max_y-2) ---
        list_start = 3
        list_height = max(1, max_y - 4)

        # Scrolling: ensure selected is visible
        if len(rows) > 0:
            selected = max(0, min(selected, len(rows) - 1))
        scroll_offset = 0
        if selected >= list_height:
            scroll_offset = selected - list_height + 1

        for i in range(list_height):
            row_idx = scroll_offset + i
            if row_idx >= len(rows):
                break
            screen_row = list_start + i
            if screen_row >= max_y - 1:
                break
            line, attr = _format_row(rows[row_idx], row_idx == selected, max_x)
            try:
                stdscr.addnstr(screen_row, 0, line, max_x, attr)
            except curses.error:
                pass

        stdscr.refresh()

        # --- Handle input ---
        try:
            key = stdscr.getch()
        except curses.error:
            key = -1

        # Auto-refresh on timeout (-1 means no key pressed within interval)
        if key == -1:
            now = time.monotonic()
            if now - last_refresh >= interval:
                rows = _fetch_rows(session, stats_data)
                last_refresh = now
            continue

        # Quit
        if key in (ord("q"), 27):  # q or Escape
            break

        # Navigation
        if key in (curses.KEY_UP, ord("k")):
            if selected > 0:
                selected -= 1
            continue
        if key in (curses.KEY_DOWN, ord("j")):
            if selected < len(rows) - 1:
                selected += 1
            continue

        # Jump to pane (Enter or f)
        if key in (10, 13, ord("f")):
            if rows and 0 <= selected < len(rows):
                target = rows[selected].get("pane_id")
                if target:
                    tmux(["select-window", "-t", f"{session}:teams"], timeout=5)
                    tmux(["select-pane", "-t", target], timeout=5)
                    break
            continue

        # Launch claude in selected pane
        if key == ord("c"):
            if rows and 0 <= selected < len(rows):
                target = rows[selected].get("pane_id")
                if target:
                    pane_send(target, "claude", enter=True)
            continue

        # Launch codex in selected pane
        if key == ord("x"):
            if rows and 0 <= selected < len(rows):
                target = rows[selected].get("pane_id")
                if target:
                    pane_send(target, "codex", enter=True)
            continue

        # Send command to selected pane
        if key == ord("s"):
            if rows and 0 <= selected < len(rows):
                target = rows[selected].get("pane_id")
                if target:
                    try:
                        curses.echo()
                        curses.curs_set(1)
                        stdscr.addnstr(max_y - 1, 0, "cmd> ", max_x, curses.A_BOLD)
                        stdscr.clrtoeol()
                        stdscr.refresh()
                        raw = stdscr.getstr(max_y - 1, 5, max_x - 6)
                        curses.noecho()
                        curses.curs_set(0)
                        if raw:
                            cmd = raw.decode("utf-8", errors="replace").strip()
                            if cmd:
                                pane_send(target, cmd, enter=True)
                    except curses.error:
                        curses.noecho()
                        curses.curs_set(0)
            continue

        # Send Ctrl-C to selected pane (K = uppercase, since lowercase k is nav)
        if key == ord("K"):
            if rows and 0 <= selected < len(rows):
                target = rows[selected].get("pane_id")
                if target:
                    tmux(["send-keys", "-t", target, "C-c", ""], timeout=5)
            continue

        # Hide/show toggle (h)
        if key == ord("h"):
            if rows and 0 <= selected < len(rows):
                row = rows[selected]
                if row.get("hidden"):
                    # Show: bring hidden pane back to grid
                    wname = row.get("hidden_window")
                    if wname:
                        try:
                            show_pane(session, wname)
                        except ValueError:
                            pass
                else:
                    # Hide: move pane to background
                    target = row.get("pane_id")
                    project = row.get("project_id") or row.get("title") or target
                    if target:
                        try:
                            hide_pane(session, target, project)
                        except ValueError:
                            pass
                rows = _fetch_rows(session, stats_data)
                last_refresh = time.monotonic()
                if selected >= len(rows):
                    selected = max(0, len(rows) - 1)
            continue

        # Manual refresh
        if key == ord("r"):
            rows = _fetch_rows(session, stats_data)
            last_refresh = time.monotonic()
            continue

        # Switch to manager window (tab or 1)
        if key in (9, ord("1")):  # 9 = tab
            tmux(["select-window", "-t", f"{session}:{manager_window}"], timeout=5)
            break

        # Fallback: also handle Ctrl-C to quit gracefully
        if key == 3:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid Navigator")
    parser.add_argument("--session", default=None)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--manager-window", default="manager")
    args = parser.parse_args()

    ensure_tmux()
    store = load_store()
    session = args.session or store.get("default_session") or "amir-grid"
    if not session_exists(session):
        raise SystemExit(f"Session '{session}' not found")

    curses.wrapper(grid_main, session, args.interval, args.manager_window)


if __name__ == "__main__":
    main()
