#!/usr/bin/env python3
"""Rail rendering — auto-refreshing status display for Agent Wrangler."""

from __future__ import annotations

import argparse
import select
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from aw.core import (
    ROOT, SELF_PATH, DEFAULT_SESSION,
    ensure_tmux, load_store, session_exists,
    refresh_pane_health, list_hidden_panes,
    play_sound,
    _log_transitions, _append_activity,
    _load_active_times, _save_active_times,
    # Import mutable state dicts (modifications propagate by reference)
    _health_history, _prev_rail_health, _prev_rail_costs,
    _sparkle_countdown,
    _last_active_time, _STALL_THRESHOLD_MIN,
    # Auto-recovery
    _auto_recover_enabled, attempt_recovery, _AUTO_RECOVER_THRESHOLD_MIN,
    # _campfire_frame is a single-element list in core for cross-module mutability
    _campfire_frame,
)


# ── Rail rendering functions ───────────────────────────────────────

def _sparkline(history: list[int]) -> str:
    """Render a colored health sparkline from history values (0=red, 1=yellow, 2=green)."""
    bars = "▁▂▃▄▅▆▇█"
    color_map = {0: "\033[31m", 1: "\033[33m", 2: "\033[32m"}
    parts = []
    for val in history[-10:]:
        val = max(0, min(2, val))
        bar_idx = [0, 4, 7][val]  # red→▁, yellow→▅, green→█
        parts.append(f"{color_map[val]}{bars[bar_idx]}")
    return "".join(parts) + "\033[0m" if parts else ""


def _context_bar(pct: int, width: int = 20) -> str:
    """Render a mini context usage bar: [████░░░░] with threshold coloring."""
    filled = int(pct * width / 100)
    empty = width - filled
    color = "\033[31m" if pct >= 80 else "\033[33m" if pct >= 50 else "\033[32m"
    return f"{color}{'█' * filled}\033[2m{'░' * empty}\033[0m"


def _campfire_header(frame: int, counts: dict[str, int]) -> list[str]:
    """Render ranch board header with cowboy hat."""
    total = sum(counts.values())
    g = counts.get("green", 0)
    y = counts.get("yellow", 0)
    r = counts.get("red", 0)

    hat_color = 172  # amber
    lines = [
        f"\033[38;5;{hat_color}m      .~~~~`\\~~\\\033[0m",
        f"\033[38;5;{hat_color}m     ;       ~~ \\\033[0m",
        f"\033[38;5;{hat_color}m     |           ;\033[0m",
        f"\033[38;5;{hat_color}m ,--------,______|---.\033[0m",
        f"\033[38;5;{hat_color}m/          \\-----`    \\\033[0m",
        f"\033[38;5;{hat_color}m`.__________`-_______-'\033[0m",
        "",
        f" \033[38;5;130m\033[1mRANCH BOARD\033[0m",
        (
            f" {total} head · "
            f"\033[32m{g}●\033[0m "
            f"\033[33m{y}●\033[0m "
            f"\033[31m{r}●\033[0m"
        ),
    ]
    return lines


# --- Barn discovery cache ---
_barn_cache: list[dict[str, str]] = []
_barn_cache_time = [0.0]  # list for cross-module mutability
_BARN_CACHE_TTL = 60.0  # rescan ~/  every 60 seconds


def _discover_barn_repos(active_project_ids: set[str]) -> list[dict[str, str]]:
    """Scan ~/ for git repos not currently active in the grid. Cached."""
    now = time.time()
    if _barn_cache and (now - _barn_cache_time[0]) < _BARN_CACHE_TTL:
        # Return cached list filtered against current active set
        return [r for r in _barn_cache if r["id"] not in active_project_ids]

    home = Path.home()
    repos: list[dict[str, str]] = []
    try:
        for entry in sorted(home.iterdir()):
            if entry.name.startswith(".") or not entry.is_dir():
                continue
            if str(entry) == SELF_PATH:
                continue  # exclude agent-wrangler itself
            if (entry / ".git").is_dir():
                repos.append({
                    "id": entry.name.replace(" ", "-").lower(),
                    "name": entry.name,
                    "path": str(entry),
                })
    except OSError:
        pass

    _barn_cache.clear()
    _barn_cache.extend(repos)
    _barn_cache_time[0] = now
    return [r for r in repos if r["id"] not in active_project_ids]


def _graze_project(path: str) -> None:
    """Open a project in a new Ghostty tab and launch Claude in it."""
    escaped_path = path.replace("\\", "\\\\").replace('"', '\\"')
    script = (
        'tell application "Ghostty"\n'
        "    set cfg to new surface configuration\n"
        f'    set initial working directory of cfg to "{escaped_path}"\n'
        "    set t to new tab in front window with configuration cfg\n"
        '    input text "claude --dangerously-skip-permissions" to t\n'
        '    send key "enter" to t\n'
        "end tell"
    )
    try:
        subprocess.Popen(
            ["osascript", "-e", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        pass


def run_rail(args: argparse.Namespace) -> int:
    """Auto-refreshing ranch board status rail for a narrow tmux split."""
    import tty
    import termios

    ensure_tmux()
    store = load_store()
    session = args.session or store.get("default_session") or DEFAULT_SESSION
    interval = max(1, int(args.interval))

    # Set stdin to raw mode for non-blocking keypress detection (barn graze)
    old_tty_settings = None
    if sys.stdin.isatty():
        try:
            old_tty_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            sys.stdout.write("\033[?1000h\033[?1006h")  # enable SGR mouse tracking
            sys.stdout.flush()
        except termios.error:
            pass

    def _restore_tty() -> None:
        try:
            sys.stdout.write("\033[?1000l\033[?1006l")  # disable mouse tracking
            sys.stdout.flush()
        except OSError:
            pass
        if old_tty_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty_settings)
            except termios.error:
                pass

    try:
        return _rail_loop(args, session, interval)
    finally:
        _restore_tty()


def _rail_loop(args: argparse.Namespace, session: str, interval: int) -> int:
    """Inner rail loop (separated so tty restore happens in finally)."""
    _last_active_time.update(_load_active_times())

    _append_activity([{
        "project": "_system",
        "event": "rail_started",
    }])

    while True:
        if not session_exists(session):
            print(f"Session '{session}' not found.")
            time.sleep(interval)
            continue

        rows = refresh_pane_health(
            session=session,
            capture_lines=40,
            wait_attention_min=5,
        )
        _log_transitions(rows)

        lines: list[str] = []
        lines.append("\033[2J\033[H")  # clear screen

        # --- Campfire header with herd count ---
        counts: dict[str, int] = {"green": 0, "yellow": 0, "red": 0}
        waiting = 0
        total_cost = 0.0
        total_tokens_k = 0
        claude_count = 0

        for row in rows:
            health = str(row.get("health") or "green")
            counts[health] = counts.get(health, 0) + 1
            if str(row.get("status") or "idle") == "waiting":
                waiting += 1

        _campfire_frame[0] += 1
        lines.extend(_campfire_header(_campfire_frame[0], counts))
        lines.append("\033[2m" + "─" * 34 + "\033[0m")

        # --- Ranch Status summary box ---
        total = sum(counts.values())
        g, y, r = counts.get("green", 0), counts.get("yellow", 0), counts.get("red", 0)
        ranch_terms = []
        if g:
            ranch_terms.append(f"\033[32m{g} grazing\033[0m")
        if y:
            ranch_terms.append(f"\033[33m{y} at fence\033[0m")
        if r:
            ranch_terms.append(f"\033[31m{r} down\033[0m")
        lines.append(f" {' · '.join(ranch_terms)}")
        lines.append("\033[2m" + "─" * 34 + "\033[0m")

        # --- Per-pane entries with sparklines ---
        health_val = {"green": 2, "yellow": 1, "red": 0}
        for row in rows:
            health = str(row.get("health") or "green")
            project = str(row.get("project_id") or row.get("pane_title") or "?")
            agent = str(row.get("agent") or row.get("ai_tool") or "")
            status = str(row.get("status") or "idle")
            if len(project) > 16:
                project = project[:15] + "~"

            # Update health history
            h_val = health_val.get(health, 2)
            hist = _health_history.setdefault(project, [])
            hist.append(h_val)
            if len(hist) > 10:
                _health_history[project] = hist[-10:]

            # Check for state change sparkle
            prev = _prev_rail_health.get(project)
            if prev and prev != health:
                _sparkle_countdown[project] = 2
                # Play sound on health transitions
                if health == "red":
                    play_sound("Basso", 0.4, key=f"red-{project}")
                elif health == "green" and prev == "red":
                    play_sound("Bottle", 0.3, key=f"green-{project}")
            _prev_rail_health[project] = health

            sparkle = ""
            if _sparkle_countdown.get(project, 0) > 0:
                sparkle = " \033[38;5;220m✦\033[0m"
                _sparkle_countdown[project] -= 1

            dot_color = {"green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m"}.get(health, "\033[0m")
            spark = _sparkline(_health_history.get(project, []))

            line = f" {dot_color}●\033[0m{sparkle} {project:<16}"
            if agent and agent != "-":
                line += f" \033[2m{agent:<8}\033[0m"

            # Stall detection: track when last green, show idle time if yellow too long
            if health == "green":
                _last_active_time[project] = time.time()
            elif health == "yellow" and project in _last_active_time:
                idle_min = (time.time() - _last_active_time[project]) / 60.0
                if idle_min >= _STALL_THRESHOLD_MIN:
                    line += f" \033[31m{int(idle_min)}m idle\033[0m"
                elif idle_min >= 2:
                    line += f" \033[2m{int(idle_min)}m\033[0m"

            # Auto-recovery: restart stalled agents
            if (health == "yellow" and project in _last_active_time
                    and agent and agent != "-"
                    and _auto_recover_enabled()):
                idle_min_check = (time.time() - _last_active_time[project]) / 60.0
                if idle_min_check >= _AUTO_RECOVER_THRESHOLD_MIN:
                    pane_id = str(row.get("pane_id") or "")
                    if pane_id and attempt_recovery(session, project, pane_id):
                        line += f" \033[33m⟳\033[0m"
                        _last_active_time[project] = time.time()

            lines.append(line)

            # Sparkline on its own line (compact, below the pane entry)
            if len(_health_history.get(project, [])) > 1:
                lines.append(f"   {spark}")

            # Claude Code stats with context bar
            cc = row.get("cc_stats")
            if cc:
                claude_count += 1
                ctx = cc.get("context_pct")
                cost = cc.get("cost")

                if ctx is not None:
                    bar = _context_bar(ctx)
                    lines.append(f"   ctx {bar} {ctx}%")

                if cost is not None:
                    # Flash gold when cost increases
                    prev_cost = _prev_rail_costs.get(project, 0.0)
                    cost_color = "\033[38;5;214m" if cost > prev_cost else "\033[2m"
                    _prev_rail_costs[project] = cost
                    lines.append(f"   {cost_color}${cost:.2f}\033[0m")
                    total_cost += cost
                total_tokens_k += cc.get("tokens_k", 0)

        # --- Totals ---
        lines.append("\033[2m" + "─" * 34 + "\033[0m")
        if claude_count > 0:
            lines.append(
                f" \033[36m{claude_count} claude\033[0m  "
                f"\033[2m{total_tokens_k}k tok  ${total_cost:.2f}\033[0m"
            )

        _save_active_times(_last_active_time)

        # --- Hidden panes ---
        hidden = list_hidden_panes(session)
        if hidden:
            lines.append(f" \033[2m{len(hidden)} hidden\033[0m")
            for h in hidden:
                hp = str(h.get("project_id") or "?")
                if len(hp) > 16:
                    hp = hp[:15] + "~"
                ha = str(h.get("agent") or "")
                hs = str(h.get("status") or "")
                agent_label = f"  {ha}" if ha and ha != "-" else ""
                lines.append(f" \033[2m○ {hp:<16}{agent_label}  {hs}\033[0m")

        # --- Last roundup timestamp ---
        now_str = datetime.now().strftime("%H:%M:%S")
        lines.append(f"\n\033[2m  last roundup: {now_str}\033[0m")

        # --- Barn: discovered repos not in the grid ---
        active_ids = {
            str(row.get("project_id") or row.get("pane_title") or "")
            for row in rows
        }
        # Also include hidden pane IDs
        for h in (hidden if hidden else []):
            hid = str(h.get("project_id") or "")
            if hid:
                active_ids.add(hid)

        barn_repos = _discover_barn_repos(active_ids)
        barn_item_y: dict[int, int] = {}  # terminal Y coord → barn index (0-based)
        if barn_repos:
            lines.append("\033[2m" + "─" * 34 + "\033[0m")
            lines.append(f" \033[38;5;130mIN THE BARN ({len(barn_repos)})\033[0m")
            # Show up to 9 numbered entries
            for idx, repo in enumerate(barn_repos[:9], 1):
                name = repo["name"]
                if len(name) > 22:
                    name = name[:21] + "~"
                barn_item_y[len(lines) + 1] = idx - 1  # Y = lines index + 1
                lines.append(f" \033[2m[{idx}] {name}\033[0m")
            if len(barn_repos) > 9:
                lines.append(f" \033[2m    +{len(barn_repos) - 9} more\033[0m")
            lines.append(f" \033[2m click or press 1-{min(len(barn_repos), 9)} to graze\033[0m")

        print("\n".join(lines), flush=True)

        # --- Non-blocking keypress + mouse click detection for barn graze ---
        try:
            deadline = time.time() + interval
            while time.time() < deadline:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                ready, _, _ = select.select([sys.stdin], [], [], min(remaining, 0.2))
                if ready:
                    key = sys.stdin.read(1)
                    if key == '\033':
                        # Read rest of escape sequence (mouse click or other)
                        buf = '\033'
                        while len(buf) < 32:
                            r2, _, _ = select.select([sys.stdin], [], [], 0.05)
                            if not r2:
                                break
                            ch = sys.stdin.read(1)
                            buf += ch
                            if ch in ('M', 'm') and '[<' in buf:
                                break  # SGR mouse sequence complete
                        # Parse SGR mouse press: \033[<button;x;yM
                        if len(buf) > 5 and buf[1:3] == '[<' and buf[-1] == 'M':
                            parts = buf[3:-1].split(';')
                            if len(parts) == 3:
                                try:
                                    btn, _mx, my = int(parts[0]), int(parts[1]), int(parts[2])
                                    if btn == 0 and my in barn_item_y:
                                        chosen = barn_repos[barn_item_y[my]]
                                        _graze_project(chosen["path"])
                                        _barn_cache_time[0] = 0.0
                                        break
                                except (ValueError, IndexError):
                                    pass
                    elif key.isdigit() and 1 <= int(key) <= min(len(barn_repos) if barn_repos else 0, 9):
                        chosen = barn_repos[int(key) - 1]
                        _graze_project(chosen["path"])
                        _barn_cache_time[0] = 0.0
                        break
        except KeyboardInterrupt:
            break
    _append_activity([{
        "project": "_system",
        "event": "rail_stopped",
    }])
    return 0
