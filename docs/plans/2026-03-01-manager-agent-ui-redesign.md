# Manager Agent UI Redesign

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the curses 2-page UI with a tmux-native layout: a Claude Code manager agent (Page 1) and a simplified grid navigator (Page 2), connected by tab switching, with a cowboy-themed ASCII welcome banner.

**Architecture:** The manager is a real Claude Code session running in a tmux window alongside a small auto-refreshing status rail. The grid navigator is a stripped-down curses app (panels page only) in a second tmux window. Navigation between them uses tmux window switching. The `agent-wrangler start` command wires it all up and shows a branded welcome screen.

**Tech Stack:** Python 3.10+ (curses, subprocess, argparse), Bash, tmux

---

### Task 1: ASCII Welcome Banner

Create a cowboy-themed welcome banner that displays when `agent-wrangler start` runs.

**Files:**
- Create: `scripts/welcome_banner.sh`
- Modify: `scripts/agent-wrangler` (add banner call to `start` case)

**Step 1: Create the welcome banner script**

Create `scripts/welcome_banner.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

CYAN='\033[0;36m'
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'

cat <<'ART'

        ____
       /    \
      | ^  ^ |
      |  __  |      ___                    __
       \____/      /   | ____ ____  ____  / /_
        |  |      / /| |/ __ `/ _ \/ __ \/ __/
   /|   |  |   | / ___ / /_/ /  __/ / / / /_
  / |   |  |   |/_/  |_\__, /\___/_/ /_/\__/
 /  |___|  |___|     /____/
|   |          |  _       __                        __
|   |  |    |  | | |     / /________ _____  ____ _/ /__  _____
|   |  |    |  | | | /| / / ___/ __ `/ __ \/ __ `/ / _ \/ ___/
 \  |  |    |  |/  |/ |/ / /  / /_/ / / / / /_/ / /  __/ /
  \_|__|____|__|   |__/|_/_/   \__,_/_/ /_/\__, /_/\___/_/
                                           /____/

ART

printf "${DIM}  Steering agents. Wrangling terminals. Shipping code.${RESET}\n\n"
```

**Step 2: Make it executable and call from start**

```bash
chmod +x scripts/welcome_banner.sh
```

In `scripts/agent-wrangler`, add the banner call inside the `start)` case, before the `set --` line:

```bash
  start)
    shift
    bash "$SCRIPT_DIR/welcome_banner.sh"
    MAX_PANES="$(resolve_max_panes)"
    ...
```

**Step 3: Verify manually**

Run: `./scripts/welcome_banner.sh`
Expected: Cowboy figure + "Agent Wrangler" in ASCII art with dim tagline.

**Step 4: Commit**

```bash
git add scripts/welcome_banner.sh scripts/agent-wrangler
git commit -m "feat: add cowboy-themed ASCII welcome banner"
```

---

### Task 2: Status Rail Command

Create `agent-wrangler rail` - a compact, auto-refreshing status view designed to run in a narrow tmux split pane alongside the Claude Code manager.

**Files:**
- Modify: `scripts/tmux_teams.py` (add `run_rail` function + `rail` subcommand)

**Step 1: Write the `run_rail` function**

Add to `tmux_teams.py` after the `run_status` function (around line 1282). This is a compact, looping display for a ~30-column tmux pane:

```python
def run_rail(args: argparse.Namespace) -> int:
    """Auto-refreshing compact status rail for a narrow tmux split."""
    import time as _time

    ensure_tmux()
    store = load_store()
    session = args.session or store.get("default_session") or DEFAULT_SESSION
    interval = max(1, int(args.interval))

    while True:
        if not session_exists(session):
            print(f"Session '{session}' not found.")
            _time.sleep(interval)
            continue

        proj_map = project_map()
        panes = backfill_pane_project_ids(session, list_panes(session), proj_map)
        rows = refresh_pane_health(
            session=session,
            panes=panes,
            capture_lines=40,
            wait_attention_min=1,
            apply_colors=False,
        )

        lines: list[str] = []
        lines.append("\033[2J\033[H")  # clear screen
        lines.append("\033[1;36m STATUS RAIL\033[0m")
        lines.append("\033[2m" + "─" * 30 + "\033[0m")

        counts = {"green": 0, "yellow": 0, "red": 0, "total": 0}
        waiting = 0
        for row in rows:
            counts["total"] += 1
            health = str(row.get("health") or "green")
            counts[health] = counts.get(health, 0) + 1
            status = str(row.get("status") or "idle")
            if status == "waiting":
                waiting += 1

            dot_color = {"green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m"}.get(health, "\033[0m")
            agent = str(row.get("agent") or row.get("ai_tool") or "")
            project = str(row.get("project_id") or row.get("pane_title") or "?")
            if len(project) > 16:
                project = project[:15] + "~"
            agent_label = f"  {agent}" if agent else ""
            status_label = status if status != "idle" else ""

            line = f" {dot_color}●\033[0m {project:<16}{agent_label}"
            if status_label:
                line += f"  \033[2m{status_label}\033[0m"
            lines.append(line)

        lines.append("\033[2m" + "─" * 30 + "\033[0m")
        lines.append(f" {counts['total']} panes  {waiting} waiting")
        lines.append(
            f" \033[32m{counts['green']}g\033[0m "
            f"\033[33m{counts['yellow']}y\033[0m "
            f"\033[31m{counts['red']}r\033[0m"
        )

        print("\n".join(lines), flush=True)

        try:
            _time.sleep(interval)
        except KeyboardInterrupt:
            break
    return 0
```

**Step 2: Register the `rail` subcommand in the argparse section**

In the argparse setup area (around line 2780+), add:

```python
rail = subparsers.add_parser("rail", help="Compact auto-refreshing status rail for narrow split panes")
rail.add_argument("--session", default=None)
rail.add_argument("--interval", type=int, default=5)
rail.set_defaults(handler=run_rail)
```

**Step 3: Route `rail` through agent-wrangler**

In `scripts/agent-wrangler`, add `rail` to the list of commands routed to `teams`:

```bash
  up|bootstrap|import|attach|status|paint|watch|manager|nav|send|stop|restart|agent|focus|kill|shell|layout|capture|projects|persistence|profile|hooks|doctor|drift|rail)
```

**Step 4: Verify manually**

Run: `./scripts/agent-wrangler rail --interval 3`
Expected: Compact colored status list that refreshes every 3 seconds. Ctrl-C to stop.

**Step 5: Commit**

```bash
git add scripts/tmux_teams.py scripts/agent-wrangler
git commit -m "feat: add compact status rail command for manager sidebar"
```

---

### Task 3: Manager Window (Claude Code + Status Rail)

Replace the current manager window (which runs the curses UI or `watch`) with a split layout: Claude Code on the left (~75%) and the status rail on the right (~25%).

**Files:**
- Modify: `scripts/tmux_teams.py` (rewrite `run_manager` function)

**Step 1: Rewrite `run_manager`**

Replace the `run_manager` function (line 1396-1444 of `tmux_teams.py`). The new version creates a tmux window with two panes:

```python
def run_manager(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    session = args.session or store.get("default_session") or DEFAULT_SESSION
    if not session_exists(session):
        raise ValueError(f"Session '{session}' does not exist")

    window = args.window
    if manager_window_exists(session, window):
        if args.replace:
            code, _, err = tmux(["kill-window", "-t", f"{session}:{window}"], timeout=5)
            if code != 0:
                raise ValueError(err.strip() or f"failed to replace manager window '{window}'")
        else:
            print(f"Manager window '{window}' already exists.")
            if args.focus:
                tmux(["select-window", "-t", f"{session}:{window}"], timeout=5)
            if args.attach:
                return attach_session(session)
            return 0

    # Create manager window with Claude Code
    wrangler_root = str(ROOT)
    claude_cmd = "claude"
    shell_tail = "; exec zsh"
    shell_command = "zsh -lc " + shlex.quote(claude_cmd + shell_tail)
    code, _, err = tmux(
        ["new-window", "-d", "-t", session, "-n", window, "-c", wrangler_root, shell_command],
        timeout=8,
    )
    if code != 0:
        raise ValueError(err.strip() or f"failed to create manager window '{window}'")

    # Split right pane for status rail (~25% width)
    rail_script = ROOT / "scripts" / "tmux_teams.py"
    rail_cmd = (
        f"python3 {shlex.quote(str(rail_script))} teams rail "
        f"--session {shlex.quote(session)} --interval {max(1, int(args.interval))}"
    )
    rail_shell = "zsh -lc " + shlex.quote(rail_cmd + shell_tail)
    code, _, err = tmux(
        ["split-window", "-h", "-t", f"{session}:{window}", "-l", "25%",
         "-c", wrangler_root, rail_shell],
        timeout=8,
    )
    if code != 0:
        print(f"Warning: failed to create status rail split: {err.strip()}")

    # Focus the left pane (Claude Code) within the manager window
    tmux(["select-pane", "-t", f"{session}:{window}.0"], timeout=5)

    if not manager_window_exists(session, window):
        raise ValueError(f"manager window '{window}' did not persist")
    print(f"Manager window started: {session}:{window} (claude + rail)")

    if args.focus:
        tmux(["select-window", "-t", f"{session}:{window}"], timeout=5)
    if args.attach:
        return attach_session(session)
    return 0
```

**Step 2: Update the `--manager-ui` / `--manager-watch` args**

These flags are now obsolete. In `run_up` (line 1235-1249), simplify the `run_manager` call:

```python
    if args.manager:
        run_manager(
            argparse.Namespace(
                session=session,
                window=args.manager_window,
                interval=args.manager_interval,
                replace=args.manager_replace,
                focus=True,
                attach=False,
            )
        )
```

Remove or deprecate the `--manager-ui` and `--manager-watch` argparse definitions (lines ~2686-2693). They are no longer needed since the manager always uses Claude Code + rail.

**Step 3: Verify manually**

Run: `./scripts/agent-wrangler start`
Expected: tmux session with project panes + a "manager" window that has Claude Code on the left and the status rail on the right. The manager window is focused.

**Step 4: Commit**

```bash
git add scripts/tmux_teams.py
git commit -m "feat: replace manager window with Claude Code + status rail"
```

---

### Task 4: Grid Navigator Window

Create a dedicated "grid" tmux window running a simplified curses navigator (panels page only, no Gastown/wrangler page). This is Page 2.

**Files:**
- Create: `scripts/grid_navigator.py` - standalone curses app, panels-only
- Modify: `scripts/tmux_teams.py` (add grid window creation to `run_up`)
- Modify: `scripts/agent-wrangler` (add `grid` route)

**Step 1: Create `grid_navigator.py`**

This is a standalone curses app that shows the pane list with health colors and supports:
- `up/down` or `j/k`: select pane
- `Enter` or `f`: jump to selected pane (tmux select-pane)
- `c`: launch claude in selected pane
- `x`: launch codex in selected pane
- `s`: send command to selected pane
- `k`: send Ctrl-C to selected pane
- `r`: refresh
- `Escape` or `q`: quit (or go back to manager)
- `tab` or `1`: switch to manager window

Create `scripts/grid_navigator.py`. The core structure:

```python
#!/usr/bin/env python3
"""Grid Navigator - tmux pane browser with health coloring and direct actions."""
from __future__ import annotations

import argparse
import curses
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# Import pane health functions from tmux_teams
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from tmux_teams import (
    backfill_pane_project_ids,
    ensure_tmux,
    list_panes,
    load_store,
    pane_ctrl_c,
    pane_send,
    project_map,
    refresh_pane_health,
    session_exists,
    tmux,
)


def grid_main(stdscr: Any, session: str, interval: int, manager_window: str) -> int:
    """Main curses loop for the grid navigator."""
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    stdscr.nodelay(True)
    stdscr.timeout(200)

    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)    # title
        curses.init_pair(2, curses.COLOR_GREEN, -1)    # healthy
        curses.init_pair(3, curses.COLOR_YELLOW, -1)   # warning
        curses.init_pair(4, curses.COLOR_RED, -1)      # critical
        curses.init_pair(5, curses.COLOR_MAGENTA, -1)  # footer

    selected = 0
    last_tick = 0.0
    rows: list[dict[str, Any]] = []

    def refresh() -> list[dict[str, Any]]:
        nonlocal last_tick
        proj = project_map()
        panes = backfill_pane_project_ids(session, list_panes(session), proj)
        result = refresh_pane_health(
            session=session,
            panes=panes,
            capture_lines=40,
            wait_attention_min=1,
            apply_colors=False,
        )
        last_tick = time.time()
        return result

    rows = refresh()

    while True:
        now = time.time()
        if now - last_tick >= interval:
            rows = refresh()

        h, w = stdscr.getmaxyx()
        stdscr.erase()

        # Title bar
        title = f" GRID NAVIGATOR  |  {session}  |  {len(rows)} panes "
        try:
            stdscr.addstr(0, 0, title[:w-1], curses.color_pair(1) | curses.A_BOLD)
        except curses.error:
            pass

        # Pane list
        selected = max(0, min(selected, len(rows) - 1))
        list_top = 2
        list_h = h - 4
        # Scroll offset
        scroll = max(0, selected - list_h + 3)

        for i, row in enumerate(rows):
            if i < scroll:
                continue
            y = list_top + (i - scroll)
            if y >= h - 2:
                break

            health = str(row.get("health") or "green")
            color = {"green": 2, "yellow": 3, "red": 4}.get(health, 2)
            project = str(row.get("project_id") or row.get("pane_title") or "?")
            agent = str(row.get("agent") or row.get("ai_tool") or "")
            status = str(row.get("status") or "")
            pane_id = str(row.get("pane_id") or "")

            prefix = " > " if i == selected else "   "
            attr = curses.A_BOLD if i == selected else 0
            line = f"{prefix}{project:<20} {agent:<8} {status:<10} {pane_id}"

            try:
                stdscr.addstr(y, 0, line[:w-1], curses.color_pair(color) | attr)
            except curses.error:
                pass

        # Footer
        footer = " [j/k] select  [enter] jump  [c] claude  [x] codex  [s] send  [k] stop  [tab/1] manager  [q] quit "
        try:
            stdscr.addstr(h - 1, 0, footer[:w-1], curses.color_pair(5) | curses.A_BOLD)
        except curses.error:
            pass

        stdscr.refresh()

        try:
            ch = stdscr.getch()
        except curses.error:
            ch = -1

        if ch < 0:
            continue

        if ch in (ord("q"), ord("Q"), 27):  # q or Escape
            break

        if ch in (9, ord("1")):  # tab or 1 -> switch to manager
            tmux(["select-window", "-t", f"{session}:{manager_window}"], timeout=5)
            continue

        if ch in (curses.KEY_UP, ord("k"), ord("K")):
            selected = max(0, selected - 1)
            continue
        if ch in (curses.KEY_DOWN, ord("j"), ord("J")):
            selected = min(len(rows) - 1, selected + 1)
            continue

        if not rows:
            continue

        target = rows[selected]
        target_pane_id = str(target.get("pane_id") or "")

        if ch in (10, 13, ord("f"), ord("F")):  # Enter or f -> jump
            if target_pane_id:
                tmux(["select-window", "-t", f"{session}:teams"], timeout=5)
                tmux(["select-pane", "-t", target_pane_id], timeout=5)
            continue

        if ch in (ord("c"), ord("C")):
            if target_pane_id:
                pane_send(target_pane_id, "claude", enter=True)
            continue

        if ch in (ord("x"), ord("X")):
            if target_pane_id:
                pane_send(target_pane_id, "codex", enter=True)
            continue

        if ch in (ord("s"), ord("S")):
            # Simple prompt for command
            curses.echo()
            try:
                stdscr.addstr(h - 1, 0, " send> " + " " * (w - 8))
                stdscr.move(h - 1, 7)
                stdscr.refresh()
                cmd_bytes = stdscr.getstr(h - 1, 7, w - 10)
                cmd = cmd_bytes.decode("utf-8", errors="replace").strip()
                if cmd and target_pane_id:
                    pane_send(target_pane_id, cmd, enter=True)
            except curses.error:
                pass
            finally:
                curses.noecho()
            continue

        if ch in (ord("k"), ord("K")):
            # Already handled above for navigation, this is for Ctrl-C
            pass

        if ch in (ord("r"), ord("R")):
            rows = refresh()
            continue

    return 0


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
```

**Step 2: Add grid window creation to `run_up`**

In `tmux_teams.py`, after the manager window creation block in `run_up` (line ~1249), add grid window creation:

```python
    if args.manager:
        # ... existing manager call ...

        # Create grid navigator window
        grid_script = ROOT / "scripts" / "grid_navigator.py"
        grid_cmd = (
            f"python3 {shlex.quote(str(grid_script))} "
            f"--session {shlex.quote(session)} --interval 5 "
            f"--manager-window {shlex.quote(args.manager_window)}"
        )
        grid_shell_tail = "; exec zsh"
        grid_shell = "zsh -lc " + shlex.quote(grid_cmd + grid_shell_tail)
        grid_window = "grid"
        if not manager_window_exists(session, grid_window):
            tmux(
                ["new-window", "-d", "-t", session, "-n", grid_window, "-c", str(ROOT), grid_shell],
                timeout=8,
            )
```

**Step 3: Route `grid` in `agent-wrangler`**

Add a `grid)` case in `scripts/agent-wrangler` before the catch-all:

```bash
  grid)
    shift
    exec python3 "$SCRIPT_DIR/grid_navigator.py" "$@"
    ;;
```

**Step 4: Verify manually**

Run: `./scripts/agent-wrangler start`
Expected: Three tmux windows: `teams` (project panes), `manager` (Claude + rail), `grid` (curses navigator). Tab between manager and grid via tmux, Enter on grid jumps to a project pane.

**Step 5: Commit**

```bash
git add scripts/grid_navigator.py scripts/tmux_teams.py scripts/agent-wrangler
git commit -m "feat: add grid navigator window with pane browser"
```

---

### Task 5: Tmux Navigation Bindings

Set up keybindings so the user can easily switch between manager (Page 1), grid (Page 2), and project panes.

**Files:**
- Modify: `scripts/tmux_teams.py` (update `run_nav` to add manager/grid window bindings)

**Step 1: Add window navigation bindings**

Find the `run_nav` function in `tmux_teams.py` and add these bindings alongside the existing Option+Arrow pane navigation:

```python
# Window navigation for manager/grid workflow
# Option+m -> manager window
tmux(["bind-key", "-n", "M-m", "select-window", "-t", f"{session}:{manager_window}"], timeout=5)
# Option+g -> grid window
tmux(["bind-key", "-n", "M-g", "select-window", "-t", f"{session}:grid"], timeout=5)
# Option+t -> teams (project panes) window
tmux(["bind-key", "-n", "M-t", "select-window", "-t", f"{session}:teams"], timeout=5)
```

These give three fast shortcuts:
- `Option+m` → Manager (Claude Code + rail)
- `Option+g` → Grid Navigator
- `Option+t` → Project panes

**Step 2: Verify manually**

After `agent-wrangler start`, press Option+m, Option+g, Option+t to switch between windows.

**Step 3: Commit**

```bash
git add scripts/tmux_teams.py
git commit -m "feat: add Option+m/g/t bindings for manager/grid/teams windows"
```

---

### Task 6: Clean Up Legacy Curses UI

The old 2-page curses UI (`command_center.py ui`) is replaced by the manager + grid setup. Deprecate it gracefully.

**Files:**
- Modify: `scripts/command_center.py` (add deprecation notice to `run_ui`)
- Modify: `scripts/agent-wrangler` (route `ui` to new start flow or show deprecation)

**Step 1: Add deprecation to `run_ui`**

At the top of `run_ui` (line 1625 in `command_center.py`), add:

```python
def run_ui(args: argparse.Namespace) -> int:
    print("NOTE: 'ui' is deprecated. Use 'agent-wrangler start' for the new manager + grid experience.")
    print("Starting legacy UI...\n")
    # ... rest of existing code unchanged ...
```

**Step 2: Update `agent-wrangler` help text**

In the help text of `scripts/agent-wrangler`, update the examples to reflect the new workflow:

```
Agent Wrangler shortcuts:
  start
    -> Manager (Claude Code + status rail) + Grid Navigator + project panes
  ops
    -> Interactive operator console (numbered menu)
  grid
    -> Standalone grid navigator
  ...
```

**Step 3: Commit**

```bash
git add scripts/command_center.py scripts/agent-wrangler
git commit -m "chore: deprecate legacy curses UI in favor of manager + grid"
```

---

### Task 7: Session Context Stats

Add per-session context/usage stats to the grid navigator header. Two data sources:
- **Cheap (every refresh):** process uptime, tmux pane scrollback size, activity status, CPU time
- **Periodic (every ~5 min, idle sessions only):** send `/usage` to waiting Claude Code sessions, capture output, parse and store

**Files:**
- Create: `scripts/session_stats.py` - stats collection + storage
- Modify: `scripts/grid_navigator.py` (add stats header bar)
- Modify: `scripts/tmux_teams.py` (integrate stats into rail)

**Step 1: Create `session_stats.py`**

This module handles both cheap and periodic stats collection:

```python
#!/usr/bin/env python3
"""Session context stats: cheap approximation + periodic /usage ground truth."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STATS_FILE = ROOT / ".state" / "session_stats.json"
USAGE_POLL_INTERVAL = 300  # 5 minutes

# Ensure .state directory exists
STATS_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_stats() -> dict[str, Any]:
    if STATS_FILE.exists():
        try:
            return json.loads(STATS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"sessions": {}, "last_poll": 0.0}


def save_stats(data: dict[str, Any]) -> None:
    STATS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def cheap_stats_for_pane(pane_id: str, tmux_fn) -> dict[str, Any]:
    """Gather cheap stats from tmux without interrupting the session."""
    stats: dict[str, Any] = {}

    # Scrollback size (rough proxy for output volume / context usage)
    code, out, _ = tmux_fn(["capture-pane", "-t", pane_id, "-p", "-S", "-"], timeout=5)
    if code == 0:
        line_count = len(out.splitlines())
        stats["scrollback_lines"] = line_count
        stats["scrollback_kb"] = round(len(out) / 1024, 1)

    # Pane uptime via pid start time
    code, out, _ = tmux_fn(["display-message", "-t", pane_id, "-p", "#{pane_pid}"], timeout=5)
    if code == 0:
        pid = out.strip()
        if pid:
            stats["pid"] = pid
            try:
                import subprocess
                result = subprocess.run(
                    ["ps", "-o", "etime=", "-p", pid],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    stats["uptime"] = result.stdout.strip()
            except Exception:
                pass

    return stats


def poll_usage_for_pane(pane_id: str, tmux_fn) -> dict[str, Any] | None:
    """Send /usage to a Claude Code session and capture the output.

    Only call this on panes that are in a 'waiting' state (at the Claude prompt).
    Returns parsed usage data or None if capture failed.
    """
    # Send /usage command
    tmux_fn(["send-keys", "-t", pane_id, "/usage", "Enter"], timeout=5)
    # Wait for output
    time.sleep(2)
    # Capture the output
    code, out, _ = tmux_fn(["capture-pane", "-t", pane_id, "-p", "-S", "-30"], timeout=5)
    if code != 0:
        return None

    usage: dict[str, Any] = {}
    for line in out.splitlines():
        line_lower = line.strip().lower()
        # Parse common /usage output patterns
        if "cost" in line_lower and "$" in line:
            usage["cost_line"] = line.strip()
        if "token" in line_lower:
            usage["token_line"] = line.strip()
        if "context" in line_lower and ("%" in line or "/" in line):
            usage["context_line"] = line.strip()

    if usage:
        usage["polled_at"] = time.time()
    return usage if usage else None


def should_poll(stats_data: dict[str, Any]) -> bool:
    """Check if enough time has passed since last /usage poll."""
    last = stats_data.get("last_poll", 0.0)
    return (time.time() - last) >= USAGE_POLL_INTERVAL


def update_session_stats(
    pane_id: str,
    project_id: str,
    cheap: dict[str, Any],
    usage: dict[str, Any] | None,
    stats_data: dict[str, Any],
) -> None:
    """Merge cheap + usage stats into the stats store."""
    sessions = stats_data.setdefault("sessions", {})
    entry = sessions.get(project_id, {})
    entry["pane_id"] = pane_id
    entry["updated_at"] = time.time()
    entry.update(cheap)
    if usage:
        entry["usage"] = usage
        stats_data["last_poll"] = time.time()
    sessions[project_id] = entry


def format_stats_line(project_id: str, stats_data: dict[str, Any]) -> str:
    """Format a compact one-line stats summary for display."""
    sessions = stats_data.get("sessions", {})
    entry = sessions.get(project_id, {})
    if not entry:
        return ""

    parts = []
    uptime = entry.get("uptime", "")
    if uptime:
        parts.append(f"up:{uptime}")

    kb = entry.get("scrollback_kb", 0)
    if kb:
        parts.append(f"out:{kb}kb")

    usage = entry.get("usage", {})
    ctx = usage.get("context_line", "")
    if ctx:
        # Try to extract just the percentage or fraction
        parts.append(ctx)
    cost = usage.get("cost_line", "")
    if cost:
        parts.append(cost)

    return "  ".join(parts)
```

**Step 2: Integrate cheap stats into grid navigator refresh**

In `grid_navigator.py`, import session_stats and call `cheap_stats_for_pane` during each refresh cycle. Add a header bar above the pane list:

```python
from session_stats import (
    cheap_stats_for_pane, load_stats, save_stats,
    update_session_stats, format_stats_line, should_poll,
    poll_usage_for_pane,
)
```

In the `refresh()` function, after getting rows, collect cheap stats:

```python
def refresh() -> list[dict[str, Any]]:
    nonlocal last_tick, stats_data
    # ... existing pane refresh ...

    # Cheap stats for every pane
    for row in result:
        pid = str(row.get("pane_id") or "")
        proj = str(row.get("project_id") or "")
        if pid and proj:
            cheap = cheap_stats_for_pane(pid, tmux)
            update_session_stats(pid, proj, cheap, None, stats_data)

    # Periodic /usage poll for ONE idle claude session per cycle
    if should_poll(stats_data):
        for row in result:
            if (str(row.get("status") or "") == "waiting"
                    and str(row.get("ai_tool") or row.get("agent") or "") == "claude"):
                pid = str(row.get("pane_id") or "")
                proj = str(row.get("project_id") or "")
                usage = poll_usage_for_pane(pid, tmux)
                if usage:
                    update_session_stats(pid, proj, None, usage, stats_data)
                break  # Only one per cycle to avoid disruption

    save_stats(stats_data)
    last_tick = time.time()
    return result
```

In the draw loop, add a stats summary header between the title bar and the pane list:

```python
# Stats header (line 1)
total_kb = sum(
    stats_data.get("sessions", {}).get(
        str(r.get("project_id") or ""), {}
    ).get("scrollback_kb", 0)
    for r in rows
)
claude_count = sum(1 for r in rows if str(r.get("ai_tool") or r.get("agent") or "") == "claude")
stats_header = f" Agents: {claude_count}  Total output: {total_kb:.0f}kb"

# If any session has /usage data, show the most recent
latest_usage = ""
for sid, entry in stats_data.get("sessions", {}).items():
    ctx = entry.get("usage", {}).get("context_line", "")
    if ctx:
        latest_usage = f"  |  Latest: {sid} {ctx}"
        break

try:
    stdscr.addstr(1, 0, (stats_header + latest_usage)[:w-1], curses.color_pair(1))
except curses.error:
    pass
```

Shift `list_top` from 2 to 3 to make room for the stats header.

**Step 3: Verify manually**

Run: `./scripts/agent-wrangler start`, switch to grid (Option+g).
Expected: Stats header shows agent count and output volume. After 5 minutes with a waiting Claude session, context/cost data appears from `/usage` poll.

**Step 4: Commit**

```bash
git add scripts/session_stats.py scripts/grid_navigator.py
git commit -m "feat: add session context stats to grid navigator header"
```

---

### Task 8: Update CLAUDE.md

Update the project documentation to reflect the new architecture and stats system.

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update architecture section**

Replace the layer model and key sections to reflect:
- Manager window = Claude Code + status rail (Page 1)
- Grid navigator = curses pane browser (Page 2)
- Navigation: Option+m/g/t for fast switching
- `agent-wrangler start` is the single entry point

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for manager + grid architecture"
```
