# Ghostty-Native Architecture Rewrite

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the tmux grid/pane layer with direct terminal monitoring via TTY + optional Ghostty AppleScript integration, keeping the ranch board as the primary UI.

**Architecture:** terminal_sentinel discovers all terminals (any app). Health is TTY-mtime-based. Management commands write directly to TTY device files. Optional Ghostty module uses AppleScript to create tabs, set titles, and navigate. Ranch board runs standalone in any terminal.

**Tech Stack:** Python 3.10+ stdlib, osascript (macOS), no external dependencies.

---

### Task 1: Create the TTY manager module

**Files:**
- Create: `scripts/tty_manager.py`

**Step 1: Write tty_manager.py**

Core module for sending commands to terminals via TTY device files. This replaces tmux send-keys with direct TTY writes that work in any terminal.

```python
#!/usr/bin/env python3
"""Direct TTY management — send commands to any terminal without tmux."""

from __future__ import annotations

import os
import signal
from pathlib import Path


def tty_write(tty: str, text: str) -> bool:
    """Write text to a TTY device. Returns True on success."""
    dev = f"/dev/{tty}" if not tty.startswith("/dev/") else tty
    try:
        fd = os.open(dev, os.O_WRONLY | os.O_NOCTTY)
        try:
            os.write(fd, text.encode())
            return True
        finally:
            os.close(fd)
    except OSError:
        return False


def tty_send_command(tty: str, command: str) -> bool:
    """Send a command + newline to a TTY."""
    return tty_write(tty, command + "\n")


def tty_send_ctrl_c(tty: str) -> bool:
    """Send Ctrl-C (interrupt) to a TTY."""
    return tty_write(tty, "\x03")


def tty_send_ctrl_d(tty: str) -> bool:
    """Send Ctrl-D (EOF) to a TTY."""
    return tty_write(tty, "\x04")
```

**Step 2: Verify it works**

Run: `python3 -c "import scripts.tty_manager; print('OK')"`

**Step 3: Commit**

```
feat: add tty_manager for direct terminal writes
```

---

### Task 2: Create the Ghostty AppleScript module

**Files:**
- Create: `scripts/ghostty_bridge.py`

**Step 1: Write ghostty_bridge.py**

Optional Ghostty integration via AppleScript. Gracefully returns None/False when not on macOS or Ghostty isn't running.

```python
#!/usr/bin/env python3
"""Optional Ghostty integration via AppleScript (macOS only)."""

from __future__ import annotations

import subprocess
import sys
from typing import Any


def _osascript(script: str, timeout: int = 10) -> tuple[bool, str]:
    """Run AppleScript. Returns (success, output)."""
    if sys.platform != "darwin":
        return False, "not macOS"
    try:
        proc = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
        return proc.returncode == 0, proc.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, ""


def is_available() -> bool:
    """Check if Ghostty is running and scriptable."""
    ok, _ = _osascript('tell application "System Events" to (name of processes) contains "Ghostty"')
    return ok and _ == "true"


def list_tabs() -> list[dict[str, Any]]:
    """List all Ghostty tabs with their working directories."""
    script = '''
    tell application "Ghostty"
        set results to {}
        repeat with w in windows
            repeat with t in tabs of w
                set term to focused terminal of t
                set tid to id of term
                set tname to name of t
                set tdir to working directory of term
                set tidx to index of t
                set end of results to (tid & "||" & tname & "||" & tdir & "||" & tidx)
            end repeat
        end repeat
        return results
    end tell
    '''
    ok, output = _osascript(script)
    if not ok or not output:
        return []
    tabs = []
    for line in output.split(", "):
        parts = line.strip().split("||")
        if len(parts) >= 4:
            tabs.append({
                "terminal_id": parts[0],
                "name": parts[1],
                "working_directory": parts[2],
                "index": parts[3],
            })
    return tabs


def create_tab(working_directory: str, command: str | None = None) -> bool:
    """Create a new Ghostty tab with the given working directory."""
    cmd_part = ""
    if command:
        cmd_part = f'\n        set command of cfg to "{command}"'
    script = f'''
    tell application "Ghostty"
        set cfg to new surface configuration
        set initial working directory of cfg to "{working_directory}"{cmd_part}
        new tab in front window with configuration cfg
    end tell
    '''
    ok, _ = _osascript(script)
    return ok


def focus_tab(terminal_id: str) -> bool:
    """Focus a specific terminal by ID."""
    script = f'''
    tell application "Ghostty"
        repeat with w in windows
            repeat with t in tabs of w
                set term to focused terminal of t
                if id of term is "{terminal_id}" then
                    select t
                    focus term
                    return true
                end if
            end repeat
        end repeat
        return false
    end tell
    '''
    ok, _ = _osascript(script)
    return ok


def set_tab_title(terminal_id: str, title: str) -> bool:
    """Set the title of a specific tab."""
    script = f'''
    tell application "Ghostty"
        repeat with w in windows
            repeat with t in tabs of w
                set term to focused terminal of t
                if id of term is "{terminal_id}" then
                    perform action "set_tab_title:{title}" on term
                    return true
                end if
            end repeat
        end repeat
        return false
    end tell
    '''
    ok, _ = _osascript(script)
    return ok


def send_text(terminal_id: str, text: str) -> bool:
    """Send text input to a specific terminal."""
    escaped = text.replace('\\', '\\\\').replace('"', '\\"')
    script = f'''
    tell application "Ghostty"
        repeat with w in windows
            repeat with t in tabs of w
                set term to focused terminal of t
                if id of term is "{terminal_id}" then
                    input text "{escaped}" to term
                    return true
                end if
            end repeat
        end repeat
        return false
    end tell
    '''
    ok, _ = _osascript(script)
    return ok


def close_tab(terminal_id: str) -> bool:
    """Close a specific tab by terminal ID."""
    script = f'''
    tell application "Ghostty"
        repeat with w in windows
            repeat with t in tabs of w
                set term to focused terminal of t
                if id of term is "{terminal_id}" then
                    close t
                    return true
                end if
            end repeat
        end repeat
        return false
    end tell
    '''
    ok, _ = _osascript(script)
    return ok
```

**Step 2: Verify it loads**

Run: `python3 -c "from scripts import ghostty_bridge; print('available:', ghostty_bridge.is_available())"`

**Step 3: Commit**

```
feat: add ghostty_bridge for optional AppleScript integration
```

---

### Task 3: Create the unified session manager

**Files:**
- Create: `scripts/session_manager.py`

**Step 1: Write session_manager.py**

This is the new brain. It uses terminal_sentinel for discovery, tty_manager for commands, and optionally ghostty_bridge for tab creation. Replaces the tmux grid layer.

```python
#!/usr/bin/env python3
"""Unified session manager — discovers and manages terminals without tmux."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import terminal_sentinel
import tty_manager

ROOT = Path(__file__).resolve().parents[1]
PROJECTS_CONFIG = ROOT / "config" / "projects.json"

# Try to import Ghostty bridge (optional)
try:
    import ghostty_bridge
    _HAS_GHOSTTY = ghostty_bridge.is_available()
except ImportError:
    _HAS_GHOSTTY = False


def load_projects() -> list[dict[str, Any]]:
    """Load projects from config."""
    if not PROJECTS_CONFIG.exists():
        return []
    try:
        config = json.loads(PROJECTS_CONFIG.read_text(encoding="utf-8"))
        return config.get("projects", [])
    except (json.JSONDecodeError, KeyError):
        return []


def project_map() -> dict[str, dict[str, Any]]:
    """Map project IDs to their config."""
    return {p["id"]: p for p in load_projects() if "id" in p}


def active_projects() -> list[dict[str, Any]]:
    """Projects not in the barn."""
    return [p for p in load_projects() if not p.get("barn")]


def barned_projects() -> list[dict[str, Any]]:
    """Projects in the barn."""
    return [p for p in load_projects() if p.get("barn")]


def discover_sessions() -> list[dict[str, Any]]:
    """Discover all terminal sessions via terminal_sentinel.

    Returns a list of session dicts with:
    - tty, pid, status, agent, source, cwd, command, waiting_minutes
    - project_id (inferred from cwd matching projects.json)
    """
    snapshot, _ = terminal_sentinel.classify_sessions(
        source_filter="all", include_idle=True,
    )
    sessions = list(snapshot.get("sessions", []))
    proj_map_data = project_map()

    # Exclude agent-wrangler's own directory
    self_path = str(ROOT)

    enriched = []
    for sess in sessions:
        cwd = sess.get("cwd") or ""

        # Skip agent-wrangler's own sessions
        if cwd and os.path.realpath(cwd) == os.path.realpath(self_path):
            continue

        # Match to project by path
        matched_project = None
        for pid, proj in proj_map_data.items():
            proj_path = proj.get("path", "")
            if proj_path and os.path.realpath(cwd) == os.path.realpath(proj_path):
                matched_project = pid
                break

        # Fallback: match by directory basename
        if not matched_project and cwd:
            basename = Path(cwd).name.lower().replace(" ", "-")
            if basename in proj_map_data:
                matched_project = basename

        sess["project_id"] = matched_project
        enriched.append(sess)

    return enriched


def send_command(project_id: str, command: str) -> bool:
    """Send a command to the terminal running in a project's directory."""
    sessions = discover_sessions()
    for sess in sessions:
        if sess.get("project_id") == project_id:
            tty = sess.get("tty")
            if tty:
                return tty_manager.tty_send_command(tty, command)
    return False


def stop_agent(project_id: str) -> bool:
    """Send Ctrl-C to a project's terminal."""
    sessions = discover_sessions()
    for sess in sessions:
        if sess.get("project_id") == project_id:
            tty = sess.get("tty")
            if tty:
                return tty_manager.tty_send_ctrl_c(tty)
    return False


def start_agent(project_id: str, tool: str = "claude") -> bool:
    """Start an AI agent in a project's terminal."""
    return send_command(project_id, tool)


def open_project(project_id: str, tool: str | None = None) -> bool:
    """Open a new terminal tab for a project (Ghostty only)."""
    if not _HAS_GHOSTTY:
        return False
    proj = project_map().get(project_id)
    if not proj:
        return False
    path = proj.get("path", "")
    if not path:
        return False
    return ghostty_bridge.create_tab(path, command=tool)


def focus_project(project_id: str) -> bool:
    """Focus the terminal tab for a project (Ghostty only)."""
    if not _HAS_GHOSTTY:
        return False
    tabs = ghostty_bridge.list_tabs()
    proj = project_map().get(project_id)
    if not proj:
        return False
    proj_path = os.path.realpath(proj.get("path", ""))
    for tab in tabs:
        tab_path = os.path.realpath(tab.get("working_directory", ""))
        if tab_path == proj_path:
            return ghostty_bridge.focus_tab(tab["terminal_id"])
    return False
```

**Step 2: Commit**

```
feat: add session_manager — unified discovery + management layer
```

---

### Task 4: Rewrite the CLI entry points

**Files:**
- Create: `scripts/wrangler.py` (new clean CLI — the old agent_wrangler.py stays for now)

**Step 1: Write wrangler.py**

New clean CLI that uses session_manager. This will eventually replace agent_wrangler.py but we keep both during transition.

The CLI provides: status, rail, agent, stop, send, open, focus, barn, unbarn, barn-list, projects.

```python
#!/usr/bin/env python3
"""Agent Wrangler v2 — terminal-agnostic agent management."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add scripts dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import session_manager
import terminal_sentinel

ROOT = Path(__file__).resolve().parents[1]
PROJECTS_CONFIG = ROOT / "config" / "projects.json"


# ── Health classification ──────────────────────────────────────────────────

def classify_health(sess: dict[str, Any]) -> str:
    """Classify a session as green/yellow/red based on TTY activity."""
    status = str(sess.get("status") or "idle")
    agent = str(sess.get("agent") or "")

    if status == "active":
        return "green"
    if status == "waiting":
        return "yellow"
    if agent and agent != "-":
        return "yellow"  # has agent but status unclear
    return "yellow"  # idle at shell


# ── Ranch board rendering ──────────────────────────────────────────────────

_health_history: dict[str, list[int]] = {}
_prev_health: dict[str, str] = {}


def _sparkline(history: list[int]) -> str:
    blocks = {0: "\033[31m▁\033[0m", 1: "\033[33m▄\033[0m", 2: "\033[32m█\033[0m"}
    return "".join(blocks.get(v, "▁") for v in history[-10:])


def _ranch_header(counts: dict[str, int]) -> list[str]:
    total = sum(counts.values())
    g = counts.get("green", 0)
    y = counts.get("yellow", 0)
    r = counts.get("red", 0)
    hat_color = 172
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


def render_rail(sessions: list[dict[str, Any]]) -> list[str]:
    """Render the ranch board status rail."""
    health_val = {"green": 2, "yellow": 1, "red": 0}
    counts: dict[str, int] = {"green": 0, "yellow": 0, "red": 0}

    for sess in sessions:
        health = classify_health(sess)
        counts[health] = counts.get(health, 0) + 1

    lines: list[str] = ["\033[2J\033[H"]  # clear
    lines.extend(_ranch_header(counts))
    lines.append("\033[2m" + "─" * 34 + "\033[0m")

    # Summary
    ranch_terms = []
    g, y, r = counts.get("green", 0), counts.get("yellow", 0), counts.get("red", 0)
    if g:
        ranch_terms.append(f"\033[32m{g} grazing\033[0m")
    if y:
        ranch_terms.append(f"\033[33m{y} at fence\033[0m")
    if r:
        ranch_terms.append(f"\033[31m{r} down\033[0m")
    lines.append(f" {' · '.join(ranch_terms)}")
    lines.append("\033[2m" + "─" * 34 + "\033[0m")

    # Per-session entries
    for sess in sessions:
        health = classify_health(sess)
        project = str(sess.get("project_id") or sess.get("cwd", "?").split("/")[-1])
        agent = str(sess.get("agent") or "")
        status = str(sess.get("status") or "idle")
        source = str(sess.get("source") or "")
        wait_min = sess.get("waiting_minutes")

        if len(project) > 16:
            project = project[:15] + "~"

        # Health history sparkline
        h_val = health_val.get(health, 1)
        hist = _health_history.setdefault(project, [])
        hist.append(h_val)
        if len(hist) > 10:
            _health_history[project] = hist[-10:]
        spark = _sparkline(_health_history.get(project, []))

        dot_color = {"green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m"}.get(health, "\033[0m")
        line = f" {dot_color}●\033[0m {project:<16}"
        if agent and agent != "-":
            line += f" \033[2m{agent}\033[0m"
        lines.append(line)
        lines.append(f"   {spark}")

        # Waiting time
        if wait_min and int(wait_min) > 0:
            lines.append(f"   \033[2m⏳ {int(wait_min)}m waiting\033[0m")

    # Barn count
    barn = session_manager.barned_projects()
    if barn:
        lines.append("\033[2m" + "─" * 34 + "\033[0m")
        lines.append(f" \033[38;5;130m{len(barn)} in barn\033[0m")

    # Timestamp
    now_str = datetime.now().strftime("%H:%M:%S")
    lines.append(f"\n\033[2m  last roundup: {now_str}\033[0m")

    return lines


# ── CLI commands ───────────────────────────────────────────────────────────

def cmd_status(args: argparse.Namespace) -> int:
    sessions = session_manager.discover_sessions()
    if not sessions:
        print("No terminal sessions discovered.")
        return 0

    print(f"{'PROJECT':<20} {'AGENT':<10} {'STATUS':<10} {'SOURCE':<10} {'TTY':<10}")
    print("─" * 60)
    for sess in sessions:
        project = str(sess.get("project_id") or "-")
        agent = str(sess.get("agent") or "-")
        status = str(sess.get("status") or "idle")
        source = str(sess.get("source") or "-")
        tty = str(sess.get("tty") or "-")
        print(f"{project:<20} {agent:<10} {status:<10} {source:<10} {tty:<10}")
    return 0


def cmd_rail(args: argparse.Namespace) -> int:
    interval = max(1, int(args.interval))
    while True:
        sessions = session_manager.discover_sessions()
        # Filter to known projects only (skip unknown terminals)
        known = [s for s in sessions if s.get("project_id")]
        lines = render_rail(known)
        print("\n".join(lines), flush=True)
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            break
    return 0


def cmd_agent(args: argparse.Namespace) -> int:
    if session_manager.start_agent(args.project, args.tool):
        print(f"Started {args.tool} in {args.project}")
        return 0
    # Fallback: try opening a new Ghostty tab
    if session_manager.open_project(args.project, args.tool):
        print(f"Opened new tab for {args.project} with {args.tool}")
        return 0
    print(f"Could not find terminal for '{args.project}'. Open a tab in that directory first.")
    return 1


def cmd_stop(args: argparse.Namespace) -> int:
    if session_manager.stop_agent(args.project):
        print(f"Sent Ctrl-C to {args.project}")
        return 0
    print(f"Could not find terminal for '{args.project}'")
    return 1


def cmd_send(args: argparse.Namespace) -> int:
    if session_manager.send_command(args.project, args.command):
        print(f"Sent to {args.project}: {args.command}")
        return 0
    print(f"Could not find terminal for '{args.project}'")
    return 1


def cmd_open(args: argparse.Namespace) -> int:
    tool = getattr(args, "tool", None)
    if session_manager.open_project(args.project, tool):
        print(f"Opened {args.project}" + (f" with {tool}" if tool else ""))
        return 0
    print("Could not open tab. Ghostty may not be available.")
    print(f"Manually: open a terminal, cd to the project directory.")
    return 1


def cmd_focus(args: argparse.Namespace) -> int:
    if session_manager.focus_project(args.project):
        return 0
    print(f"Could not focus '{args.project}'. Ghostty may not be available.")
    return 1


def _set_barn_flag(proj_id: str, barn: bool) -> bool:
    if not PROJECTS_CONFIG.exists():
        return False
    config = json.loads(PROJECTS_CONFIG.read_text(encoding="utf-8"))
    found = False
    for p in config.get("projects", []):
        if p.get("id") == proj_id:
            if barn:
                p["barn"] = True
            else:
                p.pop("barn", None)
            found = True
            break
    if found:
        PROJECTS_CONFIG.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return found


def cmd_barn(args: argparse.Namespace) -> int:
    if _set_barn_flag(args.project, barn=True):
        print(f"Sent '{args.project}' to the barn.")
        return 0
    print(f"Project '{args.project}' not found in config.")
    return 1


def cmd_unbarn(args: argparse.Namespace) -> int:
    if _set_barn_flag(args.project, barn=False):
        print(f"Let '{args.project}' out of the barn.")
        # Try to open a tab if Ghostty is available
        session_manager.open_project(args.project)
        return 0
    print(f"Project '{args.project}' not found in config.")
    return 1


def cmd_barn_list(args: argparse.Namespace) -> int:
    active = session_manager.active_projects()
    barn = session_manager.barned_projects()
    if active:
        print(f"\033[32mGrazing ({len(active)}):\033[0m")
        for p in active:
            print(f"  {p['id']}")
    if barn:
        print(f"\n\033[33mIn the barn ({len(barn)}):\033[0m")
        for p in barn:
            print(f"  {p['id']}")
    else:
        print("\nBarn is empty.")
    return 0


def cmd_projects(args: argparse.Namespace) -> int:
    projects = session_manager.load_projects()
    if not projects:
        print("No projects configured. Run: agent-wrangler add <path>")
        return 0
    for p in projects:
        barn_mark = " [barn]" if p.get("barn") else ""
        print(f"  {p['id']:<24} {p.get('path', '')}{barn_mark}")
    return 0


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="agent-wrangler",
        description="Agent Wrangler — manage AI coding agents across terminals.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show all discovered terminal sessions")

    rail = sub.add_parser("rail", help="Live ranch board status display")
    rail.add_argument("--interval", type=int, default=5, help="Refresh interval in seconds")

    agent_cmd = sub.add_parser("agent", help="Start an AI agent in a project")
    agent_cmd.add_argument("project", help="Project ID")
    agent_cmd.add_argument("tool", choices=["claude", "codex", "gemini"], default="claude", nargs="?")

    stop_cmd = sub.add_parser("stop", help="Send Ctrl-C to a project's terminal")
    stop_cmd.add_argument("project", help="Project ID")

    send_cmd = sub.add_parser("send", help="Send a command to a project's terminal")
    send_cmd.add_argument("project", help="Project ID")
    send_cmd.add_argument("--command", required=True, help="Command to send")

    open_cmd = sub.add_parser("open", help="Open a new Ghostty tab for a project")
    open_cmd.add_argument("project", help="Project ID")
    open_cmd.add_argument("--tool", help="Agent to start (claude, codex, gemini)")

    focus_cmd = sub.add_parser("focus", help="Focus the Ghostty tab for a project")
    focus_cmd.add_argument("project", help="Project ID")

    barn_cmd = sub.add_parser("barn", help="Send a project to the barn")
    barn_cmd.add_argument("project", help="Project ID")

    unbarn_cmd = sub.add_parser("unbarn", help="Let a project out of the barn")
    unbarn_cmd.add_argument("project", help="Project ID")

    sub.add_parser("barn-list", help="List grazing vs barned projects")
    sub.add_parser("projects", help="List all configured projects")

    args = parser.parse_args()

    handlers = {
        "status": cmd_status,
        "rail": cmd_rail,
        "agent": cmd_agent,
        "stop": cmd_stop,
        "send": cmd_send,
        "open": cmd_open,
        "focus": cmd_focus,
        "barn": cmd_barn,
        "unbarn": cmd_unbarn,
        "barn-list": cmd_barn_list,
        "projects": cmd_projects,
    }
    handler = handlers.get(args.command)
    if handler:
        return handler(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

**Step 2: Commit**

```
feat: add wrangler.py — clean v2 CLI with terminal-agnostic management
```

---

### Task 5: Create the new bash router

**Files:**
- Create: `scripts/aw` (new clean entry point)

**Step 1: Write scripts/aw**

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/wrangler.py" "$@"
```

**Step 2: Make executable and commit**

```bash
chmod +x scripts/aw
git add scripts/aw scripts/wrangler.py scripts/tty_manager.py scripts/ghostty_bridge.py scripts/session_manager.py
git commit -m "feat: agent-wrangler v2 — terminal-agnostic architecture"
```

---

### Task 6: Integration test

**Step 1: Test discovery**

```bash
python3 scripts/wrangler.py status
```

Should list all terminal sessions with project IDs matched.

**Step 2: Test ranch board**

```bash
python3 scripts/wrangler.py rail --interval 3
```

Should show live ranch board. Ctrl-C to exit.

**Step 3: Test Ghostty integration**

```bash
python3 -c "from scripts import ghostty_bridge; print(ghostty_bridge.list_tabs())"
```

Should list all Ghostty tabs.

**Step 4: Test sending commands**

```bash
python3 scripts/wrangler.py send <project> --command "echo hello from wrangler"
```

**Step 5: Test barn operations**

```bash
python3 scripts/wrangler.py barn-list
python3 scripts/wrangler.py barn <project>
python3 scripts/wrangler.py barn-list
python3 scripts/wrangler.py unbarn <project>
```

**Step 6: Commit any fixes and push**

```
fix: integration test fixes for v2 architecture
```
