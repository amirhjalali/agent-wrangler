#!/usr/bin/env python3
"""Agent Wrangler v2 — terminal-agnostic agent management."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add scripts dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import session_manager

ROOT = Path(__file__).resolve().parents[1]
PROJECTS_CONFIG = ROOT / "config" / "projects.json"


# ── Health classification ──────────────────────────────────────────────────

def classify_health(sess: dict[str, Any]) -> str:
    """Classify a session as green/yellow/red based on TTY activity."""
    status = str(sess.get("status") or "idle")
    if status == "active":
        return "green"
    if status == "waiting":
        return "yellow"
    return "yellow"


# ── Ranch board rendering ──────────────────────────────────────────────────

_health_history: dict[str, list[int]] = {}


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
        project = str(sess.get("project_id") or "?")
        agent = str(sess.get("agent") or "")
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

        dot_color = {"green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m"}.get(
            health, "\033[0m"
        )
        line = f" {dot_color}●\033[0m {project:<16}"
        if agent and agent != "-":
            line += f" \033[2m{agent}\033[0m"
        lines.append(line)
        lines.append(f"   {spark}")

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
    if session_manager.open_project(args.project, args.tool):
        print(f"Opened new tab for {args.project} with {args.tool}")
        return 0
    print(f"No terminal found for '{args.project}'. Open a tab in that directory first.")
    return 1


def cmd_stop(args: argparse.Namespace) -> int:
    if session_manager.stop_agent(args.project):
        print(f"Sent Ctrl-C to {args.project}")
        return 0
    print(f"No terminal found for '{args.project}'")
    return 1


def cmd_send(args: argparse.Namespace) -> int:
    if session_manager.send_command(args.project, args.command):
        print(f"Sent to {args.project}: {args.command}")
        return 0
    print(f"No terminal found for '{args.project}'")
    return 1


def cmd_open(args: argparse.Namespace) -> int:
    tool = getattr(args, "tool", None)
    if session_manager.open_project(args.project, tool):
        print(f"Opened {args.project}" + (f" with {tool}" if tool else ""))
        return 0
    print("Ghostty not available. Open a tab manually and cd to the project directory.")
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
        print("No projects configured.")
        return 0
    for p in projects:
        barn_mark = " \033[33m[barn]\033[0m" if p.get("barn") else ""
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
    rail.add_argument("--interval", type=int, default=5)

    agent_cmd = sub.add_parser("agent", help="Start an AI agent in a project")
    agent_cmd.add_argument("project")
    agent_cmd.add_argument("tool", choices=["claude", "codex", "gemini"], default="claude", nargs="?")

    stop_cmd = sub.add_parser("stop", help="Send Ctrl-C to a project's terminal")
    stop_cmd.add_argument("project")

    send_cmd = sub.add_parser("send", help="Send a command to a project's terminal")
    send_cmd.add_argument("project")
    send_cmd.add_argument("--command", required=True)

    open_cmd = sub.add_parser("open", help="Open a new Ghostty tab for a project")
    open_cmd.add_argument("project")
    open_cmd.add_argument("--tool")

    focus_cmd = sub.add_parser("focus", help="Focus the Ghostty tab for a project")
    focus_cmd.add_argument("project")

    barn_cmd = sub.add_parser("barn", help="Send a project to the barn")
    barn_cmd.add_argument("project")

    unbarn_cmd = sub.add_parser("unbarn", help="Let a project out of the barn")
    unbarn_cmd.add_argument("project")

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
