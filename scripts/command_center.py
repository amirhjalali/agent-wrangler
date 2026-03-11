#!/usr/bin/env python3
"""Agent Wrangler: Ant Farm runtime monitor + operator console."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import terminal_sentinel
import tmux_teams

ROOT = Path(__file__).resolve().parents[1]
STORE_PATH = ROOT / "config" / "command_center.json"
REPORTS_DIR = ROOT / "reports"

BRAND_NAME = "Agent Wrangler"
PRIMARY_CLI = "agent-wrangler"


def short_text(text: str, width: int) -> str:
    if width <= 3:
        return text[:width]
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_store() -> dict[str, Any]:
    return {
        "version": "1.0",
        "settings": {
            "antfarm": {
                "source": "ghostty",
                "max_ai_sessions": 4,
                "kill_waiting_ai_after_min": 120,
                "overnight_interval_sec": 300,
            },
        },
    }


def load_store() -> dict[str, Any]:
    if not STORE_PATH.exists():
        store = default_store()
        save_store(store)
        return store
    try:
        return json.loads(STORE_PATH.read_text(encoding="utf-8"))
    except Exception:
        store = default_store()
        save_store(store)
        return store


def save_store(store: dict[str, Any]) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_PATH.write_text(json.dumps(store, indent=2), encoding="utf-8")


# ─── Ant Farm (runtime session monitor) ─────────────────────────────────


def session_tool(session: dict[str, Any]) -> str:
    tagged = str(session.get("agent") or "").strip().lower()
    if tagged:
        return tagged
    command = str(session.get("command") or "").lower()
    for tool in ("claude", "codex", "aider", "gemini", "chatgpt"):
        if tool in command:
            return tool
    if "python" in command:
        return "python"
    return terminal_sentinel.command_bin(str(session.get("command") or "")) or "task"


def antfarm_snapshot(source: str, include_idle: bool) -> tuple[dict[str, Any], list[str]]:
    return terminal_sentinel.classify_sessions(source_filter=source, include_idle=include_idle)


def print_antfarm(
    snapshot: dict[str, Any],
    warnings: list[str],
    max_wait_rows: int,
    max_ai_sessions: int,
    wait_limit_minutes: int,
) -> None:
    term_width = shutil.get_terminal_size(fallback=(120, 40)).columns
    cmd_width = max(24, term_width - 56)

    print("Ant Farm")
    if warnings:
        for warning in warnings:
            print(f"WARN: {warning}")

    summary = snapshot.get("summary", {})
    print(
        "Sessions: total={total} ai={ai} waiting={waiting} active={active} background={background} idle={idle}".format(
            total=summary.get("total", 0),
            ai=summary.get("ai", 0),
            waiting=summary.get("waiting", 0),
            active=summary.get("active", 0),
            background=summary.get("background", 0),
            idle=summary.get("idle", 0),
        )
    )
    by_agent = summary.get("by_agent", {})
    if by_agent:
        chunks: list[str] = []
        for name in sorted(by_agent.keys()):
            info = by_agent[name]
            chunks.append(f"{name}:{info.get('total', 0)}(w{info.get('waiting', 0)}/a{info.get('active', 0)})")
        print("Agents: " + ", ".join(chunks))

    sessions = list(snapshot.get("sessions", []))
    waiting = [s for s in sessions if s.get("status") == "waiting"]
    waiting.sort(key=lambda s: (s.get("waiting_minutes") or 0), reverse=True)
    active = [s for s in sessions if s.get("status") == "active"]
    active.sort(key=lambda s: s.get("pcpu") or 0, reverse=True)

    print("\nSessions (top waiting/active):")
    print(f"  {'TTY':<8} {'AGENT':<8} {'STATE':<8} {'WAIT':<6} {'CPU%':<6} {'PID':<7} COMMAND")
    display_rows = waiting[:max_wait_rows] + active[: max(0, max_wait_rows - len(waiting[:max_wait_rows]))]
    if not display_rows:
        print("  -")
    else:
        for session in display_rows[:max_wait_rows]:
            mins = session.get("waiting_minutes")
            wait_text = f"{int(mins)}m" if mins is not None else "-"
            command = short_text(str(session.get("command", "")), cmd_width)
            agent = short_text(session_tool(session), 8)
            print(
                f"  {str(session.get('tty', '-')):<8} {agent:<8} {str(session.get('status', '-')):<8} {wait_text:<6} "
                f"{str(session.get('pcpu', '-')):<6} {str(session.get('pid', '-')):<7} {command}"
            )

    actions = terminal_sentinel.recommend_actions(
        snapshot=snapshot,
        max_ai_sessions=max_ai_sessions,
        wait_limit_minutes=wait_limit_minutes,
    )

    if actions:
        print("\nSuggested actions:")
        for action in actions[:5]:
            reason = short_text(str(action["reason"]), 46)
            print(f"  stop pid={action['pid']} tty={action['tty']} ({reason})")
        extra = len(actions) - 5
        if extra > 0:
            print(f"  ... and {extra} more")
    else:
        print("\nSuggested actions: none")


# ─── Ant Farm handlers ───────────────────────────────────────────────────


def run_antfarm_status(args: argparse.Namespace) -> int:
    snapshot, warnings = antfarm_snapshot(source=args.source, include_idle=not args.no_idle)
    print_antfarm(
        snapshot=snapshot,
        warnings=warnings,
        max_wait_rows=args.wait_rows,
        max_ai_sessions=args.max_ai_sessions,
        wait_limit_minutes=args.kill_waiting_ai_after,
    )
    if args.write_report:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = REPORTS_DIR / f"antfarm-{ts}.json"
        payload = {"generated_at": now_iso(), "warnings": warnings, "snapshot": snapshot}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote report: {path}")
    return 0


def run_antfarm_watch(args: argparse.Namespace) -> int:
    loops = 0
    interval = max(2, args.interval)
    try:
        while True:
            print("\033[2J\033[H", end="")
            run_antfarm_status(args)
            loops += 1
            if args.iterations > 0 and loops >= args.iterations:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        return 0
    return 0


def run_antfarm_overnight(args: argparse.Namespace) -> int:
    store = load_store()
    defaults = store.get("settings", {}).get("antfarm", {})

    source = args.source or defaults.get("source", "ghostty")
    max_ai = int(args.max_ai_sessions or defaults.get("max_ai_sessions", 4))
    wait_limit = int(args.kill_waiting_ai_after or defaults.get("kill_waiting_ai_after_min", 120))
    interval = int(args.interval or defaults.get("overnight_interval_sec", 300))

    loops = 0
    try:
        while True:
            snapshot, warnings = antfarm_snapshot(source=source, include_idle=False)
            actions = terminal_sentinel.recommend_actions(
                snapshot=snapshot,
                max_ai_sessions=max_ai,
                wait_limit_minutes=wait_limit,
            )
            results = terminal_sentinel.apply_actions(actions, dry_run=(not args.apply))

            print(f"[{now_iso()}] Overnight guard")
            if warnings:
                for warning in warnings:
                    print(f"WARN: {warning}")

            summary = snapshot.get("summary", {})
            print(
                f"ai={summary.get('ai', 0)} waiting={summary.get('waiting', 0)} "
                f"active={summary.get('active', 0)} actions={len(results)}"
            )
            for result in results:
                mode = "APPLY" if result.get("applied") else "PLAN"
                print(
                    f"- {mode} pid={result.get('pid')} tty={result.get('tty')} "
                    f"reason={result.get('reason')} => {result.get('result')}"
                )

            if args.write_log:
                REPORTS_DIR.mkdir(parents=True, exist_ok=True)
                log_path = Path(args.log_file).expanduser() if args.log_file else REPORTS_DIR / "overnight.log"
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps({
                            "generated_at": now_iso(),
                            "warnings": warnings,
                            "summary": summary,
                            "actions": results,
                        })
                        + "\n"
                    )

            loops += 1
            if args.iterations > 0 and loops >= args.iterations:
                break
            time.sleep(max(30, interval))
    except KeyboardInterrupt:
        return 0
    return 0


# ─── Operator console ────────────────────────────────────────────────────


def _run_ops_command(command: list[str]) -> int:
    cmd = [str(ROOT / "scripts" / "agent-wrangler"), *command]
    print("")
    print("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def run_ops(_: argparse.Namespace) -> int:
    actions: list[tuple[str, list[str]]] = [
        ("Start all (import + grid + manager + nav)", ["start"]),
        ("Attach grid session", ["attach"]),
        ("Show pane status", ["status"]),
        ("Focus pane by project/token and attach", ["__focus__"]),
        ("Send command to pane/project token", ["__send__"]),
        ("Launch agent in pane/project token", ["__agent__"]),
        ("Stop pane (Ctrl-C) by project/token", ["__stop__"]),
        ("Open manager window", ["manager", "--replace"]),
        ("Doctor (attention)", ["doctor", "--only-attention"]),
        ("Enable hooks", ["hooks", "enable"]),
        ("Persistence save", ["persistence", "save"]),
        ("Persistence restore (last snapshot)", ["persistence", "restore", "--force", "--attach"]),
        ("Profile status", ["profile", "status"]),
        ("Profile list", ["profile", "list"]),
        ("Program status", ["program", "status"]),
    ]

    print("Agent Wrangler Ops Console")
    print("Single-command control center. Enter number, or q to quit.")

    while True:
        print("")
        for idx, (label, _command) in enumerate(actions, start=1):
            print(f"{idx:>2}. {label}")
        print(" q. Quit")

        choice = input("ops> ").strip().lower()
        if choice in {"q", "quit", "exit"}:
            print("ops closed")
            return 0
        if not choice:
            continue
        if not choice.isdigit():
            print("Invalid choice. Enter a number or q.")
            continue

        idx = int(choice)
        if idx < 1 or idx > len(actions):
            print("Invalid choice. Pick one of the listed numbers.")
            continue

        label, command = actions[idx - 1]
        print(f"Running: {label}")
        if command == ["__focus__"]:
            token = input("pane token (project/id/index/title): ").strip()
            if not token:
                print("No token entered.")
                continue
            code = _run_ops_command(["focus", token, "--attach"])
            print(f"Exit code: {code}")
            continue
        if command == ["__send__"]:
            token = input("pane token (project/id/index/title): ").strip()
            text = input("command to send: ").strip()
            if not token or not text:
                print("Token and command are required.")
                continue
            code = _run_ops_command(["send", token, "--command", text])
            print(f"Exit code: {code}")
            continue
        if command == ["__agent__"]:
            token = input("pane token (project/id/index/title): ").strip()
            tool = input("tool (claude|codex|aider|gemini): ").strip().lower()
            if tool not in {"claude", "codex", "aider", "gemini"}:
                print("Invalid tool. Use claude, codex, aider, or gemini.")
                continue
            if not token:
                print("Pane token is required.")
                continue
            code = _run_ops_command(["agent", token, tool])
            print(f"Exit code: {code}")
            continue
        if command == ["__stop__"]:
            token = input("pane token (project/id/index/title): ").strip()
            if not token:
                print("Pane token is required.")
                continue
            code = _run_ops_command(["stop", token])
            print(f"Exit code: {code}")
            continue
        code = _run_ops_command(command)
        print(f"Exit code: {code}")


# ─── CLI parser ──────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PRIMARY_CLI, description=BRAND_NAME)
    sub = parser.add_subparsers(dest="command", required=True)

    ops = sub.add_parser("ops", help="Interactive operator console")
    ops.set_defaults(handler=run_ops)

    ant = sub.add_parser("antfarm", help="Runtime session operations")
    ant_sub = ant.add_subparsers(dest="antfarm_command", required=True)

    a_status = ant_sub.add_parser("status", help="One-shot runtime status")
    a_status.add_argument("--source", default="ghostty", choices=["ghostty", "all", "codex", "iterm", "terminal"])
    a_status.add_argument("--no-idle", action="store_true")
    a_status.add_argument("--wait-rows", type=int, default=8)
    a_status.add_argument("--max-ai-sessions", type=int, default=4)
    a_status.add_argument("--kill-waiting-ai-after", type=int, default=120)
    a_status.add_argument("--write-report", action="store_true")
    a_status.set_defaults(handler=run_antfarm_status)

    a_watch = ant_sub.add_parser("watch", help="Live runtime status")
    a_watch.add_argument("--source", default="ghostty", choices=["ghostty", "all", "codex", "iterm", "terminal"])
    a_watch.add_argument("--no-idle", action="store_true")
    a_watch.add_argument("--wait-rows", type=int, default=8)
    a_watch.add_argument("--max-ai-sessions", type=int, default=4)
    a_watch.add_argument("--kill-waiting-ai-after", type=int, default=120)
    a_watch.add_argument("--interval", type=int, default=10)
    a_watch.add_argument("--iterations", type=int, default=0)
    a_watch.set_defaults(handler=run_antfarm_watch)

    a_overnight = ant_sub.add_parser("overnight", help="Overnight guardrail loop")
    a_overnight.add_argument("--source", default=None, choices=["ghostty", "all", "codex", "iterm", "terminal"])
    a_overnight.add_argument("--max-ai-sessions", type=int, default=None)
    a_overnight.add_argument("--kill-waiting-ai-after", type=int, default=None)
    a_overnight.add_argument("--interval", type=int, default=None)
    a_overnight.add_argument("--iterations", type=int, default=0)
    a_overnight.add_argument("--apply", action="store_true")
    a_overnight.add_argument("--write-log", action="store_true")
    a_overnight.add_argument("--log-file")
    a_overnight.set_defaults(handler=run_antfarm_overnight)

    tmux_teams.register_subparser(sub)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    handler = getattr(args, "handler", None)
    if not handler:
        parser.print_help()
        return 1
    try:
        return int(handler(args))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    sys.exit(main())
