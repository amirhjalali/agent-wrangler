#!/usr/bin/env python3
"""Lightweight monitor for interactive terminal sessions (Ghostty-focused)."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / ".state"
STATE_FILE = STATE_DIR / "terminal-sentinel-state.json"
REPORTS_DIR = ROOT / "reports"

WRAPPER_BINS = {
    "login",
    "zsh",
    "bash",
    "sh",
    "fish",
    "tmux",
    "zellij",
    "screen",
}
AI_TOOL_MARKERS: dict[str, tuple[str, ...]] = {
    "claude": ("claude", "anthropic"),
    "codex": ("codex", "@openai/codex", "/openai/"),
    "aider": ("aider",),
    "chatgpt": ("chatgpt",),
    "gemini": ("gemini",),
}
SOURCE_MARKERS = {
    "ghostty": ("ghostty.app/contents/macos/ghostty", "ghostty"),
    "codex": ("codex.app/contents/macos/codex",),
    "iterm": ("iterm2",),
    "terminal": ("/applications/utilities/terminal.app",),
}


@dataclass
class Proc:
    pid: int
    ppid: int
    tty: str
    stat: str
    pcpu: float
    cpu_seconds: float
    command: str



def run_cmd(cmd: list[str], timeout: int = 10) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError) as exc:
        return 1, "", str(exc)
    return proc.returncode, proc.stdout, proc.stderr



def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()



def parse_ps_time(raw: str) -> float:
    """Parse ps time formats into seconds. Supports MM:SS.xx and HH:MM:SS."""
    raw = raw.strip()
    if not raw:
        return 0.0

    days = 0
    value = raw
    if "-" in raw:
        day_part, value = raw.split("-", 1)
        try:
            days = int(day_part)
        except ValueError:
            days = 0

    parts = value.split(":")
    try:
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return days * 86400 + hours * 3600 + minutes * 60 + seconds
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return days * 86400 + minutes * 60 + seconds
        if len(parts) == 1:
            return days * 86400 + float(parts[0])
    except ValueError:
        return 0.0

    return 0.0



def fmt_seconds(seconds: float) -> str:
    value = int(seconds)
    if value < 60:
        return f"{value}s"
    if value < 3600:
        return f"{value // 60}m"
    if value < 86400:
        return f"{value // 3600}h"
    return f"{value // 86400}d"



def command_bin(command: str) -> str:
    first = command.split(" ", 1)[0]
    base = Path(first).name
    return base.lstrip("-").lower()



def is_wrapper(command: str) -> bool:
    base = command_bin(command)
    if base in WRAPPER_BINS:
        return True
    if base == "node" and ("/usr/local/bin/codex" in command.lower()):
        return True
    return False



def detect_ai_tool(command: str) -> str | None:
    lower = command.lower()
    for tool, markers in AI_TOOL_MARKERS.items():
        if any(marker in lower for marker in markers):
            return tool
    return None



def is_ai_command(command: str) -> bool:
    return detect_ai_tool(command) is not None



def parse_ps_table() -> tuple[list[Proc], str | None]:
    code, out, err = run_cmd(["ps", "-Ao", "pid=,ppid=,tty=,stat=,pcpu=,time=,command="], timeout=12)
    if code != 0:
        detail = err.strip() if err else "unknown error"
        return [], f"ps unavailable: {detail}"

    rows: list[Proc] = []
    for line in out.splitlines():
        match = re.match(r"\s*(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)$", line)
        if not match:
            continue

        pid = int(match.group(1))
        ppid = int(match.group(2))
        tty = match.group(3)
        stat = match.group(4)

        try:
            pcpu = float(match.group(5))
        except ValueError:
            pcpu = 0.0

        cpu_seconds = parse_ps_time(match.group(6))
        command = match.group(7)

        rows.append(Proc(pid=pid, ppid=ppid, tty=tty, stat=stat, pcpu=pcpu, cpu_seconds=cpu_seconds, command=command))

    return rows, None



def build_pid_map(procs: list[Proc]) -> dict[int, Proc]:
    return {proc.pid: proc for proc in procs}



def lineage_commands(pid_map: dict[int, Proc], pid: int, max_hops: int = 24) -> list[str]:
    chain: list[str] = []
    current = pid
    hops = 0
    seen: set[int] = set()

    while current > 0 and hops < max_hops and current not in seen:
        seen.add(current)
        proc = pid_map.get(current)
        if not proc:
            break
        chain.append(proc.command.lower())
        current = proc.ppid
        hops += 1

    return chain



def detect_source(tty_procs: list[Proc], pid_map: dict[int, Proc]) -> str:
    checks = [proc.pid for proc in tty_procs[:4]]
    for pid in checks:
        lineage = "\n".join(lineage_commands(pid_map, pid))
        for source, markers in SOURCE_MARKERS.items():
            if any(marker in lineage for marker in markers):
                return source
    return "unknown"



def load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {"sessions": {}}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"sessions": {}}



def save_state(state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")



def select_primary_process(tty_procs: list[Proc]) -> tuple[Proc | None, str]:
    if not tty_procs:
        return None, "idle"

    candidates = [proc for proc in tty_procs if not is_wrapper(proc.command) or is_ai_command(proc.command)]
    if not candidates:
        shell = sorted(tty_procs, key=lambda p: p.pid)[-1]
        return shell, "idle"

    ai_candidates = [proc for proc in candidates if is_ai_command(proc.command)]
    if ai_candidates:
        # Prefer the process closest to the shell (often the top-level CLI), then highest CPU.
        shell_pids = {proc.pid for proc in tty_procs if command_bin(proc.command) in WRAPPER_BINS}
        ai_candidates.sort(
            key=lambda p: (1 if p.ppid in shell_pids else 0, p.pcpu, p.cpu_seconds),
            reverse=True,
        )
        return ai_candidates[0], "ai"

    # Non-AI: pick a leaf process with highest activity.
    child_pids = {proc.ppid for proc in candidates}
    leaf = [proc for proc in candidates if proc.pid not in child_pids]
    pool = leaf if leaf else candidates
    pool.sort(key=lambda p: (p.pcpu, p.cpu_seconds), reverse=True)
    return pool[0], "task"



def classify_sessions(
    source_filter: str,
    include_idle: bool,
) -> tuple[dict[str, Any], list[str]]:
    procs, parse_error = parse_ps_table()
    if parse_error:
        return {
            "generated_at": now_iso(),
            "sessions": [],
            "summary": {
                "total": 0,
                "active": 0,
                "waiting": 0,
                "idle": 0,
                "background": 0,
                "ai": 0,
                "by_agent": {},
            },
        }, [parse_error]

    pid_map = build_pid_map(procs)
    by_tty: dict[str, list[Proc]] = {}
    for proc in procs:
        if proc.tty == "??":
            continue
        by_tty.setdefault(proc.tty, []).append(proc)

    state = load_state()
    state_sessions = state.setdefault("sessions", {})
    touched_keys: set[str] = set()

    sessions: list[dict[str, Any]] = []
    for tty, tty_procs in sorted(by_tty.items()):
        tty_procs.sort(key=lambda p: p.pid)
        source = detect_source(tty_procs, pid_map)
        if source_filter != "all" and source != source_filter:
            continue

        primary, kind = select_primary_process(tty_procs)
        if not primary:
            continue

        status = "idle"
        agent: str | None = None
        waiting_minutes: float | None = None

        if kind == "ai":
            agent = detect_ai_tool(primary.command) or "ai"
            key = tty
            touched_keys.add(key)
            prev = state_sessions.get(key)

            same_proc = (
                isinstance(prev, dict)
                and prev.get("pid") == primary.pid
                and prev.get("command") == primary.command
            )

            delta_cpu = None
            if same_proc:
                try:
                    delta_cpu = primary.cpu_seconds - float(prev.get("cpu_seconds", 0.0))
                except Exception:
                    delta_cpu = None

            is_active = (
                primary.pcpu >= 1.0
                or primary.stat.startswith("R")
                or (delta_cpu is not None and delta_cpu >= 0.8)
            )

            if is_active:
                status = "active"
                waiting_since = None
            else:
                status = "waiting"
                waiting_since = prev.get("waiting_since") if same_proc else None
                if not waiting_since:
                    waiting_since = now_iso()

            state_sessions[key] = {
                "pid": primary.pid,
                "command": primary.command,
                "agent": agent,
                "cpu_seconds": primary.cpu_seconds,
                "waiting_since": waiting_since,
                "last_seen": now_iso(),
            }

            if waiting_since:
                try:
                    started = datetime.fromisoformat(waiting_since)
                    waiting_minutes = max(0.0, (datetime.now(timezone.utc) - started).total_seconds() / 60.0)
                except Exception:
                    waiting_minutes = None

        elif kind == "task":
            if primary.pcpu >= 1.0 or primary.stat.startswith("R"):
                status = "active"
            else:
                status = "background"

        if not include_idle and status == "idle":
            continue

        sessions.append(
            {
                "tty": tty,
                "source": source,
                "kind": kind,
                "agent": agent,
                "status": status,
                "pid": primary.pid,
                "pcpu": round(primary.pcpu, 2),
                "cpu_seconds": round(primary.cpu_seconds, 2),
                "runtime": fmt_seconds(primary.cpu_seconds),
                "waiting_minutes": round(waiting_minutes, 1) if waiting_minutes is not None else None,
                "command": primary.command,
            }
        )

    # Clean stale state entries.
    stale_keys = [key for key in state_sessions.keys() if key not in touched_keys]
    for key in stale_keys:
        state_sessions.pop(key, None)

    save_state(state)

    status_counts = {"active": 0, "waiting": 0, "idle": 0, "background": 0}
    ai_count = 0
    by_agent: dict[str, dict[str, int]] = {}
    for session in sessions:
        status_counts[session["status"]] = status_counts.get(session["status"], 0) + 1
        if session["kind"] == "ai":
            ai_count += 1
            agent_name = str(session.get("agent") or "ai")
            stats = by_agent.setdefault(agent_name, {"total": 0, "active": 0, "waiting": 0, "background": 0, "idle": 0})
            stats["total"] += 1
            status = str(session.get("status") or "")
            if status in stats:
                stats[status] += 1

    summary = {
        "total": len(sessions),
        "active": status_counts.get("active", 0),
        "waiting": status_counts.get("waiting", 0),
        "idle": status_counts.get("idle", 0),
        "background": status_counts.get("background", 0),
        "ai": ai_count,
        "by_agent": by_agent,
    }

    sessions.sort(key=lambda s: ({"waiting": 0, "active": 1, "background": 2, "idle": 3}.get(s["status"], 9), s["tty"]))

    return {"generated_at": now_iso(), "sessions": sessions, "summary": summary}, []



def shorten(text: str, length: int = 72) -> str:
    return text if len(text) <= length else text[: length - 3] + "..."



def print_snapshot(snapshot: dict[str, Any], warnings: list[str]) -> None:
    print(f"[{snapshot['generated_at']}] Terminal Sentinel")
    if warnings:
        for warning in warnings:
            print(f"WARN: {warning}")

    summary = snapshot["summary"]
    print(
        "Summary: total={total} ai={ai} waiting={waiting} active={active} background={background} idle={idle}".format(
            **summary
        )
    )
    by_agent = summary.get("by_agent", {})
    if by_agent:
        agent_chunks: list[str] = []
        for name in sorted(by_agent.keys()):
            info = by_agent[name]
            agent_chunks.append(
                f"{name}:{info.get('total', 0)}(w{info.get('waiting', 0)}/a{info.get('active', 0)})"
            )
        print("Agents: " + ", ".join(agent_chunks))

    if not snapshot["sessions"]:
        print("No matching terminal sessions found.")
        return

    print("TTY       SOURCE   KIND  AGENT   STATUS     WAIT   CPU%   PID     COMMAND")
    for session in snapshot["sessions"]:
        wait = "-"
        if session["waiting_minutes"] is not None:
            wait = f"{int(session['waiting_minutes'])}m"
        agent = str(session.get("agent") or "-")
        line = (
            f"{session['tty']:<9} {session['source']:<8} {session['kind']:<5} {agent:<7} {session['status']:<10} "
            f"{wait:<6} {session['pcpu']:<5} {session['pid']:<7} {shorten(session['command'])}"
        )
        print(line)



def write_snapshot(snapshot: dict[str, Any], prefix: str = "terminals") -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = REPORTS_DIR / f"{prefix}-{ts}.json"
    out_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return out_path



def run_summary(args: argparse.Namespace) -> int:
    snapshot, warnings = classify_sessions(source_filter=args.source, include_idle=not args.no_idle)
    if args.json:
        print(json.dumps({"warnings": warnings, **snapshot}, indent=2))
    else:
        print_snapshot(snapshot, warnings)

    if args.write_report:
        out_path = write_snapshot({"warnings": warnings, **snapshot}, prefix="terminals")
        print(f"Wrote report: {out_path}")

    return 0



def run_watch(args: argparse.Namespace) -> int:
    interval = max(2, args.interval)
    loops = 0
    try:
        while True:
            snapshot, warnings = classify_sessions(source_filter=args.source, include_idle=not args.no_idle)
            print("\033[2J\033[H", end="")
            print_snapshot(snapshot, warnings)
            loops += 1
            if args.iterations > 0 and loops >= args.iterations:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        return 0
    return 0



def recommend_actions(snapshot: dict[str, Any], max_ai_sessions: int, wait_limit_minutes: int) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    ai_sessions = [s for s in snapshot["sessions"] if s["kind"] == "ai"]

    long_waiting = [
        session
        for session in ai_sessions
        if session["status"] == "waiting"
        and session.get("waiting_minutes") is not None
        and session["waiting_minutes"] >= wait_limit_minutes
    ]
    for session in long_waiting:
        actions.append(
            {
                "type": "stop",
                "pid": session["pid"],
                "tty": session["tty"],
                "reason": f"waiting {int(session['waiting_minutes'])}m (limit {wait_limit_minutes}m)",
            }
        )

    if len(ai_sessions) > max_ai_sessions:
        overflow = len(ai_sessions) - max_ai_sessions
        candidates = sorted(
            ai_sessions,
            key=lambda s: (
                0 if s["status"] == "waiting" else 1,
                -(s.get("waiting_minutes") or 0),
                s["cpu_seconds"],
            ),
            reverse=False,
        )
        for session in candidates[:overflow]:
            actions.append(
                {
                    "type": "stop",
                    "pid": session["pid"],
                    "tty": session["tty"],
                    "reason": f"ai sessions {len(ai_sessions)} > max {max_ai_sessions}",
                }
            )

    # De-duplicate by PID.
    unique: dict[int, dict[str, Any]] = {}
    for action in actions:
        unique[action["pid"]] = action
    return list(unique.values())



def apply_actions(actions: list[dict[str, Any]], dry_run: bool) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for action in actions:
        pid = action["pid"]
        if dry_run:
            results.append({**action, "applied": False, "result": "dry-run"})
            continue

        code, _, err = run_cmd(["kill", "-TERM", str(pid)], timeout=3)
        if code == 0:
            results.append({**action, "applied": True, "result": "sent SIGTERM"})
        else:
            results.append({**action, "applied": True, "result": f"failed: {err.strip() or 'unknown'}"})
    return results



def run_overnight(args: argparse.Namespace) -> int:
    interval = max(30, args.interval)
    loops = 0
    log_path: Path | None = None

    if args.log_file:
        log_path = Path(args.log_file).expanduser()
    elif args.write_log:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = REPORTS_DIR / f"overnight-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"

    try:
        while True:
            snapshot, warnings = classify_sessions(source_filter=args.source, include_idle=False)
            actions = recommend_actions(
                snapshot=snapshot,
                max_ai_sessions=args.max_ai_sessions,
                wait_limit_minutes=args.kill_waiting_ai_after,
            )
            results = apply_actions(actions, dry_run=(not args.apply))

            print(f"[{snapshot['generated_at']}] overnight summary")
            if warnings:
                for warning in warnings:
                    print(f"WARN: {warning}")

            summary = snapshot["summary"]
            print(
                f"ai={summary['ai']} waiting={summary['waiting']} active={summary['active']} actions={len(results)}"
            )
            for result in results:
                mode = "APPLY" if result["applied"] else "PLAN"
                print(
                    f"- {mode} pid={result['pid']} tty={result['tty']} reason={result['reason']} => {result['result']}"
                )

            if log_path:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"snapshot": snapshot, "warnings": warnings, "actions": results}) + "\n")

            loops += 1
            if args.iterations > 0 and loops >= args.iterations:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        pass

    if log_path:
        print(f"Log file: {log_path}")
    return 0



def main() -> int:
    parser = argparse.ArgumentParser(description="Ghostty terminal sentinel")
    sub = parser.add_subparsers(dest="command", required=True)

    summary = sub.add_parser("summary", help="One-shot terminal status snapshot")
    summary.add_argument("--source", choices=["ghostty", "all", "codex", "iterm", "terminal"], default="ghostty")
    summary.add_argument("--json", action="store_true", help="Emit JSON output")
    summary.add_argument("--no-idle", action="store_true", help="Hide idle shell sessions")
    summary.add_argument("--write-report", action="store_true", help="Write JSON report into reports/")

    watch = sub.add_parser("watch", help="Continuously monitor terminal sessions")
    watch.add_argument("--source", choices=["ghostty", "all", "codex", "iterm", "terminal"], default="ghostty")
    watch.add_argument("--interval", type=int, default=10, help="Refresh interval (seconds)")
    watch.add_argument("--iterations", type=int, default=0, help="Stop after N loops (0 = infinite)")
    watch.add_argument("--no-idle", action="store_true", help="Hide idle shell sessions")

    overnight = sub.add_parser("overnight", help="Guardrail mode for long unattended runs")
    overnight.add_argument("--source", choices=["ghostty", "all", "codex", "iterm", "terminal"], default="ghostty")
    overnight.add_argument("--interval", type=int, default=300, help="Check interval in seconds")
    overnight.add_argument("--iterations", type=int, default=0, help="Stop after N loops (0 = infinite)")
    overnight.add_argument("--max-ai-sessions", type=int, default=4, help="Max concurrent AI sessions")
    overnight.add_argument(
        "--kill-waiting-ai-after",
        type=int,
        default=120,
        help="Stop AI sessions waiting longer than this many minutes",
    )
    overnight.add_argument("--apply", action="store_true", help="Apply actions (default is dry-run)")
    overnight.add_argument("--write-log", action="store_true", help="Write NDJSON log into reports/")
    overnight.add_argument("--log-file", help="Custom log file path")

    args = parser.parse_args()

    if args.command == "summary":
        return run_summary(args)
    if args.command == "watch":
        return run_watch(args)
    if args.command == "overnight":
        return run_overnight(args)
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    sys.exit(main())
