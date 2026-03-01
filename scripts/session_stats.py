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


def load_stats() -> dict[str, Any]:
    """Load stats from disk."""
    STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if STATS_FILE.exists():
        try:
            return json.loads(STATS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"sessions": {}, "last_poll": 0.0}


def save_stats(data: dict[str, Any]) -> None:
    """Persist stats to disk."""
    STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def cheap_stats_for_pane(pane_id: str, tmux_fn) -> dict[str, Any]:
    """Gather cheap stats from tmux without interrupting the session.

    tmux_fn should be the tmux() function from tmux_teams.py that takes
    a list of args and returns (code, stdout, stderr).
    """
    stats: dict[str, Any] = {}

    # Scrollback size (rough proxy for output volume / context usage)
    code, out, _ = tmux_fn(["capture-pane", "-t", pane_id, "-p", "-S", "-"], timeout=5)
    if code == 0:
        stats["scrollback_lines"] = len(out.splitlines())
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

    Only call on panes in a 'waiting' state (at the Claude prompt).
    Returns parsed usage data or None if capture failed.
    """
    # Send /usage command
    tmux_fn(["send-keys", "-t", pane_id, "/usage", "Enter"], timeout=5)
    # Wait for output
    time.sleep(2)
    # Capture recent output
    code, out, _ = tmux_fn(["capture-pane", "-t", pane_id, "-p", "-S", "-30"], timeout=5)
    if code != 0:
        return None

    usage: dict[str, Any] = {}
    for line in out.splitlines():
        line_s = line.strip()
        line_lower = line_s.lower()
        # Parse common /usage output patterns
        if "cost" in line_lower and "$" in line_s:
            usage["cost_line"] = line_s
        if "token" in line_lower:
            usage["token_line"] = line_s
        if "context" in line_lower and ("%" in line_s or "/" in line_s):
            usage["context_line"] = line_s

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
    cheap: dict[str, Any] | None,
    usage: dict[str, Any] | None,
    stats_data: dict[str, Any],
) -> None:
    """Merge cheap + usage stats into the stats store."""
    sessions = stats_data.setdefault("sessions", {})
    entry = sessions.get(project_id, {})
    entry["pane_id"] = pane_id
    entry["updated_at"] = time.time()
    if cheap:
        entry.update(cheap)
    if usage:
        entry["usage"] = usage
        stats_data["last_poll"] = time.time()
    sessions[project_id] = entry


def format_stats_summary(rows: list[dict[str, Any]], stats_data: dict[str, Any]) -> str:
    """Format a compact one-line stats summary for the grid header."""
    sessions_data = stats_data.get("sessions", {})
    total_kb = sum(
        sessions_data.get(str(r.get("project_id") or ""), {}).get("scrollback_kb", 0)
        for r in rows
    )
    claude_count = sum(
        1 for r in rows
        if str(r.get("ai_tool") or r.get("agent") or "") == "claude"
    )

    parts = [f"Agents: {claude_count}", f"Output: {total_kb:.0f}kb"]

    # Find most recent /usage data
    latest_ctx = ""
    latest_time = 0.0
    for sid, entry in sessions_data.items():
        usage = entry.get("usage", {})
        polled = usage.get("polled_at", 0.0)
        if polled > latest_time:
            ctx = usage.get("context_line", "")
            if ctx:
                latest_ctx = f"{sid}: {ctx}"
                latest_time = polled

    if latest_ctx:
        parts.append(latest_ctx)

    return "  |  ".join(parts)
