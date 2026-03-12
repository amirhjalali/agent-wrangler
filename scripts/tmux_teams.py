#!/usr/bin/env python3
"""Tmux team-grid orchestration for multi-repo agent sessions."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import terminal_sentinel
import workflow_agent

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "team_grid.json"
PERSISTENCE_DIR = ROOT / ".state" / "persistence"

DEFAULT_SESSION = "amir-grid"
DEFAULT_LAYOUT = "auto"
VALID_LAYOUTS = {"tiled", "even-horizontal", "even-vertical", "main-horizontal", "main-vertical"}
LAYOUT_CHOICES = sorted(VALID_LAYOUTS | {"auto"})
DEFAULT_LIMIT = 6
ERROR_MARKERS = (
    "elifecycle",
    "command failed",
    "exited (1)",
    "npm err!",
    "traceback",
    "exception",
    "fatal",
    "zsh: command not found",
    "err_pnpm_recursive_run_first_fail",
)
MISSING_COMMAND_PATTERN = re.compile(r"command not found:\s*([a-z0-9._/+:-]+)")
PORT_IN_USE_PATTERN = re.compile(r"port\s+(\d+)\s+is\s+in\s+use")
HOOK_EVENTS = ("after-split-window", "after-new-window", "client-session-changed")


@dataclass
class TmuxPane:
    pane_id: str
    pane_index: int
    pane_active: bool
    project_id: str
    pane_title: str
    pane_command: str
    pane_pid: int
    pane_tty: str
    pane_path: str


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run(cmd: list[str], timeout: int = 10) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError) as exc:
        return 1, "", str(exc)
    return proc.returncode, proc.stdout, proc.stderr


def tmux(args: list[str], timeout: int = 10) -> tuple[int, str, str]:
    for attempt in range(2):
        code, out, err = run(["tmux", *args], timeout=timeout)
        if code == 0:
            return code, out, err
        if "server exited unexpectedly" in (err or "").lower() and attempt == 0:
            time.sleep(0.15)
            continue
        return code, out, err
    return 1, "", "tmux command failed"


def ensure_tmux() -> None:
    if shutil.which("tmux"):
        return
    raise ValueError("tmux is not installed. Install with: brew install tmux")


def default_store() -> dict[str, Any]:
    return {
        "default_session": DEFAULT_SESSION,
        "default_layout": DEFAULT_LAYOUT,
        "default_projects": [],
        "persistence": {
            "enabled": False,
            "autosave_minutes": 15,
            "last_snapshot": "",
        },
        "profiles": {
            "current": "default",
            "items": {
                "default": {
                    "managed_sessions": [],
                    "max_panes": 10,
                }
            },
        },
        "updated_at": now_iso(),
    }


def _normalize_store(data: dict[str, Any] | None) -> dict[str, Any]:
    base = default_store()
    current = data if isinstance(data, dict) else {}
    merged = {
        **base,
        **current,
    }
    persistence = current.get("persistence", {}) if isinstance(current.get("persistence"), dict) else {}
    merged["persistence"] = {
        "enabled": bool(persistence.get("enabled", base.get("persistence", {}).get("enabled", False))),
        "autosave_minutes": int(persistence.get("autosave_minutes", base.get("persistence", {}).get("autosave_minutes", 15))),
        "last_snapshot": str(persistence.get("last_snapshot", base.get("persistence", {}).get("last_snapshot", ""))),
    }
    profiles = current.get("profiles", {}) if isinstance(current.get("profiles"), dict) else {}
    profile_items = profiles.get("items", {}) if isinstance(profiles.get("items"), dict) else {}
    normalized_items: dict[str, dict[str, Any]] = {}
    for key, value in profile_items.items():
        if not isinstance(value, dict):
            continue
        name = str(key).strip()
        if not name:
            continue
        managed_sessions = value.get("managed_sessions", [])
        if not isinstance(managed_sessions, list):
            managed_sessions = []
        normalized_items[name] = {
            "managed_sessions": [str(item).strip() for item in managed_sessions if str(item).strip()],
            "max_panes": int(value.get("max_panes", 10) or 10),
        }
    if "default" not in normalized_items:
        normalized_items["default"] = {"managed_sessions": [], "max_panes": 10}
    current_profile = str(profiles.get("current") or "default").strip() or "default"
    if current_profile not in normalized_items:
        current_profile = "default"
    merged["profiles"] = {
        "current": current_profile,
        "items": normalized_items,
    }
    return merged


def load_store() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return default_store()
    try:
        parsed = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return default_store()
    return _normalize_store(parsed)


def save_store(store: dict[str, Any]) -> None:
    normalized = _normalize_store(store)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    normalized["updated_at"] = now_iso()
    CONFIG_PATH.write_text(json.dumps(normalized, indent=2), encoding="utf-8")


def sanitize_snapshot_name(name: str) -> str:
    raw = (name or "").strip()
    if not raw:
        raw = "snapshot"
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", raw).strip("-.")
    if not safe:
        safe = "snapshot"
    if not safe.endswith(".json"):
        safe += ".json"
    return safe


def persistence_snapshot_path(name: str) -> Path:
    return PERSISTENCE_DIR / sanitize_snapshot_name(name)


def session_window_layout(session: str) -> str:
    code, out, err = tmux(["list-windows", "-t", f"{session}:0", "-F", "#{window_layout}"], timeout=5)
    if code != 0:
        raise ValueError(err.strip() or f"failed to read window layout for session '{session}'")
    first = out.splitlines()[0].strip() if out else ""
    return first or "tiled"


def tmux_resurrect_scripts() -> tuple[Path, Path]:
    base = Path.home() / ".tmux" / "plugins" / "tmux-resurrect" / "scripts"
    return (base / "save.sh", base / "restore.sh")


def project_map() -> dict[str, dict[str, Any]]:
    config = workflow_agent.load_config()
    projects = config.get("projects", [])
    return {project["id"]: project for project in projects}


def infer_project_id_from_path(path: str, proj_map: dict[str, dict[str, Any]]) -> str | None:
    path_norm = os.path.abspath(path or "")
    if not path_norm:
        return None
    matches: list[tuple[int, str]] = []
    for project_id, project in proj_map.items():
        project_path = str(project.get("path") or "")
        if not project_path:
            continue
        project_norm = os.path.abspath(project_path)
        if path_norm == project_norm or path_norm.startswith(project_norm + os.sep):
            matches.append((len(project_norm), project_id))
    if not matches:
        return None
    matches.sort(reverse=True)
    return matches[0][1]


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def choose_projects(
    projects_value: str | None,
    limit: int,
    group: str | None,
) -> list[str]:
    proj_map = project_map()
    explicit = split_csv(projects_value)
    if explicit:
        missing = [project_id for project_id in explicit if project_id not in proj_map]
        if missing:
            known = ", ".join(sorted(proj_map.keys()))
            raise ValueError(f"Unknown project id(s): {', '.join(missing)}. Known: {known}")
        return explicit

    selected: list[str] = []
    for project in workflow_agent.load_config().get("projects", []):
        if group and str(project.get("group", "")).lower() != group.lower():
            continue
        selected.append(project["id"])
        if len(selected) >= max(1, limit):
            break
    if not selected:
        raise ValueError("No projects selected. Pass --projects id1,id2 or use --group with matching projects.")
    return selected


def choose_layout(layout: str | None, pane_count: int) -> str:
    requested = (layout or DEFAULT_LAYOUT).strip().lower()
    if requested != "auto":
        if requested not in VALID_LAYOUTS:
            known = ", ".join(sorted(VALID_LAYOUTS))
            raise ValueError(f"Invalid layout '{requested}'. Use one of: auto, {known}")
        return requested

    if pane_count <= 2:
        return "even-horizontal"
    if pane_count <= 4:
        return "tiled"
    if pane_count <= 6:
        return "main-vertical"
    if pane_count <= 9:
        return "tiled"
    return "even-vertical"


def process_cwd(pid: int) -> str | None:
    if pid <= 0:
        return None
    code, out, _ = run(["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"], timeout=5)
    if code != 0 or not out:
        return None
    for line in out.splitlines():
        if line.startswith("n"):
            value = line[1:].strip()
            if value:
                return value
    return None


def infer_project_id_from_command(command: str, proj_map: dict[str, dict[str, Any]]) -> str | None:
    lower = command.lower()
    matches: list[tuple[int, str]] = []
    for project_id, project in proj_map.items():
        project_path = str(project.get("path") or "")
        if not project_path:
            continue
        norm = os.path.abspath(project_path).lower()
        if norm in lower:
            matches.append((len(norm), project_id))
    if not matches:
        return None
    matches.sort(reverse=True)
    return matches[0][1]


def infer_project_id_from_session(session: dict[str, Any], proj_map: dict[str, dict[str, Any]]) -> tuple[str | None, str | None]:
    pid = int(session.get("pid") or 0)
    cwd = process_cwd(pid)
    if cwd:
        project_id = infer_project_id_from_path(cwd, proj_map)
        if project_id:
            return project_id, cwd

    command = str(session.get("command") or "")
    project_id = infer_project_id_from_command(command, proj_map)
    return project_id, cwd


def session_exists(session: str) -> bool:
    code, _, _ = tmux(["has-session", "-t", session], timeout=5)
    return code == 0


def list_tmux_sessions() -> list[str]:
    code, out, err = tmux(["list-sessions", "-F", "#{session_name}"], timeout=5)
    if code != 0:
        msg = (err or "").strip().lower()
        if "no server running" in msg:
            return []
        return []
    sessions = [line.strip() for line in out.splitlines() if line.strip()]
    return sorted(set(sessions))



def pane_format() -> str:
    return (
        "#{pane_id}\t#{pane_index}\t#{pane_active}\t#{@project_id}\t#{pane_title}\t#{pane_current_command}\t"
        "#{pane_pid}\t#{pane_tty}\t#{pane_current_path}"
    )


def list_panes(session: str) -> list[TmuxPane]:
    code, out, err = tmux(["list-panes", "-t", f"{session}:0", "-F", pane_format()], timeout=10)
    if code != 0:
        detail = err.strip() or f"unable to list panes for session '{session}'"
        raise ValueError(detail)

    panes: list[TmuxPane] = []
    for raw_line in out.splitlines():
        parts = raw_line.split("\t")
        if len(parts) < 9:
            continue
        pane_id = parts[0]
        try:
            pane_index = int(parts[1])
        except ValueError:
            pane_index = 0
        pane_active = parts[2] == "1"
        project_id = parts[3]
        pane_title = parts[4]
        pane_command = parts[5]
        try:
            pane_pid = int(parts[6])
        except ValueError:
            pane_pid = 0
        pane_tty = parts[7]
        pane_path = parts[8]
        panes.append(
            TmuxPane(
                pane_id=pane_id,
                pane_index=pane_index,
                pane_active=pane_active,
                project_id=project_id,
                pane_title=pane_title,
                pane_command=pane_command,
                pane_pid=pane_pid,
                pane_tty=pane_tty,
                pane_path=pane_path,
            )
        )

    panes.sort(key=lambda pane: pane.pane_index)
    return panes


def session_monitor_by_tty() -> dict[str, dict[str, Any]]:
    snapshot, _ = terminal_sentinel.classify_sessions(source_filter="all", include_idle=True)
    return {str(item.get("tty")): item for item in snapshot.get("sessions", [])}


def capture_pane_text(pane_id: str, lines: int) -> str:
    code, out, _ = tmux(["capture-pane", "-p", "-t", pane_id, "-S", f"-{max(5, lines)}"], timeout=8)
    if code != 0:
        return ""
    return out.lower()


def capture_pane_raw(pane_id: str, lines: int) -> str:
    """Capture pane text without lowercasing (needed for status bar parsing)."""
    code, out, _ = tmux(["capture-pane", "-p", "-t", pane_id, "-S", f"-{max(5, lines)}"], timeout=8)
    if code != 0:
        return ""
    return out


def batch_set_pane_options(pane_id: str, options: list[tuple[str, str]]) -> None:
    """Set multiple pane options in a single tmux command using \\; separators."""
    if not options:
        return
    args: list[str] = []
    for key, value in options:
        if args:
            args.append(";")
        args.extend(["set-option", "-p", "-t", pane_id, key, value])
    tmux(args, timeout=8)


# ── Claude Code status bar parser ──────────────────────────────────
# Parses the native status line rendered at the bottom of Claude Code sessions:
#   Opus 4.6 | ●●●●○○○○○○ 86k/200k (42%) | ~$2.93
#   5hr ○○○○○○○○○○ 2% in 3h 33m | 7d ●●●○○○○○○○ 39% in 1d 15h | extra $0.00/$50

_CC_MODEL_RE = re.compile(
    r"((?:Opus|Sonnet|Haiku)\s+[\d.]+)"
    r"\s*\|.*?(\d+)k/(\d+)k\s*\((\d+)%\)"
    r"\s*\|.*?~?\$?([\d.]+)"
)
_CC_RATE_RE = re.compile(r"(\d+(?:hr|d))\s+[●○]+\s+(\d+)%")


def _parse_claude_status_from_text(raw: str) -> dict[str, Any] | None:
    """Parse Claude Code status bar from already-captured pane text."""
    if not raw:
        return None

    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None

    # Search the last 8 non-empty lines for the model line
    tail = lines[-8:]
    result: dict[str, Any] = {}

    for line in tail:
        m = _CC_MODEL_RE.search(line)
        if m:
            result["model"] = m.group(1)
            result["tokens_k"] = int(m.group(2))
            result["tokens_max_k"] = int(m.group(3))
            result["context_pct"] = int(m.group(4))
            result["cost"] = float(m.group(5))

        for rm in _CC_RATE_RE.finditer(line):
            window = rm.group(1)
            pct = int(rm.group(2))
            result[f"rate_{window}"] = pct

    return result if result else None


def parse_claude_status(pane_id: str) -> dict[str, Any] | None:
    """Scrape Claude Code's status bar from the bottom of a pane."""
    raw = capture_pane_raw(pane_id, lines=8)
    return _parse_claude_status_from_text(raw)


def detect_error_marker(text: str) -> str | None:
    if not text:
        return None
    for marker in ERROR_MARKERS:
        if marker in text:
            return marker
    return None


def detect_missing_command(text: str) -> str | None:
    if not text:
        return None
    match = MISSING_COMMAND_PATTERN.search(text)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def detect_port_in_use(text: str) -> str | None:
    if not text:
        return None
    match = PORT_IN_USE_PATTERN.search(text)
    if not match:
        return None
    return match.group(1).strip() or None


def pane_health_level(
    monitor: dict[str, Any],
    error_marker: str | None,
    wait_attention_min: int,
) -> tuple[str, bool, str]:
    status = str(monitor.get("status") or "idle")
    agent = str(monitor.get("agent") or "")
    wait = monitor.get("waiting_minutes")

    if error_marker:
        return "red", True, f"error: {error_marker}"

    if status == "waiting" and agent:
        if wait is None:
            return "red", True, "waiting"
        if float(wait) >= float(wait_attention_min):
            return "red", True, f"waiting {int(wait)}m"
        return "yellow", False, f"waiting {int(wait)}m"

    if status == "background":
        return "yellow", False, "background"
    if status in {"active", "idle"}:
        return "green", False, status
    return "yellow", False, status


def style_for_level(level: str) -> tuple[str, str]:
    """Return (inactive_border_style, active_border_style) for a health level.

    Health is encoded via border color (hue). Active pane gets a bright white
    border so it's unmistakably distinct regardless of health color.
    """
    if level == "red":
        return "fg=colour88", "fg=colour255,bold"
    if level == "yellow":
        return "fg=colour136", "fg=colour255,bold"
    return "fg=colour22", "fg=colour255,bold"


def set_window_orchestrator_format(session: str) -> None:
    """Apply pane border format, indicators, and inactive dimming.

    All calls are idempotent set-option — safe to call on every refresh.
    """
    target = f"{session}:0"
    tmux(["set-option", "-w", "-t", target, "pane-border-status", "top"], timeout=5)
    tmux(["set-option", "-w", "-t", target, "pane-border-lines", "single"], timeout=5)
    tmux(["set-option", "-w", "-t", target, "pane-border-indicators", "both"], timeout=5)
    tmux(["set-option", "-w", "-t", target, "window-style", "fg=colour245,bg=colour234"], timeout=5)
    tmux(["set-option", "-w", "-t", target, "window-active-style", "fg=default,bg=default"], timeout=5)
    # Border format: active marker (▶) + health symbol (● ⚑ ✖) + project info
    fmt = (
        "#{?pane_active,#[fg=colour255 bold]▶ ,  }"
        "#{?#{==:#{@health},RED},#[fg=colour196]✖ ,"
        "#{?#{==:#{@health},YELLOW},#[fg=colour220]⚑ ,"
        "#[fg=colour34]● }}"
        "#[fg=colour250]#{@project_id}"
        "#{?@health_reason, #[fg=colour246](#{@health_reason}),}"
        "#[default]"
    )
    tmux(["set-option", "-w", "-t", target, "pane-border-format", fmt], timeout=5)
    # Status bar: left shows session name, right shows zoom + time
    tmux(["set-option", "-t", session, "status-style", "bg=colour235,fg=colour250"], timeout=5)
    tmux(["set-option", "-t", session, "status-left",
          " #[fg=colour39 bold]AW#[default] #[fg=colour130]#{session_name}#[default] "], timeout=5)
    tmux(["set-option", "-t", session, "status-left-length", "30"], timeout=5)
    tmux(["set-option", "-t", session, "status-right",
          "#{?window_zoomed_flag,#[fg=colour214 bold] ZOOM #[default],} "
          "#[fg=colour250]%H:%M#[default] "], timeout=5)
    tmux(["set-option", "-t", session, "status-right-length", "30"], timeout=5)


_WINDOW_FORMAT_APPLIED: set[str] = set()


def refresh_pane_health(
    session: str,
    capture_lines: int,
    wait_attention_min: int,
    apply_colors: bool = True,
) -> list[dict[str, Any]]:
    panes = list_panes(session)
    by_tty = session_monitor_by_tty()
    rows: list[dict[str, Any]] = []

    # Window format only needs to be set once per session, not every refresh
    if apply_colors and session not in _WINDOW_FORMAT_APPLIED:
        set_window_orchestrator_format(session)
        _WINDOW_FORMAT_APPLIED.add(session)

    for pane in panes:
        tty_short = pane.pane_tty.split("/")[-1]
        monitor = by_tty.get(tty_short, {})
        agent = str(monitor.get("agent") or "-")
        status = str(monitor.get("status") or "idle")
        wait = monitor.get("waiting_minutes")

        # Single capture per pane — use raw text for both health detection and status bar
        raw_text = capture_pane_raw(pane.pane_id, lines=capture_lines)
        pane_text = raw_text.lower()
        error_marker = detect_error_marker(pane_text)
        missing_command = detect_missing_command(pane_text)
        port_in_use = detect_port_in_use(pane_text)
        level, needs_attention, reason = pane_health_level(
            monitor=monitor,
            error_marker=error_marker,
            wait_attention_min=wait_attention_min,
        )
        if reason.startswith("error: zsh: command not found") and missing_command:
            reason = f"missing command: {missing_command}"
        elif port_in_use and not reason.startswith("error:"):
            reason = f"port in use: {port_in_use}"

        # Parse Claude Code status bar from the already-captured raw text
        cc_stats = None
        if agent == "claude":
            cc_stats = _parse_claude_status_from_text(raw_text)

        if apply_colors:
            # Batch all pane options into a single tmux command
            border_style, active_style = style_for_level(level)
            batch_set_pane_options(pane.pane_id, [
                ("@agent", agent),
                ("@health", level.upper()),
                ("@health_reason", reason),
                ("@needs_attention", "1" if needs_attention else "0"),
                ("pane-border-style", border_style),
                ("pane-active-border-style", active_style),
            ])

        rows.append(
            {
                "pane_id": pane.pane_id,
                "index": pane.pane_index,
                "project_id": pane.project_id or "-",
                "title": pane.pane_title,
                "tty": tty_short,
                "agent": agent,
                "status": status,
                "wait": (int(wait) if wait is not None else None),
                "health": level,
                "needs_attention": needs_attention,
                "reason": reason,
                "error_marker": error_marker,
                "missing_command": missing_command,
                "port_in_use": port_in_use,
                "cc_stats": cc_stats,
            }
        )

    # Update status bar with aggregate stats
    if apply_colors and rows:
        counts = {"green": 0, "yellow": 0, "red": 0}
        agents = 0
        total_ctx = 0
        total_cost = 0.0
        ctx_count = 0
        for row in rows:
            lev = str(row.get("health") or "").lower()
            if lev in counts:
                counts[lev] += 1
            if row.get("agent") not in (None, "-", ""):
                agents += 1
            cc = row.get("cc_stats")
            if cc and cc.get("context_pct") is not None:
                total_ctx += int(cc["context_pct"])
                ctx_count += 1
                total_cost += float(cc.get("cost") or 0)
        avg_ctx = total_ctx // ctx_count if ctx_count else 0
        status_right = (
            f"#[fg=colour34]{counts['green']}g#[default] "
            f"#[fg=colour220]{counts['yellow']}y#[default] "
            f"#[fg=colour196]{counts['red']}r#[default]"
        )
        if agents:
            status_right += f" #[fg=colour75]|#[default] {agents} agents"
        if ctx_count:
            ctx_color = "colour196" if avg_ctx >= 80 else "colour220" if avg_ctx >= 50 else "colour34"
            status_right += f" #[fg={ctx_color}]{avg_ctx}%#[default]"
        if total_cost > 0:
            status_right += f" #[fg=colour245]${total_cost:.2f}#[default]"
        status_right += (
            " #{?window_zoomed_flag,#[fg=colour214 bold] ZOOM #[default],}"
            " #[fg=colour250]%H:%M#[default] "
        )
        tmux(["set-option", "-t", session, "status-right", status_right], timeout=3)
        tmux(["set-option", "-t", session, "status-right-length", "60"], timeout=3)

    # Desktop notifications on health state changes
    if apply_colors:
        check_and_notify(rows)

    return rows


def apply_layout(session: str, layout: str) -> None:
    if layout not in VALID_LAYOUTS:
        known = ", ".join(sorted(VALID_LAYOUTS))
        raise ValueError(f"Invalid layout '{layout}'. Use one of: {known}")
    code, _, err = tmux(["select-layout", "-t", f"{session}:0", layout], timeout=5)
    if code != 0:
        raise ValueError(err.strip() or f"failed to set layout '{layout}'")


def list_pane_sizes(session: str) -> list[tuple[str, int, int]]:
    code, out, err = tmux(
        ["list-panes", "-t", f"{session}:0", "-F", "#{pane_id}\t#{pane_width}\t#{pane_height}"],
        timeout=10,
    )
    if code != 0:
        raise ValueError(err.strip() or f"failed to inspect pane sizes for session '{session}'")

    rows: list[tuple[str, int, int]] = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        pane_id = parts[0]
        try:
            width = int(parts[1])
            height = int(parts[2])
        except ValueError:
            continue
        rows.append((pane_id, width, height))
    return rows


def split_for_path(session: str, path: str) -> None:
    sizes = list_pane_sizes(session)
    if not sizes:
        raise ValueError(f"no panes found in session '{session}'")

    target, width, height = max(sizes, key=lambda item: item[1] * item[2])
    orientation = "-h" if width >= height else "-v"

    code, _, err = tmux(["split-window", orientation, "-d", "-t", target, "-c", path], timeout=8)
    if code == 0:
        return

    other = "-v" if orientation == "-h" else "-h"
    code, _, err = tmux(["split-window", other, "-d", "-t", target, "-c", path], timeout=8)
    if code == 0:
        return

    code, _, err2 = tmux(["split-window", "-d", "-t", f"{session}:0", "-c", path], timeout=8)
    if code == 0:
        return
    detail = (err2 or err or "").strip()
    raise ValueError(detail or f"failed creating pane for path '{path}'")


def pane_target(session: str, token: str) -> TmuxPane:
    panes = list_panes(session)
    if not panes:
        raise ValueError("No panes found")

    value = token.strip()
    for pane in panes:
        if pane.pane_id == value:
            return pane

    if value.isdigit():
        index = int(value)
        for pane in panes:
            if pane.pane_index == index:
                return pane

    lower = value.lower()
    for pane in panes:
        if pane.project_id and pane.project_id.lower() == lower:
            return pane
    for pane in panes:
        if pane.pane_title.lower() == lower:
            return pane

    for pane in panes:
        if lower in pane.pane_title.lower() or (pane.project_id and lower in pane.project_id.lower()):
            return pane
    for pane in panes:
        if lower in pane.pane_path.lower():
            return pane

    raise ValueError(f"Pane not found: {token}")


def pane_send(pane_id: str, command: str, enter: bool = True) -> None:
    args = ["send-keys", "-t", pane_id, command]
    if enter:
        args.append("C-m")
    code, _, err = tmux(args, timeout=5)
    if code != 0:
        raise ValueError(err.strip() or f"failed to send keys to {pane_id}")


def pane_set_project_id(pane_id: str, project_id: str) -> None:
    code, _, err = tmux(["set-option", "-p", "-t", pane_id, "@project_id", project_id], timeout=5)
    if code != 0:
        raise ValueError(err.strip() or f"failed to set project id for {pane_id}")


def pane_ctrl_c(pane_id: str) -> None:
    code, _, err = tmux(["send-keys", "-t", pane_id, "C-c"], timeout=5)
    if code != 0:
        raise ValueError(err.strip() or f"failed to send Ctrl-C to {pane_id}")


# ── Hide / Show pane toggling ─────────────────────────────────────
HIDDEN_PREFIX = "_hid_"


def hide_pane(session: str, pane_id: str, project_id: str) -> str:
    """Move a pane to a hidden background window. Agent keeps running.

    Returns the hidden window name.
    """
    hidden_name = f"{HIDDEN_PREFIX}{project_id}"
    # break-pane moves the pane to its own new window, -d stays in current window
    code, _, err = tmux(["break-pane", "-d", "-s", pane_id, "-n", hidden_name], timeout=8)
    if code != 0:
        raise ValueError(err.strip() or f"failed to hide pane {pane_id}")
    # Rebalance the grid after removing a pane
    try:
        apply_layout(session, "tiled")
    except ValueError:
        pass  # May fail if grid is now empty
    return hidden_name


def show_pane(session: str, hidden_window: str) -> None:
    """Bring a hidden pane back to the grid window."""
    # join-pane moves the pane from hidden window into the grid window
    grid_target = f"{session}:grid"
    source = f"{hidden_window}.0"
    code, _, err = tmux(["join-pane", "-d", "-s", source, "-t", grid_target], timeout=8)
    if code != 0:
        raise ValueError(err.strip() or f"failed to show pane from {hidden_window}")
    # Rebalance
    try:
        apply_layout(session, "tiled")
    except ValueError:
        pass


def list_hidden_panes(session: str) -> list[dict[str, Any]]:
    """List all hidden panes (windows with _hid_ prefix) in the session."""
    code, out, _ = tmux(
        ["list-windows", "-t", session, "-F", "#{window_name}\t#{window_id}\t#{pane_id}\t#{pane_tty}\t#{pane_current_path}"],
        timeout=5,
    )
    if code != 0:
        return []

    hidden: list[dict[str, Any]] = []
    by_tty = session_monitor_by_tty()
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        name, window_id, pane_id, tty, path = parts[0], parts[1], parts[2], parts[3], parts[4]
        if not name.startswith(HIDDEN_PREFIX):
            continue
        project_id = name[len(HIDDEN_PREFIX):]
        tty_short = tty.split("/")[-1]
        monitor = by_tty.get(tty_short, {})
        hidden.append({
            "window_name": name,
            "window_id": window_id,
            "pane_id": pane_id,
            "project_id": project_id,
            "path": path,
            "agent": str(monitor.get("agent") or "-"),
            "status": str(monitor.get("status") or "idle"),
        })
    return hidden


def attach_session(session: str) -> int:
    proc = subprocess.run(["tmux", "attach-session", "-t", session], check=False)
    return int(proc.returncode)


def print_panes(session: str, panes: list[TmuxPane]) -> None:
    snapshot, _ = terminal_sentinel.classify_sessions(source_filter="all", include_idle=True)
    by_tty = {str(item.get("tty")): item for item in snapshot.get("sessions", [])}

    print(f"Session: {session}  panes={len(panes)}")
    print(
        f"{'IDX':<4} {'PANE':<6} {'PROJECT':<22} {'TITLE':<24} {'AGENT':<8} {'STATUS':<10} "
        f"{'CMD':<10} {'TTY':<10} PATH"
    )
    for pane in panes:
        tty_short = pane.pane_tty.split("/")[-1]
        monitor = by_tty.get(tty_short, {})
        agent = str(monitor.get("agent") or "-")
        status = str(monitor.get("status") or "-")
        marker = "*" if pane.pane_active else " "
        project = (pane.project_id or "-")[:22]
        title = pane.pane_title[:24]
        path = pane.pane_path
        print(
            f"{pane.pane_index:<4} {pane.pane_id:<6} {project:<22} {marker}{title:<23} {agent:<8} {status:<10} "
            f"{pane.pane_command:<10} {tty_short:<10} {path}"
        )


def backfill_pane_project_ids(session: str, panes: list[TmuxPane], proj_map: dict[str, dict[str, Any]]) -> list[TmuxPane]:
    changed = False
    for pane in panes:
        if pane.project_id:
            continue
        inferred = infer_project_id_from_path(pane.pane_path, proj_map)
        if not inferred:
            continue
        pane_set_project_id(pane.pane_id, inferred)
        pane.project_id = inferred
        changed = True
    if changed:
        return list_panes(session)
    return panes


def session_health_summary(
    *,
    session: str,
    capture_lines: int,
    wait_attention_min: int,
) -> dict[str, Any]:
    rows = refresh_pane_health(
        session=session,
        capture_lines=capture_lines,
        wait_attention_min=wait_attention_min,
    )
    counts = {
        "red": 0,
        "yellow": 0,
        "green": 0,
    }
    for row in rows:
        level = str(row.get("health") or "").lower()
        if level in counts:
            counts[level] += 1
    attention = len([row for row in rows if row.get("needs_attention")])
    waiting = len([row for row in rows if str(row.get("status")) == "waiting"])
    active = len([row for row in rows if str(row.get("status")) == "active"])
    top_reason = "-"
    for row in rows:
        if row.get("needs_attention"):
            top_reason = str(row.get("reason") or "-")
            break
    if top_reason == "-" and rows:
        top_reason = str(rows[0].get("reason") or "-")
    return {
        "session": session,
        "rows": rows,
        "panes": len(rows),
        "attention": attention,
        "waiting": waiting,
        "active": active,
        "red": counts["red"],
        "yellow": counts["yellow"],
        "green": counts["green"],
        "top_reason": top_reason,
        "ok": True,
    }


def git_project_snapshot(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    code, out, _ = run(["git", "-C", path, "rev-parse", "--is-inside-work-tree"], timeout=5)
    if code != 0 or out.strip() != "true":
        return None

    code, branch, _ = run(["git", "-C", path, "symbolic-ref", "--quiet", "--short", "HEAD"], timeout=5)
    if code != 0:
        code2, detached, _ = run(["git", "-C", path, "rev-parse", "--short", "HEAD"], timeout=5)
        branch_name = detached.strip() if code2 == 0 and detached.strip() else "detached"
    else:
        branch_name = branch.strip() or "-"

    code, status_out, _ = run(["git", "-C", path, "status", "--porcelain"], timeout=8)
    dirty = 0
    staged = 0
    unstaged = 0
    untracked = 0
    if code == 0 and status_out:
        for line in status_out.splitlines():
            if not line:
                continue
            dirty += 1
            if line.startswith("??"):
                untracked += 1
                continue
            if len(line) >= 2:
                if line[0] not in {" ", "?"}:
                    staged += 1
                if line[1] not in {" ", "?"}:
                    unstaged += 1

    ahead = 0
    behind = 0
    code, upstream, _ = run(["git", "-C", path, "rev-parse", "--abbrev-ref", "@{upstream}"], timeout=5)
    if code == 0 and upstream.strip():
        code2, counts, _ = run(
            ["git", "-C", path, "rev-list", "--left-right", "--count", f"{upstream.strip()}...HEAD"],
            timeout=5,
        )
        if code2 == 0:
            parts = counts.strip().split()
            if len(parts) == 2:
                try:
                    behind = int(parts[0])
                    ahead = int(parts[1])
                except ValueError:
                    ahead = 0
                    behind = 0

    return {
        "path": path,
        "branch": branch_name,
        "dirty": dirty,
        "staged": staged,
        "unstaged": unstaged,
        "untracked": untracked,
        "ahead": ahead,
        "behind": behind,
    }


def project_rows_for_session(session: str, proj_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    panes = backfill_pane_project_ids(session, list_panes(session), proj_map)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for pane in panes:
        project_id = pane.project_id or infer_project_id_from_path(pane.pane_path, proj_map) or "-"
        if project_id in seen:
            continue
        seen.add(project_id)
        project = proj_map.get(project_id, {})
        path = str(project.get("path") or pane.pane_path or "")
        git = git_project_snapshot(path)
        rows.append(
            {
                "project_id": project_id,
                "path": path,
                "git": git,
            }
        )

    rows.sort(key=lambda item: str(item.get("project_id") or ""))
    return rows


def create_grid_session(
    *,
    session: str,
    layout: str | None,
    project_ids: list[str],
    proj_map: dict[str, dict[str, Any]],
    project_overrides: dict[str, dict[str, Any]] | None,
    no_startup: bool,
    agent_default: str | None,
    agent_by_project: dict[str, str] | None,
    force: bool,
) -> tuple[str, list[TmuxPane]]:
    if session_exists(session):
        if force:
            code, _, err = tmux(["kill-session", "-t", session], timeout=5)
            if code != 0:
                raise ValueError(err.strip() or f"failed to replace existing session '{session}'")
        else:
            raise ValueError(f"Session '{session}' already exists. Use --force to replace it.")

    if not project_ids:
        raise ValueError("No projects to create panes for.")

    merged_map = dict(proj_map)
    if project_overrides:
        merged_map.update(project_overrides)

    first = merged_map.get(project_ids[0], {})
    first_path = str(first.get("path") or "")
    if not first_path:
        raise ValueError(f"Project '{project_ids[0]}' has no path")

    code, _, err = tmux(
        ["new-session", "-d", "-s", session, "-n", "grid", "-x", "260", "-y", "90", "-c", first_path],
        timeout=8,
    )
    if code != 0:
        raise ValueError(err.strip() or f"failed to create session '{session}'")

    for project_id in project_ids[1:]:
        project = merged_map.get(project_id, {})
        path = str(project.get("path") or "")
        if not path:
            raise ValueError(f"Project '{project_id}' has no path")
        split_for_path(session, path)
        # Rebalance as the grid grows to prevent "no space for new pane" on repeated splits.
        apply_layout(session, "tiled")

    resolved_layout = choose_layout(layout, pane_count=len(project_ids))
    apply_layout(session, resolved_layout)

    panes = list_panes(session)
    for idx, pane in enumerate(panes):
        project_id = project_ids[idx] if idx < len(project_ids) else f"pane-{idx}"
        project = merged_map.get(project_id, {})

        pane_set_project_id(pane.pane_id, project_id)
        code, _, err = tmux(["select-pane", "-t", pane.pane_id, "-T", project_id], timeout=5)
        if code != 0:
            raise ValueError(err.strip() or f"failed to set pane title for {pane.pane_id}")

        # Show project context without clearing — preserves shell state
        pane_send(pane.pane_id, f"echo '\\n[{project_id}] ready'", enter=True)

        startup_command = str(project.get("startup_command") or "").strip()
        if startup_command and not no_startup:
            pane_send(pane.pane_id, startup_command, enter=True)

        agent_cmd = None
        if agent_by_project and project_id in agent_by_project:
            agent_cmd = agent_by_project[project_id]
        elif agent_default and agent_default.strip():
            agent_cmd = agent_default.strip()

        if agent_cmd:
            pane_send(pane.pane_id, agent_cmd, enter=True)

    return resolved_layout, list_panes(session)


def ghostty_import_plan(
    *,
    proj_map: dict[str, dict[str, Any]],
    max_panes: int,
    include_idle: bool,
    preserve_duplicates: bool,
) -> tuple[list[str], dict[str, str], dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    snapshot, _ = terminal_sentinel.classify_sessions(source_filter="ghostty", include_idle=include_idle)
    sessions = list(snapshot.get("sessions", []))
    sessions.sort(
        key=lambda item: (
            {"active": 0, "waiting": 1, "background": 2, "idle": 3}.get(str(item.get("status")), 9),
            -(float(item.get("waiting_minutes") or 0.0)),
        )
    )

    project_ids: list[str] = []
    agent_by_project: dict[str, str] = {}
    project_overrides: dict[str, dict[str, Any]] = {}
    duplicate_counts: dict[str, int] = {}
    mapped: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []

    for session in sessions:
        # Skip background sessions — only import active/waiting terminals
        if str(session.get("status")) == "background":
            continue

        project_id, cwd = infer_project_id_from_session(session, proj_map)
        if not project_id:
            unmatched.append(
                {
                    "tty": session.get("tty"),
                    "pid": session.get("pid"),
                    "status": session.get("status"),
                    "agent": session.get("agent"),
                    "command": session.get("command"),
                    "cwd": cwd,
                }
            )
            continue

        mapped_project_id = project_id
        if preserve_duplicates:
            if len(project_ids) >= max(1, max_panes):
                continue
            seen = duplicate_counts.get(project_id, 0) + 1
            duplicate_counts[project_id] = seen
            if seen > 1:
                mapped_project_id = f"{project_id}__dup{seen}"
                project_overrides[mapped_project_id] = dict(proj_map.get(project_id, {}))
            project_ids.append(mapped_project_id)
        else:
            if project_id not in project_ids:
                if len(project_ids) >= max(1, max_panes):
                    continue
                project_ids.append(project_id)

        agent = str(session.get("agent") or "").strip().lower()
        if agent in {"claude", "codex", "aider", "gemini"} and mapped_project_id not in agent_by_project:
            agent_by_project[mapped_project_id] = agent

        mapped.append(
            {
                "project_id": project_id,
                "mapped_project_id": mapped_project_id,
                "tty": session.get("tty"),
                "pid": session.get("pid"),
                "status": session.get("status"),
                "agent": session.get("agent"),
                "cwd": cwd,
                "command": session.get("command"),
            }
        )

    return project_ids, agent_by_project, project_overrides, mapped, unmatched


def run_bootstrap(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()

    session = args.session or store.get("default_session") or DEFAULT_SESSION
    layout = args.layout or store.get("default_layout") or DEFAULT_LAYOUT

    project_ids = choose_projects(args.projects, limit=args.limit, group=args.group)
    proj_map = project_map()

    resolved_layout, _ = create_grid_session(
        session=session,
        layout=layout,
        project_ids=project_ids,
        proj_map=proj_map,
        project_overrides=None,
        no_startup=args.no_startup,
        agent_default=args.agent,
        agent_by_project=None,
        force=args.force,
    )

    store["default_session"] = session
    store["default_layout"] = resolved_layout
    store["default_projects"] = project_ids
    save_store(store)

    print(f"Created tmux team grid '{session}' with {len(project_ids)} panes")
    print("Projects: " + ", ".join(project_ids))
    print(f"Layout: {resolved_layout}")
    print(f"Attach: tmux attach -t {session}")

    if args.attach:
        return attach_session(session)
    return 0


def run_import(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    proj_map = project_map()

    session = args.session or store.get("default_session") or DEFAULT_SESSION
    layout = args.layout or store.get("default_layout") or DEFAULT_LAYOUT
    run_startup = bool(args.startup) and (not bool(args.no_startup))
    run_agents = bool(args.agent) and (not bool(args.no_agent))

    project_ids, agent_by_project, project_overrides, mapped, unmatched = ghostty_import_plan(
        proj_map=proj_map,
        max_panes=args.max_panes,
        include_idle=args.include_idle,
        preserve_duplicates=args.preserve_duplicates,
    )
    if not project_ids:
        raise ValueError("No Ghostty sessions matched known project paths. Keep current setup or pass explicit projects.")

    resolved_layout = choose_layout(layout, pane_count=len(project_ids))
    if args.dry_run:
        print("Ghostty import plan (dry-run)")
        print(f"Session: {session}")
        print(f"Panes: {len(project_ids)}  Layout: {resolved_layout}")
        print("Projects: " + ", ".join(project_ids))
        print(f"Preserve duplicates: {'yes' if args.preserve_duplicates else 'no'}")
        print(f"Will run startup commands: {'yes' if run_startup else 'no'}")
        print(f"Will launch detected agents: {'yes' if run_agents else 'no'}")
        if run_agents and agent_by_project:
            pairs = [f"{pid}:{agent_by_project[pid]}" for pid in project_ids if pid in agent_by_project]
            if pairs:
                print("Agent hints: " + ", ".join(pairs))
        if unmatched:
            print(f"Unmatched sessions: {len(unmatched)}")
        return 0

    resolved_layout, _ = create_grid_session(
        session=session,
        layout=layout,
        project_ids=project_ids,
        proj_map=proj_map,
        project_overrides=project_overrides,
        no_startup=(not run_startup),
        agent_default=None,
        agent_by_project=(agent_by_project if run_agents else None),
        force=args.force,
    )

    store["default_session"] = session
    store["default_layout"] = resolved_layout
    store["default_projects"] = project_ids
    save_store(store)

    print(f"Imported Ghostty sessions into tmux grid '{session}'")
    print(f"Panes: {len(project_ids)}  Layout: {resolved_layout}")
    print("Projects: " + ", ".join(project_ids))
    print(f"Preserve duplicates: {'enabled' if args.preserve_duplicates else 'disabled'}")
    print(f"Startup commands: {'enabled' if run_startup else 'disabled'}")
    print(f"Detected agents: {'enabled' if run_agents else 'disabled'}")
    if run_agents and agent_by_project:
        pairs = [f"{pid}:{agent_by_project[pid]}" for pid in project_ids if pid in agent_by_project]
        if pairs:
            print("Agent hints: " + ", ".join(pairs))

    print("")
    print("Mapped sessions:")
    for item in mapped[: max(1, args.max_panes)]:
        mapped_project_id = item.get("mapped_project_id") or item.get("project_id") or "-"
        mapped_text = (
            f"{item.get('project_id', '-')}" if mapped_project_id == item.get("project_id") else
            f"{item.get('project_id', '-')} -> {mapped_project_id}"
        )
        print(
            "- tty={tty} status={status} project={project} agent={agent} cwd={cwd}".format(
                tty=item.get("tty", "-"),
                status=item.get("status", "-"),
                project=mapped_text,
                agent=item.get("agent", "-"),
                cwd=item.get("cwd") or "-",
            )
        )

    if unmatched:
        print("")
        print(f"Unmatched sessions: {len(unmatched)} (left out)")

    print("")
    print(f"Attach: tmux attach -t {session}")
    print("You do not need to reset Ghostty first. Keep both until this grid feels stable, then close old tabs.")

    if args.attach:
        return attach_session(session)
    return 0


def run_up(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    proj_map = project_map()

    session = args.session or store.get("default_session") or DEFAULT_SESSION
    layout = args.layout or store.get("default_layout") or DEFAULT_LAYOUT
    run_startup = bool(args.startup) and (not bool(args.no_startup))
    run_agents = bool(args.agent) and (not bool(args.no_agent))

    exists = session_exists(session)
    if exists and not args.rebuild:
        print(f"Using existing session '{session}'")
    else:
        if args.mode == "bootstrap":
            project_ids = choose_projects(args.projects, limit=args.max_panes, group=args.group)
            agent_by_project = None
            project_overrides = None
        else:
            project_ids, detected_agents, project_overrides, mapped, unmatched = ghostty_import_plan(
                proj_map=proj_map,
                max_panes=args.max_panes,
                include_idle=args.include_idle,
                preserve_duplicates=args.preserve_duplicates,
            )
            agent_by_project = detected_agents if run_agents else None

            if not project_ids:
                if args.projects or args.group:
                    project_ids = choose_projects(args.projects, limit=args.max_panes, group=args.group)
                    agent_by_project = None
                    project_overrides = None
                    print("No Ghostty matches found; using configured project selection fallback.")
                else:
                    raise ValueError(
                        "No Ghostty sessions matched known project paths. Pass --mode bootstrap or provide --projects."
                    )
            else:
                print(f"Ghostty mapping: matched={len(mapped)} unmatched={len(unmatched)}")

        resolved_layout, _ = create_grid_session(
            session=session,
            layout=layout,
            project_ids=project_ids,
            proj_map=proj_map,
            project_overrides=project_overrides,
            no_startup=(not run_startup),
            agent_default=None,
            agent_by_project=agent_by_project,
            force=exists,
        )

        store["default_session"] = session
        store["default_layout"] = resolved_layout
        store["default_projects"] = project_ids
        save_store(store)

        print(f"Session ready: {session}")
        print(f"Panes: {len(project_ids)}  Layout: {resolved_layout}")

    if args.nav:
        run_nav(argparse.Namespace(remove=False))

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

    # Apply pane orchestrator format, status bar, and pane dimming
    set_window_orchestrator_format(session)

    show_status = bool(getattr(args, "status", True))
    attach = bool(getattr(args, "attach", True))

    if show_status:
        status_args = argparse.Namespace(session=session)
        run_status(status_args)

    if attach:
        return attach_session(session)
    return 0


def run_attach(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    session = args.session or store.get("default_session") or DEFAULT_SESSION
    if not session_exists(session):
        raise ValueError(f"Session '{session}' does not exist")
    return attach_session(session)


def run_status(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    session = args.session or store.get("default_session") or DEFAULT_SESSION
    if not session_exists(session):
        raise ValueError(f"Session '{session}' does not exist")
    proj_map = project_map()
    panes = backfill_pane_project_ids(session, list_panes(session), proj_map)
    print_panes(session, panes)
    return 0


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

        rows = refresh_pane_health(
            session=session,
            capture_lines=40,
            wait_attention_min=5,

        )

        lines: list[str] = []
        lines.append("\033[2J\033[H")  # clear screen
        lines.append("\033[1;36m STATUS RAIL\033[0m")
        lines.append("\033[2m" + "─" * 30 + "\033[0m")

        counts = {"green": 0, "yellow": 0, "red": 0, "total": 0}
        waiting = 0
        total_cost = 0.0
        total_tokens_k = 0
        claude_count = 0
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

            # Append Claude Code stats when available
            cc = row.get("cc_stats")
            if cc:
                claude_count += 1
                ctx = cc.get("context_pct")
                cost = cc.get("cost")
                if ctx is not None:
                    ctx_color = "\033[31m" if ctx >= 80 else "\033[33m" if ctx >= 50 else "\033[32m"
                    line += f"  {ctx_color}{ctx}%\033[0m"
                if cost is not None:
                    line += f"  \033[2m${cost:.2f}\033[0m"
                    total_cost += cost
                total_tokens_k += cc.get("tokens_k", 0)

            lines.append(line)

        lines.append("\033[2m" + "─" * 30 + "\033[0m")
        lines.append(f" {counts['total']} panes  {waiting} waiting")
        lines.append(
            f" \033[32m{counts['green']}g\033[0m "
            f"\033[33m{counts['yellow']}y\033[0m "
            f"\033[31m{counts['red']}r\033[0m"
        )
        if claude_count > 0:
            lines.append(
                f" \033[36m{claude_count} claude\033[0m  "
                f"\033[2m{total_tokens_k}k tok  ${total_cost:.2f}\033[0m"
            )

        # Show hidden panes
        hidden = list_hidden_panes(session)
        if hidden:
            lines.append("\033[2m" + "─" * 30 + "\033[0m")
            lines.append(f" \033[2m{len(hidden)} hidden\033[0m")
            for h in hidden:
                hp = str(h.get("project_id") or "?")
                if len(hp) > 16:
                    hp = hp[:15] + "~"
                ha = str(h.get("agent") or "")
                hs = str(h.get("status") or "")
                agent_label = f"  {ha}" if ha and ha != "-" else ""
                lines.append(f" \033[2m○ {hp:<16}{agent_label}  {hs}\033[0m")

        print("\n".join(lines), flush=True)

        try:
            _time.sleep(interval)
        except KeyboardInterrupt:
            break
    return 0


def run_paint(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    session = args.session or store.get("default_session") or DEFAULT_SESSION
    if not session_exists(session):
        raise ValueError(f"Session '{session}' does not exist")

    rows = refresh_pane_health(
        session=session,
        capture_lines=args.capture_lines,
        wait_attention_min=args.wait_attention_min,

    )
    attention = len([row for row in rows if row.get("needs_attention")])
    print(f"Painted session '{session}' panes={len(rows)} needs_attention={attention}")
    for row in rows:
        wait = f"{row['wait']}m" if row.get("wait") is not None else "-"
        mark = "!" if row.get("needs_attention") else " "
        cc = row.get("cc_stats") or {}
        ctx_str = f"ctx={cc['context_pct']}%" if cc.get("context_pct") is not None else ""
        cost_str = f"${cc['cost']:.2f}" if cc.get("cost") is not None else ""
        cc_label = f" {ctx_str} {cost_str}".rstrip() if (ctx_str or cost_str) else ""
        print(
            f"{mark} {row['index']:<2} {row['project_id']:<22} {row['health']:<6} {row['status']:<10} "
            f"wait={wait:<4} agent={row['agent']:<8} reason={row['reason']}{cc_label}"
        )
    return 0


def run_watch(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    session = args.session or store.get("default_session") or DEFAULT_SESSION
    if not session_exists(session):
        raise ValueError(f"Session '{session}' does not exist")

    loops = 0
    interval = max(1, int(args.interval))
    try:
        while True:
            rows = refresh_pane_health(
                session=session,
                capture_lines=args.capture_lines,
                wait_attention_min=args.wait_attention_min,
        
            )
            if not args.no_clear:
                print("\033[2J\033[H", end="")
            attention = len([row for row in rows if row.get("needs_attention")])
            print(f"[{now_iso()}] Agent Wrangler Manager  session={session} panes={len(rows)} attention={attention}")
            print(f"{'IDX':<4} {'PROJECT':<22} {'HLTH':<6} {'STATUS':<10} {'WAIT':<6} {'AGENT':<8} {'CTX':<6} {'COST':<8} REASON")
            for row in rows:
                wait = f"{row['wait']}m" if row.get("wait") is not None else "-"
                cc = row.get("cc_stats") or {}
                ctx_str = f"{cc['context_pct']}%" if cc.get("context_pct") is not None else "-"
                cost_str = f"${cc['cost']:.2f}" if cc.get("cost") is not None else "-"
                print(
                    f"{row['index']:<4} {row['project_id']:<22} {row['health']:<6} {row['status']:<10} "
                    f"{wait:<6} {row['agent']:<8} {ctx_str:<6} {cost_str:<8} {row['reason']}"
                )

            attention_rows = [row for row in rows if row.get("needs_attention") or str(row.get("health")) == "red"]
            if attention_rows:
                attention_rows.sort(
                    key=lambda row: (
                        0 if str(row.get("health")) == "red" else 1,
                        -(int(row.get("wait") or 0)),
                    )
                )
                print("")
                print("Top attention panes:")
                for row in attention_rows[:5]:
                    fix = ""
                    missing_command = str(row.get("missing_command") or "")
                    if missing_command:
                        if missing_command == "code":
                            fix = "fix: use `codex` or install VS Code `code` shell command"
                        else:
                            fix = f"fix: install `{missing_command}` or update startup_command"
                    elif row.get("port_in_use"):
                        fix = f"fix: free port {row['port_in_use']} or change PORT"
                    elif str(row.get("reason", "")).startswith("waiting"):
                        fix = f"fix: inspect pane: agent-wrangler capture {row['project_id']} --lines 60"
                    print(
                        "- {project} ({tty}) {reason}{fix}".format(
                            project=row.get("project_id", "-"),
                            tty=row.get("tty", "-"),
                            reason=row.get("reason", "-"),
                            fix=(f" | {fix}" if fix else ""),
                        )
                    )

            print("")
            print(
                "Navigation: Option+Arrow panes | Option+[ / Option+] windows | Option+1..9 window jump "
                "(fallback: Ctrl-b + arrows)"
            )
            print(
                "Control: agent-wrangler agent <project> claude|codex | "
                "agent-wrangler stop <project> | agent-wrangler shell <project>"
            )

            loops += 1
            if args.iterations > 0 and loops >= args.iterations:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        return 0
    return 0


def manager_window_exists(session: str, window_name: str) -> bool:
    code, out, _ = tmux(["list-windows", "-t", session, "-F", "#{window_name}"], timeout=5)
    if code != 0:
        return False
    names = {line.strip() for line in out.splitlines() if line.strip()}
    return window_name in names


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
            if getattr(args, "attach", False):
                return attach_session(session)
            return 0

    if not manager_window_exists(session, window):
        # Create manager window with Claude Code
        wrangler_root = str(ROOT)
        claude_cmd = "claude --dangerously-skip-permissions"
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
    if getattr(args, "attach", False):
        return attach_session(session)
    return 0



def run_profile_list(_: argparse.Namespace) -> int:
    store = load_store()
    profiles = store.get("profiles", {})
    current = str(profiles.get("current") or "default")
    items = profiles.get("items", {}) if isinstance(profiles.get("items"), dict) else {}

    print(f"Current profile: {current}")
    if not items:
        print("No profiles configured.")
        return 0

    print(f"{'PROFILE':<16} {'CURRENT':<7} {'MAX_PANES':<9} SESSIONS")
    for name in sorted(items.keys()):
        item = items.get(name, {})
        sessions = item.get("managed_sessions", []) if isinstance(item.get("managed_sessions"), list) else []
        max_panes = int(item.get("max_panes") or 10)
        mark = "yes" if name == current else "no"
        print(f"{name:<16} {mark:<7} {max_panes:<9} {', '.join(sessions) if sessions else '-'}")
    return 0


def run_profile_status(_: argparse.Namespace) -> int:
    store = load_store()
    profiles = store.get("profiles", {})
    current = str(profiles.get("current") or "default")
    items = profiles.get("items", {}) if isinstance(profiles.get("items"), dict) else {}
    item = items.get(current, {}) if isinstance(items.get(current), dict) else {}
    sessions = item.get("managed_sessions", []) if isinstance(item.get("managed_sessions"), list) else []
    max_panes = int(item.get("max_panes") or 10)

    print(f"Profile: {current}")
    print(f"- max_panes: {max_panes}")
    print(f"- managed_sessions: {', '.join(sessions) if sessions else '-'}")
    print(f"- start hint: AW_MAX_PANES={max_panes} ./scripts/agent-wrangler start")
    return 0


def run_profile_save(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    profiles = store.setdefault("profiles", {})
    items = profiles.setdefault("items", {})
    if not isinstance(items, dict):
        items = {}
        profiles["items"] = items

    name = str(args.name or "").strip().lower()
    if not name:
        raise ValueError("Profile name is required")

    sessions = split_csv(args.sessions)
    if not sessions and args.auto_running:
        sessions = [s for s in list_tmux_sessions() if s]

    item = items.setdefault(name, {})
    item["managed_sessions"] = sorted(set(sessions))
    item["max_panes"] = max(1, int(args.max_panes))
    save_store(store)

    print(f"Saved profile '{name}'")
    print(f"- max_panes: {item['max_panes']}")
    print(f"- managed_sessions: {', '.join(item['managed_sessions']) if item['managed_sessions'] else '-'}")
    return 0


def run_profile_use(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    profiles = store.setdefault("profiles", {})
    items = profiles.get("items", {}) if isinstance(profiles.get("items"), dict) else {}

    name = str(args.name or "").strip().lower()
    if name not in items:
        known = ", ".join(sorted(items.keys()))
        raise ValueError(f"Unknown profile '{name}'. Known: {known}")

    profiles["current"] = name
    item = items.get(name, {})
    save_store(store)

    print(f"Active profile: {name}")
    print(f"- max_panes: {int(item.get('max_panes') or 10)}")
    print(f"- start hint: AW_MAX_PANES={int(item.get('max_panes') or 10)} ./scripts/agent-wrangler start")
    return 0




def run_drift(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    proj_map = project_map()

    session = args.session or store.get("default_session") or DEFAULT_SESSION
    sessions = [session]

    total_projects = 0
    total_dirty = 0
    high_drift: list[tuple[str, str, int]] = []

    for session in sessions:
        if not session_exists(session):
            print(f"\n[{session}] missing")
            continue
        try:
            projects = project_rows_for_session(session, proj_map)
        except ValueError as exc:
            print(f"\n[{session}] error: {exc}")
            continue

        print(f"\n[{session}] project drift")
        print(f"{'PROJECT':<22} {'BRANCH':<18} {'DIRTY':<5} {'STG':<3} {'UNS':<3} {'UNT':<3} {'A':<3} {'B':<3} PATH")
        if not projects:
            print("(no projects)")
            continue

        for item in projects:
            total_projects += 1
            project_id = str(item.get("project_id") or "-")
            path = str(item.get("path") or "-")
            git = item.get("git")
            if not git:
                print(f"{project_id:<22} {'-':<18} {'-':<5} {'-':<3} {'-':<3} {'-':<3} {'-':<3} {'-':<3} {path}")
                continue
            dirty = int(git.get("dirty") or 0)
            total_dirty += dirty
            if dirty >= args.alert_dirty:
                high_drift.append((session, project_id, dirty))
            print(
                f"{project_id:<22} {str(git.get('branch') or '-'):<18} {dirty:<5} "
                f"{int(git.get('staged') or 0):<3} {int(git.get('unstaged') or 0):<3} {int(git.get('untracked') or 0):<3} "
                f"{int(git.get('ahead') or 0):<3} {int(git.get('behind') or 0):<3} {path}"
            )

    print("")
    print(
        "Drift totals: sessions={sessions} projects={projects} dirty_files={dirty}".format(
            sessions=len(sessions),
            projects=total_projects,
            dirty=total_dirty,
        )
    )
    if high_drift:
        print("High drift alerts (dirty >= {limit}):".format(limit=args.alert_dirty))
        for session, project_id, dirty in sorted(high_drift, key=lambda item: item[2], reverse=True):
            print(f"- {session}:{project_id} dirty={dirty}")
    return 0


def run_persistence_status(_: argparse.Namespace) -> int:
    store = load_store()
    persistence = store.get("persistence", {})
    enabled = bool(persistence.get("enabled"))
    autosave = int(persistence.get("autosave_minutes") or 15)
    last_snapshot = str(persistence.get("last_snapshot") or "")
    save_script, restore_script = tmux_resurrect_scripts()

    print("Persistence status")
    print(f"- enabled: {'yes' if enabled else 'no'}")
    print(f"- autosave_minutes: {autosave}")
    print(f"- last_snapshot: {last_snapshot or '-'}")
    print(f"- tmux-resurrect save script: {'yes' if save_script.exists() else 'no'} ({save_script})")
    print(f"- tmux-resurrect restore script: {'yes' if restore_script.exists() else 'no'} ({restore_script})")

    snapshots: list[Path] = []
    if PERSISTENCE_DIR.exists():
        snapshots = sorted(PERSISTENCE_DIR.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    print(f"- local snapshots: {len(snapshots)}")
    for path in snapshots[:10]:
        stamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat()
        print(f"  {path.name}  {stamp}")
    return 0


def run_persistence_enable(args: argparse.Namespace) -> int:
    store = load_store()
    persistence = store.setdefault("persistence", {})
    persistence["enabled"] = True
    persistence["autosave_minutes"] = max(1, int(args.autosave_minutes))
    save_store(store)
    print(
        "Persistence enabled (autosave_minutes={mins}).".format(
            mins=int(persistence.get("autosave_minutes") or 15)
        )
    )
    print("Tip: run `agent-wrangler persistence save` at key checkpoints.")
    return 0


def run_persistence_disable(_: argparse.Namespace) -> int:
    store = load_store()
    persistence = store.setdefault("persistence", {})
    persistence["enabled"] = False
    save_store(store)
    print("Persistence disabled.")
    return 0


def run_persistence_save(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    session = args.session or store.get("default_session") or DEFAULT_SESSION
    if not session_exists(session):
        raise ValueError(f"Session '{session}' does not exist")

    proj_map = project_map()
    panes = backfill_pane_project_ids(session, list_panes(session), proj_map)
    by_tty = session_monitor_by_tty()
    pane_rows: list[dict[str, Any]] = []
    for pane in panes:
        tty_short = pane.pane_tty.split("/")[-1]
        monitor = by_tty.get(tty_short, {})
        pane_rows.append(
            {
                "index": pane.pane_index,
                "project_id": pane.project_id or pane.pane_title or f"pane-{pane.pane_index}",
                "title": pane.pane_title,
                "path": pane.pane_path,
                "tty": tty_short,
                "agent": str(monitor.get("agent") or ""),
                "status": str(monitor.get("status") or ""),
            }
        )

    layout_hint = str(store.get("default_layout") or DEFAULT_LAYOUT)
    data = {
        "saved_at": now_iso(),
        "session": session,
        "layout": (layout_hint if layout_hint in LAYOUT_CHOICES else DEFAULT_LAYOUT),
        "pane_count": len(pane_rows),
        "panes": pane_rows,
    }

    if args.file:
        snapshot_path = Path(args.file).expanduser()
    else:
        snapshot_path = persistence_snapshot_path(args.name or session)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    persistence = store.setdefault("persistence", {})
    persistence["last_snapshot"] = str(snapshot_path)
    save_store(store)

    print(f"Saved persistence snapshot: {snapshot_path}")
    print(f"Session: {session}  panes={len(pane_rows)}")

    if args.tmux_resurrect:
        save_script, _ = tmux_resurrect_scripts()
        if not save_script.exists():
            raise ValueError(f"tmux-resurrect save script not found: {save_script}")
        code, _, err = run([str(save_script)], timeout=45)
        if code != 0:
            detail = (err or "").strip()
            raise ValueError(detail or "tmux-resurrect save failed")
        print("tmux-resurrect save completed.")
    return 0


def run_persistence_restore(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    if args.file:
        snapshot_path = Path(args.file).expanduser()
    elif args.name:
        snapshot_path = persistence_snapshot_path(args.name)
    else:
        remembered = str(store.get("persistence", {}).get("last_snapshot") or "")
        snapshot_path = Path(remembered).expanduser() if remembered else persistence_snapshot_path(
            store.get("default_session") or DEFAULT_SESSION
        )
    if not snapshot_path.exists():
        raise ValueError(f"snapshot file not found: {snapshot_path}")

    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"invalid snapshot file: {snapshot_path} ({exc})") from exc

    pane_items = payload.get("panes", [])
    if not isinstance(pane_items, list) or not pane_items:
        raise ValueError("snapshot has no panes")

    session = args.session or str(payload.get("session") or store.get("default_session") or DEFAULT_SESSION)
    proj_map = project_map()
    project_ids: list[str] = []
    project_overrides: dict[str, dict[str, Any]] = {}
    agent_by_project: dict[str, str] = {}
    seen: dict[str, int] = {}

    for idx, pane in enumerate(pane_items):
        if not isinstance(pane, dict):
            continue
        base_project_id = str(pane.get("project_id") or pane.get("title") or f"pane-{idx + 1}").strip()
        if not base_project_id or base_project_id == "-":
            base_project_id = f"pane-{idx + 1}"
        count = seen.get(base_project_id, 0) + 1
        seen[base_project_id] = count
        project_id = base_project_id if count == 1 else f"{base_project_id}__rest{count}"

        path = str(pane.get("path") or "").strip()
        if not path:
            path = str(proj_map.get(base_project_id, {}).get("path") or "").strip()
        if not path:
            continue

        base_project = dict(proj_map.get(base_project_id, {}))
        base_project["path"] = path
        base_project.setdefault("startup_command", "")
        project_overrides[project_id] = base_project
        project_ids.append(project_id)

        agent = str(pane.get("agent") or "").strip().lower()
        if agent in {"claude", "codex", "aider", "gemini"}:
            agent_by_project[project_id] = agent

    if not project_ids:
        raise ValueError("snapshot has no restorable pane paths")

    layout_hint = str(payload.get("layout") or DEFAULT_LAYOUT)
    layout = args.layout or (layout_hint if layout_hint in LAYOUT_CHOICES else DEFAULT_LAYOUT)
    resolved_layout, _ = create_grid_session(
        session=session,
        layout=layout,
        project_ids=project_ids,
        proj_map=proj_map,
        project_overrides=project_overrides,
        no_startup=(not args.startup),
        agent_default=None,
        agent_by_project=(agent_by_project if args.agent else None),
        force=args.force,
    )

    store["default_session"] = session
    store["default_layout"] = resolved_layout
    store["default_projects"] = project_ids
    persistence = store.setdefault("persistence", {})
    persistence["last_snapshot"] = str(snapshot_path)
    save_store(store)

    print(f"Restored snapshot into session '{session}'")
    print(f"Panes: {len(project_ids)}  Layout: {resolved_layout}")
    print(f"Source: {snapshot_path}")
    print(f"Startup commands: {'enabled' if args.startup else 'disabled'}")
    print(f"Agent relaunch: {'enabled' if args.agent else 'disabled'}")

    if args.tmux_resurrect:
        _, restore_script = tmux_resurrect_scripts()
        if not restore_script.exists():
            raise ValueError(f"tmux-resurrect restore script not found: {restore_script}")
        code, _, err = run([str(restore_script)], timeout=60)
        if code != 0:
            detail = (err or "").strip()
            raise ValueError(detail or "tmux-resurrect restore failed")
        print("tmux-resurrect restore completed.")

    if args.attach:
        return attach_session(session)
    return 0


def run_hooks_enable(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    session = args.session or store.get("default_session") or DEFAULT_SESSION
    if not session_exists(session):
        raise ValueError(f"Session '{session}' does not exist")

    script = Path(__file__).resolve()
    paint_cmd = (
        f"python3 {shlex.quote(str(script))} teams paint --session {shlex.quote(session)} "
        f"--capture-lines {max(20, int(args.capture_lines))} --wait-attention-min {max(0, int(args.wait_attention_min))} "
        "--no-colorize"
    )
    hook_cmd = f"run-shell -b {shlex.quote(paint_cmd + ' >/dev/null 2>&1')}"
    for name in HOOK_EVENTS:
        code, _, err = tmux(["set-hook", "-t", session, name, hook_cmd], timeout=5)
        if code != 0:
            raise ValueError(err.strip() or f"failed to set hook '{name}'")
    print(f"Enabled hooks for session '{session}'")
    for name in HOOK_EVENTS:
        print(f"- {name}")
    return 0


def run_hooks_disable(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    session = args.session or store.get("default_session") or DEFAULT_SESSION
    if not session_exists(session):
        raise ValueError(f"Session '{session}' does not exist")

    for name in HOOK_EVENTS:
        tmux(["set-hook", "-u", "-t", session, name], timeout=5)
    print(f"Disabled hooks for session '{session}'")
    return 0


def run_hooks_status(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    session = args.session or store.get("default_session") or DEFAULT_SESSION
    if not session_exists(session):
        raise ValueError(f"Session '{session}' does not exist")

    code, out, err = tmux(["show-hooks", "-t", session], timeout=5)
    if code != 0:
        raise ValueError(err.strip() or f"failed to read hooks for session '{session}'")

    found = 0
    print(f"Hooks status for '{session}'")
    for name in HOOK_EVENTS:
        matched = [line.strip() for line in out.splitlines() if line.strip().startswith(name)]
        if matched:
            found += 1
            print(f"- {name}: enabled")
        else:
            print(f"- {name}: disabled")
    print(f"Enabled hooks: {found}/{len(HOOK_EVENTS)}")
    return 0


def doctor_fix_for_row(row: dict[str, Any]) -> str:
    missing_command = str(row.get("missing_command") or "")
    if missing_command:
        if missing_command == "code":
            return "Use `codex` for OpenAI Codex CLI, or install VS Code shell command `code`."
        if missing_command in {"claude", "codex", "aider", "gemini"} and not shutil.which(missing_command):
            return f"Install `{missing_command}` and ensure it is on PATH."
        if missing_command == "pnpm" and not shutil.which("pnpm"):
            return "Install pnpm (`brew install pnpm`) or update startup_command to npm/bun."
        if missing_command == "bun" and not shutil.which("bun"):
            return "Install bun (`brew install oven-sh/bun/bun`) or switch startup_command."
        return f"Install `{missing_command}` or fix the startup command for this project."
    port = str(row.get("port_in_use") or "")
    if port:
        return f"Free port {port} (`lsof -i :{port}`) or run this app on a different port."
    reason = str(row.get("reason") or "")
    if reason.startswith("waiting"):
        return "Session is waiting for input; inspect prompt and continue/stop task."
    if str(row.get("error_marker") or "") in {"elifecycle", "command failed", "exited (1)", "err_pnpm_recursive_run_first_fail"}:
        return "Open pane logs and rerun startup command manually to verify package scripts."
    return "Inspect pane output and restart shell or command if needed."


def run_doctor(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()

    session = args.session or store.get("default_session") or DEFAULT_SESSION
    sessions = [session]

    print("Agent Wrangler Doctor")
    print("Tool availability:")
    for tool in ["claude", "codex", "aider", "gemini", "npm", "pnpm", "bun"]:
        ok = "yes" if shutil.which(tool) else "no"
        print(f"- {tool:<7} installed={ok}")

    total_findings = 0
    for session in sessions:
        print("")
        print(f"[{session}]")
        if not session_exists(session):
            print("- missing session")
            continue

        rows = refresh_pane_health(
            session=session,
            capture_lines=max(20, int(args.capture_lines)),
            wait_attention_min=max(0, int(args.wait_attention_min)),
    
        )
        if args.only_attention:
            rows = [row for row in rows if row.get("needs_attention") or str(row.get("health")) == "red"]

        findings: list[dict[str, Any]] = []
        for row in rows:
            if row.get("missing_command"):
                findings.append(row)
                continue
            if row.get("port_in_use"):
                findings.append(row)
                continue
            if str(row.get("health")) == "red":
                findings.append(row)
                continue
            if str(row.get("reason") or "").startswith("waiting") and int(row.get("wait") or 0) >= max(
                1, int(args.wait_attention_min)
            ):
                findings.append(row)

        if not findings:
            print("- no critical findings")
            continue

        findings.sort(
            key=lambda row: (
                0 if str(row.get("health")) == "red" else 1,
                -(int(row.get("wait") or 0)),
            )
        )
        total_findings += len(findings)
        for row in findings:
            wait = f"{row['wait']}m" if row.get("wait") is not None else "-"
            print(
                "! pane={index:<2} project={project:<22} health={health:<6} status={status:<10} wait={wait:<6} reason={reason}".format(
                    index=int(row.get("index") or 0),
                    project=str(row.get("project_id") or "-")[:22],
                    health=str(row.get("health") or "-"),
                    status=str(row.get("status") or "-"),
                    wait=wait,
                    reason=str(row.get("reason") or "-"),
                )
            )
            print(f"  action: {doctor_fix_for_row(row)}")

    print("")
    print(f"Doctor summary: sessions={len(sessions)} findings={total_findings}")
    if total_findings > 0:
        print("Next: fix highest-red panes, then rerun `agent-wrangler doctor`.")
    return 0


def run_nav(args: argparse.Namespace) -> int:
    ensure_tmux()
    pane_bindings = [
        ("M-Left", ["select-pane", "-L"]),
        ("M-Right", ["select-pane", "-R"]),
        ("M-Up", ["select-pane", "-U"]),
        ("M-Down", ["select-pane", "-D"]),
    ]
    window_bindings = [
        ("M-[", ["previous-window"]),
        ("M-]", ["next-window"]),
    ]
    index_bindings: list[tuple[str, list[str]]] = []
    for idx in range(1, 10):
        index_bindings.append((f"M-{idx}", ["select-window", "-t", f":{idx - 1}"]))
    # Named window shortcuts for manager/grid workflow
    store = load_store()
    session = store.get("default_session") or DEFAULT_SESSION
    named_window_bindings = [
        ("M-m", ["select-window", "-t", f"{session}:manager"]),
        ("M-g", ["select-window", "-t", f"{session}:grid"]),
    ]
    exit_script = str(ROOT / "scripts" / "agent-wrangler")
    summary_script = str(ROOT / "scripts" / "tmux_teams.py")
    utility_bindings = [
        ("M-z", ["resize-pane", "-Z"]),          # zoom toggle
        ("M-j", ["display-panes", "-d", "2000"]),  # jump by number overlay
        ("M-q", ["confirm-before", "-p", "Exit Agent Wrangler? (y/n)", f"run-shell '{exit_script} exit --force'"]),
        ("M-s", ["display-popup", "-E", "-w", "80", "-h", "30",
                  f"python3 {summary_script} teams summary #{{pane_title}} --session {session} --lines 50; read"]),
    ]
    all_bindings = pane_bindings + window_bindings + index_bindings + named_window_bindings + utility_bindings

    if args.remove:
        for key, _cmd in all_bindings:
            tmux(["unbind-key", "-n", key], timeout=5)
        print("Removed no-prefix Alt navigation bindings (pane + window).")
        return 0

    for key, cmd in all_bindings:
        tmux(["bind-key", "-n", key, *cmd], timeout=5)

    # Enable mouse: click to select pane, scroll to browse pane content (not shell history)
    tmux(["set-option", "-g", "mouse", "on"], timeout=5)
    # Scroll sends scroll events to the pane (enters copy mode), not Up/Down keys
    tmux(["set-option", "-g", "terminal-overrides", "xterm*:smcup@:rmcup@"], timeout=5)

    print("Enabled no-prefix Alt navigation bindings.")
    print("Pane navigation: Option+Arrow | mouse click")
    print("Window navigation: Option+[ / Option+]")
    print("Window direct jump: Option+1..9")
    print("Named windows: Option+m (manager) | Option+g (grid)")
    print("Utilities: Option+z (zoom) | Option+j (jump) | Option+q (exit) | Option+s (summary)")
    print("Mouse: click pane to select | scroll to browse output")
    return 0


def run_send(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    session = args.session or store.get("default_session") or DEFAULT_SESSION
    pane = pane_target(session, args.pane)
    pane_send(pane.pane_id, args.command, enter=(not args.no_enter))
    print(f"sent to {pane.pane_id} ({pane.pane_title}): {args.command}")
    return 0


def run_stop(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    session = args.session or store.get("default_session") or DEFAULT_SESSION
    pane = pane_target(session, args.pane)
    pane_ctrl_c(pane.pane_id)
    print(f"sent Ctrl-C to {pane.pane_id} ({pane.pane_title})")
    return 0


def run_restart(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    session = args.session or store.get("default_session") or DEFAULT_SESSION
    pane = pane_target(session, args.pane)

    pane_ctrl_c(pane.pane_id)

    proj_map = project_map()
    project_id = pane.project_id or pane.pane_title
    if project_id not in proj_map:
        inferred = infer_project_id_from_path(pane.pane_path, proj_map)
        if inferred:
            project_id = inferred
            pane_set_project_id(pane.pane_id, project_id)
    project = proj_map.get(project_id)
    if not project:
        raise ValueError(
            f"Pane project id '{project_id}' is not a known project id."
        )

    startup_command = str(project.get("startup_command") or "").strip()
    if not startup_command:
        raise ValueError(f"Project '{project_id}' has no startup_command")

    pane_send(pane.pane_id, startup_command, enter=True)
    print(f"restarted {pane.pane_id} ({project_id}) with: {startup_command}")
    return 0


def run_agent(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    session = args.session or store.get("default_session") or DEFAULT_SESSION
    pane = pane_target(session, args.pane)

    tool = args.tool.strip()
    tokens: list[str] = [tool]
    if args.flags:
        tokens.extend(args.flags.strip().split())
    extra = list(args.agent_args or [])
    if extra and extra[0] == "--":
        extra = extra[1:]
    tokens.extend(extra)
    command = " ".join(token for token in tokens if token)
    pane_send(pane.pane_id, command, enter=True)
    print(f"launched agent in {pane.pane_id} ({pane.pane_title}): {command}")
    return 0


def run_focus(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    session = args.session or store.get("default_session") or DEFAULT_SESSION
    pane = pane_target(session, args.pane)
    code, _, err = tmux(["select-pane", "-t", pane.pane_id], timeout=5)
    if code != 0:
        raise ValueError(err.strip() or f"failed to focus pane {pane.pane_id}")
    print(f"focused {pane.pane_id} ({pane.pane_title})")
    if args.attach:
        return attach_session(session)
    return 0


def run_kill(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    session = args.session or store.get("default_session") or DEFAULT_SESSION
    pane = pane_target(session, args.pane)
    code, _, err = tmux(["kill-pane", "-t", pane.pane_id], timeout=5)
    if code != 0:
        raise ValueError(err.strip() or f"failed to kill pane {pane.pane_id}")
    print(f"killed {pane.pane_id} ({pane.pane_title})")
    return 0


def run_exit(args: argparse.Namespace) -> int:
    """Kill the entire Agent Wrangler tmux session."""
    ensure_tmux()
    store = load_store()
    session = args.session or store.get("default_session") or DEFAULT_SESSION

    if not args.force:
        # Check for running agents and warn
        panes = list_panes(session)
        agents = [p for p in panes if p.agent]
        if agents:
            names = ", ".join(f"{p.pane_title}({p.agent})" for p in agents)
            print(f"Active agents: {names}")
            try:
                answer = input("Kill session and all agents? [y/N] ")
            except (EOFError, KeyboardInterrupt):
                print()
                return 1
            if answer.strip().lower() not in ("y", "yes"):
                print("Aborted.")
                return 1

    # Remove nav bindings first (best-effort)
    for key in ("M-Left", "M-Right", "M-Up", "M-Down", "M-[", "M-]",
                "M-m", "M-g", "M-z", "M-j", "M-q"):
        tmux(["unbind-key", "-n", key], timeout=3)
    for idx in range(1, 10):
        tmux(["unbind-key", "-n", f"M-{idx}"], timeout=3)

    code, _, err = tmux(["kill-session", "-t", session], timeout=5)
    if code != 0:
        raise ValueError(err.strip() or f"failed to kill session {session}")
    print(f"Agent Wrangler session '{session}' exited.")
    return 0


def run_shell(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    session = args.session or store.get("default_session") or DEFAULT_SESSION
    pane = pane_target(session, args.pane)

    shell_bin = args.shell or os.environ.get("SHELL") or "zsh"
    code, _, err = tmux(
        ["respawn-pane", "-k", "-t", pane.pane_id, "-c", pane.pane_path, shell_bin],
        timeout=8,
    )
    if code != 0:
        raise ValueError(err.strip() or f"failed to respawn shell in pane {pane.pane_id}")

    if pane.project_id:
        pane_set_project_id(pane.pane_id, pane.project_id)
        tmux(["select-pane", "-t", pane.pane_id, "-T", pane.project_id], timeout=5)

    print(f"reset {pane.pane_id} to shell '{shell_bin}'")
    return 0


def run_layout(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    session = args.session or store.get("default_session") or DEFAULT_SESSION
    panes = list_panes(session)
    resolved = choose_layout(args.layout, pane_count=len(panes))
    apply_layout(session, resolved)
    print(f"layout for {session}: {resolved}")
    return 0


def run_capture(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    session = args.session or store.get("default_session") or DEFAULT_SESSION
    pane = pane_target(session, args.pane)
    lines = max(5, args.lines)
    code, out, err = tmux(["capture-pane", "-p", "-t", pane.pane_id, "-S", f"-{lines}"], timeout=10)
    if code != 0:
        raise ValueError(err.strip() or f"failed to capture pane {pane.pane_id}")
    print(out.rstrip("\n"))
    return 0


def run_hide(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    session = args.session or store.get("default_session") or DEFAULT_SESSION
    pane = pane_target(session, args.pane)
    project_id = pane.project_id or pane.pane_title or pane.pane_id
    hidden_name = hide_pane(session, pane.pane_id, project_id)
    print(f"Hidden: {project_id} -> {hidden_name}")
    return 0


def run_show(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    session = args.session or store.get("default_session") or DEFAULT_SESSION
    hidden = list_hidden_panes(session)
    if not hidden:
        print("No hidden panes.")
        return 0

    # Find by project_id match
    target = args.pane
    match = None
    for h in hidden:
        if h["project_id"] == target or h["window_name"] == target or h["pane_id"] == target:
            match = h
            break

    if not match:
        print(f"No hidden pane matching '{target}'. Hidden panes:")
        for h in hidden:
            print(f"  {h['project_id']}  ({h['agent']}, {h['status']})")
        return 1

    show_pane(session, match["window_name"])
    print(f"Restored: {match['project_id']}")
    return 0


def run_hidden(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    session = args.session or store.get("default_session") or DEFAULT_SESSION
    hidden = list_hidden_panes(session)
    if not hidden:
        print("No hidden panes.")
        return 0
    print(f"Hidden panes in '{session}':")
    for h in hidden:
        print(f"  {h['project_id']:<22} agent={h['agent']:<8} status={h['status']}")
    return 0


def run_list_projects(args: argparse.Namespace) -> int:
    config = workflow_agent.load_config()
    group = args.group.lower() if args.group else None
    print(f"{'ID':<22} {'GROUP':<10} PATH")
    for project in config.get("projects", []):
        pg = str(project.get("group", ""))
        if group and pg.lower() != group:
            continue
        print(f"{project['id']:<22} {pg:<10} {project.get('path', '')}")
    return 0


PROJECTS_CONFIG = ROOT / "config" / "projects.json"
NOTIFY_STATE_PATH = ROOT / ".state" / "health_state.json"
NOTIFY_APP_PATH = ROOT / "assets" / "AgentWrangler.app"
NOTIFY_COOLDOWN_SEC = 120  # Minimum seconds between notifications
NOTIFY_DEBOUNCE = 2  # Pane must stay in new state for N checks before notifying


def _load_health_state() -> dict[str, Any]:
    """Load previous pane health levels, streak counts, and last notify time."""
    try:
        if NOTIFY_STATE_PATH.exists():
            return json.loads(NOTIFY_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_health_state(state: dict[str, Any]) -> None:
    """Persist current pane health levels for change detection."""
    try:
        NOTIFY_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        NOTIFY_STATE_PATH.write_text(json.dumps(state), encoding="utf-8")
    except Exception:
        pass


def _notifications_enabled() -> bool:
    """Check if notifications are enabled. Off by default.

    Enable via: "notifications": true in team_grid.json, or AW_NOTIFY=1 env var.
    """
    if os.environ.get("AW_NOTIFY", "").strip() in ("1", "true", "yes"):
        return True
    try:
        store = load_store()
        return bool(store.get("notifications", False))
    except Exception:
        return False


def _notify_desktop(title: str, message: str) -> None:
    """Send a macOS desktop notification via AgentWrangler.app (cowboy hat icon)."""
    try:
        if NOTIFY_APP_PATH.is_dir():
            # Use the bundled .app — shows cowboy hat icon
            Path("/tmp/aw_notify.txt").write_text(f"{title}\n{message}")
            subprocess.run(
                ["open", "-g", "-n", str(NOTIFY_APP_PATH)],
                timeout=5, check=False, capture_output=True,
            )
        else:
            # Fallback to plain osascript
            safe_msg = message.replace('"', '\\"')
            safe_title = title.replace('"', '\\"')
            subprocess.run(
                [
                    "osascript", "-e",
                    f'display notification "{safe_msg}" with title "{safe_title}" sound name "Ping"',
                ],
                timeout=5, check=False, capture_output=True,
            )
    except Exception:
        pass


def check_and_notify(rows: list[dict[str, Any]]) -> None:
    """Compare health state, send desktop notifications on confirmed error transitions.

    Notifications are off by default. Enable with "notifications": true in team_grid.json
    or pass --notify to paint/watch commands.

    Debounce: a pane must stay red for NOTIFY_DEBOUNCE consecutive checks before
    alerting. Same for recovery. Waiting states are never notified. Cooldown prevents
    rapid-fire notifications.
    """
    if not _notifications_enabled():
        return
    state = _load_health_state()
    prev_levels = state.get("levels", {})
    streaks: dict[str, int] = state.get("streaks", {})
    notified: dict[str, str] = state.get("notified", {})
    last_notify = float(state.get("last_notify", 0))
    current: dict[str, str] = {}
    alerts: list[str] = []

    for row in rows:
        key = str(row.get("project_id") or row.get("pane_id") or "")
        level = str(row.get("health") or "green")
        reason = str(row.get("reason") or "")
        current[key] = level

        prev_level = prev_levels.get(key, level)

        # Skip waiting states entirely — not worth notifying
        if reason.startswith("waiting"):
            streaks[key] = 0
            continue

        # Track consecutive checks at same level
        if level == prev_level:
            streaks[key] = streaks.get(key, 0) + 1
        else:
            streaks[key] = 1

        # Only alert after NOTIFY_DEBOUNCE consecutive checks in new state
        if streaks.get(key, 0) < NOTIFY_DEBOUNCE:
            continue

        last_notified_level = notified.get(key)
        if level == "red" and last_notified_level != "red":
            alerts.append(f"{key}: {reason or 'needs attention'}")
            notified[key] = "red"
        elif level == "green" and last_notified_level == "red":
            alerts.append(f"{key}: recovered")
            notified[key] = "green"

    now = time.time()
    new_state: dict[str, Any] = {
        "levels": current,
        "streaks": streaks,
        "notified": notified,
        "last_notify": last_notify,
    }
    if alerts and (now - last_notify) >= NOTIFY_COOLDOWN_SEC:
        _notify_desktop("Agent Wrangler", "\n".join(alerts[:3]))
        new_state["last_notify"] = now

    _save_health_state(new_state)


def run_init(_: argparse.Namespace) -> int:
    """Interactive project setup — scan for git repos and create projects.json."""
    if PROJECTS_CONFIG.exists():
        try:
            answer = input(f"{PROJECTS_CONFIG} already exists. Overwrite? [y/N] ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 1
        if answer.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return 1

    home = Path.home()
    print(f"Scanning {home} for git repositories...")

    repos: list[Path] = []
    for entry in sorted(home.iterdir()):
        if entry.name.startswith(".") or not entry.is_dir():
            continue
        if (entry / ".git").is_dir():
            repos.append(entry)

    if not repos:
        print("No git repos found in home directory.")
        return 1

    print(f"\nFound {len(repos)} repos:\n")
    for idx, repo in enumerate(repos):
        print(f"  {idx + 1:>2}. {repo.name:<30} {repo}")

    print(f"\nEnter numbers to include (e.g. 1,3,5-8), or 'all':")
    try:
        selection = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return 1

    selected: list[Path] = []
    if selection.lower() == "all":
        selected = list(repos)
    else:
        for part in selection.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                for idx in range(int(lo), int(hi) + 1):
                    if 1 <= idx <= len(repos):
                        selected.append(repos[idx - 1])
            elif part.isdigit():
                idx = int(part)
                if 1 <= idx <= len(repos):
                    selected.append(repos[idx - 1])

    if not selected:
        print("No projects selected.")
        return 1

    projects: list[dict[str, Any]] = []
    for repo in selected:
        proj_id = repo.name.replace(" ", "-").lower()
        projects.append({
            "id": proj_id,
            "name": repo.name,
            "path": str(repo),
            "default_branch": "main",
        })

    config = {"projects": projects}
    PROJECTS_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    PROJECTS_CONFIG.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    print(f"\nCreated {PROJECTS_CONFIG} with {len(projects)} projects:")
    for p in projects:
        print(f"  - {p['id']}")

    # Also create team_grid.json if missing
    if not CONFIG_PATH.exists():
        grid_config = {
            "default_session": "agent-grid",
            "default_layout": "tiled",
            "default_projects": [p["id"] for p in projects[:10]],
            "persistence": {"enabled": False, "autosave_minutes": 15},
            "profiles": {"current": "default", "items": {"default": {"max_panes": 10}}},
            "updated_at": now_iso(),
        }
        CONFIG_PATH.write_text(json.dumps(grid_config, indent=2) + "\n", encoding="utf-8")
        print(f"Created {CONFIG_PATH}")

    print("\nRun: ./scripts/agent-wrangler start")
    return 0


def run_add(args: argparse.Namespace) -> int:
    """Add current directory (or specified path) as a project and hot-add to running grid."""
    path = Path(args.path).resolve() if args.path else Path.cwd()
    if not path.is_dir():
        raise ValueError(f"Not a directory: {path}")

    proj_id = args.name or path.name.replace(" ", "-").lower()

    # Add to projects.json
    config: dict[str, Any] = {}
    if PROJECTS_CONFIG.exists():
        config = json.loads(PROJECTS_CONFIG.read_text(encoding="utf-8"))
    projects = config.setdefault("projects", [])

    # Check if already exists
    existing = [p for p in projects if p.get("id") == proj_id]
    if existing:
        print(f"Project '{proj_id}' already in config.")
    else:
        projects.append({
            "id": proj_id,
            "name": path.name,
            "path": str(path),
            "default_branch": "main",
        })
        PROJECTS_CONFIG.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(f"Added '{proj_id}' to {PROJECTS_CONFIG}")

    # Hot-add to running grid if tmux session exists
    store = load_store()
    session = args.session or store.get("default_session") or DEFAULT_SESSION
    if session_exists(session):
        split_for_path(session, str(path))
        apply_layout(session, "tiled")
        # Tag the new pane
        panes = list_panes(session)
        if panes:
            last_pane = panes[-1]
            pane_set_project_id(last_pane.pane_id, proj_id)
            tmux(["select-pane", "-t", last_pane.pane_id, "-T", proj_id], timeout=5)
            pane_send(last_pane.pane_id, f"echo '\\n[{proj_id}] ready'", enter=True)
        set_window_orchestrator_format(session)
        print(f"Added pane for '{proj_id}' to session '{session}'")
    else:
        print(f"Session '{session}' not running. Pane will appear on next start.")

    return 0


def run_summary(args: argparse.Namespace) -> int:
    """Show a summary of recent output from a pane."""
    ensure_tmux()
    store = load_store()
    session = args.session or store.get("default_session") or DEFAULT_SESSION
    pane = pane_target(session, args.pane)
    lines = max(10, int(args.lines))
    raw = capture_pane_raw(pane.pane_id, lines=lines)

    # Extract meaningful lines (skip empty, prompts-only)
    meaningful: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Skip bare shell prompts
        if stripped in ("$", "%", ">", "❯"):
            continue
        meaningful.append(stripped)

    project = pane.project_id or pane.pane_title
    print(f"[{project}] Last {lines} lines ({len(meaningful)} non-empty):")
    print("─" * 60)
    for line in meaningful[-30:]:
        print(line)
    print("─" * 60)
    return 0


def _run_ops_command(command: list[str]) -> int:
    cmd = [str(ROOT / "scripts" / "agent-wrangler"), *command]
    print("")
    print("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def run_ops(_: argparse.Namespace) -> int:
    """Interactive operator console — numbered menu for common actions."""
    actions: list[tuple[str, list[str]]] = [
        ("Start all (import + grid + manager + nav)", ["start"]),
        ("Attach grid session", ["attach"]),
        ("Show pane status", ["status"]),
        ("Focus pane by project/token", ["__focus__"]),
        ("Send command to pane", ["__send__"]),
        ("Launch agent in pane", ["__agent__"]),
        ("Stop pane (Ctrl-C)", ["__stop__"]),
        ("Open manager window", ["manager", "--replace"]),
        ("Doctor (attention)", ["doctor", "--only-attention"]),
        ("Program status", ["program", "status"]),
    ]

    print("Agent Wrangler Ops Console")
    print("Enter number, or q to quit.\n")

    while True:
        for idx, (label, _) in enumerate(actions, start=1):
            print(f"{idx:>2}. {label}")
        print(" q. Quit")

        choice = input("\nops> ").strip().lower()
        if choice in {"q", "quit", "exit"}:
            return 0
        if not choice or not choice.isdigit():
            continue

        idx = int(choice)
        if idx < 1 or idx > len(actions):
            print("Invalid choice.")
            continue

        _, command = actions[idx - 1]
        if command == ["__focus__"]:
            token = input("pane token: ").strip()
            if token:
                _run_ops_command(["focus", token, "--attach"])
        elif command == ["__send__"]:
            token = input("pane token: ").strip()
            text = input("command: ").strip()
            if token and text:
                _run_ops_command(["send", token, "--command", text])
        elif command == ["__agent__"]:
            token = input("pane token: ").strip()
            tool = input("tool (claude|codex|aider|gemini): ").strip().lower()
            if token and tool in {"claude", "codex", "aider", "gemini"}:
                _run_ops_command(["agent", token, tool])
        elif command == ["__stop__"]:
            token = input("pane token: ").strip()
            if token:
                _run_ops_command(["stop", token])
        else:
            _run_ops_command(command)
        print("")


def register_subparser(root_subparsers: argparse._SubParsersAction[Any]) -> None:
    teams = root_subparsers.add_parser("teams", help="Tmux team grid operations")
    teams_sub = teams.add_subparsers(dest="teams_command", required=True)

    up = teams_sub.add_parser("up", help="One-command entry: build/reuse grid, show status, attach")
    up.add_argument("--session", default=None)
    up.add_argument("--mode", choices=["import", "bootstrap"], default="import")
    up.add_argument("--layout", choices=LAYOUT_CHOICES, default=None)
    up.add_argument("--max-panes", type=int, default=10)
    up.add_argument("--projects", help="Comma-separated project ids (used for bootstrap or fallback)")
    up.add_argument("--group", choices=["business", "personal"], help="Project group for bootstrap/fallback")
    up.add_argument("--include-idle", action="store_true", help="Include idle Ghostty sessions when importing")
    up.add_argument(
        "--preserve-duplicates",
        action="store_true",
        help="Keep one pane per matched Ghostty session, even when multiple sessions map to the same project",
    )
    up.add_argument("--startup", action="store_true", help="Run startup commands in panes")
    up.add_argument("--agent", action="store_true", help="Launch detected agents in panes")
    up.add_argument("--no-startup", action="store_true")
    up.add_argument("--no-agent", action="store_true")
    up.add_argument("--rebuild", action="store_true", help="Recreate session from current source even if it exists")
    up.add_argument("--force", dest="rebuild", action="store_true", help="Alias for --rebuild")
    up.add_argument("--nav", action="store_true", help="Enable Option+Arrow pane navigation bindings")
    up.add_argument("--manager", action="store_true", help="Open orchestrator manager window")
    up.add_argument("--manager-window", default="manager")
    up.add_argument("--manager-interval", type=int, default=3)
    up.add_argument("--manager-replace", action="store_true")
    up.add_argument("--status", action="store_true", default=True, help="Print status before attach (default true)")
    up.add_argument("--no-status", dest="status", action="store_false", help="Skip status output")
    up.add_argument("--attach", action="store_true", default=True, help="Attach to session (default true)")
    up.add_argument("--no-attach", dest="attach", action="store_false", help="Do not attach")
    up.set_defaults(handler=run_up)

    bootstrap = teams_sub.add_parser("bootstrap", help="Create a tmux grid session from projects")
    bootstrap.add_argument("--session", default=None)
    bootstrap.add_argument("--projects", help="Comma-separated project ids")
    bootstrap.add_argument("--group", choices=["business", "personal"], help="Autoselect from a project group")
    bootstrap.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Max panes when auto-selecting")
    bootstrap.add_argument("--layout", choices=LAYOUT_CHOICES, default=None)
    bootstrap.add_argument("--force", action="store_true", help="Replace existing session with same name")
    bootstrap.add_argument("--no-startup", action="store_true", help="Do not run per-project startup commands")
    bootstrap.add_argument("--agent", help="Agent command to run in each pane (example: claude or codex)")
    bootstrap.add_argument("--attach", action="store_true", help="Attach immediately after creation")
    bootstrap.set_defaults(handler=run_bootstrap)

    imp = teams_sub.add_parser("import", help="Import current Ghostty sessions into a tmux grid")
    imp.add_argument("--session", default=None)
    imp.add_argument("--layout", choices=LAYOUT_CHOICES, default=None)
    imp.add_argument("--max-panes", type=int, default=10, help="Max panes to import from Ghostty")
    imp.add_argument("--include-idle", action="store_true", help="Include idle Ghostty sessions when mapping")
    imp.add_argument(
        "--preserve-duplicates",
        action="store_true",
        help="Keep one pane per matched Ghostty session, even when multiple sessions map to the same project",
    )
    imp.add_argument("--startup", action="store_true", help="Run project startup commands in imported panes")
    imp.add_argument("--agent", action="store_true", help="Start detected agent tools in imported panes")
    imp.add_argument("--no-startup", action="store_true", help="Do not run per-project startup commands")
    imp.add_argument("--no-agent", action="store_true", help="Do not auto-start detected agent tools")
    imp.add_argument("--dry-run", action="store_true", help="Preview mapping only; do not create/replace tmux session")
    imp.add_argument("--force", action="store_true", help="Replace existing session with same name")
    imp.add_argument("--attach", action="store_true", help="Attach immediately after import")
    imp.set_defaults(handler=run_import)

    attach = teams_sub.add_parser("attach", help="Attach to the tmux team session")
    attach.add_argument("--session", default=None)
    attach.set_defaults(handler=run_attach)

    status = teams_sub.add_parser("status", help="Show pane grid with per-pane status")
    status.add_argument("--session", default=None)
    status.set_defaults(handler=run_status)

    rail = teams_sub.add_parser("rail", help="Compact auto-refreshing status rail for narrow split panes")
    rail.add_argument("--session", default=None)
    rail.add_argument("--interval", type=int, default=5)
    rail.set_defaults(handler=run_rail)

    paint = teams_sub.add_parser("paint", help="Color panes by attention state (green/red)")
    paint.add_argument("--session", default=None)
    paint.add_argument("--capture-lines", type=int, default=80, help="Recent lines to inspect for error markers")
    paint.add_argument("--wait-attention-min", type=int, default=5, help="Waiting minutes before marking red")
    paint.add_argument("--no-colorize", action="store_true", help="Compute health only, do not set pane colors")
    paint.set_defaults(handler=run_paint)

    watch = teams_sub.add_parser("watch", help="Live manager loop with health + attention states")
    watch.add_argument("--session", default=None)
    watch.add_argument("--interval", type=int, default=3)
    watch.add_argument("--iterations", type=int, default=0, help="0 means infinite")
    watch.add_argument("--capture-lines", type=int, default=80)
    watch.add_argument("--wait-attention-min", type=int, default=5)
    watch.add_argument("--no-colorize", action="store_true")
    watch.add_argument("--no-clear", action="store_true")
    watch.set_defaults(handler=run_watch)

    manager = teams_sub.add_parser("manager", help="Create/open orchestrator manager screen as tmux window")
    manager.add_argument("--session", default=None)
    manager.add_argument("--window", default="manager")
    manager.add_argument("--interval", type=int, default=3)
    manager.add_argument("--replace", action="store_true", help="Replace existing manager window")
    manager.add_argument("--focus", action="store_true", default=True, help="Focus manager window (default true)")
    manager.add_argument("--no-focus", dest="focus", action="store_false")
    manager.add_argument("--attach", action="store_true", default=True, help="Attach session (default true)")
    manager.add_argument("--no-attach", dest="attach", action="store_false")
    manager.set_defaults(handler=run_manager)

    nav = teams_sub.add_parser("nav", help="Enable no-prefix Option navigation for panes and windows")
    nav.add_argument("--remove", action="store_true", help="Remove bindings")
    nav.set_defaults(handler=run_nav)

    send = teams_sub.add_parser("send", help="Send command to one pane")
    send.add_argument("pane", help="Pane id (%1), pane index (0), or pane title/project id")
    send.add_argument("--session", default=None)
    send.add_argument("--command", required=True)
    send.add_argument("--no-enter", action="store_true")
    send.set_defaults(handler=run_send)

    stop = teams_sub.add_parser("stop", help="Send Ctrl-C to one pane")
    stop.add_argument("pane")
    stop.add_argument("--session", default=None)
    stop.set_defaults(handler=run_stop)

    restart = teams_sub.add_parser("restart", help="Restart pane using project startup command")
    restart.add_argument("pane")
    restart.add_argument("--session", default=None)
    restart.set_defaults(handler=run_restart)

    agent = teams_sub.add_parser("agent", help="Start an agent command in one pane")
    agent.add_argument("pane")
    agent.add_argument("tool", choices=["claude", "codex", "aider", "gemini"])
    agent.add_argument("--session", default=None)
    agent.add_argument("--flags", help="Additional flags passed after tool command")
    agent.add_argument("agent_args", nargs=argparse.REMAINDER, help="Extra args. Use after --, e.g. -- --help")
    agent.set_defaults(handler=run_agent)

    focus = teams_sub.add_parser("focus", help="Focus a pane (optionally attach)")
    focus.add_argument("pane")
    focus.add_argument("--session", default=None)
    focus.add_argument("--attach", action="store_true")
    focus.set_defaults(handler=run_focus)

    kill = teams_sub.add_parser("kill", help="Kill a pane")
    kill.add_argument("pane")
    kill.add_argument("--session", default=None)
    kill.set_defaults(handler=run_kill)

    exit_cmd = teams_sub.add_parser("exit", help="Exit Agent Wrangler (kill the tmux session)")
    exit_cmd.add_argument("--session", default=None)
    exit_cmd.add_argument("--force", "-f", action="store_true", help="Skip confirmation even if agents are running")
    exit_cmd.set_defaults(handler=run_exit)

    hide = teams_sub.add_parser("hide", help="Hide a pane (move to background, agent keeps running)")
    hide.add_argument("pane")
    hide.add_argument("--session", default=None)
    hide.set_defaults(handler=run_hide)

    show = teams_sub.add_parser("show", help="Restore a hidden pane back to the grid")
    show.add_argument("pane")
    show.add_argument("--session", default=None)
    show.set_defaults(handler=run_show)

    hidden = teams_sub.add_parser("hidden", help="List hidden panes")
    hidden.add_argument("--session", default=None)
    hidden.set_defaults(handler=run_hidden)

    shell = teams_sub.add_parser("shell", help="Reset a pane to a fresh shell")
    shell.add_argument("pane")
    shell.add_argument("--session", default=None)
    shell.add_argument("--shell", help="Shell binary to start (default from $SHELL)")
    shell.set_defaults(handler=run_shell)

    layout = teams_sub.add_parser("layout", help="Change tmux layout")
    layout.add_argument("layout", choices=LAYOUT_CHOICES)
    layout.add_argument("--session", default=None)
    layout.set_defaults(handler=run_layout)

    capture = teams_sub.add_parser("capture", help="Capture recent pane output")
    capture.add_argument("pane")
    capture.add_argument("--session", default=None)
    capture.add_argument("--lines", type=int, default=30)
    capture.set_defaults(handler=run_capture)

    projects = teams_sub.add_parser("projects", help="List known projects for team panes")
    projects.add_argument("--group", choices=["business", "personal"])
    projects.set_defaults(handler=run_list_projects)

    persistence = teams_sub.add_parser("persistence", help="Save/restore tmux workspace state")
    persistence_sub = persistence.add_subparsers(dest="persistence_command", required=True)

    persistence_status = persistence_sub.add_parser("status", help="Show persistence settings and snapshots")
    persistence_status.set_defaults(handler=run_persistence_status)

    persistence_enable = persistence_sub.add_parser("enable", help="Enable persistence mode")
    persistence_enable.add_argument("--autosave-minutes", type=int, default=15)
    persistence_enable.set_defaults(handler=run_persistence_enable)

    persistence_disable = persistence_sub.add_parser("disable", help="Disable persistence mode")
    persistence_disable.set_defaults(handler=run_persistence_disable)

    persistence_save = persistence_sub.add_parser("save", help="Save current session snapshot")
    persistence_save.add_argument("--session", default=None)
    persistence_save.add_argument("--name", help="Snapshot name (without .json is fine)")
    persistence_save.add_argument("--file", help="Explicit snapshot file path")
    persistence_save.add_argument("--tmux-resurrect", action="store_true", help="Also run tmux-resurrect save script")
    persistence_save.set_defaults(handler=run_persistence_save)

    persistence_restore = persistence_sub.add_parser("restore", help="Restore a saved session snapshot")
    persistence_restore.add_argument("--session", default=None)
    persistence_restore.add_argument("--name", help="Snapshot name (without .json is fine)")
    persistence_restore.add_argument("--file", help="Explicit snapshot file path")
    persistence_restore.add_argument("--layout", choices=LAYOUT_CHOICES, default=None)
    persistence_restore.add_argument("--startup", action="store_true", help="Run startup commands after restore")
    persistence_restore.add_argument("--agent", action="store_true", help="Relaunch detected agents after restore")
    persistence_restore.add_argument("--force", action="store_true", help="Replace existing session")
    persistence_restore.add_argument("--attach", action="store_true", help="Attach after restore")
    persistence_restore.add_argument(
        "--tmux-resurrect",
        action="store_true",
        help="Also run tmux-resurrect restore script after local restore",
    )
    persistence_restore.set_defaults(handler=run_persistence_restore)

    profile = teams_sub.add_parser("profile", help="Manage workspace profiles (gabooja/personal/etc.)")
    profile_sub = profile.add_subparsers(dest="profile_command", required=True)

    profile_list = profile_sub.add_parser("list", help="List available profiles")
    profile_list.set_defaults(handler=run_profile_list)

    profile_status = profile_sub.add_parser("status", help="Show current profile")
    profile_status.set_defaults(handler=run_profile_status)

    profile_save = profile_sub.add_parser("save", help="Save/update a profile")
    profile_save.add_argument("name", help="Profile name")
    profile_save.add_argument("--sessions", help="Comma-separated tmux sessions for this profile")
    profile_save.add_argument("--max-panes", type=int, default=10)
    profile_save.add_argument("--auto-running", action="store_true", help="Fallback to all running tmux sessions")
    profile_save.set_defaults(handler=run_profile_save)

    profile_use = profile_sub.add_parser("use", help="Activate a profile and apply managed sessions")
    profile_use.add_argument("name", help="Profile name")
    profile_use.set_defaults(handler=run_profile_use)

    hooks = teams_sub.add_parser("hooks", help="Enable/disable tmux hooks for event-driven repaint")
    hooks_sub = hooks.add_subparsers(dest="hooks_command", required=True)

    hooks_status = hooks_sub.add_parser("status", help="Show hook status")
    hooks_status.add_argument("--session", default=None)
    hooks_status.set_defaults(handler=run_hooks_status)

    hooks_enable = hooks_sub.add_parser("enable", help="Enable hooks for a session")
    hooks_enable.add_argument("--session", default=None)
    hooks_enable.add_argument("--capture-lines", type=int, default=80)
    hooks_enable.add_argument("--wait-attention-min", type=int, default=5)
    hooks_enable.set_defaults(handler=run_hooks_enable)

    hooks_disable = hooks_sub.add_parser("disable", help="Disable hooks for a session")
    hooks_disable.add_argument("--session", default=None)
    hooks_disable.set_defaults(handler=run_hooks_disable)

    init_cmd = teams_sub.add_parser("init", help="Interactive project setup — scan for repos and create config")
    init_cmd.set_defaults(handler=run_init)

    add_cmd = teams_sub.add_parser("add", help="Add a project to config and running grid")
    add_cmd.add_argument("path", nargs="?", default=None, help="Directory path (default: current directory)")
    add_cmd.add_argument("--name", help="Project ID override (default: directory name)")
    add_cmd.add_argument("--session", default=None)
    add_cmd.set_defaults(handler=run_add)

    summary_cmd = teams_sub.add_parser("summary", help="Show recent output summary from a pane")
    summary_cmd.add_argument("pane", help="Pane token (project id, index, or pane id)")
    summary_cmd.add_argument("--session", default=None)
    summary_cmd.add_argument("--lines", type=int, default=50, help="Lines to capture")
    summary_cmd.set_defaults(handler=run_summary)

    doctor = teams_sub.add_parser("doctor", help="Diagnose broken/waiting agent panes")
    doctor.add_argument("--session", default=None, help="Tmux session (default: configured default_session)")
    doctor.add_argument("--capture-lines", type=int, default=120, help="Recent pane lines to inspect for issues")
    doctor.add_argument("--wait-attention-min", type=int, default=5, help="Waiting threshold in minutes")
    doctor.add_argument("--only-attention", action="store_true", help="Only print panes that need attention")
    doctor.set_defaults(handler=run_doctor)

    drift = teams_sub.add_parser("drift", help="Show git drift for pane projects")
    drift.add_argument("--session", default=None, help="Tmux session (default: configured default_session)")
    drift.add_argument("--alert-dirty", type=int, default=25, help="Dirty-file threshold for high-drift alerts")
    drift.set_defaults(handler=run_drift)


def main() -> int:
    parser = argparse.ArgumentParser(prog="agent-wrangler", description="Agent Wrangler")
    sub = parser.add_subparsers(dest="command", required=True)

    # Top-level ops command
    ops_parser = sub.add_parser("ops", help="Interactive operator console")
    ops_parser.set_defaults(handler=run_ops)

    # All teams subcommands registered under "teams" namespace
    register_subparser(sub)

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
    sys.exit(main())
