#!/usr/bin/env python3
"""Tmux team-grid orchestration for multi-repo agent sessions."""

from __future__ import annotations

import argparse
import json
import os
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

DEFAULT_SESSION = "amir-grid"
DEFAULT_LAYOUT = "auto"
DEFAULT_FLEET_MANAGER_SESSION = "wrangler-hq"
DEFAULT_FLEET_MANAGER_WINDOW = "fleet"
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
        "fleet": {
            "managed_sessions": [],
            "manager_session": DEFAULT_FLEET_MANAGER_SESSION,
            "manager_window": DEFAULT_FLEET_MANAGER_WINDOW,
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
    base_fleet = base.get("fleet", {})
    fleet = current.get("fleet", {}) if isinstance(current.get("fleet"), dict) else {}
    merged["fleet"] = {
        **base_fleet,
        **fleet,
    }
    managed = merged.get("fleet", {}).get("managed_sessions")
    if not isinstance(managed, list):
        merged["fleet"]["managed_sessions"] = []
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


def store_managed_sessions(store: dict[str, Any]) -> list[str]:
    fleet = store.get("fleet", {})
    values = fleet.get("managed_sessions", [])
    if not isinstance(values, list):
        return []
    return [str(item).strip() for item in values if str(item).strip()]


def resolve_fleet_sessions(
    *,
    store: dict[str, Any],
    explicit_csv: str | None,
    pattern: str | None,
    include_manager: bool = False,
) -> list[str]:
    running = list_tmux_sessions()
    fleet_cfg = store.get("fleet", {})
    manager_session = str(fleet_cfg.get("manager_session") or DEFAULT_FLEET_MANAGER_SESSION).strip()

    explicit = split_csv(explicit_csv)
    if explicit:
        sessions = [name for name in explicit if name in running]
    else:
        managed = [name for name in store_managed_sessions(store) if name in running]
        sessions = managed if managed else list(running)

    if pattern:
        needle = pattern.lower()
        sessions = [name for name in sessions if needle in name.lower()]

    if not include_manager and manager_session:
        sessions = [name for name in sessions if name != manager_session]

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


def detect_error_marker(text: str) -> str | None:
    if not text:
        return None
    for marker in ERROR_MARKERS:
        if marker in text:
            return marker
    return None


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
    if level == "red":
        return "fg=colour160", "fg=colour196,bold"
    if level == "yellow":
        return "fg=colour220", "fg=colour214,bold"
    return "fg=colour34", "fg=colour82,bold"


def set_window_orchestrator_format(session: str) -> None:
    tmux(["set-option", "-w", "-t", f"{session}:0", "pane-border-status", "top"], timeout=5)
    tmux(["set-option", "-w", "-t", f"{session}:0", "pane-border-lines", "single"], timeout=5)
    fmt = (
        "#{?pane_active,#[bold],}"
        "#{?@needs_attention,#[fg=colour196],#[fg=colour34]}"
        "#{pane_index} #{@project_id} #{@agent} #{@health}"
        "#{?@health_reason, (#{@health_reason}),}"
        "#[default]"
    )
    tmux(["set-option", "-w", "-t", f"{session}:0", "pane-border-format", fmt], timeout=5)


def refresh_pane_health(
    session: str,
    capture_lines: int,
    wait_attention_min: int,
    apply_colors: bool,
) -> list[dict[str, Any]]:
    panes = list_panes(session)
    by_tty = session_monitor_by_tty()
    rows: list[dict[str, Any]] = []

    set_window_orchestrator_format(session)

    for pane in panes:
        tty_short = pane.pane_tty.split("/")[-1]
        monitor = by_tty.get(tty_short, {})
        agent = str(monitor.get("agent") or "-")
        status = str(monitor.get("status") or "idle")
        wait = monitor.get("waiting_minutes")
        error_marker = detect_error_marker(capture_pane_text(pane.pane_id, lines=capture_lines))
        level, needs_attention, reason = pane_health_level(
            monitor=monitor,
            error_marker=error_marker,
            wait_attention_min=wait_attention_min,
        )

        tmux(["set-option", "-p", "-t", pane.pane_id, "@agent", agent], timeout=5)
        tmux(["set-option", "-p", "-t", pane.pane_id, "@health", level.upper()], timeout=5)
        tmux(["set-option", "-p", "-t", pane.pane_id, "@health_reason", reason], timeout=5)
        tmux(["set-option", "-p", "-t", pane.pane_id, "@needs_attention", "1" if needs_attention else "0"], timeout=5)

        if apply_colors:
            border_style, active_style = style_for_level(level)
            tmux(["set-option", "-p", "-t", pane.pane_id, "pane-border-style", border_style], timeout=5)
            tmux(["set-option", "-p", "-t", pane.pane_id, "pane-active-border-style", active_style], timeout=5)

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
            }
        )
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
    apply_colors: bool,
) -> dict[str, Any]:
    rows = refresh_pane_health(
        session=session,
        capture_lines=capture_lines,
        wait_attention_min=wait_attention_min,
        apply_colors=apply_colors,
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

    first = proj_map[project_ids[0]]
    first_path = str(first.get("path") or "")
    if not first_path:
        raise ValueError(f"Project '{project_ids[0]}' has no path")

    code, _, err = tmux(
        ["new-session", "-d", "-s", session, "-n", "teams", "-x", "260", "-y", "90", "-c", first_path],
        timeout=8,
    )
    if code != 0:
        raise ValueError(err.strip() or f"failed to create session '{session}'")

    for project_id in project_ids[1:]:
        project = proj_map[project_id]
        path = str(project.get("path") or "")
        split_for_path(session, path)
        # Rebalance as the grid grows to prevent "no space for new pane" on repeated splits.
        apply_layout(session, "tiled")

    resolved_layout = choose_layout(layout, pane_count=len(project_ids))
    apply_layout(session, resolved_layout)

    panes = list_panes(session)
    for idx, pane in enumerate(panes):
        project_id = project_ids[idx] if idx < len(project_ids) else f"pane-{idx}"
        project = proj_map.get(project_id, {})

        pane_set_project_id(pane.pane_id, project_id)
        code, _, err = tmux(["select-pane", "-t", pane.pane_id, "-T", project_id], timeout=5)
        if code != 0:
            raise ValueError(err.strip() or f"failed to set pane title for {pane.pane_id}")

        banner = f"clear; echo '[{project_id}] {project.get('path', '')}'"
        pane_send(pane.pane_id, banner, enter=True)

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
) -> tuple[list[str], dict[str, str], list[dict[str, Any]], list[dict[str, Any]]]:
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
    mapped: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []

    for session in sessions:
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

        if project_id not in project_ids:
            if len(project_ids) >= max(1, max_panes):
                continue
            project_ids.append(project_id)

        agent = str(session.get("agent") or "").strip().lower()
        if agent in {"claude", "codex", "aider", "gemini"} and project_id not in agent_by_project:
            agent_by_project[project_id] = agent

        mapped.append(
            {
                "project_id": project_id,
                "tty": session.get("tty"),
                "pid": session.get("pid"),
                "status": session.get("status"),
                "agent": session.get("agent"),
                "cwd": cwd,
                "command": session.get("command"),
            }
        )

    return project_ids, agent_by_project, mapped, unmatched


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

    project_ids, agent_by_project, mapped, unmatched = ghostty_import_plan(
        proj_map=proj_map,
        max_panes=args.max_panes,
        include_idle=args.include_idle,
    )
    if not project_ids:
        raise ValueError("No Ghostty sessions matched known project paths. Keep current setup or pass explicit projects.")

    resolved_layout = choose_layout(layout, pane_count=len(project_ids))
    if args.dry_run:
        print("Ghostty import plan (dry-run)")
        print(f"Session: {session}")
        print(f"Panes: {len(project_ids)}  Layout: {resolved_layout}")
        print("Projects: " + ", ".join(project_ids))
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
    print(f"Startup commands: {'enabled' if run_startup else 'disabled'}")
    print(f"Detected agents: {'enabled' if run_agents else 'disabled'}")
    if run_agents and agent_by_project:
        pairs = [f"{pid}:{agent_by_project[pid]}" for pid in project_ids if pid in agent_by_project]
        if pairs:
            print("Agent hints: " + ", ".join(pairs))

    print("")
    print("Mapped sessions:")
    for item in mapped[: max(1, args.max_panes)]:
        print(
            "- tty={tty} status={status} project={project_id} agent={agent} cwd={cwd}".format(
                tty=item.get("tty", "-"),
                status=item.get("status", "-"),
                project_id=item.get("project_id", "-"),
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
        else:
            project_ids, detected_agents, mapped, unmatched = ghostty_import_plan(
                proj_map=proj_map,
                max_panes=args.max_panes,
                include_idle=args.include_idle,
            )
            agent_by_project = detected_agents if run_agents else None

            if not project_ids:
                if args.projects or args.group:
                    project_ids = choose_projects(args.projects, limit=args.max_panes, group=args.group)
                    agent_by_project = None
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
                capture_lines=80,
                wait_attention_min=1,
                replace=args.manager_replace,
                no_colorize=False,
                focus=True,
                attach=False,
            )
        )

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
        apply_colors=(not args.no_colorize),
    )
    attention = len([row for row in rows if row.get("needs_attention")])
    print(f"Painted session '{session}' panes={len(rows)} needs_attention={attention}")
    for row in rows:
        wait = f"{row['wait']}m" if row.get("wait") is not None else "-"
        mark = "!" if row.get("needs_attention") else " "
        print(
            f"{mark} {row['index']:<2} {row['project_id']:<22} {row['health']:<6} {row['status']:<10} "
            f"wait={wait:<4} agent={row['agent']:<8} reason={row['reason']}"
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
                apply_colors=(not args.no_colorize),
            )
            if not args.no_clear:
                print("\033[2J\033[H", end="")
            attention = len([row for row in rows if row.get("needs_attention")])
            print(f"[{now_iso()}] Agent Wrangler Manager  session={session} panes={len(rows)} attention={attention}")
            print(f"{'IDX':<4} {'PROJECT':<22} {'HLTH':<6} {'STATUS':<10} {'WAIT':<6} {'AGENT':<8} REASON")
            for row in rows:
                wait = f"{row['wait']}m" if row.get("wait") is not None else "-"
                print(
                    f"{row['index']:<4} {row['project_id']:<22} {row['health']:<6} {row['status']:<10} "
                    f"{wait:<6} {row['agent']:<8} {row['reason']}"
                )

            print("")
            print("Navigation: Ctrl-b + arrows | Ctrl-b + o next pane | Ctrl-b + q show pane numbers")
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
    if not manager_window_exists(session, window):
        script = Path(__file__).resolve()
        cmd = (
            f"python3 {shlex.quote(str(script))} teams watch --session {shlex.quote(session)} "
            f"--interval {max(1, int(args.interval))} --capture-lines {max(20, int(args.capture_lines))} "
            f"--wait-attention-min {max(0, int(args.wait_attention_min))}"
        )
        if args.no_colorize:
            cmd += " --no-colorize"
        shell_tail = "; EXIT_CODE=$?; echo manager_exited:$EXIT_CODE; exec zsh"
        shell_command = "zsh -lc " + shlex.quote(cmd + shell_tail)
        code, _, err = tmux(
            ["new-window", "-d", "-t", session, "-n", window, "-c", str(ROOT), shell_command],
            timeout=8,
        )
        if code != 0:
            raise ValueError(err.strip() or f"failed to create manager window '{window}'")
        if not manager_window_exists(session, window):
            raise ValueError(f"manager window '{window}' did not persist")
        print(f"Manager window started: {session}:{window}")

    if args.focus:
        tmux(["select-window", "-t", f"{session}:{window}"], timeout=5)
    if args.attach:
        return attach_session(session)
    return 0


def fleet_health_rows(
    *,
    sessions: list[str],
    capture_lines: int,
    wait_attention_min: int,
    apply_colors: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session in sessions:
        if not session_exists(session):
            rows.append(
                {
                    "session": session,
                    "ok": False,
                    "error": "session not found",
                    "panes": 0,
                    "attention": 0,
                    "waiting": 0,
                    "active": 0,
                    "red": 0,
                    "yellow": 0,
                    "green": 0,
                    "top_reason": "-",
                }
            )
            continue
        try:
            rows.append(
                session_health_summary(
                    session=session,
                    capture_lines=capture_lines,
                    wait_attention_min=wait_attention_min,
                    apply_colors=apply_colors,
                )
            )
        except ValueError as exc:
            rows.append(
                {
                    "session": session,
                    "ok": False,
                    "error": str(exc),
                    "panes": 0,
                    "attention": 0,
                    "waiting": 0,
                    "active": 0,
                    "red": 0,
                    "yellow": 0,
                    "green": 0,
                    "top_reason": "-",
                }
            )
    return rows


def print_fleet_table(rows: list[dict[str, Any]]) -> None:
    print(
        f"{'SESSION':<22} {'OK':<3} {'PANES':<5} {'ATTN':<4} {'WAIT':<4} {'ACT':<4} "
        f"{'R':<2} {'Y':<2} {'G':<2} TOP_REASON"
    )
    for row in rows:
        ok = "yes" if row.get("ok") else "no"
        top_reason = row.get("top_reason") or row.get("error") or "-"
        print(
            f"{str(row.get('session') or '-'):<22} {ok:<3} {int(row.get('panes') or 0):<5} "
            f"{int(row.get('attention') or 0):<4} {int(row.get('waiting') or 0):<4} {int(row.get('active') or 0):<4} "
            f"{int(row.get('red') or 0):<2} {int(row.get('yellow') or 0):<2} {int(row.get('green') or 0):<2} "
            f"{str(top_reason)}"
        )


def run_fleet_list(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    running = list_tmux_sessions()
    managed = set(store_managed_sessions(store))
    fleet_cfg = store.get("fleet", {})
    manager_session = str(fleet_cfg.get("manager_session") or DEFAULT_FLEET_MANAGER_SESSION)
    manager_window = str(fleet_cfg.get("manager_window") or DEFAULT_FLEET_MANAGER_WINDOW)

    print(f"Fleet manager session: {manager_session}  window: {manager_window}")
    if not running:
        print("No running tmux sessions.")
        return 0
    print(f"{'SESSION':<24} {'MANAGED':<7} {'ROLE':<10}")
    for session in running:
        is_managed = session in managed if managed else (session != manager_session)
        role = "manager" if session == manager_session else "work"
        print(f"{session:<24} {('yes' if is_managed else 'no'):<7} {role:<10}")
    return 0


def run_fleet_set(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    sessions = split_csv(args.sessions)
    if not sessions:
        raise ValueError("Pass --sessions s1,s2,...")

    running = set(list_tmux_sessions())
    missing = [name for name in sessions if name not in running]
    if missing and not args.allow_missing:
        raise ValueError(
            "Session(s) not running: {vals}. Use --allow-missing to store anyway.".format(vals=", ".join(missing))
        )

    fleet = store.setdefault("fleet", {})
    fleet["managed_sessions"] = sorted(set(sessions))
    if args.manager_session:
        fleet["manager_session"] = args.manager_session
    if args.manager_window:
        fleet["manager_window"] = args.manager_window
    save_store(store)

    print("Saved managed fleet sessions: " + ", ".join(fleet["managed_sessions"]))
    print(
        "Manager target: {session}:{window}".format(
            session=fleet.get("manager_session", DEFAULT_FLEET_MANAGER_SESSION),
            window=fleet.get("manager_window", DEFAULT_FLEET_MANAGER_WINDOW),
        )
    )
    return 0


def run_fleet_clear(_: argparse.Namespace) -> int:
    store = load_store()
    fleet = store.setdefault("fleet", {})
    fleet["managed_sessions"] = []
    save_store(store)
    print("Cleared managed fleet sessions. Fleet commands will auto-target all running tmux sessions.")
    return 0


def run_fleet_status(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    sessions = resolve_fleet_sessions(
        store=store,
        explicit_csv=args.sessions,
        pattern=args.pattern,
        include_manager=args.include_manager,
    )
    if not sessions:
        print("No fleet sessions found. Start a team session or configure with: agent-wrangler fleet set --sessions ...")
        return 0

    rows = fleet_health_rows(
        sessions=sessions,
        capture_lines=args.capture_lines,
        wait_attention_min=args.wait_attention_min,
        apply_colors=(not args.no_colorize),
    )
    totals = {
        "sessions": len(rows),
        "panes": sum(int(row.get("panes") or 0) for row in rows),
        "attention": sum(int(row.get("attention") or 0) for row in rows),
        "waiting": sum(int(row.get("waiting") or 0) for row in rows),
        "active": sum(int(row.get("active") or 0) for row in rows),
        "red": sum(int(row.get("red") or 0) for row in rows),
        "yellow": sum(int(row.get("yellow") or 0) for row in rows),
        "green": sum(int(row.get("green") or 0) for row in rows),
    }

    print(
        "[{ts}] Fleet status  sessions={sessions} panes={panes} attention={attention} waiting={waiting} active={active}".format(
            ts=now_iso(),
            sessions=totals["sessions"],
            panes=totals["panes"],
            attention=totals["attention"],
            waiting=totals["waiting"],
            active=totals["active"],
        )
    )
    print_fleet_table(rows)
    print("")
    print(
        "Color totals: red={red} yellow={yellow} green={green}".format(
            red=totals["red"],
            yellow=totals["yellow"],
            green=totals["green"],
        )
    )
    return 0


def run_fleet_watch(args: argparse.Namespace) -> int:
    ensure_tmux()
    loops = 0
    interval = max(1, int(args.interval))
    try:
        while True:
            store = load_store()
            sessions = resolve_fleet_sessions(
                store=store,
                explicit_csv=args.sessions,
                pattern=args.pattern,
                include_manager=args.include_manager,
            )
            rows = fleet_health_rows(
                sessions=sessions,
                capture_lines=args.capture_lines,
                wait_attention_min=args.wait_attention_min,
                apply_colors=(not args.no_colorize),
            )
            if not args.no_clear:
                print("\033[2J\033[H", end="")
            totals = {
                "sessions": len(rows),
                "panes": sum(int(row.get("panes") or 0) for row in rows),
                "attention": sum(int(row.get("attention") or 0) for row in rows),
                "waiting": sum(int(row.get("waiting") or 0) for row in rows),
                "active": sum(int(row.get("active") or 0) for row in rows),
            }
            print(
                "[{ts}] Agent Wrangler Fleet  sessions={sessions} panes={panes} attention={attention} waiting={waiting} active={active}".format(
                    ts=now_iso(),
                    sessions=totals["sessions"],
                    panes=totals["panes"],
                    attention=totals["attention"],
                    waiting=totals["waiting"],
                    active=totals["active"],
                )
            )
            if not sessions:
                print("No sessions matched. Use `agent-wrangler fleet set --sessions ...` or start tmux sessions.")
            else:
                print_fleet_table(rows)
            print("")
            print("Control: agent-wrangler fleet focus <session> | agent-wrangler fleet status | agent-wrangler drift --fleet")

            loops += 1
            if args.iterations > 0 and loops >= args.iterations:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        return 0
    return 0


def run_fleet_manager(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    fleet_cfg = store.setdefault("fleet", {})
    manager_session = args.manager_session or str(fleet_cfg.get("manager_session") or DEFAULT_FLEET_MANAGER_SESSION)
    manager_window = args.window or str(fleet_cfg.get("manager_window") or DEFAULT_FLEET_MANAGER_WINDOW)
    if args.update_defaults:
        fleet_cfg["manager_session"] = manager_session
        fleet_cfg["manager_window"] = manager_window
        save_store(store)

    script = Path(__file__).resolve()
    cmd_parts = [
        "python3",
        shlex.quote(str(script)),
        "teams",
        "fleet",
        "watch",
        "--interval",
        str(max(1, int(args.interval))),
        "--capture-lines",
        str(max(20, int(args.capture_lines))),
        "--wait-attention-min",
        str(max(0, int(args.wait_attention_min))),
    ]
    if args.sessions:
        cmd_parts.extend(["--sessions", shlex.quote(args.sessions)])
    if args.pattern:
        cmd_parts.extend(["--pattern", shlex.quote(args.pattern)])
    if args.include_manager:
        cmd_parts.append("--include-manager")
    if args.no_colorize:
        cmd_parts.append("--no-colorize")
    shell_tail = "; EXIT_CODE=$?; echo fleet_manager_exited:$EXIT_CODE; exec zsh"
    shell_command = "zsh -lc " + shlex.quote(" ".join(cmd_parts) + shell_tail)

    if not session_exists(manager_session):
        code, _, err = tmux(
            [
                "new-session",
                "-d",
                "-s",
                manager_session,
                "-n",
                manager_window,
                "-x",
                "260",
                "-y",
                "90",
                "-c",
                str(ROOT),
                shell_command,
            ],
            timeout=10,
        )
        if code != 0:
            raise ValueError(err.strip() or f"failed to create manager session '{manager_session}'")
    else:
        if manager_window_exists(manager_session, manager_window):
            if args.replace:
                code, _, err = tmux(["kill-window", "-t", f"{manager_session}:{manager_window}"], timeout=5)
                if code != 0:
                    raise ValueError(err.strip() or f"failed to replace fleet manager window '{manager_window}'")
            else:
                print(f"Fleet manager window '{manager_window}' already exists.")
        if not manager_window_exists(manager_session, manager_window):
            code, _, err = tmux(
                ["new-window", "-d", "-t", manager_session, "-n", manager_window, "-c", str(ROOT), shell_command],
                timeout=8,
            )
            if code != 0:
                raise ValueError(err.strip() or f"failed to create fleet manager window '{manager_window}'")

    if args.focus:
        tmux(["select-window", "-t", f"{manager_session}:{manager_window}"], timeout=5)
    print(f"Fleet manager ready: {manager_session}:{manager_window}")
    if args.attach:
        return attach_session(manager_session)
    return 0


def run_fleet_focus(args: argparse.Namespace) -> int:
    ensure_tmux()
    target = args.session_name
    if not session_exists(target):
        raise ValueError(f"Session '{target}' does not exist")

    if os.environ.get("TMUX"):
        code, _, _ = tmux(["switch-client", "-t", target], timeout=5)
        if code == 0:
            print(f"switched tmux client to session '{target}'")
            return 0

    return attach_session(target)


def run_fleet_jump(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    sessions = resolve_fleet_sessions(
        store=store,
        explicit_csv=args.sessions,
        pattern=args.pattern,
        include_manager=args.include_manager,
    )
    if not sessions:
        print("No sessions available for jump.")
        return 0

    if args.session_name:
        target = args.session_name.strip()
        if target not in sessions:
            known = ", ".join(sessions)
            raise ValueError(f"Session '{target}' is not in current fleet set. Known: {known}")
        return run_fleet_focus(argparse.Namespace(session_name=target))

    if os.environ.get("TMUX"):
        # Native tmux picker for fastest in-client jump navigation.
        code, _, err = tmux(["choose-tree", "-Zw"], timeout=20)
        if code != 0:
            raise ValueError(err.strip() or "failed to open tmux choose-tree")
        return 0

    print("Jump picker is best inside tmux.")
    print("Use one of:")
    for session in sessions:
        print(f"- agent-wrangler fleet jump --session-name {session}")
    return 0


def run_fleet_popup(args: argparse.Namespace) -> int:
    ensure_tmux()
    if not os.environ.get("TMUX"):
        print("fleet popup requires running inside an attached tmux client.")
        print("Fallback: agent-wrangler fleet manager --replace")
        return 0

    script = Path(__file__).resolve()
    watch_cmd = (
        f"python3 {shlex.quote(str(script))} teams fleet watch "
        f"--interval {max(1, int(args.interval))} "
        f"--capture-lines {max(20, int(args.capture_lines))} "
        f"--wait-attention-min {max(0, int(args.wait_attention_min))}"
    )
    if args.sessions:
        watch_cmd += f" --sessions {shlex.quote(args.sessions)}"
    if args.pattern:
        watch_cmd += f" --pattern {shlex.quote(args.pattern)}"
    if args.include_manager:
        watch_cmd += " --include-manager"
    if args.no_colorize:
        watch_cmd += " --no-colorize"

    popup_cmd = "zsh -lc " + shlex.quote(watch_cmd)
    code, _, err = tmux(
        [
            "display-popup",
            "-E",
            "-w",
            str(max(80, int(args.width))),
            "-h",
            str(max(18, int(args.height))),
            popup_cmd,
        ],
        timeout=20,
    )
    if code != 0:
        raise ValueError(err.strip() or "failed to open fleet popup")
    return 0


def run_drift(args: argparse.Namespace) -> int:
    ensure_tmux()
    store = load_store()
    proj_map = project_map()

    if args.fleet:
        sessions = resolve_fleet_sessions(
            store=store,
            explicit_csv=args.sessions,
            pattern=args.pattern,
            include_manager=args.include_manager,
        )
    else:
        session = args.session or store.get("default_session") or DEFAULT_SESSION
        sessions = [session]

    if not sessions:
        print("No sessions selected for drift view.")
        return 0

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
    all_bindings = pane_bindings + window_bindings + index_bindings

    if args.remove:
        for key, _cmd in all_bindings:
            tmux(["unbind-key", "-n", key], timeout=5)
        print("Removed no-prefix Alt navigation bindings (pane + window).")
        return 0

    for key, cmd in all_bindings:
        tmux(["bind-key", "-n", key, *cmd], timeout=5)
    print("Enabled no-prefix Alt navigation bindings.")
    print("Pane navigation: Option+Arrow")
    print("Window navigation: Option+[ / Option+]")
    print("Window direct jump: Option+1..9")
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

    paint = teams_sub.add_parser("paint", help="Color panes by attention state (green/red)")
    paint.add_argument("--session", default=None)
    paint.add_argument("--capture-lines", type=int, default=80, help="Recent lines to inspect for error markers")
    paint.add_argument("--wait-attention-min", type=int, default=1, help="Waiting minutes before marking red")
    paint.add_argument("--no-colorize", action="store_true", help="Compute health only, do not set pane colors")
    paint.set_defaults(handler=run_paint)

    watch = teams_sub.add_parser("watch", help="Live manager loop with health + attention states")
    watch.add_argument("--session", default=None)
    watch.add_argument("--interval", type=int, default=3)
    watch.add_argument("--iterations", type=int, default=0, help="0 means infinite")
    watch.add_argument("--capture-lines", type=int, default=80)
    watch.add_argument("--wait-attention-min", type=int, default=1)
    watch.add_argument("--no-colorize", action="store_true")
    watch.add_argument("--no-clear", action="store_true")
    watch.set_defaults(handler=run_watch)

    manager = teams_sub.add_parser("manager", help="Create/open orchestrator manager screen as tmux window")
    manager.add_argument("--session", default=None)
    manager.add_argument("--window", default="manager")
    manager.add_argument("--interval", type=int, default=3)
    manager.add_argument("--capture-lines", type=int, default=80)
    manager.add_argument("--wait-attention-min", type=int, default=1)
    manager.add_argument("--replace", action="store_true", help="Replace existing manager window")
    manager.add_argument("--no-colorize", action="store_true")
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

    drift = teams_sub.add_parser("drift", help="Show git drift for pane projects (AOE-style)")
    drift.add_argument("--session", default=None, help="Single tmux session (default: configured default_session)")
    drift.add_argument("--fleet", action="store_true", help="Run drift check across fleet sessions")
    drift.add_argument("--sessions", help="Comma-separated sessions override for --fleet")
    drift.add_argument("--pattern", help="Filter fleet sessions by substring")
    drift.add_argument("--include-manager", action="store_true", help="Include fleet manager session in --fleet mode")
    drift.add_argument("--alert-dirty", type=int, default=25, help="Dirty-file threshold for high-drift alerts")
    drift.set_defaults(handler=run_drift)

    fleet = teams_sub.add_parser("fleet", help="Multi-session orchestrator for all team grids")
    fleet_sub = fleet.add_subparsers(dest="fleet_command", required=True)

    fleet_list = fleet_sub.add_parser("list", help="List running tmux sessions and managed fleet set")
    fleet_list.set_defaults(handler=run_fleet_list)

    fleet_set = fleet_sub.add_parser("set", help="Persist managed fleet sessions")
    fleet_set.add_argument("--sessions", required=True, help="Comma-separated tmux session names")
    fleet_set.add_argument("--allow-missing", action="store_true", help="Allow storing sessions not currently running")
    fleet_set.add_argument("--manager-session", help="Default fleet manager tmux session name")
    fleet_set.add_argument("--manager-window", help="Default fleet manager window name")
    fleet_set.set_defaults(handler=run_fleet_set)

    fleet_clear = fleet_sub.add_parser("clear", help="Clear managed fleet sessions (fallback to all running)")
    fleet_clear.set_defaults(handler=run_fleet_clear)

    fleet_status = fleet_sub.add_parser("status", help="One-shot fleet health rollup")
    fleet_status.add_argument("--sessions", help="Comma-separated session override")
    fleet_status.add_argument("--pattern", help="Filter sessions by substring")
    fleet_status.add_argument("--include-manager", action="store_true")
    fleet_status.add_argument("--capture-lines", type=int, default=80)
    fleet_status.add_argument("--wait-attention-min", type=int, default=1)
    fleet_status.add_argument("--no-colorize", action="store_true")
    fleet_status.set_defaults(handler=run_fleet_status)

    fleet_watch = fleet_sub.add_parser("watch", help="Live fleet manager loop across sessions")
    fleet_watch.add_argument("--sessions", help="Comma-separated session override")
    fleet_watch.add_argument("--pattern", help="Filter sessions by substring")
    fleet_watch.add_argument("--include-manager", action="store_true")
    fleet_watch.add_argument("--capture-lines", type=int, default=80)
    fleet_watch.add_argument("--wait-attention-min", type=int, default=1)
    fleet_watch.add_argument("--interval", type=int, default=3)
    fleet_watch.add_argument("--iterations", type=int, default=0, help="0 means infinite")
    fleet_watch.add_argument("--no-colorize", action="store_true")
    fleet_watch.add_argument("--no-clear", action="store_true")
    fleet_watch.set_defaults(handler=run_fleet_watch)

    fleet_manager = fleet_sub.add_parser("manager", help="Create/open dedicated fleet manager tmux session")
    fleet_manager.add_argument("--manager-session", default=None, help="Fleet manager tmux session name")
    fleet_manager.add_argument("--window", default=None, help="Fleet manager window name")
    fleet_manager.add_argument("--sessions", help="Comma-separated session override for manager watch loop")
    fleet_manager.add_argument("--pattern", help="Filter sessions by substring for manager watch loop")
    fleet_manager.add_argument("--include-manager", action="store_true")
    fleet_manager.add_argument("--capture-lines", type=int, default=80)
    fleet_manager.add_argument("--wait-attention-min", type=int, default=1)
    fleet_manager.add_argument("--interval", type=int, default=3)
    fleet_manager.add_argument("--replace", action="store_true")
    fleet_manager.add_argument("--no-colorize", action="store_true")
    fleet_manager.add_argument("--focus", action="store_true", default=True)
    fleet_manager.add_argument("--no-focus", dest="focus", action="store_false")
    fleet_manager.add_argument("--attach", action="store_true", default=True)
    fleet_manager.add_argument("--no-attach", dest="attach", action="store_false")
    fleet_manager.add_argument(
        "--update-defaults",
        action="store_true",
        help="Persist manager session/window defaults to team_grid.json",
    )
    fleet_manager.set_defaults(handler=run_fleet_manager)

    fleet_focus = fleet_sub.add_parser("focus", help="Switch/attach to a target tmux session")
    fleet_focus.add_argument("session_name")
    fleet_focus.set_defaults(handler=run_fleet_focus)

    fleet_jump = fleet_sub.add_parser("jump", help="Fast jump between sessions (choose-tree inside tmux)")
    fleet_jump.add_argument("--session-name", help="Jump directly to a specific session")
    fleet_jump.add_argument("--sessions", help="Comma-separated session override")
    fleet_jump.add_argument("--pattern", help="Filter sessions by substring")
    fleet_jump.add_argument("--include-manager", action="store_true")
    fleet_jump.set_defaults(handler=run_fleet_jump)

    fleet_popup = fleet_sub.add_parser("popup", help="Open fleet watch in a tmux popup")
    fleet_popup.add_argument("--sessions", help="Comma-separated session override")
    fleet_popup.add_argument("--pattern", help="Filter sessions by substring")
    fleet_popup.add_argument("--include-manager", action="store_true")
    fleet_popup.add_argument("--capture-lines", type=int, default=80)
    fleet_popup.add_argument("--wait-attention-min", type=int, default=1)
    fleet_popup.add_argument("--interval", type=int, default=3)
    fleet_popup.add_argument("--width", type=int, default=220, help="Popup width (cells)")
    fleet_popup.add_argument("--height", type=int, default=50, help="Popup height (cells)")
    fleet_popup.add_argument("--no-colorize", action="store_true")
    fleet_popup.set_defaults(handler=run_fleet_popup)


def main() -> int:
    parser = argparse.ArgumentParser(description="Tmux Team Grid")
    sub = parser.add_subparsers(dest="command", required=True)
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
