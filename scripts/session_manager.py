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
    self_path = os.path.realpath(str(ROOT))

    enriched = []
    for sess in sessions:
        cwd = sess.get("cwd") or ""

        # Skip agent-wrangler's own sessions
        if cwd and os.path.realpath(cwd) == self_path:
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
