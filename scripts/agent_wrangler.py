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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure aw package is importable
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from aw.core import *  # noqa: F403
import aw.core as _core

# Re-export mutable module-level state from aw.core so that rail and run_*
# functions can modify dicts/lists by reference (mutations propagate).
# Scalar globals (_campfire_frame) were converted to single-element lists in
# aw.core, so mutations via _core._campfire_frame[0] also propagate.

from aw.rail import run_rail  # noqa: F401


# ── Command handlers (run_*) ──────────────────────────────────────

def run_bootstrap(args: argparse.Namespace) -> int:
    ensure_tmux()  # noqa: F405
    store = load_store()  # noqa: F405

    session = args.session or store.get("default_session") or _core.DEFAULT_SESSION
    layout = args.layout or store.get("default_layout") or _core.DEFAULT_LAYOUT

    project_ids = choose_projects(args.projects, limit=args.limit, group=args.group)  # noqa: F405
    proj_map_data = project_map()  # noqa: F405

    resolved_layout, _ = create_grid_session(  # noqa: F405
        session=session,
        layout=layout,
        project_ids=project_ids,
        proj_map=proj_map_data,
        project_overrides=None,
        no_startup=args.no_startup,
        agent_default=args.agent,
        agent_by_project=None,
        force=args.force,
    )

    store["default_session"] = session
    store["default_layout"] = resolved_layout
    store["default_projects"] = project_ids
    save_store(store)  # noqa: F405

    print(f"Created tmux team grid '{session}' with {len(project_ids)} panes")
    print("Projects: " + ", ".join(project_ids))
    print(f"Layout: {resolved_layout}")
    print(f"Attach: tmux attach -t {session}")

    if args.attach:
        return attach_session(session)  # noqa: F405
    return 0


def run_import(args: argparse.Namespace) -> int:
    ensure_tmux()  # noqa: F405
    store = load_store()  # noqa: F405
    proj_map_data = project_map()  # noqa: F405

    session = args.session or store.get("default_session") or _core.DEFAULT_SESSION
    layout = args.layout or store.get("default_layout") or _core.DEFAULT_LAYOUT
    run_startup = bool(args.startup) and (not bool(args.no_startup))
    run_agents = bool(args.agent) and (not bool(args.no_agent))

    project_ids, agent_by_project, project_overrides, mapped, unmatched = ghostty_import_plan(  # noqa: F405
        proj_map=proj_map_data,
        max_panes=args.max_panes,
        include_idle=args.include_idle,
        preserve_duplicates=args.preserve_duplicates,
    )
    if not project_ids:
        project_ids = choose_projects(None, limit=args.max_panes, group=None)  # noqa: F405
        agent_by_project = {}
        project_overrides = {}
        mapped = []
        unmatched = []
        print("No Ghostty terminals detected — starting from projects.json.")

    # Auto-register discovered terminals into projects.json
    _auto_register_projects(project_ids, project_overrides, proj_map_data)  # noqa: F405

    resolved_layout = choose_layout(layout, pane_count=len(project_ids))  # noqa: F405
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

    resolved_layout, _ = create_grid_session(  # noqa: F405
        session=session,
        layout=layout,
        project_ids=project_ids,
        proj_map=proj_map_data,
        project_overrides=project_overrides,
        no_startup=(not run_startup),
        agent_default=None,
        agent_by_project=(agent_by_project if run_agents else None),
        force=args.force,
    )

    store["default_session"] = session
    store["default_layout"] = resolved_layout
    store["default_projects"] = project_ids
    save_store(store)  # noqa: F405

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
        return attach_session(session)  # noqa: F405
    return 0


def run_up(args: argparse.Namespace) -> int:
    ensure_tmux()  # noqa: F405
    store = load_store()  # noqa: F405
    proj_map_data = project_map()  # noqa: F405

    session = args.session or store.get("default_session") or _core.DEFAULT_SESSION
    layout = args.layout or store.get("default_layout") or _core.DEFAULT_LAYOUT
    run_startup = bool(args.startup) and (not bool(args.no_startup))
    run_agents = bool(args.agent) and (not bool(args.no_agent))

    exists = session_exists(session)  # noqa: F405
    if exists and not args.rebuild:
        print(f"Using existing session '{session}'")
    else:
        if args.mode == "bootstrap":
            project_ids = choose_projects(args.projects, limit=args.max_panes, group=args.group)  # noqa: F405
            agent_by_project = None
            project_overrides = None
        else:
            project_ids, detected_agents, project_overrides, mapped, unmatched = ghostty_import_plan(  # noqa: F405
                proj_map=proj_map_data,
                max_panes=args.max_panes,
                include_idle=args.include_idle,
                preserve_duplicates=args.preserve_duplicates,
            )
            agent_by_project = detected_agents if run_agents else None

            if not project_ids:
                project_ids = choose_projects(args.projects, limit=args.max_panes, group=args.group)  # noqa: F405
                agent_by_project = None
                project_overrides = None
                if args.projects or args.group:
                    print("No Ghostty matches found; using configured project selection fallback.")
                else:
                    print("No Ghostty terminals detected — starting from projects.json.")
            else:
                _auto_register_projects(project_ids, project_overrides, proj_map_data)  # noqa: F405
                print(f"Ghostty mapping: matched={len(mapped)} unmatched={len(unmatched)}")

        resolved_layout, _ = create_grid_session(  # noqa: F405
            session=session,
            layout=layout,
            project_ids=project_ids,
            proj_map=proj_map_data,
            project_overrides=project_overrides,
            no_startup=(not run_startup),
            agent_default=None,
            agent_by_project=agent_by_project,
            force=exists,
        )

        store["default_session"] = session
        store["default_layout"] = resolved_layout
        store["default_projects"] = project_ids
        save_store(store)  # noqa: F405

        # --- Roundup animation: show each project being wrangled ---
        skip_anim = os.environ.get("AW_SKIP_ANIM", "").strip() in ("1", "true", "yes")
        RUST = "\033[38;5;130m"
        GREEN = "\033[32m"
        DIM = "\033[2m"
        RST = "\033[0m"

        print(f"\n  {RUST}Rounding up the herd...{RST}\n")

        for pid in project_ids:
            if skip_anim:
                print(f"  {DIM}◦ ─ ─{RST} {GREEN}●{RST} {RUST}{pid:<24}{RST} {GREEN}wrangled ✓{RST}")
            else:
                # Lasso animation: rope extends, catches the project
                sys.stdout.write(f"  {DIM}◦{RST}")
                sys.stdout.flush()
                time.sleep(0.08)
                sys.stdout.write(f" {DIM}─{RST}")
                sys.stdout.flush()
                time.sleep(0.06)
                sys.stdout.write(f" {DIM}─{RST}")
                sys.stdout.flush()
                time.sleep(0.06)
                sys.stdout.write(f" {GREEN}●{RST} {RUST}{pid:<24}{RST}")
                sys.stdout.flush()
                time.sleep(0.08)
                sys.stdout.write(f" {GREEN}wrangled ✓{RST}\n")
                sys.stdout.flush()

        n = len(project_ids)
        print(f"\n  {GREEN}✓{RST} All {n} head accounted for. Let's ride.\n")
        play_sound("Bottle", 0.35)  # noqa: F405

        print(f"Session ready: {session}")
        print(f"Panes: {n}  Layout: {resolved_layout}")

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
    set_window_orchestrator_format(session)  # noqa: F405

    show_status = bool(getattr(args, "status", True))
    attach = bool(getattr(args, "attach", True))

    if show_status:
        status_args = argparse.Namespace(session=session)
        run_status(status_args)

    if attach:
        return attach_session(session)  # noqa: F405
    return 0


def run_attach(args: argparse.Namespace) -> int:
    ensure_tmux()  # noqa: F405
    store = load_store()  # noqa: F405
    session = args.session or store.get("default_session") or _core.DEFAULT_SESSION
    if not session_exists(session):  # noqa: F405
        raise ValueError(f"Session '{session}' does not exist")
    return attach_session(session)  # noqa: F405


def run_status(args: argparse.Namespace) -> int:
    ensure_tmux()  # noqa: F405
    store = load_store()  # noqa: F405
    session = args.session or store.get("default_session") or _core.DEFAULT_SESSION
    if not session_exists(session):  # noqa: F405
        raise ValueError(f"Session '{session}' does not exist")
    proj_map_data = project_map()  # noqa: F405
    panes = backfill_pane_project_ids(session, list_panes(session), proj_map_data)  # noqa: F405
    print_panes(session, panes)  # noqa: F405
    return 0


def run_paint(args: argparse.Namespace) -> int:
    ensure_tmux()  # noqa: F405
    store = load_store()  # noqa: F405
    session = args.session or store.get("default_session") or _core.DEFAULT_SESSION
    if not session_exists(session):  # noqa: F405
        raise ValueError(f"Session '{session}' does not exist")

    rows = refresh_pane_health(  # noqa: F405
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
    ensure_tmux()  # noqa: F405
    store = load_store()  # noqa: F405
    session = args.session or store.get("default_session") or _core.DEFAULT_SESSION
    if not session_exists(session):  # noqa: F405
        raise ValueError(f"Session '{session}' does not exist")

    loops = 0
    interval = max(1, int(args.interval))
    try:
        while True:
            rows = refresh_pane_health(  # noqa: F405
                session=session,
                capture_lines=args.capture_lines,
                wait_attention_min=args.wait_attention_min,

            )
            _core._log_transitions(rows)
            if not args.no_clear:
                print("\033[2J\033[H", end="")
            attention = len([row for row in rows if row.get("needs_attention")])
            print(f"[{now_iso()}] Agent Wrangler Manager  session={session} panes={len(rows)} attention={attention}")  # noqa: F405
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
    code, out, _ = tmux(["list-windows", "-t", session, "-F", "#{window_name}"], timeout=5)  # noqa: F405
    if code != 0:
        return False
    names = {line.strip() for line in out.splitlines() if line.strip()}
    return window_name in names


def run_manager(args: argparse.Namespace) -> int:
    ensure_tmux()  # noqa: F405
    store = load_store()  # noqa: F405
    session = args.session or store.get("default_session") or _core.DEFAULT_SESSION
    if not session_exists(session):  # noqa: F405
        raise ValueError(f"Session '{session}' does not exist")

    window = args.window
    if manager_window_exists(session, window):
        if args.replace:
            code, _, err = tmux(["kill-window", "-t", f"{session}:{window}"], timeout=5)  # noqa: F405
            if code != 0:
                raise ValueError(err.strip() or f"failed to replace manager window '{window}'")
        else:
            print(f"Manager window '{window}' already exists.")
            if args.focus:
                tmux(["select-window", "-t", f"{session}:{window}"], timeout=5)  # noqa: F405
            if getattr(args, "attach", False):
                return attach_session(session)  # noqa: F405
            return 0

    if not manager_window_exists(session, window):
        # Create manager window with Claude Code
        wrangler_root = str(_core.ROOT)
        claude_cmd = "claude --dangerously-skip-permissions"
        shell_tail = "; exec zsh"
        shell_command = "zsh -lc " + shlex.quote(claude_cmd + shell_tail)
        code, _, err = tmux(  # noqa: F405
            ["new-window", "-d", "-t", session, "-n", window, "-c", wrangler_root, shell_command],
            timeout=8,
        )
        if code != 0:
            print(f"Warning: failed to create manager window '{window}': {err.strip()}")
            return 1

        # Split right pane for status rail (~25% width)
        rail_script = _core.ROOT / "scripts" / "agent_wrangler.py"
        rail_cmd = (
            f"python3 {shlex.quote(str(rail_script))} teams rail "
            f"--session {shlex.quote(session)} --interval {max(1, int(args.interval))}"
        )
        rail_shell = "zsh -lc " + shlex.quote(rail_cmd + shell_tail)
        code, _, err = tmux(  # noqa: F405
            ["split-window", "-h", "-t", f"{session}:{window}", "-l", "25%",
             "-c", wrangler_root, rail_shell],
            timeout=8,
        )
        if code != 0:
            print(f"Warning: failed to create status rail split: {err.strip()}")

        # Focus the left pane (Claude Code) within the manager window
        tmux(["select-pane", "-t", f"{session}:{window}.0"], timeout=5)  # noqa: F405

        if not manager_window_exists(session, window):
            raise ValueError(f"manager window '{window}' did not persist")
        print(f"Manager window started: {session}:{window} (claude + rail)")

    if args.focus:
        tmux(["select-window", "-t", f"{session}:{window}"], timeout=5)  # noqa: F405
    if getattr(args, "attach", False):
        return attach_session(session)  # noqa: F405
    return 0



def run_profile_list(_: argparse.Namespace) -> int:
    store = load_store()  # noqa: F405
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
    store = load_store()  # noqa: F405
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
    ensure_tmux()  # noqa: F405
    store = load_store()  # noqa: F405
    profiles = store.setdefault("profiles", {})
    items = profiles.setdefault("items", {})
    if not isinstance(items, dict):
        items = {}
        profiles["items"] = items

    name = str(args.name or "").strip().lower()
    if not name:
        raise ValueError("Profile name is required")

    sessions = split_csv(args.sessions)  # noqa: F405
    if not sessions and args.auto_running:
        sessions = [s for s in list_tmux_sessions() if s]  # noqa: F405

    item = items.setdefault(name, {})
    item["managed_sessions"] = sorted(set(sessions))
    item["max_panes"] = max(1, int(args.max_panes))
    save_store(store)  # noqa: F405

    print(f"Saved profile '{name}'")
    print(f"- max_panes: {item['max_panes']}")
    print(f"- managed_sessions: {', '.join(item['managed_sessions']) if item['managed_sessions'] else '-'}")
    return 0


def run_profile_use(args: argparse.Namespace) -> int:
    ensure_tmux()  # noqa: F405
    store = load_store()  # noqa: F405
    profiles = store.setdefault("profiles", {})
    items = profiles.get("items", {}) if isinstance(profiles.get("items"), dict) else {}

    name = str(args.name or "").strip().lower()
    if name not in items:
        known = ", ".join(sorted(items.keys()))
        raise ValueError(f"Unknown profile '{name}'. Known: {known}")

    profiles["current"] = name
    item = items.get(name, {})
    save_store(store)  # noqa: F405

    print(f"Active profile: {name}")
    print(f"- max_panes: {int(item.get('max_panes') or 10)}")
    print(f"- start hint: AW_MAX_PANES={int(item.get('max_panes') or 10)} ./scripts/agent-wrangler start")
    return 0




def run_drift(args: argparse.Namespace) -> int:
    ensure_tmux()  # noqa: F405
    store = load_store()  # noqa: F405
    proj_map_data = project_map()  # noqa: F405

    session = args.session or store.get("default_session") or _core.DEFAULT_SESSION
    sessions = [session]

    total_projects = 0
    total_dirty = 0
    high_drift: list[tuple[str, str, int]] = []

    for session in sessions:
        if not session_exists(session):  # noqa: F405
            print(f"\n[{session}] missing")
            continue
        try:
            projects = project_rows_for_session(session, proj_map_data)  # noqa: F405
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
    store = load_store()  # noqa: F405
    persistence = store.get("persistence", {})
    enabled = bool(persistence.get("enabled"))
    autosave = int(persistence.get("autosave_minutes") or 15)
    last_snapshot = str(persistence.get("last_snapshot") or "")
    save_script, restore_script = tmux_resurrect_scripts()  # noqa: F405

    print("Persistence status")
    print(f"- enabled: {'yes' if enabled else 'no'}")
    print(f"- autosave_minutes: {autosave}")
    print(f"- last_snapshot: {last_snapshot or '-'}")
    print(f"- tmux-resurrect save script: {'yes' if save_script.exists() else 'no'} ({save_script})")
    print(f"- tmux-resurrect restore script: {'yes' if restore_script.exists() else 'no'} ({restore_script})")

    snapshots: list[Path] = []
    if _core.PERSISTENCE_DIR.exists():
        snapshots = sorted(_core.PERSISTENCE_DIR.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    print(f"- local snapshots: {len(snapshots)}")
    for path in snapshots[:10]:
        stamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat()
        print(f"  {path.name}  {stamp}")
    return 0


def run_persistence_enable(args: argparse.Namespace) -> int:
    store = load_store()  # noqa: F405
    persistence = store.setdefault("persistence", {})
    persistence["enabled"] = True
    persistence["autosave_minutes"] = max(1, int(args.autosave_minutes))
    save_store(store)  # noqa: F405
    print(
        "Persistence enabled (autosave_minutes={mins}).".format(
            mins=int(persistence.get("autosave_minutes") or 15)
        )
    )
    print("Tip: run `agent-wrangler persistence save` at key checkpoints.")
    return 0


def run_persistence_disable(_: argparse.Namespace) -> int:
    store = load_store()  # noqa: F405
    persistence = store.setdefault("persistence", {})
    persistence["enabled"] = False
    save_store(store)  # noqa: F405
    print("Persistence disabled.")
    return 0


def run_persistence_save(args: argparse.Namespace) -> int:
    ensure_tmux()  # noqa: F405
    store = load_store()  # noqa: F405
    session = args.session or store.get("default_session") or _core.DEFAULT_SESSION
    if not session_exists(session):  # noqa: F405
        raise ValueError(f"Session '{session}' does not exist")

    proj_map_data = project_map()  # noqa: F405
    panes = backfill_pane_project_ids(session, list_panes(session), proj_map_data)  # noqa: F405
    import terminal_sentinel as _ts
    snapshot, _ = _ts.classify_sessions(source_filter="all", include_idle=True)
    by_tty, by_path = _core._build_session_indexes(snapshot)
    pane_rows: list[dict[str, Any]] = []
    for pane in panes:
        tty_short = pane.pane_tty.split("/")[-1]
        monitor = _core._resolve_monitor(tty_short, pane.pane_path, by_tty, by_path)
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

    layout_hint = str(store.get("default_layout") or _core.DEFAULT_LAYOUT)
    data = {
        "saved_at": now_iso(),  # noqa: F405
        "session": session,
        "layout": (layout_hint if layout_hint in _core.LAYOUT_CHOICES else _core.DEFAULT_LAYOUT),
        "pane_count": len(pane_rows),
        "panes": pane_rows,
    }

    if args.file:
        snapshot_path = Path(args.file).expanduser()
    else:
        snapshot_path = persistence_snapshot_path(args.name or session)  # noqa: F405
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    persistence = store.setdefault("persistence", {})
    persistence["last_snapshot"] = str(snapshot_path)
    save_store(store)  # noqa: F405

    print(f"Saved persistence snapshot: {snapshot_path}")
    print(f"Session: {session}  panes={len(pane_rows)}")

    if args.tmux_resurrect:
        save_script, _ = tmux_resurrect_scripts()  # noqa: F405
        if not save_script.exists():
            raise ValueError(f"tmux-resurrect save script not found: {save_script}")
        code, _, err = run([str(save_script)], timeout=45)  # noqa: F405
        if code != 0:
            detail = (err or "").strip()
            raise ValueError(detail or "tmux-resurrect save failed")
        print("tmux-resurrect save completed.")
    return 0


def run_persistence_restore(args: argparse.Namespace) -> int:
    ensure_tmux()  # noqa: F405
    store = load_store()  # noqa: F405
    if args.file:
        snapshot_path = Path(args.file).expanduser()
    elif args.name:
        snapshot_path = persistence_snapshot_path(args.name)  # noqa: F405
    else:
        remembered = str(store.get("persistence", {}).get("last_snapshot") or "")
        snapshot_path = Path(remembered).expanduser() if remembered else persistence_snapshot_path(  # noqa: F405
            store.get("default_session") or _core.DEFAULT_SESSION
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

    session = args.session or str(payload.get("session") or store.get("default_session") or _core.DEFAULT_SESSION)
    proj_map_data = project_map()  # noqa: F405
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
            path = str(proj_map_data.get(base_project_id, {}).get("path") or "").strip()
        if not path:
            continue

        base_project = dict(proj_map_data.get(base_project_id, {}))
        base_project["path"] = path
        base_project.setdefault("startup_command", "")
        project_overrides[project_id] = base_project
        project_ids.append(project_id)

        agent = str(pane.get("agent") or "").strip().lower()
        if agent in {"claude", "codex", "gemini"}:
            agent_by_project[project_id] = agent

    if not project_ids:
        raise ValueError("snapshot has no restorable pane paths")

    layout_hint = str(payload.get("layout") or _core.DEFAULT_LAYOUT)
    layout = args.layout or (layout_hint if layout_hint in _core.LAYOUT_CHOICES else _core.DEFAULT_LAYOUT)
    resolved_layout, _ = create_grid_session(  # noqa: F405
        session=session,
        layout=layout,
        project_ids=project_ids,
        proj_map=proj_map_data,
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
    save_store(store)  # noqa: F405

    print(f"Restored snapshot into session '{session}'")
    print(f"Panes: {len(project_ids)}  Layout: {resolved_layout}")
    print(f"Source: {snapshot_path}")
    print(f"Startup commands: {'enabled' if args.startup else 'disabled'}")
    print(f"Agent relaunch: {'enabled' if args.agent else 'disabled'}")

    if args.tmux_resurrect:
        _, restore_script = tmux_resurrect_scripts()  # noqa: F405
        if not restore_script.exists():
            raise ValueError(f"tmux-resurrect restore script not found: {restore_script}")
        code, _, err = run([str(restore_script)], timeout=60)  # noqa: F405
        if code != 0:
            detail = (err or "").strip()
            raise ValueError(detail or "tmux-resurrect restore failed")
        print("tmux-resurrect restore completed.")

    if args.attach:
        return attach_session(session)  # noqa: F405
    return 0


def run_hooks_enable(args: argparse.Namespace) -> int:
    ensure_tmux()  # noqa: F405
    store = load_store()  # noqa: F405
    session = args.session or store.get("default_session") or _core.DEFAULT_SESSION
    if not session_exists(session):  # noqa: F405
        raise ValueError(f"Session '{session}' does not exist")

    script = Path(__file__).resolve()
    paint_cmd = (
        f"python3 {shlex.quote(str(script))} teams paint --session {shlex.quote(session)} "
        f"--capture-lines {max(20, int(args.capture_lines))} --wait-attention-min {max(0, int(args.wait_attention_min))} "
        "--no-colorize"
    )
    hook_cmd = f"run-shell -b {shlex.quote(paint_cmd + ' >/dev/null 2>&1')}"
    for name in _core.HOOK_EVENTS:
        code, _, err = tmux(["set-hook", "-t", session, name, hook_cmd], timeout=5)  # noqa: F405
        if code != 0:
            raise ValueError(err.strip() or f"failed to set hook '{name}'")
    print(f"Enabled hooks for session '{session}'")
    for name in _core.HOOK_EVENTS:
        print(f"- {name}")
    return 0


def run_hooks_disable(args: argparse.Namespace) -> int:
    ensure_tmux()  # noqa: F405
    store = load_store()  # noqa: F405
    session = args.session or store.get("default_session") or _core.DEFAULT_SESSION
    if not session_exists(session):  # noqa: F405
        raise ValueError(f"Session '{session}' does not exist")

    for name in _core.HOOK_EVENTS:
        tmux(["set-hook", "-u", "-t", session, name], timeout=5)  # noqa: F405
    print(f"Disabled hooks for session '{session}'")
    return 0


def run_hooks_status(args: argparse.Namespace) -> int:
    ensure_tmux()  # noqa: F405
    store = load_store()  # noqa: F405
    session = args.session or store.get("default_session") or _core.DEFAULT_SESSION
    if not session_exists(session):  # noqa: F405
        raise ValueError(f"Session '{session}' does not exist")

    code, out, err = tmux(["show-hooks", "-t", session], timeout=5)  # noqa: F405
    if code != 0:
        raise ValueError(err.strip() or f"failed to read hooks for session '{session}'")

    found = 0
    print(f"Hooks status for '{session}'")
    for name in _core.HOOK_EVENTS:
        matched = [line.strip() for line in out.splitlines() if line.strip().startswith(name)]
        if matched:
            found += 1
            print(f"- {name}: enabled")
        else:
            print(f"- {name}: disabled")
    print(f"Enabled hooks: {found}/{len(_core.HOOK_EVENTS)}")
    return 0


def doctor_fix_for_row(row: dict[str, Any]) -> str:
    missing_command = str(row.get("missing_command") or "")
    if missing_command:
        if missing_command == "code":
            return "Use `codex` for OpenAI Codex CLI, or install VS Code shell command `code`."
        if missing_command in {"claude", "codex", "gemini"} and not shutil.which(missing_command):
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
    ensure_tmux()  # noqa: F405
    store = load_store()  # noqa: F405

    session = args.session or store.get("default_session") or _core.DEFAULT_SESSION
    sessions = [session]

    print("Agent Wrangler Doctor")
    print("Tool availability:")
    for tool in ["claude", "codex", "gemini", "npm", "pnpm", "bun"]:
        ok = "yes" if shutil.which(tool) else "no"
        print(f"- {tool:<7} installed={ok}")

    total_findings = 0
    for session in sessions:
        print("")
        print(f"[{session}]")
        if not session_exists(session):  # noqa: F405
            print("- missing session")
            continue

        rows = refresh_pane_health(  # noqa: F405
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


def _build_context_menu_cmd(session: str) -> list[str]:
    """Build a tmux display-menu command for right-click pane management."""
    aw = str(_core.ROOT / "scripts" / "agent-wrangler")
    aw_py = str(_core.ROOT / "scripts" / "agent_wrangler.py")
    # Use pane_id (e.g. %5) — tmux expands #{pane_id} reliably in display-menu
    # pane_target() resolves pane IDs so all commands work with this
    pid = "#{pane_id}"
    menu_items: list[tuple[str, str, str]] = [
        ("Zoom In", "z", "resize-pane -Z"),
        ("Check Output", "o",
         f"display-popup -E -w 80 -h 30 "
         f"'python3 {aw_py} teams summary "
         f"{pid} --session {session} --lines 50; read'"),
        ("Send Command...", "c",
         f"command-prompt -p 'command:' "
         f"\"run-shell '{aw} send {pid} --command \\\"%%\\\"'\""),
        ("", "", ""),
        ("Start Claude", "1", f"run-shell '{aw} agent {pid} claude'"),
        ("Start Codex", "2", f"run-shell '{aw} agent {pid} codex'"),
        ("Start Gemini", "3", f"run-shell '{aw} agent {pid} gemini'"),
        ("", "", ""),
        ("Restart", "r", f"run-shell '{aw} restart {pid}'"),
        ("Stop (Ctrl-C)", "s", f"run-shell '{aw} stop {pid}'"),
        ("", "", ""),
        ("Send to Barn", "b", f"run-shell '{aw} barn {pid}'"),
    ]
    args_list = ["display-menu", "-T", "#[bold]⟨ #{pane_title} ⟩", "-x", "R", "-y", "S"]
    for label, key, cmd in menu_items:
        if not label:
            args_list.append("")
        else:
            args_list.extend([label, key, cmd])
    return args_list


def run_nav(args: argparse.Namespace) -> int:
    ensure_tmux()  # noqa: F405
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
    store = load_store()  # noqa: F405
    session = store.get("default_session") or _core.DEFAULT_SESSION
    named_window_bindings = [
        ("M-m", ["select-window", "-t", f"{session}:manager"]),
        ("M-g", ["select-window", "-t", f"{session}:grid"]),
    ]
    exit_script = str(_core.ROOT / "scripts" / "agent-wrangler")
    summary_script = str(_core.ROOT / "scripts" / "agent_wrangler.py")
    utility_bindings = [
        ("M-z", ["resize-pane", "-Z"]),          # zoom toggle
        ("M-j", ["display-panes", "-d", "2000"]),  # jump by number overlay
        # Zoomed navigation: cycle panes while staying fullscreen
        ("M-n", ["select-pane", "-t", ":.+"]),   # next pane (works zoomed)
        ("M-p", ["select-pane", "-t", ":.-"]),   # prev pane (works zoomed)
        ("M-q", ["confirm-before", "-p", "Exit Agent Wrangler? (y/n)", f"run-shell '{exit_script} exit --force'"]),
        ("M-s", ["display-popup", "-E", "-w", "80", "-h", "30",
                  f"python3 {summary_script} teams summary #{{pane_title}} --session {session} --lines 50; read"]),
    ]
    all_bindings = pane_bindings + window_bindings + index_bindings + named_window_bindings + utility_bindings

    if args.remove:
        for key, _cmd in all_bindings:
            tmux(["unbind-key", "-n", key], timeout=5)  # noqa: F405
        print("Removed no-prefix Alt navigation bindings (pane + window).")
        return 0

    for key, cmd in all_bindings:
        tmux(["bind-key", "-n", key, *cmd], timeout=5)  # noqa: F405

    # Double-click to zoom/unzoom a pane
    tmux(["bind-key", "-n", "DoubleClick1Pane", "resize-pane", "-Z"], timeout=5)  # noqa: F405

    # Right-click context menu for pane management
    menu_cmd = _build_context_menu_cmd(session)
    tmux(["bind-key", "-n", "MouseDown3Pane", *menu_cmd], timeout=5)  # noqa: F405

    # Source Ghostty-optimized tmux config if available
    tmux_conf = _core.ROOT / "config" / "tmux.conf"
    if tmux_conf.exists():
        tmux(["source-file", str(tmux_conf)], timeout=5)  # noqa: F405

    print("Enabled navigation bindings.")
    print("Mouse: click select | double-click zoom | right-click menu | scroll browse")
    print("Zoomed: Option+n (next) | Option+p (prev) | Option+z (unzoom)")
    print("Windows: Option+m (manager) | Option+g (grid) | Option+[ / ]")
    return 0


def run_send(args: argparse.Namespace) -> int:
    ensure_tmux()  # noqa: F405
    store = load_store()  # noqa: F405
    session = args.session or store.get("default_session") or _core.DEFAULT_SESSION
    pane = pane_target(session, args.pane)  # noqa: F405
    pane_send(pane.pane_id, args.command, enter=(not args.no_enter))  # noqa: F405
    print(f"sent to {pane.pane_id} ({pane.pane_title}): {args.command}")
    return 0


def run_stop(args: argparse.Namespace) -> int:
    ensure_tmux()  # noqa: F405
    store = load_store()  # noqa: F405
    session = args.session or store.get("default_session") or _core.DEFAULT_SESSION
    pane = pane_target(session, args.pane)  # noqa: F405
    pane_ctrl_c(pane.pane_id)  # noqa: F405
    print(f"sent Ctrl-C to {pane.pane_id} ({pane.pane_title})")
    return 0


def run_restart(args: argparse.Namespace) -> int:
    ensure_tmux()  # noqa: F405
    store = load_store()  # noqa: F405
    session = args.session or store.get("default_session") or _core.DEFAULT_SESSION
    pane = pane_target(session, args.pane)  # noqa: F405

    pane_ctrl_c(pane.pane_id)  # noqa: F405

    proj_map_data = project_map()  # noqa: F405
    project_id = pane.project_id or pane.pane_title
    if project_id not in proj_map_data:
        inferred = infer_project_id_from_path(pane.pane_path, proj_map_data)  # noqa: F405
        if inferred:
            project_id = inferred
            pane_set_project_id(pane.pane_id, project_id)  # noqa: F405
    project = proj_map_data.get(project_id)
    if not project:
        raise ValueError(
            f"Pane project id '{project_id}' is not a known project id."
        )

    startup_command = str(project.get("startup_command") or "").strip()
    if not startup_command:
        raise ValueError(f"Project '{project_id}' has no startup_command")

    pane_send(pane.pane_id, startup_command, enter=True)  # noqa: F405
    print(f"restarted {pane.pane_id} ({project_id}) with: {startup_command}")
    return 0


def run_agent(args: argparse.Namespace) -> int:
    ensure_tmux()  # noqa: F405
    store = load_store()  # noqa: F405
    session = args.session or store.get("default_session") or _core.DEFAULT_SESSION
    pane = pane_target(session, args.pane)  # noqa: F405

    tool = args.tool.strip()
    tokens: list[str] = [tool]
    if args.flags:
        tokens.extend(args.flags.strip().split())
    extra = list(args.agent_args or [])
    if extra and extra[0] == "--":
        extra = extra[1:]
    tokens.extend(extra)
    command = " ".join(token for token in tokens if token)

    # Auto-pilot: Claude gets --dangerously-skip-permissions unless opted out
    if tool == "claude" and "--dangerously-skip-permissions" not in command:
        if not getattr(args, "no_auto", False):
            tokens.insert(1, "--dangerously-skip-permissions")
            command = " ".join(token for token in tokens if token)

    pane_send(pane.pane_id, command, enter=True)  # noqa: F405
    play_sound("Morse", 0.3, key=f"agent-{pane.pane_id}")  # noqa: F405
    print(f"launched agent in {pane.pane_id} ({pane.pane_title}): {command}")
    return 0


def run_focus(args: argparse.Namespace) -> int:
    ensure_tmux()  # noqa: F405
    store = load_store()  # noqa: F405
    session = args.session or store.get("default_session") or _core.DEFAULT_SESSION
    pane = pane_target(session, args.pane)  # noqa: F405
    code, _, err = tmux(["select-pane", "-t", pane.pane_id], timeout=5)  # noqa: F405
    if code != 0:
        raise ValueError(err.strip() or f"failed to focus pane {pane.pane_id}")
    print(f"focused {pane.pane_id} ({pane.pane_title})")
    if args.attach:
        return attach_session(session)  # noqa: F405
    return 0


def run_kill(args: argparse.Namespace) -> int:
    ensure_tmux()  # noqa: F405
    store = load_store()  # noqa: F405
    session = args.session or store.get("default_session") or _core.DEFAULT_SESSION
    pane = pane_target(session, args.pane)  # noqa: F405
    code, _, err = tmux(["kill-pane", "-t", pane.pane_id], timeout=5)  # noqa: F405
    if code != 0:
        raise ValueError(err.strip() or f"failed to kill pane {pane.pane_id}")
    print(f"killed {pane.pane_id} ({pane.pane_title})")
    return 0


def run_exit(args: argparse.Namespace) -> int:
    """Kill the entire Agent Wrangler tmux session."""
    ensure_tmux()  # noqa: F405
    store = load_store()  # noqa: F405
    session = args.session or store.get("default_session") or _core.DEFAULT_SESSION

    if not args.force:
        # Check for running agents and warn
        panes = list_panes(session)  # noqa: F405
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
        tmux(["unbind-key", "-n", key], timeout=3)  # noqa: F405
    for idx in range(1, 10):
        tmux(["unbind-key", "-n", f"M-{idx}"], timeout=3)  # noqa: F405

    play_sound("Submarine", 0.3)  # noqa: F405
    code, _, err = tmux(["kill-session", "-t", session], timeout=5)  # noqa: F405
    if code != 0:
        raise ValueError(err.strip() or f"failed to kill session {session}")
    print(f"Agent Wrangler session '{session}' exited.")
    return 0


def run_shell(args: argparse.Namespace) -> int:
    ensure_tmux()  # noqa: F405
    store = load_store()  # noqa: F405
    session = args.session or store.get("default_session") or _core.DEFAULT_SESSION
    pane = pane_target(session, args.pane)  # noqa: F405

    shell_bin = args.shell or os.environ.get("SHELL") or "zsh"
    code, _, err = tmux(  # noqa: F405
        ["respawn-pane", "-k", "-t", pane.pane_id, "-c", pane.pane_path, shell_bin],
        timeout=8,
    )
    if code != 0:
        raise ValueError(err.strip() or f"failed to respawn shell in pane {pane.pane_id}")

    if pane.project_id:
        pane_set_project_id(pane.pane_id, pane.project_id)  # noqa: F405
        tmux(["select-pane", "-t", pane.pane_id, "-T", pane.project_id], timeout=5)  # noqa: F405

    print(f"reset {pane.pane_id} to shell '{shell_bin}'")
    return 0


def run_layout(args: argparse.Namespace) -> int:
    ensure_tmux()  # noqa: F405
    store = load_store()  # noqa: F405
    session = args.session or store.get("default_session") or _core.DEFAULT_SESSION
    panes = list_panes(session)  # noqa: F405
    resolved = choose_layout(args.layout, pane_count=len(panes))  # noqa: F405
    apply_layout(session, resolved)  # noqa: F405
    print(f"layout for {session}: {resolved}")
    return 0


def run_capture(args: argparse.Namespace) -> int:
    ensure_tmux()  # noqa: F405
    store = load_store()  # noqa: F405
    session = args.session or store.get("default_session") or _core.DEFAULT_SESSION
    pane = pane_target(session, args.pane)  # noqa: F405
    lines = max(5, args.lines)
    code, out, err = tmux(["capture-pane", "-p", "-t", pane.pane_id, "-S", f"-{lines}"], timeout=10)  # noqa: F405
    if code != 0:
        raise ValueError(err.strip() or f"failed to capture pane {pane.pane_id}")
    print(out.rstrip("\n"))
    return 0


def run_hide(args: argparse.Namespace) -> int:
    ensure_tmux()  # noqa: F405
    store = load_store()  # noqa: F405
    session = args.session or store.get("default_session") or _core.DEFAULT_SESSION
    pane = pane_target(session, args.pane)  # noqa: F405
    project_id = pane.project_id or pane.pane_title or pane.pane_id
    hidden_name = hide_pane(session, pane.pane_id, project_id)  # noqa: F405
    print(f"Hidden: {project_id} -> {hidden_name}")
    return 0


def run_show(args: argparse.Namespace) -> int:
    ensure_tmux()  # noqa: F405
    store = load_store()  # noqa: F405
    session = args.session or store.get("default_session") or _core.DEFAULT_SESSION
    hidden = list_hidden_panes(session)  # noqa: F405
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

    show_pane(session, match["window_name"])  # noqa: F405
    print(f"Restored: {match['project_id']}")
    return 0


def run_hidden(args: argparse.Namespace) -> int:
    ensure_tmux()  # noqa: F405
    store = load_store()  # noqa: F405
    session = args.session or store.get("default_session") or _core.DEFAULT_SESSION
    hidden = list_hidden_panes(session)  # noqa: F405
    if not hidden:
        print("No hidden panes.")
        return 0
    print(f"Hidden panes in '{session}':")
    for h in hidden:
        print(f"  {h['project_id']:<22} agent={h['agent']:<8} status={h['status']}")
    return 0


def run_list_projects(args: argparse.Namespace) -> int:
    config = load_projects_config()  # noqa: F405
    group = args.group.lower() if args.group else None
    print(f"{'ID':<22} {'GROUP':<10} PATH")
    for project in config.get("projects", []):
        pg = str(project.get("group", ""))
        if group and pg.lower() != group:
            continue
        print(f"{project['id']:<22} {pg:<10} {project.get('path', '')}")
    return 0


def run_init(_: argparse.Namespace) -> int:
    """Interactive project setup — scan for git repos and create projects.json."""
    if _core.PROJECTS_CONFIG.exists():
        try:
            answer = input(f"{_core.PROJECTS_CONFIG} already exists. Overwrite? [y/N] ")
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
    _core.PROJECTS_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    _core.PROJECTS_CONFIG.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    print(f"\nCreated {_core.PROJECTS_CONFIG} with {len(projects)} projects:")
    for p in projects:
        print(f"  - {p['id']}")

    # Also create team_grid.json if missing
    if not _core.CONFIG_PATH.exists():
        grid_config = {
            "default_session": "agent-grid",
            "default_layout": "tiled",
            "default_projects": [p["id"] for p in projects[:10]],
            "persistence": {"enabled": False, "autosave_minutes": 15},
            "profiles": {"current": "default", "items": {"default": {"max_panes": 10}}},
            "updated_at": now_iso(),  # noqa: F405
        }
        _core.CONFIG_PATH.write_text(json.dumps(grid_config, indent=2) + "\n", encoding="utf-8")
        print(f"Created {_core.CONFIG_PATH}")

    print("\nRun: ./scripts/agent-wrangler start")
    return 0


def run_add(args: argparse.Namespace) -> int:
    """Add current directory (or specified path) as a project and hot-add to running grid."""
    path = Path(args.path).resolve() if args.path else Path.cwd()
    if not path.is_dir():
        raise ValueError(f"Not a directory: {path}")
    # Agent-wrangler is the cowboy, not a horse
    if path == Path(_core.SELF_PATH).resolve():
        print("Agent Wrangler is the cowboy, not a horse. It manages the grid from the manager window.")
        return 1

    proj_id = args.name or path.name.replace(" ", "-").lower()

    # Add to projects.json
    config: dict[str, Any] = {}
    if _core.PROJECTS_CONFIG.exists():
        config = json.loads(_core.PROJECTS_CONFIG.read_text(encoding="utf-8"))
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
        _core.PROJECTS_CONFIG.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(f"Added '{proj_id}' to {_core.PROJECTS_CONFIG}")

    # Hot-add to running grid if tmux session exists
    store = load_store()  # noqa: F405
    session = args.session or store.get("default_session") or _core.DEFAULT_SESSION
    if session_exists(session):  # noqa: F405
        split_for_path(session, str(path))  # noqa: F405
        apply_layout(session, "tiled")  # noqa: F405
        # Tag the new pane
        panes = list_panes(session)  # noqa: F405
        if panes:
            last_pane = panes[-1]
            pane_set_project_id(last_pane.pane_id, proj_id)  # noqa: F405
            tmux(["select-pane", "-t", last_pane.pane_id, "-T", proj_id], timeout=5)  # noqa: F405
            pane_send(last_pane.pane_id, f"echo '\\n[{proj_id}] ready'", enter=True)  # noqa: F405
        set_window_orchestrator_format(session)  # noqa: F405
        print(f"Added pane for '{proj_id}' to session '{session}'")
    else:
        print(f"Session '{session}' not running. Pane will appear on next start.")

    return 0


def run_remove(args: argparse.Namespace) -> int:
    """Remove a project from config and optionally kill its grid pane."""
    proj_id = args.project
    store = load_store()  # noqa: F405
    session = args.session or store.get("default_session") or _core.DEFAULT_SESSION

    # Remove from projects.json
    removed_from_config = False
    if _core.PROJECTS_CONFIG.exists():
        config = json.loads(_core.PROJECTS_CONFIG.read_text(encoding="utf-8"))
        projects = config.get("projects", [])
        before = len(projects)
        config["projects"] = [p for p in projects if p.get("id") != proj_id]
        if len(config["projects"]) < before:
            _core.PROJECTS_CONFIG.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            removed_from_config = True
            print(f"Removed '{proj_id}' from {_core.PROJECTS_CONFIG}")

    if not removed_from_config:
        print(f"Project '{proj_id}' not found in config.")

    # Kill the pane in the running grid if it exists
    if session_exists(session):  # noqa: F405
        try:
            pane = pane_target(session, proj_id)  # noqa: F405
            tmux(["kill-pane", "-t", pane.pane_id], timeout=5)  # noqa: F405
            apply_layout(session, "tiled")  # noqa: F405
            set_window_orchestrator_format(session)  # noqa: F405
            print(f"Killed pane for '{proj_id}' in session '{session}'")
        except (ValueError, RuntimeError):
            if removed_from_config:
                print(f"No running pane for '{proj_id}' (will be excluded on next start)")

    return 0


def run_barn(args: argparse.Namespace) -> int:
    """Send a project to the barn — remove from grid, keep in config."""
    store = load_store()  # noqa: F405
    session = args.session or store.get("default_session") or _core.DEFAULT_SESSION
    proj_id, pane_id = _resolve_project_id(args.project, session)  # noqa: F405

    if not _set_barn_flag(proj_id, barn=True):  # noqa: F405
        print(f"Project '{proj_id}' not found in config.")
        return 1

    print(f"Sent '{proj_id}' to the barn.")

    # Kill the pane in the running grid
    if pane_id:
        try:
            tmux(["kill-pane", "-t", pane_id], timeout=5)  # noqa: F405
            apply_layout(session, "tiled")  # noqa: F405
            set_window_orchestrator_format(session)  # noqa: F405
            print(f"Removed pane from grid.")
        except (ValueError, RuntimeError):
            pass

    return 0


def run_unbarn(args: argparse.Namespace) -> int:
    """Let a project out of the barn — add back to grid."""
    proj_id = args.project
    store = load_store()  # noqa: F405
    session = args.session or store.get("default_session") or _core.DEFAULT_SESSION

    if not _set_barn_flag(proj_id, barn=False):  # noqa: F405
        print(f"Project '{proj_id}' not found in config.")
        return 1

    print(f"Let '{proj_id}' out of the barn.")

    # Hot-add to running grid
    if session_exists(session):  # noqa: F405
        proj_map_data = project_map()  # noqa: F405
        proj = proj_map_data.get(proj_id)
        path = str(proj.get("path", "")) if proj else ""
        if path:
            split_for_path(session, path)  # noqa: F405
            apply_layout(session, "tiled")  # noqa: F405
            panes = list_panes(session)  # noqa: F405
            if panes:
                last_pane = panes[-1]
                pane_set_project_id(last_pane.pane_id, proj_id)  # noqa: F405
                tmux(["select-pane", "-t", last_pane.pane_id, "-T", proj_id], timeout=5)  # noqa: F405
            set_window_orchestrator_format(session)  # noqa: F405
            print(f"Added pane to grid.")
    else:
        print(f"Session not running. Will appear on next start.")

    return 0


def run_barn_list(args: argparse.Namespace) -> int:
    """List projects in the barn."""
    config = load_projects_config()  # noqa: F405
    barn_projects = [p for p in config.get("projects", []) if p.get("barn")]
    active_projects = [p for p in config.get("projects", []) if not p.get("barn")]

    if active_projects:
        print(f"\033[32mGrazing ({len(active_projects)}):\033[0m")
        for p in active_projects:
            print(f"  {p['id']}")

    if barn_projects:
        print(f"\n\033[33mIn the barn ({len(barn_projects)}):\033[0m")
        for p in barn_projects:
            print(f"  {p['id']}")
    else:
        print("\nBarn is empty — all projects are grazing.")

    return 0


def run_summary(args: argparse.Namespace) -> int:
    """Show a summary of recent output from a pane."""
    ensure_tmux()  # noqa: F405
    store = load_store()  # noqa: F405
    session = args.session or store.get("default_session") or _core.DEFAULT_SESSION
    pane = pane_target(session, args.pane)  # noqa: F405
    lines = max(10, int(args.lines))
    raw = capture_pane_raw(pane.pane_id, lines=lines)  # noqa: F405

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
    cmd = [str(_core.ROOT / "scripts" / "agent-wrangler"), *command]
    print("")
    print("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, check=False, timeout=120)
    return int(proc.returncode)


def run_ops(_: argparse.Namespace) -> int:
    """Interactive operator console — ranch operations board."""
    RUST = "\033[38;5;130m"
    AMBER = "\033[38;5;208m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    actions: list[tuple[str, list[str]]] = [
        ("Start all (import + grid + manager)", ["start"]),
        ("Attach grid session", ["attach"]),
        ("Show herd status", ["status"]),
        ("Focus pane by name", ["__focus__"]),
        ("Send command to pane", ["__send__"]),
        ("Saddle up agent", ["__agent__"]),
        ("Whoa (stop pane)", ["__stop__"]),
        ("Open manager window", ["manager", "--replace"]),
        ("Doctor (check attention)", ["doctor", "--only-attention"]),
    ]

    # Ranch operations header
    print(f"{DIM}╭─ Ranch Operations ──────────────────╮{RESET}")
    print(f"{DIM}│{RESET}     {RUST}/\\{RESET}                              {DIM}│{RESET}")
    print(f"{DIM}│{RESET}    {RUST}/  \\{RESET}   Agent Wrangler            {DIM}│{RESET}")
    print(f"{DIM}│{RESET}   {RUST}/    \\{RESET}  Enter number, or q to quit{DIM}│{RESET}")
    print(f"{DIM}│{RESET}  {RUST}'──────'{RESET}                           {DIM}│{RESET}")
    print(f"{DIM}╰─────────────────────────────────────╯{RESET}")
    print()

    # Try to show quick health summary
    try:
        store = load_store()  # noqa: F405
        session = store.get("default_session") or _core.DEFAULT_SESSION
        if session_exists(session):  # noqa: F405
            rows = refresh_pane_health(session, capture_lines=40, wait_attention_min=5, apply_colors=False)  # noqa: F405
            dot_map = {"green": "\033[32m●", "yellow": "\033[33m●", "red": "\033[31m●"}
            herd_line = "  Herd: "
            for row in rows:
                pid = str(row.get("project_id") or "?")
                lev = str(row.get("health") or "green")
                herd_line += f" {dot_map.get(lev, '●')}{RESET} {pid}"
            print(herd_line + RESET)
            print()
    except Exception:
        pass

    while True:
        for idx, (label, _) in enumerate(actions, start=1):
            print(f"  {RUST}{idx:>2}.{RESET} {label}")
        print(f"   {DIM}q. Quit{RESET}")

        choice = input(f"\n{AMBER}ranch>{RESET} ").strip().lower()
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
            tool = input("tool (claude|codex|gemini): ").strip().lower()
            if token and tool in {"claude", "codex", "gemini"}:
                _run_ops_command(["agent", token, tool])
        elif command == ["__stop__"]:
            token = input("pane token: ").strip()
            if token:
                _run_ops_command(["stop", token])
        else:
            _run_ops_command(command)
        print("")


def run_briefing(args: argparse.Namespace) -> int:
    """Show what happened while the user was away."""
    RUST = "\033[38;5;130m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    DIM = "\033[2m"
    RST = "\033[0m"

    since = getattr(args, "since", 60)
    entries = _core._read_activity(since_minutes=since)

    if not entries:
        print(f"{DIM}No activity recorded in the last {since} minutes.{RST}")
        print(f"{DIM}The rail must be running to collect activity data.{RST}")
        return 0

    # Calculate time span
    all_ts: list[datetime] = []
    for e in entries:
        ts_str = e.get("ts", "")
        if ts_str:
            try:
                all_ts.append(datetime.fromisoformat(ts_str))
            except (ValueError, TypeError):
                pass
    if len(all_ts) >= 2:
        span_minutes = (max(all_ts) - min(all_ts)).total_seconds() / 60
    elif all_ts:
        span_minutes = 0
    else:
        span_minutes = 0

    health_colors = {"green": GREEN, "yellow": YELLOW, "red": RED}
    health_dots = {"green": "\u25cf", "yellow": "\u25cf", "red": "\u25cf"}

    print(f"\n{RUST}{'=' * 60}{RST}")
    print(f"{RUST}  BRIEFING{RST}  {DIM}last {since} minutes{RST}")
    print(f"{RUST}{'=' * 60}{RST}\n")

    # Show monitoring windows
    system_events = [e for e in entries if e.get("project") == "_system"]
    if system_events:
        starts = [e for e in system_events if e.get("event") == "rail_started"]
        stops = [e for e in system_events if e.get("event") == "rail_stopped"]
        if starts or stops:
            if starts:
                # Parse ISO ts for display
                last_start_ts = starts[-1].get("ts", "")
                try:
                    t = datetime.fromisoformat(last_start_ts).strftime("%H:%M:%S")
                except Exception:
                    t = "?"
                print(f"  {DIM}Monitoring started: {t}{RST}")
            if stops:
                last_stop_ts = stops[-1].get("ts", "")
                try:
                    t = datetime.fromisoformat(last_stop_ts).strftime("%H:%M:%S")
                except Exception:
                    t = "?"
                print(f"  {DIM}Monitoring stopped: {t}{RST}")
            print()

    # Filter out system events from per-project display
    project_entries = [e for e in entries if e.get("project") != "_system"]

    # Group entries by project
    by_project: dict[str, list[dict[str, Any]]] = {}
    for e in project_entries:
        proj = e.get("project", "unknown")
        by_project.setdefault(proj, []).append(e)

    needs_attention: list[str] = []
    total_health_counts: dict[str, int] = {}

    for proj, proj_entries in sorted(by_project.items()):
        latest = proj_entries[-1]
        health = latest.get("health", "green")
        status = latest.get("status", "idle")
        agent = latest.get("agent", "-")
        color = health_colors.get(health, DIM)

        total_health_counts[health] = total_health_counts.get(health, 0) + 1

        # Project header with colored dot
        dot = f"{color}{health_dots.get(health, '?')}{RST}"
        header = f"  {dot} {RUST}{proj}{RST}"
        if agent and agent != "-":
            header += f"  {DIM}({agent}){RST}"
        header += f"  {DIM}{health}/{status}{RST}"
        print(header)

        # Timeline of events (skip first_seen)
        timeline_events = [e for e in proj_entries if e.get("event") != "first_seen"]
        if timeline_events:
            for ev in timeline_events:
                ts_str = ev.get("ts", "")
                event = ev.get("event", "?")
                reason = ev.get("reason", "")
                time_label = ""
                if ts_str:
                    try:
                        dt = datetime.fromisoformat(ts_str)
                        time_label = dt.strftime("%H:%M")
                    except (ValueError, TypeError):
                        pass
                ev_health = ev.get("health", "green")
                ev_color = health_colors.get(ev_health, DIM)
                line = f"    {DIM}{time_label}{RST}  {ev_color}{event}{RST}"
                if reason:
                    line += f"  {DIM}{reason}{RST}"
                print(line)

        # Error count
        red_count = sum(1 for e in proj_entries if e.get("health") == "red")
        if red_count:
            print(f"    {RED}errors: {red_count}{RST}")

        # Cost info
        costs = [e.get("cost") for e in proj_entries if e.get("cost") is not None]
        if costs:
            current_cost = costs[-1]
            cost_line = f"    {DIM}cost: ${current_cost:.2f}{RST}"
            if len(costs) >= 2:
                delta = costs[-1] - costs[0]
                if delta > 0:
                    cost_line += f"  {DIM}(+${delta:.2f}){RST}"
            print(cost_line)

        if health == "red":
            needs_attention.append(proj)

        print()

    # Overall summary
    print(f"{RUST}{'─' * 60}{RST}")
    summary_parts = [f"{len(by_project)} projects"]
    if span_minutes > 0:
        summary_parts.append(f"{span_minutes:.0f}m span")
    health_summary = []
    for h in ("green", "yellow", "red"):
        count = total_health_counts.get(h, 0)
        if count:
            c = health_colors.get(h, "")
            health_summary.append(f"{c}{count} {h}{RST}")
    if health_summary:
        summary_parts.append(" / ".join(health_summary))
    print(f"  {' | '.join(summary_parts)}")

    # Needs attention section
    if needs_attention:
        print(f"\n  {RED}Needs attention:{RST}")
        for proj in needs_attention:
            latest = by_project[proj][-1]
            reason = latest.get("reason", "")
            line = f"    {RED}\u25cf{RST} {proj}"
            if reason:
                line += f"  {DIM}{reason}{RST}"
            print(line)

    print()
    return 0


# ── Subparser registration & main ─────────────────────────────────

def register_subparser(root_subparsers: argparse._SubParsersAction[Any]) -> None:
    teams = root_subparsers.add_parser("teams", help="Tmux team grid operations")
    teams_sub = teams.add_subparsers(dest="teams_command", required=True)

    up = teams_sub.add_parser("up", help="One-command entry: build/reuse grid, show status, attach")
    up.add_argument("--session", default=None)
    up.add_argument("--mode", choices=["import", "bootstrap"], default="import")
    up.add_argument("--layout", choices=_core.LAYOUT_CHOICES, default=None)
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
    bootstrap.add_argument("--limit", type=int, default=_core.DEFAULT_LIMIT, help="Max panes when auto-selecting")
    bootstrap.add_argument("--layout", choices=_core.LAYOUT_CHOICES, default=None)
    bootstrap.add_argument("--force", action="store_true", help="Replace existing session with same name")
    bootstrap.add_argument("--no-startup", action="store_true", help="Do not run per-project startup commands")
    bootstrap.add_argument("--agent", help="Agent command to run in each pane (example: claude or codex)")
    bootstrap.add_argument("--attach", action="store_true", help="Attach immediately after creation")
    bootstrap.set_defaults(handler=run_bootstrap)

    imp = teams_sub.add_parser("import", help="Import current Ghostty sessions into a tmux grid")
    imp.add_argument("--session", default=None)
    imp.add_argument("--layout", choices=_core.LAYOUT_CHOICES, default=None)
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
    send.add_argument("pane", help="Pane id (%%1), pane index (0), or pane title/project id")
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
    agent.add_argument("tool", choices=["claude", "codex", "gemini"])
    agent.add_argument("--session", default=None)
    agent.add_argument("--flags", help="Additional flags passed after tool command")
    agent.add_argument("--no-auto", action="store_true",
                        help="Don't add --dangerously-skip-permissions to claude")
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
    layout.add_argument("layout", choices=_core.LAYOUT_CHOICES)
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
    persistence_restore.add_argument("--layout", choices=_core.LAYOUT_CHOICES, default=None)
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

    remove_cmd = teams_sub.add_parser("remove", help="Remove a project from config and kill its grid pane")
    remove_cmd.add_argument("project", help="Project ID to remove")
    remove_cmd.add_argument("--session", default=None)
    remove_cmd.set_defaults(handler=run_remove)

    barn_cmd = teams_sub.add_parser("barn", help="Send a project to the barn (keep in config, remove from grid)")
    barn_cmd.add_argument("project", help="Project ID to barn")
    barn_cmd.add_argument("--session", default=None)
    barn_cmd.set_defaults(handler=run_barn)

    unbarn_cmd = teams_sub.add_parser("unbarn", help="Let a project out of the barn (add back to grid)")
    unbarn_cmd.add_argument("project", help="Project ID to unbarn")
    unbarn_cmd.add_argument("--session", default=None)
    unbarn_cmd.set_defaults(handler=run_unbarn)

    barn_list_cmd = teams_sub.add_parser("barn-list", help="List projects: grazing vs in the barn")
    barn_list_cmd.set_defaults(handler=run_barn_list)

    summary_cmd = teams_sub.add_parser("summary", help="Show recent output summary from a pane")
    summary_cmd.add_argument("pane", help="Pane token (project id, index, or pane id)")
    summary_cmd.add_argument("--session", default=None)
    summary_cmd.add_argument("--lines", type=int, default=50, help="Lines to capture")
    summary_cmd.set_defaults(handler=run_summary)

    briefing_cmd = teams_sub.add_parser("briefing", help="Show what happened while you were away")
    briefing_cmd.add_argument("--since", type=int, default=60, help="Look back N minutes (default: 60)")
    briefing_cmd.set_defaults(handler=run_briefing)

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
