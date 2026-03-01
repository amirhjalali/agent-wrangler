#!/usr/bin/env python3
"""Agent Wrangler v1: Gastown (planning) + Ant Farm (runtime)."""

from __future__ import annotations

import argparse
import curses
import json
import os
import signal
import subprocess
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import terminal_sentinel
import tmux_teams
import workflow_agent

ROOT = Path(__file__).resolve().parents[1]
STORE_PATH = ROOT / "config" / "command_center.json"
REPORTS_DIR = ROOT / "reports"

LANE_ORDER = ["now", "next", "week", "later"]
VALID_STATUSES = {"open", "done", "canceled"}
UI_PAGES = ("wrangler", "panels")
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
        "next_card_number": 1,
        "settings": {
            "lanes": {
                "now_max": 3,
                "next_max": 6,
            },
            "antfarm": {
                "source": "ghostty",
                "max_ai_sessions": 4,
                "kill_waiting_ai_after_min": 120,
                "overnight_interval_sec": 300,
            },
        },
        "cards": [],
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



def get_project_map() -> dict[str, dict[str, Any]]:
    config = workflow_agent.load_config()
    return {project["id"]: project for project in config.get("projects", [])}



def ensure_repo(repo_id: str | None) -> str | None:
    if repo_id is None:
        return None
    project_map = get_project_map()
    if repo_id not in project_map:
        known = ", ".join(sorted(project_map.keys()))
        raise ValueError(f"Unknown repo id '{repo_id}'. Known ids: {known}")
    return repo_id



def make_card_id(store: dict[str, Any]) -> str:
    number = int(store.get("next_card_number", 1))
    store["next_card_number"] = number + 1
    return f"GAS-{number:03d}"



def get_card(store: dict[str, Any], card_id: str) -> dict[str, Any]:
    for card in store.get("cards", []):
        if card.get("id") == card_id:
            return card
    raise ValueError(f"Card not found: {card_id}")



def card_progress(card: dict[str, Any]) -> tuple[int, int]:
    steps = card.get("steps", [])
    total = len(steps)
    done = len([step for step in steps if step.get("done")])
    return done, total



def is_card_open(card: dict[str, Any]) -> bool:
    return card.get("status", "open") == "open"



def unresolved_dependencies(store: dict[str, Any], card: dict[str, Any]) -> list[str]:
    unresolved: list[str] = []
    for dep_id in card.get("depends_on", []):
        try:
            dep = get_card(store, dep_id)
        except ValueError:
            unresolved.append(dep_id)
            continue
        if dep.get("status") != "done":
            unresolved.append(dep_id)
    return unresolved



def add_card(args: argparse.Namespace) -> int:
    store = load_store()

    lane = args.lane.lower()
    if lane not in LANE_ORDER:
        raise ValueError(f"Invalid lane: {lane}. Use one of {', '.join(LANE_ORDER)}")

    repo_id = ensure_repo(args.repo)

    card_id = make_card_id(store)
    steps: list[dict[str, Any]] = []
    for idx, text in enumerate(args.step or [], start=1):
        steps.append({"id": idx, "text": text, "done": False})

    card = {
        "id": card_id,
        "title": args.title,
        "repo": repo_id,
        "lane": lane,
        "status": "open",
        "depends_on": [],
        "steps": steps,
        "notes": args.notes or "",
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }

    store.setdefault("cards", []).append(card)
    save_store(store)

    print(f"Created card {card_id} in lane '{lane}'")
    print(f"Title: {args.title}")
    if repo_id:
        print(f"Repo: {repo_id}")
    if steps:
        print(f"Steps: {len(steps)}")
    return 0



def list_cards(args: argparse.Namespace) -> int:
    store = load_store()
    cards = list(store.get("cards", []))

    if args.status:
        cards = [card for card in cards if card.get("status") == args.status]

    if args.lane:
        lane = args.lane.lower()
        cards = [card for card in cards if card.get("lane") == lane]

    if args.repo:
        cards = [card for card in cards if card.get("repo") == args.repo]

    if args.open_only:
        cards = [card for card in cards if is_card_open(card)]

    cards.sort(key=lambda card: (LANE_ORDER.index(card.get("lane", "later")), card.get("id", "")))

    if not cards:
        print("No cards found for the selected filter.")
        return 0

    print("ID       LANE   STATUS   REPO             PROGRESS  TITLE")
    for card in cards:
        done, total = card_progress(card)
        repo = card.get("repo") or "-"
        print(
            f"{card['id']:<8} {card.get('lane', '-'):<6} {card.get('status', '-'):<8} "
            f"{repo:<16} {done}/{total:<7} {card.get('title', '')}"
        )

        if args.verbose:
            deps = card.get("depends_on", [])
            if deps:
                print(f"  deps: {', '.join(deps)}")
            for step in card.get("steps", []):
                mark = "x" if step.get("done") else " "
                print(f"  [{mark}] {step.get('id')}. {step.get('text')}")
            if card.get("notes"):
                print(f"  notes: {card.get('notes')}")
    return 0



def add_step(args: argparse.Namespace) -> int:
    store = load_store()
    card = get_card(store, args.card_id)

    steps = card.setdefault("steps", [])
    next_id = max((int(step.get("id", 0)) for step in steps), default=0) + 1
    steps.append({"id": next_id, "text": args.text, "done": False})
    card["updated_at"] = now_iso()

    save_store(store)
    print(f"Added step {next_id} to {args.card_id}")
    return 0



def set_step_done(args: argparse.Namespace, done: bool) -> int:
    store = load_store()
    card = get_card(store, args.card_id)
    target = int(args.step)

    for step in card.get("steps", []):
        if int(step.get("id", -1)) == target:
            step["done"] = done
            card["updated_at"] = now_iso()
            save_store(store)
            print(f"Step {target} in {args.card_id} marked {'done' if done else 'open'}")
            return 0

    raise ValueError(f"Step {target} not found in {args.card_id}")



def move_card(args: argparse.Namespace) -> int:
    store = load_store()
    card = get_card(store, args.card_id)
    lane = args.lane.lower()

    if lane not in LANE_ORDER:
        raise ValueError(f"Invalid lane: {lane}")

    card["lane"] = lane
    card["updated_at"] = now_iso()
    save_store(store)
    print(f"Moved {args.card_id} -> {lane}")
    return 0



def set_card_status(args: argparse.Namespace) -> int:
    store = load_store()
    card = get_card(store, args.card_id)
    status = args.status.lower()

    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status: {status}. Use one of {', '.join(sorted(VALID_STATUSES))}")

    card["status"] = status
    card["updated_at"] = now_iso()
    save_store(store)
    print(f"Card {args.card_id} status -> {status}")
    return 0



def add_dependency(args: argparse.Namespace) -> int:
    store = load_store()
    card = get_card(store, args.card_id)
    dep = get_card(store, args.depends_on)

    if card["id"] == dep["id"]:
        raise ValueError("Card cannot depend on itself")

    deps = card.setdefault("depends_on", [])
    if dep["id"] not in deps:
        deps.append(dep["id"])
        card["updated_at"] = now_iso()
        save_store(store)

    print(f"Added dependency: {card['id']} depends on {dep['id']}")
    return 0



def remove_dependency(args: argparse.Namespace) -> int:
    store = load_store()
    card = get_card(store, args.card_id)
    deps = card.setdefault("depends_on", [])

    if args.depends_on in deps:
        deps.remove(args.depends_on)
        card["updated_at"] = now_iso()
        save_store(store)
        print(f"Removed dependency: {card['id']} no longer depends on {args.depends_on}")
        return 0

    print(f"Dependency {args.depends_on} was not set on {card['id']}")
    return 0



def plan_summary(store: dict[str, Any]) -> dict[str, Any]:
    open_cards = [card for card in store.get("cards", []) if is_card_open(card)]

    lanes: dict[str, list[dict[str, Any]]] = {lane: [] for lane in LANE_ORDER}
    for card in open_cards:
        lane = card.get("lane", "later")
        if lane not in lanes:
            lane = "later"
        lanes[lane].append(card)

    for lane in LANE_ORDER:
        lanes[lane].sort(key=lambda card: card.get("id", ""))

    blocked = []
    for card in open_cards:
        missing = unresolved_dependencies(store, card)
        if missing:
            blocked.append({"card": card, "missing": missing})

    return {
        "open_total": len(open_cards),
        "lanes": lanes,
        "blocked": blocked,
    }



def print_gastown(store: dict[str, Any], show_steps: bool, limit_per_lane: int) -> None:
    summary = plan_summary(store)
    settings = store.get("settings", {}).get("lanes", {})
    now_max = int(settings.get("now_max", 3))
    next_max = int(settings.get("next_max", 6))

    now_cards = summary["lanes"]["now"]
    next_cards = summary["lanes"]["next"]
    week_cards = summary["lanes"]["week"]
    later_cards = summary["lanes"]["later"]

    print("Gastown")
    print(
        "Open={open_total} | NOW={now}/{now_max} | NEXT={next_count}/{next_max} | WEEK={week_count} | LATER={later_count}".format(
            open_total=summary["open_total"],
            now=len(now_cards),
            now_max=now_max,
            next_count=len(next_cards),
            next_max=next_max,
            week_count=len(week_cards),
            later_count=len(later_cards),
        )
    )

    for lane in LANE_ORDER:
        cards = summary["lanes"][lane]
        print(f"\n[{lane.upper()}] {len(cards)}")
        if not cards:
            print("  -")
            continue

        for card in cards[:limit_per_lane]:
            done, total = card_progress(card)
            repo = card.get("repo") or "-"
            title = short_text(card["title"], 52)
            print(f"  {card['id']} [{done}/{total}] ({repo}) {title}")
            if show_steps:
                for step in card.get("steps", []):
                    mark = "x" if step.get("done") else " "
                    text = short_text(str(step.get("text", "")), 58)
                    print(f"    [{mark}] {step.get('id')}. {text}")

        extra = len(cards) - limit_per_lane
        if extra > 0:
            print(f"  ... and {extra} more")

    blocked = summary["blocked"]
    if blocked:
        print("\n[BLOCKED]")
        for item in blocked[:8]:
            card = item["card"]
            missing = ", ".join(item["missing"])
            print(f"  {card['id']} blocked by {missing}")



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


def build_gastown_lines(store: dict[str, Any], line_width: int, limit_per_lane: int) -> list[str]:
    summary = plan_summary(store)
    settings = store.get("settings", {}).get("lanes", {})
    now_max = int(settings.get("now_max", 3))
    next_max = int(settings.get("next_max", 6))

    lines: list[str] = []
    lines.append(
        short_text(
            "Open={open_total}  NOW={now}/{now_max}  NEXT={next_count}/{next_max}  WEEK={week_count}  LATER={later_count}".format(
                open_total=summary["open_total"],
                now=len(summary["lanes"]["now"]),
                now_max=now_max,
                next_count=len(summary["lanes"]["next"]),
                next_max=next_max,
                week_count=len(summary["lanes"]["week"]),
                later_count=len(summary["lanes"]["later"]),
            ),
            line_width,
        )
    )
    lines.append("")

    for lane in LANE_ORDER:
        cards = summary["lanes"][lane]
        lines.append(f"[{lane.upper()}] {len(cards)}")
        if not cards:
            lines.append("  -")
        else:
            for card in cards[:limit_per_lane]:
                done, total = card_progress(card)
                repo = card.get("repo") or "-"
                title = short_text(card.get("title", ""), max(10, line_width - 28))
                lines.append(f"  {card['id']} [{done}/{total}] ({repo}) {title}")
            extra = len(cards) - limit_per_lane
            if extra > 0:
                lines.append(f"  ... and {extra} more")
        lines.append("")

    blocked = summary["blocked"]
    if blocked:
        lines.append("[BLOCKED]")
        for item in blocked[:4]:
            card = item["card"]
            missing = ",".join(item["missing"])
            lines.append(short_text(f"  {card['id']} <- {missing}", line_width))

    return lines


def build_antfarm_lines(
    snapshot: dict[str, Any],
    warnings: list[str],
    line_width: int,
    max_wait_rows: int,
    max_ai_sessions: int,
    wait_limit_minutes: int,
) -> list[str]:
    lines: list[str] = []
    summary = snapshot.get("summary", {})
    by_agent = summary.get("by_agent", {})
    lines.append(
        short_text(
            "total={total} ai={ai} waiting={waiting} active={active} bg={background} idle={idle}".format(
                total=summary.get("total", 0),
                ai=summary.get("ai", 0),
                waiting=summary.get("waiting", 0),
                active=summary.get("active", 0),
                background=summary.get("background", 0),
                idle=summary.get("idle", 0),
            ),
            line_width,
        )
    )
    if by_agent:
        parts: list[str] = []
        for name in sorted(by_agent.keys()):
            info = by_agent[name]
            parts.append(f"{name}:{info.get('total', 0)}")
        lines.append(short_text("agent mix " + " ".join(parts), line_width))

    if warnings:
        lines.append(short_text(f"WARN: {warnings[0]}", line_width))
    lines.append("")

    sessions = list(snapshot.get("sessions", []))
    waiting = [s for s in sessions if s.get("status") == "waiting"]
    waiting.sort(key=lambda s: (s.get("waiting_minutes") or 0), reverse=True)
    active = [s for s in sessions if s.get("status") == "active"]
    active.sort(key=lambda s: s.get("pcpu") or 0, reverse=True)

    lines.append("TTY      AGT      ST   WAIT  CPU%  PID     CMD")
    display_rows = waiting[:max_wait_rows] + active[: max(0, max_wait_rows - len(waiting[:max_wait_rows]))]
    if not display_rows:
        lines.append("-")
    else:
        cmd_width = max(10, line_width - 45)
        for session in display_rows[:max_wait_rows]:
            mins = session.get("waiting_minutes")
            wait_text = f"{int(mins)}m" if mins is not None else "-"
            command = short_text(str(session.get("command", "")), cmd_width)
            agent = short_text(session_tool(session), 8)
            lines.append(
                f"{str(session.get('tty', '-')):<8} {agent:<8} {str(session.get('status', '-')):<4} {wait_text:<5} "
                f"{str(session.get('pcpu', '-')):<5} {str(session.get('pid', '-')):<7} {command}"
            )

    lines.append("")
    lines.append("Suggested actions:")
    actions = terminal_sentinel.recommend_actions(
        snapshot=snapshot,
        max_ai_sessions=max_ai_sessions,
        wait_limit_minutes=wait_limit_minutes,
    )
    if not actions:
        lines.append("none")
    else:
        for action in actions[:5]:
            lines.append(short_text(f"stop {action['tty']}:{action['pid']} ({action['reason']})", line_width))
        extra = len(actions) - 5
        if extra > 0:
            lines.append(f"... and {extra} more")

    return lines


def session_sort_key(session: dict[str, Any]) -> tuple[int, float, float, str]:
    rank = {"waiting": 0, "active": 1, "background": 2, "idle": 3}.get(str(session.get("status")), 9)
    waiting = float(session.get("waiting_minutes") or 0.0)
    cpu = float(session.get("pcpu") or 0.0)
    tty = str(session.get("tty") or "")
    return (rank, -waiting, -cpu, tty)


def sorted_sessions(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    sessions = list(snapshot.get("sessions", []))
    sessions.sort(key=session_sort_key)
    return sessions


def clamp_selected_index(selected_index: int, total: int) -> int:
    if total <= 0:
        return 0
    return max(0, min(selected_index, total - 1))


def selected_session(sessions: list[dict[str, Any]], selected_index: int) -> dict[str, Any] | None:
    if not sessions:
        return None
    idx = clamp_selected_index(selected_index, len(sessions))
    return sessions[idx]


def session_key(session: dict[str, Any]) -> tuple[str, int]:
    tty = str(session.get("tty") or "")
    pid = int(session.get("pid") or 0)
    return tty, pid


def session_tool(session: dict[str, Any]) -> str:
    tagged = str(session.get("agent") or "").strip().lower()
    if tagged:
        return tagged

    command = str(session.get("command") or "").lower()
    if "claude" in command:
        return "claude"
    if "codex" in command:
        return "codex"
    if "aider" in command:
        return "aider"
    if "gemini" in command:
        return "gemini"
    if "chatgpt" in command:
        return "chatgpt"
    if "python" in command:
        return "python"
    return terminal_sentinel.command_bin(str(session.get("command") or "")) or "task"


def infer_repo_from_command(command: str, project_map: dict[str, dict[str, Any]]) -> str | None:
    command_lower = command.lower()
    matches: list[tuple[int, str]] = []
    for repo_id, project in project_map.items():
        path = str(project.get("path") or "")
        if not path:
            continue
        path_norm = path.lower()
        if path_norm in command_lower:
            matches.append((len(path_norm), repo_id))
    if not matches:
        return None
    matches.sort(reverse=True)
    return matches[0][1]


def build_studio_overview_lines(
    store: dict[str, Any],
    snapshot: dict[str, Any],
    line_width: int,
    max_rows: int,
) -> list[str]:
    open_cards = [card for card in store.get("cards", []) if is_card_open(card)]
    repo_counts: dict[str, dict[str, int]] = {}
    for card in open_cards:
        repo = card.get("repo") or "unassigned"
        info = repo_counts.setdefault(repo, {"open": 0, "now": 0, "next": 0, "week": 0, "later": 0})
        info["open"] += 1
        lane = str(card.get("lane") or "later")
        if lane not in info:
            lane = "later"
        info[lane] += 1

    sessions = sorted_sessions(snapshot)
    team_counts: dict[str, dict[str, int]] = {}
    for session in sessions:
        team = session_tool(session)
        info = team_counts.setdefault(team, {"total": 0, "waiting": 0, "active": 0})
        info["total"] += 1
        status = str(session.get("status") or "")
        if status in ("waiting", "active"):
            info[status] += 1

    lines: list[str] = []
    lines.append("Repos")
    if not repo_counts:
        lines.append("none")
    else:
        repos = sorted(repo_counts.items(), key=lambda item: (-item[1]["open"], item[0]))
        max_repo = max(2, max_rows // 2)
        for repo, info in repos[:max_repo]:
            row = "{repo:<16} open={open} now={now} next={next_count}".format(
                repo=repo[:16],
                open=info["open"],
                now=info["now"],
                next_count=info["next"],
            )
            lines.append(short_text(row, line_width))
        extra = len(repos) - max_repo
        if extra > 0:
            lines.append(f"... and {extra} more")

    lines.append("")
    lines.append("Teams")
    if not team_counts:
        lines.append("none")
    else:
        teams = sorted(team_counts.items(), key=lambda item: (-item[1]["total"], item[0]))
        max_team = max(2, max_rows // 2)
        for team, info in teams[:max_team]:
            row = "{team:<12} total={total} waiting={waiting} active={active}".format(
                team=team[:12],
                total=info["total"],
                waiting=info["waiting"],
                active=info["active"],
            )
            lines.append(short_text(row, line_width))
        extra = len(teams) - max_team
        if extra > 0:
            lines.append(f"... and {extra} more")

    return lines[:max_rows]


def build_health_lines(
    snapshot: dict[str, Any],
    warnings: list[str],
    suggested_actions: list[dict[str, Any]],
    line_width: int,
    max_rows: int,
) -> list[str]:
    lines: list[str] = []
    summary = snapshot.get("summary", {})
    by_agent = summary.get("by_agent", {})
    lines.append(
        short_text(
            "Sessions total={total} ai={ai} waiting={waiting} active={active} bg={background}".format(
                total=summary.get("total", 0),
                ai=summary.get("ai", 0),
                waiting=summary.get("waiting", 0),
                active=summary.get("active", 0),
                background=summary.get("background", 0),
            ),
            line_width,
        )
    )
    if by_agent:
        parts: list[str] = []
        for name in sorted(by_agent.keys()):
            info = by_agent[name]
            parts.append(f"{name}:{info.get('total', 0)}")
        lines.append(short_text("Agent mix " + " ".join(parts), line_width))
    if warnings:
        lines.append(short_text(f"WARN: {warnings[0]}", line_width))
    lines.append("")

    lines.append("Longest waiting")
    waiting = [session for session in sorted_sessions(snapshot) if session.get("status") == "waiting"]
    if not waiting:
        lines.append("none")
    else:
        for session in waiting[: max(2, max_rows // 3)]:
            mins = int(session.get("waiting_minutes") or 0)
            row = "{tty:<8} wait={wait:>3}m cpu={cpu:<4} cmd={cmd}".format(
                tty=str(session.get("tty") or "-"),
                wait=mins,
                cpu=str(session.get("pcpu") or "-"),
                cmd=short_text(session_tool(session), 10),
            )
            lines.append(short_text(row, line_width))

    lines.append("")
    lines.append("Suggested actions")
    if not suggested_actions:
        lines.append("none")
    else:
        max_actions = max(2, max_rows // 3)
        for action in suggested_actions[:max_actions]:
            lines.append(short_text(f"stop {action['tty']}:{action['pid']} ({action['reason']})", line_width))
        extra = len(suggested_actions) - max_actions
        if extra > 0:
            lines.append(f"... and {extra} more")

    return lines[:max_rows]


def ui_active_lines(snapshot: dict[str, Any], line_width: int, max_rows: int) -> list[str]:
    sessions = sorted_sessions(snapshot)

    lines: list[str] = []
    if not sessions:
        lines.append("No active sessions.")
        return lines

    lines.append("TTY      AGT      ST       WAIT   CPU%  PID     CMD")
    cmd_width = max(10, line_width - 47)
    for session in sessions[:max_rows]:
        mins = session.get("waiting_minutes")
        wait_text = f"{int(mins)}m" if mins is not None else "-"
        command = short_text(str(session.get("command", "")), cmd_width)
        agent = short_text(session_tool(session), 8)
        lines.append(
            f"{str(session.get('tty', '-')):<8} {agent:<8} {str(session.get('status', '-')):<8} {wait_text:<6} "
            f"{str(session.get('pcpu', '-')):<5} {str(session.get('pid', '-')):<7} {command}"
        )

    extra = len(sessions) - max_rows
    if extra > 0:
        lines.append(f"... and {extra} more")
    return lines


def ui_queue_lines(store: dict[str, Any], line_width: int, max_rows: int) -> list[str]:
    summary = plan_summary(store)
    lines: list[str] = []

    now_cards = summary["lanes"]["now"]
    next_cards = summary["lanes"]["next"]
    lines.append(f"NOW {len(now_cards)} | NEXT {len(next_cards)}")
    lines.append("")

    visible = now_cards + next_cards
    if not visible:
        lines.append("0 waiting")
        return lines

    for card in visible[:max_rows]:
        done, total = card_progress(card)
        repo = card.get("repo") or "-"
        title = short_text(card.get("title", ""), max(10, line_width - 24))
        lines.append(f"{card['id']} [{done}/{total}] ({repo}) {title}")

    extra = len(visible) - max_rows
    if extra > 0:
        lines.append(f"... and {extra} more")
    return lines


def ui_history_lines(
    store: dict[str, Any],
    suggested_actions: list[dict[str, Any]],
    ui_events: list[str],
    line_width: int,
    max_rows: int,
) -> list[str]:
    lines: list[str] = []
    done_cards = [card for card in store.get("cards", []) if card.get("status") == "done"]
    done_cards.sort(key=lambda card: card.get("updated_at", ""), reverse=True)

    if ui_events:
        lines.append("Recent operator events:")
        for event in ui_events[-3:][::-1]:
            lines.append(short_text(f"- {event}", line_width))
        lines.append("")

    if suggested_actions:
        lines.append("Suggested actions:")
        for action in suggested_actions[: max(1, max_rows // 2)]:
            lines.append(short_text(f"stop {action['tty']}:{action['pid']} ({action['reason']})", line_width))
        lines.append("")

    lines.append("Completed cards:")
    if not done_cards:
        lines.append("none")
        return lines

    remaining = max_rows - len(lines)
    for card in done_cards[: max(1, remaining)]:
        title = short_text(card.get("title", ""), max(10, line_width - 12))
        lines.append(f"{card['id']} {title}")

    return lines[:max_rows]


def ui_tmux_session() -> str:
    try:
        store = tmux_teams.load_store()
    except Exception:
        return tmux_teams.DEFAULT_SESSION
    return str(store.get("default_session") or tmux_teams.DEFAULT_SESSION)


def panel_key(panel: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(panel.get("project_id") or ""),
        int(panel.get("index") or 0),
        str(panel.get("tty") or ""),
    )


def panel_sort_key(panel: dict[str, Any]) -> tuple[int, int, float, str]:
    health_rank = {"red": 0, "yellow": 1, "green": 2}.get(str(panel.get("health") or ""), 3)
    status_rank = {"waiting": 0, "active": 1, "background": 2, "idle": 3}.get(str(panel.get("status") or ""), 9)
    wait = float(panel.get("wait") or 0.0)
    project = str(panel.get("project_id") or "")
    return (health_rank, status_rank, -wait, project)


def sorted_panels(panels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(panels)
    rows.sort(key=panel_sort_key)
    return rows


def clamp_selected_panel_index(selected_index: int, total: int) -> int:
    if total <= 0:
        return 0
    return max(0, min(selected_index, total - 1))


def selected_panel(panels: list[dict[str, Any]], selected_index: int) -> dict[str, Any] | None:
    if not panels:
        return None
    idx = clamp_selected_panel_index(selected_index, len(panels))
    return panels[idx]


def panels_snapshot(
    tmux_session: str,
    capture_lines: int,
    wait_attention_min: int,
    apply_colors: bool,
) -> tuple[list[dict[str, Any]], str | None]:
    try:
        if not tmux_teams.session_exists(tmux_session):
            return [], f"tmux session '{tmux_session}' not running"
        rows = tmux_teams.refresh_pane_health(
            session=tmux_session,
            capture_lines=max(20, int(capture_lines)),
            wait_attention_min=max(0, int(wait_attention_min)),
            apply_colors=apply_colors,
        )
        pane_by_index = {pane.pane_index: pane for pane in tmux_teams.list_panes(tmux_session)}
        merged: list[dict[str, Any]] = []
        for row in rows:
            index = int(row.get("index") or 0)
            pane = pane_by_index.get(index)
            item = dict(row)
            item["pane_id"] = pane.pane_id if pane else ""
            item["pane_path"] = pane.pane_path if pane else ""
            item["pane_title"] = pane.pane_title if pane else ""
            item["pane_command"] = pane.pane_command if pane else ""
            merged.append(item)
        return sorted_panels(merged), None
    except Exception as exc:
        return [], str(exc)


def ui_panels_lines(
    panels: list[dict[str, Any]],
    selected_index: int,
    line_width: int,
    max_rows: int,
) -> list[str]:
    lines: list[str] = []
    if not panels:
        lines.append("No tmux panels found.")
        return lines

    lines.append("SEL IDX PROJECT                HLTH   ST       WAIT  AGENT")
    for idx, panel in enumerate(panels[: max(1, max_rows - 1)]):
        marker = ">" if idx == selected_index else " "
        wait = panel.get("wait")
        wait_text = f"{int(wait)}m" if wait is not None else "-"
        row = (
            f"{marker}  {int(panel.get('index') or 0):<3} {str(panel.get('project_id') or '-')[:22]:<22} "
            f"{str(panel.get('health') or '-'): <6} {str(panel.get('status') or '-'): <8} "
            f"{wait_text:<5} {str(panel.get('agent') or '-')[:8]:<8}"
        )
        lines.append(short_text(row, line_width))

    extra = len(panels) - max(1, max_rows - 1)
    if extra > 0:
        lines.append(f"... and {extra} more")
    return lines[:max_rows]


def ui_panel_detail_lines(
    panel: dict[str, Any] | None,
    tmux_session: str,
    line_width: int,
    max_rows: int,
) -> list[str]:
    lines: list[str] = []
    if not panel:
        lines.append("No panel selected.")
        return lines

    lines.append(f"Session {tmux_session}  Pane {panel.get('pane_id') or '-'}  Index {panel.get('index')}")
    lines.append(
        short_text(
            "Project {project}  Health {health}  Status {status}  Wait {wait}".format(
                project=str(panel.get("project_id") or "-"),
                health=str(panel.get("health") or "-"),
                status=str(panel.get("status") or "-"),
                wait=(f"{int(panel['wait'])}m" if panel.get("wait") is not None else "-"),
            ),
            line_width,
        )
    )
    lines.append(
        short_text(
            "TTY {tty}  Agent {agent}  Reason {reason}".format(
                tty=str(panel.get("tty") or "-"),
                agent=str(panel.get("agent") or "-"),
                reason=str(panel.get("reason") or "-"),
            ),
            line_width,
        )
    )
    lines.append("")
    lines.append(short_text(f"Path: {str(panel.get('pane_path') or '-')}", line_width))
    lines.append(short_text(f"Cmd: {str(panel.get('pane_command') or '-')}", line_width))
    lines.append("")
    lines.append("Quick actions")
    lines.append("- [enter/f] jump to panel in tmux")
    lines.append("- [k] stop panel (Ctrl-C)")
    lines.append("- [c] run claude   [x] run codex")
    lines.append("- [s] send custom command")
    lines.append("- [i] copy tty inspect command")
    return lines[:max_rows]


def ui_panel_channel_lines(
    tmux_session: str,
    panel_error: str | None,
    ui_events: list[str],
    line_width: int,
    max_rows: int,
) -> list[str]:
    lines: list[str] = []
    lines.append(f"Wrangler Channel (tmux={tmux_session})")
    if panel_error:
        lines.append(short_text(f"WARN: {panel_error}", line_width))
    lines.append("")

    if ui_events:
        for event in ui_events[-max(1, max_rows - 3) :][::-1]:
            lines.append(short_text(f"- {event}", line_width))
    else:
        lines.append("No events yet.")
    return lines[:max_rows]


def open_panel_in_tmux(tmux_session: str, panel: dict[str, Any] | None) -> str:
    if not panel:
        return "PANEL open: no selected panel"
    token = str(panel.get("project_id") or panel.get("index") or "")
    if not token:
        return "PANEL open: missing token"
    try:
        pane = tmux_teams.pane_target(tmux_session, token)
        code, _, err = tmux_teams.tmux(["select-pane", "-t", pane.pane_id], timeout=5)
        if code != 0:
            return f"PANEL open failed: {err.strip() or 'select-pane failed'}"
        if os.environ.get("TMUX"):
            code, _, err = tmux_teams.tmux(["switch-client", "-t", tmux_session], timeout=5)
            if code == 0:
                return f"PANEL open: switched to {tmux_session}:{pane.pane_id}"
            return f"PANEL open failed: {err.strip() or 'switch-client failed'}"
        tmux_teams.attach_session(tmux_session)
        return f"PANEL open: attached {tmux_session}:{pane.pane_id}"
    except Exception as exc:
        return f"PANEL open failed: {exc}"


def stop_panel(tmux_session: str, panel: dict[str, Any] | None) -> str:
    if not panel:
        return "PANEL stop: no selected panel"
    token = str(panel.get("project_id") or panel.get("index") or "")
    try:
        pane = tmux_teams.pane_target(tmux_session, token)
        tmux_teams.pane_ctrl_c(pane.pane_id)
        return f"PANEL stop: sent Ctrl-C to {pane.pane_id} ({token})"
    except Exception as exc:
        return f"PANEL stop failed: {exc}"


def send_panel_command(tmux_session: str, panel: dict[str, Any] | None, command: str) -> str:
    if not panel:
        return "PANEL send: no selected panel"
    if not command.strip():
        return "PANEL send: empty command"
    token = str(panel.get("project_id") or panel.get("index") or "")
    try:
        pane = tmux_teams.pane_target(tmux_session, token)
        tmux_teams.pane_send(pane.pane_id, command, enter=True)
        return f"PANEL send: {token} <= {command}"
    except Exception as exc:
        return f"PANEL send failed: {exc}"


def run_guard_once(source: str, max_ai_sessions: int, wait_limit_minutes: int, apply: bool) -> str:
    snapshot, _ = antfarm_snapshot(source=source, include_idle=False)
    actions = terminal_sentinel.recommend_actions(
        snapshot=snapshot,
        max_ai_sessions=max_ai_sessions,
        wait_limit_minutes=wait_limit_minutes,
    )
    results = terminal_sentinel.apply_actions(actions, dry_run=(not apply))
    mode = "APPLY" if apply else "PLAN"
    if not results:
        return f"{mode}: no actions"
    sample = results[0]
    return f"{mode}: {len(results)} action(s), first={sample.get('tty')}:{sample.get('pid')} {sample.get('result')}"


def kill_oldest_waiting(source: str) -> str:
    snapshot, _ = antfarm_snapshot(source=source, include_idle=False)
    waiting = [session for session in snapshot.get("sessions", []) if session.get("status") == "waiting"]
    if not waiting:
        return "KILL: no waiting session"
    waiting.sort(key=lambda s: (s.get("waiting_minutes") or 0), reverse=True)
    target = waiting[0]
    actions = [{"type": "stop", "pid": target["pid"], "tty": target["tty"], "reason": "manual kill-oldest"}]
    results = terminal_sentinel.apply_actions(actions, dry_run=False)
    result = results[0] if results else {"result": "unknown"}
    return f"KILL: {target['tty']}:{target['pid']} => {result.get('result')}"


def kill_specific_session(session: dict[str, Any]) -> str:
    if not session:
        return "KILL: no selected session"
    actions = [
        {
            "type": "stop",
            "pid": session.get("pid"),
            "tty": session.get("tty"),
            "reason": "manual selected kill",
        }
    ]
    results = terminal_sentinel.apply_actions(actions, dry_run=False)
    result = results[0] if results else {"result": "unknown"}
    return f"KILL: {session.get('tty')}:{session.get('pid')} => {result.get('result')}"


def resolve_session_target(
    sessions: list[dict[str, Any]], selected_index: int, token: str | None
) -> dict[str, Any] | None:
    if not sessions:
        return None
    if not token or token.lower() == "selected":
        return selected_session(sessions, selected_index)

    token_norm = token.strip()
    if token_norm.startswith("/dev/"):
        token_norm = token_norm.split("/")[-1]

    if token_norm.isdigit():
        pid = int(token_norm)
        for session in sessions:
            if int(session.get("pid") or 0) == pid:
                return session
        return None

    token_lower = token_norm.lower()
    for session in sessions:
        if str(session.get("tty") or "").lower() == token_lower:
            return session
    return None


def tty_inspect_command(tty: str) -> str:
    return f"ps -t {tty} -o pid=,ppid=,stat=,pcpu=,time=,command="


def copy_to_clipboard(text: str) -> bool:
    try:
        proc = subprocess.run(
            ["pbcopy"],
            input=text,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        return False
    return proc.returncode == 0


def launch_tty_monitor(tty: str) -> str:
    cmd = tty_inspect_command(tty)
    script = f"while true; do clear; date; echo 'Monitoring {tty}'; {cmd}; sleep 2; done"
    candidates = ["ghostty", "/Applications/Ghostty.app/Contents/MacOS/ghostty"]
    for binary in candidates:
        try:
            subprocess.Popen(
                [binary, "-e", "zsh", "-lc", script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return f"OPEN: launched monitor for {tty}"
        except FileNotFoundError:
            continue
        except Exception as exc:
            return f"OPEN failed: {exc}"
    return "OPEN failed: ghostty binary not found"


def prompt_command(stdscr: Any, prompt: str = "admin> ") -> str:
    h, w = stdscr.getmaxyx()
    width = max(1, w - len(prompt) - 1)
    text = ""

    prev_delay = 200
    if hasattr(stdscr, "getdelay"):
        try:
            prev_delay = int(stdscr.getdelay())
        except Exception:
            prev_delay = 200
    stdscr.nodelay(False)
    stdscr.timeout(-1)
    try:
        curses.echo()
        try:
            curses.curs_set(1)
        except curses.error:
            pass
        stdscr.addstr(h - 1, 0, short_text(prompt, w - 1))
        stdscr.clrtoeol()
        stdscr.refresh()
        raw = stdscr.getstr(h - 1, len(prompt), width)
        text = raw.decode("utf-8", errors="ignore").strip()
    except curses.error:
        text = ""
    finally:
        curses.noecho()
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        stdscr.timeout(prev_delay if prev_delay >= 0 else 200)
        stdscr.nodelay(True)

    return text


def execute_admin_command(
    command: str,
    sessions: list[dict[str, Any]],
    selected_index: int,
    source: str,
    max_ai_sessions: int,
    wait_limit_minutes: int,
) -> str:
    text = command.strip()
    if not text:
        return "CMD: empty"

    parts = text.split()
    op = parts[0].lower()
    target_token = parts[1] if len(parts) > 1 else None

    if op in {"help", "?"}:
        return "CMD help: kill [selected|pid|tty], inspect [selected|tty], open [selected|tty], plan, apply"

    if op == "kill":
        target = resolve_session_target(sessions, selected_index, target_token)
        if not target:
            return "CMD kill: target not found"
        return kill_specific_session(target)

    if op in {"inspect", "copy"}:
        target = resolve_session_target(sessions, selected_index, target_token)
        if not target:
            return "CMD inspect: target not found"
        command_text = tty_inspect_command(str(target.get("tty") or ""))
        copied = copy_to_clipboard(command_text)
        suffix = " (copied)" if copied else ""
        return f"INSPECT: {command_text}{suffix}"

    if op in {"open", "monitor"}:
        target = resolve_session_target(sessions, selected_index, target_token)
        if not target:
            return "CMD open: target not found"
        return launch_tty_monitor(str(target.get("tty") or ""))

    if op == "plan":
        return run_guard_once(
            source=source,
            max_ai_sessions=max_ai_sessions,
            wait_limit_minutes=wait_limit_minutes,
            apply=False,
        )

    if op == "apply":
        return run_guard_once(
            source=source,
            max_ai_sessions=max_ai_sessions,
            wait_limit_minutes=wait_limit_minutes,
            apply=True,
        )

    return f"CMD: unknown command '{op}'"


def ui_session_admin_lines(
    sessions: list[dict[str, Any]],
    selected_index: int,
    line_width: int,
    max_rows: int,
) -> list[str]:
    lines: list[str] = []
    if not sessions:
        lines.append("No sessions found.")
        return lines

    lines.append("SEL TTY      AGENT    ST       WAIT   CPU%  PID")
    tool_width = max(6, line_width - 44)
    for idx, session in enumerate(sessions[:max(1, max_rows - 1)]):
        marker = ">" if idx == selected_index else " "
        mins = session.get("waiting_minutes")
        wait_text = f"{int(mins)}m" if mins is not None else "-"
        tool = short_text(session_tool(session), tool_width)
        row = (
            f"{marker}  {str(session.get('tty', '-')):<8} {tool:<8} {str(session.get('status', '-')):<8} "
            f"{wait_text:<6} {str(session.get('pcpu', '-')):<5} {str(session.get('pid', '-')):<7}"
        )
        lines.append(short_text(row, line_width))

    extra = len(sessions) - max(1, max_rows - 1)
    if extra > 0:
        lines.append(f"... and {extra} more")
    return lines[:max_rows]


def ui_selected_detail_lines(
    session: dict[str, Any] | None,
    store: dict[str, Any],
    project_map: dict[str, dict[str, Any]],
    line_width: int,
    max_rows: int,
) -> list[str]:
    lines: list[str] = []
    if not session:
        lines.append("No session selected.")
        return lines

    tty = str(session.get("tty") or "-")
    pid = str(session.get("pid") or "-")
    status = str(session.get("status") or "-")
    wait = session.get("waiting_minutes")
    wait_text = f"{int(wait)}m" if wait is not None else "-"
    lines.append(f"TTY {tty}  PID {pid}")
    lines.append(f"Status {status}  Wait {wait_text}  CPU {session.get('pcpu')}%  Kind {session.get('kind')}")
    lines.append(f"Source {session.get('source')}  Runtime {session.get('runtime')}")
    lines.append("")

    command = str(session.get("command") or "")
    lines.append(short_text(f"Cmd: {command}", line_width))
    lines.append("")

    repo_id = infer_repo_from_command(command=command, project_map=project_map)
    if repo_id:
        lines.append(f"Repo signal: {repo_id}")
        open_cards = [
            card
            for card in store.get("cards", [])
            if is_card_open(card) and str(card.get("repo") or "") == repo_id
        ]
        if not open_cards:
            lines.append("Open cards: none")
        else:
            lines.append(f"Open cards: {len(open_cards)}")
            for card in open_cards[:2]:
                done, total = card_progress(card)
                lines.append(short_text(f"{card['id']} [{done}/{total}] {card.get('title', '')}", line_width))
    else:
        lines.append("Repo signal: not inferred")
        lines.append("Use inspect/open on this tty to drill in.")

    return lines[:max_rows]


def ui_channel_lines(
    ui_events: list[str],
    suggested_actions: list[dict[str, Any]],
    line_width: int,
    max_rows: int,
) -> list[str]:
    lines: list[str] = []
    lines.append("Admin channel")
    lines.append("")

    if ui_events:
        for event in ui_events[-max(1, max_rows - 4) :][::-1]:
            lines.append(short_text(f"- {event}", line_width))
    else:
        lines.append("No events yet. Press ':' for command mode.")

    if len(lines) < max_rows - 1:
        lines.append("")
        if suggested_actions:
            lines.append(short_text(f"Suggestion: stop {suggested_actions[0]['tty']}:{suggested_actions[0]['pid']}", line_width))

    return lines[:max_rows]


def draw_panel(
    stdscr: Any,
    y: int,
    x: int,
    h: int,
    w: int,
    title: str,
    lines: list[str],
    color_pair: int = 0,
) -> None:
    if h < 3 or w < 10:
        return

    attr = curses.color_pair(color_pair) if color_pair > 0 else curses.A_NORMAL

    try:
        stdscr.addstr(y, x, "+" + "-" * (w - 2) + "+", attr)
        stdscr.addstr(y + h - 1, x, "+" + "-" * (w - 2) + "+", attr)
        for row in range(y + 1, y + h - 1):
            stdscr.addstr(row, x, "|", attr)
            stdscr.addstr(row, x + w - 1, "|", attr)
        stdscr.addstr(y, x + 2, short_text(f"[{title}]", max(4, w - 4)), attr | curses.A_BOLD)
    except curses.error:
        return

    max_lines = h - 2
    content_width = w - 2
    for idx, line in enumerate(lines[:max_lines]):
        try:
            stdscr.addstr(y + 1 + idx, x + 1, short_text(line, content_width).ljust(content_width))
        except curses.error:
            pass


def render_ui_frame(
    stdscr: Any,
    store: dict[str, Any],
    snapshot: dict[str, Any],
    warnings: list[str],
    max_ai_sessions: int,
    wait_limit_minutes: int,
    source: str,
    limit: int,
    page: str,
    sessions: list[dict[str, Any]],
    selected_index: int,
    panels: list[dict[str, Any]],
    selected_panel_index: int,
    panel_error: str | None,
    tmux_session: str,
    ui_events: list[str],
) -> None:
    h, w = stdscr.getmaxyx()
    stdscr.erase()

    if h < 18 or w < 90:
        msg = f"Terminal too small ({w}x{h}). Resize to at least 90x18."
        try:
            stdscr.addstr(0, 0, msg)
            stdscr.addstr(2, 0, "Press q to quit.")
        except curses.error:
            pass
        stdscr.refresh()
        return

    summary = snapshot.get("summary", {})
    open_cards = plan_summary(store).get("open_total", 0)
    page = page if page in UI_PAGES else UI_PAGES[0]
    page_number = 1 if page == "wrangler" else 2
    page_label = "Wrangler" if page == "wrangler" else "Panels"
    title = (
        "AGENT WRANGLER  |  Page {page_number}/2 {page_label}  |  Source {source}  |  "
        "Open Cards {open_cards}  |  Sessions {sessions}  |  Waiting {waiting}  |  Tmux {tmux_session}".format(
            page_number=page_number,
            page_label=page_label,
            source=source,
            open_cards=open_cards,
            sessions=summary.get("total", 0),
            waiting=summary.get("waiting", 0),
            tmux_session=tmux_session,
        )
    )
    try:
        stdscr.addstr(0, 0, short_text(title, w - 1), curses.color_pair(1) | curses.A_BOLD)
    except curses.error:
        pass

    suggested_actions = terminal_sentinel.recommend_actions(
        snapshot=snapshot,
        max_ai_sessions=max_ai_sessions,
        wait_limit_minutes=wait_limit_minutes,
    )

    body_top = 1
    body_h = h - 3
    if page == "wrangler":
        top_h = max(8, int(body_h * 0.45))
        bottom_h = body_h - top_h
        if bottom_h < 6:
            top_h = body_h - 6
            bottom_h = 6

        left_w = max(36, int(w * 0.52))
        right_w = w - left_w

        studio_lines = build_studio_overview_lines(
            store=store,
            snapshot=snapshot,
            line_width=left_w - 3,
            max_rows=max(4, top_h - 3),
        )
        health_lines = build_health_lines(
            snapshot=snapshot,
            warnings=warnings,
            suggested_actions=suggested_actions,
            line_width=right_w - 3,
            max_rows=max(4, top_h - 3),
        )
        panel_red = len([row for row in panels if str(row.get("health") or "") == "red"])
        panel_waiting = len([row for row in panels if str(row.get("status") or "") == "waiting"])
        health_lines.insert(0, f"Panels total={len(panels)} red={panel_red} waiting={panel_waiting}")
        if panel_error:
            health_lines.insert(1, short_text(f"WARN: {panel_error}", right_w - 3))

        draw_panel(stdscr, body_top, 0, top_h, left_w, "Wrangler Overview", studio_lines, color_pair=2)
        draw_panel(stdscr, body_top, left_w, top_h, right_w, "Runtime + Panel Health", health_lines, color_pair=4)

        lower_y = body_top + top_h
        queue_lines = ui_queue_lines(store=store, line_width=left_w - 3, max_rows=max(3, bottom_h - 3))
        history_lines = ui_history_lines(
            store=store,
            suggested_actions=suggested_actions,
            ui_events=ui_events,
            line_width=right_w - 3,
            max_rows=max(3, bottom_h - 3),
        )
        draw_panel(stdscr, lower_y, 0, bottom_h, left_w, "Wrangler Queue", queue_lines, color_pair=3)
        draw_panel(stdscr, lower_y, left_w, bottom_h, right_w, "Wrangler Log", history_lines, color_pair=5)

        footer = "[tab/2] panels  [k] kill-oldest waiting  [o] plan  [a] apply  [r] refresh  [q] quit"
        try:
            stdscr.addstr(h - 1, 0, short_text(footer, w - 1), curses.color_pair(5) | curses.A_BOLD)
        except curses.error:
            pass
    else:
        top_h = max(8, int(body_h * 0.62))
        channel_h = body_h - top_h
        if channel_h < 6:
            top_h = body_h - 6
            channel_h = 6

        left_w = max(36, int(w * 0.46))
        right_w = w - left_w

        clamped_panel_index = clamp_selected_panel_index(selected_panel_index, len(panels))
        panel_lines = ui_panels_lines(
            panels=panels,
            selected_index=clamped_panel_index,
            line_width=left_w - 3,
            max_rows=max(4, top_h - 3),
        )
        detail_lines = ui_panel_detail_lines(
            panel=selected_panel(panels, clamped_panel_index),
            tmux_session=tmux_session,
            line_width=right_w - 3,
            max_rows=max(4, top_h - 3),
        )
        channel_lines = ui_panel_channel_lines(
            tmux_session=tmux_session,
            panel_error=panel_error,
            ui_events=ui_events,
            line_width=w - 3,
            max_rows=max(3, channel_h - 3),
        )

        draw_panel(stdscr, body_top, 0, top_h, left_w, "Panels", panel_lines, color_pair=2)
        draw_panel(stdscr, body_top, left_w, top_h, right_w, "Selected Panel", detail_lines, color_pair=4)
        draw_panel(stdscr, body_top + top_h, 0, channel_h, w, "Wrangler Channel", channel_lines, color_pair=3)

        footer = (
            "[tab/1] wrangler  [up/down] select panel  [enter/f] jump  [s] send  [c] claude  [x] codex  "
            "[k] stop  [i] inspect copy  [r] refresh  [q] quit"
        )
        try:
            stdscr.addstr(h - 1, 0, short_text(footer, w - 1), curses.color_pair(5) | curses.A_BOLD)
        except curses.error:
            pass

    stdscr.refresh()


def run_ui(args: argparse.Namespace) -> int:
    store = load_store()
    settings = store.get("settings", {}).get("antfarm", {})
    source = args.source or settings.get("source", "ghostty")
    max_ai = int(args.max_ai_sessions or settings.get("max_ai_sessions", 4))
    wait_limit = int(args.kill_waiting_ai_after or settings.get("kill_waiting_ai_after_min", 120))
    interval = max(1, args.interval)

    def _loop(stdscr: Any) -> int:
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        stdscr.nodelay(True)
        stdscr.timeout(200)

        if curses.has_colors():
            try:
                curses.start_color()
                curses.use_default_colors()
                curses.init_pair(1, curses.COLOR_CYAN, -1)
                curses.init_pair(2, curses.COLOR_GREEN, -1)
                curses.init_pair(3, curses.COLOR_YELLOW, -1)
                curses.init_pair(4, curses.COLOR_BLUE, -1)
                curses.init_pair(5, curses.COLOR_MAGENTA, -1)
            except curses.error:
                pass

        last_tick = 0.0
        snapshot: dict[str, Any] = {"sessions": [], "summary": {}}
        warnings: list[str] = []
        sessions: list[dict[str, Any]] = []
        selected_index = 0
        panels: list[dict[str, Any]] = []
        selected_panel_index = 0
        panel_error: str | None = None
        tmux_session = ui_tmux_session()
        ui_page = "wrangler"
        ui_events: list[str] = [f"{now_iso()} UI started"]

        def add_event(message: str) -> None:
            ui_events.append(f"{now_iso()} {message}")
            del ui_events[:-20]

        def refresh_view() -> None:
            nonlocal snapshot, warnings, sessions, selected_index, panels, selected_panel_index, panel_error, tmux_session, last_tick
            selected = selected_session(sessions, selected_index)
            selected_key = session_key(selected) if selected else None
            selected_panel_row = selected_panel(panels, selected_panel_index)
            selected_panel_key = panel_key(selected_panel_row) if selected_panel_row else None

            snapshot, warnings = antfarm_snapshot(source=source, include_idle=False)
            sessions = sorted_sessions(snapshot)
            tmux_session_local = ui_tmux_session()
            tmux_session = tmux_session_local
            panel_rows, panel_err = panels_snapshot(
                tmux_session=tmux_session_local,
                capture_lines=80,
                wait_attention_min=1,
                apply_colors=True,
            )
            panels = panel_rows
            panel_error = panel_err

            if selected_key:
                matched = False
                for idx, session in enumerate(sessions):
                    if session_key(session) == selected_key:
                        selected_index = idx
                        matched = True
                        break
                if not matched:
                    selected_index = clamp_selected_index(selected_index, len(sessions))
            else:
                selected_index = clamp_selected_index(selected_index, len(sessions))

            if selected_panel_key:
                matched = False
                for idx, panel_row in enumerate(panels):
                    if panel_key(panel_row) == selected_panel_key:
                        selected_panel_index = idx
                        matched = True
                        break
                if not matched:
                    selected_panel_index = clamp_selected_panel_index(selected_panel_index, len(panels))
            else:
                selected_panel_index = clamp_selected_panel_index(selected_panel_index, len(panels))

            store_local = load_store()
            render_ui_frame(
                stdscr=stdscr,
                store=store_local,
                snapshot=snapshot,
                warnings=warnings,
                max_ai_sessions=max_ai,
                wait_limit_minutes=wait_limit,
                source=source,
                limit=args.limit,
                page=ui_page,
                sessions=sessions,
                selected_index=selected_index,
                panels=panels,
                selected_panel_index=selected_panel_index,
                panel_error=panel_error,
                tmux_session=tmux_session_local,
                ui_events=ui_events,
            )
            last_tick = time.time()

        refresh_view()

        while True:
            now = time.time()
            if now - last_tick >= interval:
                refresh_view()

            try:
                ch = stdscr.getch()
            except curses.error:
                ch = -1

            if ch < 0:
                continue

            if ch in (ord("q"), ord("Q")):
                break

            if ch in (9, curses.KEY_BTAB):
                ui_page = "panels" if ui_page == "wrangler" else "wrangler"
                add_event(f"PAGE: {ui_page}")
                refresh_view()
                continue
            if ch == ord("1"):
                ui_page = "wrangler"
                refresh_view()
                continue
            if ch == ord("2"):
                ui_page = "panels"
                refresh_view()
                continue

            if ui_page == "panels" and ch in (curses.KEY_UP, ord("p"), ord("P")):
                selected_panel_index = clamp_selected_panel_index(selected_panel_index - 1, len(panels))
                refresh_view()
                continue
            if ui_page == "panels" and ch in (curses.KEY_DOWN, ord("n"), ord("N"), ord("j"), ord("J")):
                selected_panel_index = clamp_selected_panel_index(selected_panel_index + 1, len(panels))
                refresh_view()
                continue

            if ch in (ord("r"), ord("R")):
                add_event("REFRESH")
                refresh_view()
                continue

            if ui_page == "panels" and ch in (ord("g"), ord("G"), ord("f"), ord("F"), 10, 13):
                message = open_panel_in_tmux(tmux_session, selected_panel(panels, selected_panel_index))
                add_event(message)
                refresh_view()
                continue

            if ui_page == "panels" and ch in (ord("i"), ord("I")):
                target = selected_panel(panels, selected_panel_index)
                if target:
                    inspect_cmd = tty_inspect_command(str(target.get("tty") or ""))
                    copied = copy_to_clipboard(inspect_cmd)
                    suffix = " (copied)" if copied else ""
                    add_event(f"INSPECT: {inspect_cmd}{suffix}")
                else:
                    add_event("INSPECT: no selected panel")
                refresh_view()
                continue

            if ui_page == "panels" and ch in (ord("k"), ord("K")):
                message = stop_panel(tmux_session, selected_panel(panels, selected_panel_index))
                add_event(message)
                refresh_view()
                continue

            if ui_page == "wrangler" and ch in (ord("k"), ord("K")):
                message = kill_oldest_waiting(source=source)
                add_event(message)
                refresh_view()
                continue

            if ui_page == "panels" and ch in (ord("c"), ord("C")):
                message = send_panel_command(
                    tmux_session=tmux_session,
                    panel=selected_panel(panels, selected_panel_index),
                    command="claude",
                )
                add_event(message)
                refresh_view()
                continue

            if ui_page == "panels" and ch in (ord("x"), ord("X")):
                message = send_panel_command(
                    tmux_session=tmux_session,
                    panel=selected_panel(panels, selected_panel_index),
                    command="codex",
                )
                add_event(message)
                refresh_view()
                continue

            if ui_page == "panels" and ch in (ord("s"), ord("S")):
                cmd = prompt_command(stdscr, prompt="send> ")
                if cmd:
                    message = send_panel_command(
                        tmux_session=tmux_session,
                        panel=selected_panel(panels, selected_panel_index),
                        command=cmd,
                    )
                    add_event(message)
                else:
                    add_event("SEND canceled")
                refresh_view()
                continue

            if ui_page == "wrangler" and ch == ord(":"):
                cmd = prompt_command(stdscr, prompt="wrangler> ")
                if cmd:
                    message = execute_admin_command(
                        command=cmd,
                        sessions=sessions,
                        selected_index=selected_index,
                        source=source,
                        max_ai_sessions=max_ai,
                        wait_limit_minutes=wait_limit,
                    )
                    add_event(f"CMD {cmd} -> {message}")
                else:
                    add_event("CMD canceled")
                refresh_view()
                continue

            if ch in (ord("o"), ord("O")):
                message = run_guard_once(
                    source=source,
                    max_ai_sessions=max_ai,
                    wait_limit_minutes=wait_limit,
                    apply=False,
                )
                add_event(message)
                refresh_view()
                continue

            if ch in (ord("a"), ord("A")):
                message = run_guard_once(
                    source=source,
                    max_ai_sessions=max_ai,
                    wait_limit_minutes=wait_limit,
                    apply=True,
                )
                add_event(message)
                refresh_view()
                continue

        return 0

    return int(curses.wrapper(_loop))


def run_dashboard(args: argparse.Namespace) -> int:
    store = load_store()
    settings = store.get("settings", {}).get("antfarm", {})

    source = args.source or settings.get("source", "ghostty")
    max_ai = int(args.max_ai_sessions or settings.get("max_ai_sessions", 4))
    wait_limit = int(args.kill_waiting_ai_after or settings.get("kill_waiting_ai_after_min", 120))

    snapshot, warnings = antfarm_snapshot(source=source, include_idle=False)

    print(f"[{now_iso()}] {BRAND_NAME}")
    print("")
    print_gastown(store, show_steps=args.show_steps, limit_per_lane=args.limit)
    print("\n" + "-" * 72 + "\n")
    print_antfarm(
        snapshot=snapshot,
        warnings=warnings,
        max_wait_rows=8,
        max_ai_sessions=max_ai,
        wait_limit_minutes=wait_limit,
    )
    return 0



def run_dashboard_watch(args: argparse.Namespace) -> int:
    loops = 0
    interval = max(2, args.interval)
    try:
        while True:
            print("\033[2J\033[H", end="")
            run_dashboard(args)
            loops += 1
            if args.iterations > 0 and loops >= args.iterations:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        return 0
    return 0



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
        path = REPORTS_DIR / f"command-center-antfarm-{ts}.json"
        payload = {
            "generated_at": now_iso(),
            "warnings": warnings,
            "snapshot": snapshot,
        }
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
                f"ai={summary.get('ai', 0)} waiting={summary.get('waiting', 0)} active={summary.get('active', 0)} actions={len(results)}"
            )
            for result in results:
                mode = "APPLY" if result.get("applied") else "PLAN"
                print(
                    f"- {mode} pid={result.get('pid')} tty={result.get('tty')} reason={result.get('reason')} => {result.get('result')}"
                )

            if args.write_log:
                REPORTS_DIR.mkdir(parents=True, exist_ok=True)
                if args.log_file:
                    log_path = Path(args.log_file).expanduser()
                else:
                    log_path = REPORTS_DIR / "command-center-overnight.log"
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "generated_at": now_iso(),
                                "warnings": warnings,
                                "summary": summary,
                                "actions": results,
                            }
                        )
                        + "\n"
                    )

            loops += 1
            if args.iterations > 0 and loops >= args.iterations:
                break
            time.sleep(max(30, interval))
    except KeyboardInterrupt:
        return 0

    return 0



def run_palette(_: argparse.Namespace) -> int:
    print("Agent Wrangler Palette")
    print(f"1. {PRIMARY_CLI} ui")
    print("   - page switch: tab / 1 (wrangler) / 2 (panels)")
    print("   - panels controls: up/down, enter/f, s, c, x, k, i")
    print(f"2. {PRIMARY_CLI} dashboard")
    print(f"3. {PRIMARY_CLI} watch")
    print(f"4. {PRIMARY_CLI} gastown list --open-only --verbose")
    print(f"5. {PRIMARY_CLI} antfarm status")
    print(f"6. {PRIMARY_CLI} antfarm overnight --iterations 1")
    print(f"7. {PRIMARY_CLI} antfarm overnight --apply --iterations 1")
    print(f"8. {PRIMARY_CLI} gastown add --title \"...\" --repo <repo-id> --lane now --step \"...\"")
    print(f"9. {PRIMARY_CLI} start")
    print(f"10. {PRIMARY_CLI} manager --replace")
    print(f"11. {PRIMARY_CLI} nav")
    print(f"12. {PRIMARY_CLI} status")
    print(f"13. {PRIMARY_CLI} fleet status")
    print(f"14. {PRIMARY_CLI} fleet manager --replace --update-defaults")
    print(f"15. {PRIMARY_CLI} drift --fleet --alert-dirty 25")
    print(f"16. {PRIMARY_CLI} fleet jump --fzf")
    print(f"17. {PRIMARY_CLI} doctor --fleet --only-attention")
    print(f"18. {PRIMARY_CLI} persistence status")
    print(f"19. {PRIMARY_CLI} persistence save")
    print(f"20. {PRIMARY_CLI} persistence restore --force --attach")
    print(f"21. {PRIMARY_CLI} profile status")
    print(f"22. {PRIMARY_CLI} profile list")
    print(f"23. {PRIMARY_CLI} hooks enable")
    print(f"24. {PRIMARY_CLI} hooks status")
    print(f"25. {PRIMARY_CLI} agent creator-studio codex")
    print(f"26. {PRIMARY_CLI} program init")
    print(f"27. {PRIMARY_CLI} program status")
    print(f"28. {PRIMARY_CLI} program plan --write-report")
    print(f"29. {PRIMARY_CLI} program loop --iterations 1 --apply-safe --write-report")
    print(f"30. {PRIMARY_CLI} program daemon --apply-guardrails")
    print(f"31. {PRIMARY_CLI} program phases --refresh-state")
    print(f"32. {PRIMARY_CLI} program promote")
    print(f"33. {PRIMARY_CLI} program complete")
    print(f"34. {PRIMARY_CLI} ops")
    print("Legacy aliases still work: cc, hq, teams")
    return 0


def _run_ops_command(command: list[str]) -> int:
    cmd = [str(ROOT / "scripts" / "agent-wrangler"), *command]
    print("")
    print("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def run_ops(_: argparse.Namespace) -> int:
    actions: list[tuple[str, list[str]]] = [
        ("Open Wrangler UI (2-page command center)", ["ui"]),
        ("Start all (import + grid + manager + nav)", ["start"]),
        ("Attach grid session", ["attach"]),
        ("Show pane status", ["status"]),
        ("Focus pane by project/token and attach", ["__focus__"]),
        ("Send command to pane/project token", ["__send__"]),
        ("Launch agent in pane/project token", ["__agent__"]),
        ("Stop pane (Ctrl-C) by project/token", ["__stop__"]),
        ("Open manager window", ["manager", "--replace"]),
        ("Fleet status", ["fleet", "status"]),
        ("Fleet jump (fzf)", ["fleet", "jump", "--fzf"]),
        ("Doctor (fleet attention)", ["doctor", "--fleet", "--only-attention"]),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PRIMARY_CLI, description=f"{BRAND_NAME} (Gastown + Ant Farm)")
    sub = parser.add_subparsers(dest="command", required=True)

    dashboard = sub.add_parser("dashboard", help="Show combined Gastown + Ant Farm view")
    dashboard.add_argument("--source", default=None, choices=["ghostty", "all", "codex", "iterm", "terminal"])
    dashboard.add_argument("--max-ai-sessions", type=int, default=None)
    dashboard.add_argument("--kill-waiting-ai-after", type=int, default=None)
    dashboard.add_argument("--show-steps", action="store_true")
    dashboard.add_argument("--limit", type=int, default=8, help="Max cards per lane")
    dashboard.set_defaults(handler=run_dashboard)

    watch = sub.add_parser("watch", help="Live combined dashboard")
    watch.add_argument("--source", default=None, choices=["ghostty", "all", "codex", "iterm", "terminal"])
    watch.add_argument("--max-ai-sessions", type=int, default=None)
    watch.add_argument("--kill-waiting-ai-after", type=int, default=None)
    watch.add_argument("--show-steps", action="store_true")
    watch.add_argument("--limit", type=int, default=8)
    watch.add_argument("--interval", type=int, default=10)
    watch.add_argument("--iterations", type=int, default=0)
    watch.set_defaults(handler=run_dashboard_watch)

    ui = sub.add_parser("ui", help=f"Fullscreen TUI {BRAND_NAME}")
    ui.add_argument("--source", default=None, choices=["ghostty", "all", "codex", "iterm", "terminal"])
    ui.add_argument("--max-ai-sessions", type=int, default=None)
    ui.add_argument("--kill-waiting-ai-after", type=int, default=None)
    ui.add_argument("--interval", type=int, default=4, help="Refresh interval seconds")
    ui.add_argument("--limit", type=int, default=8, help="Max cards per lane")
    ui.set_defaults(handler=run_ui)

    palette = sub.add_parser("palette", help="Show quick command palette shortcuts")
    palette.set_defaults(handler=run_palette)

    ops = sub.add_parser("ops", help="Interactive one-screen operator console")
    ops.set_defaults(handler=run_ops)

    gastown = sub.add_parser("gastown", help="Planning board operations")
    gastown_sub = gastown.add_subparsers(dest="gastown_command", required=True)

    g_list = gastown_sub.add_parser("list", help="List cards")
    g_list.add_argument("--lane", choices=LANE_ORDER)
    g_list.add_argument("--status", choices=sorted(VALID_STATUSES))
    g_list.add_argument("--repo")
    g_list.add_argument("--open-only", action="store_true")
    g_list.add_argument("--verbose", action="store_true")
    g_list.set_defaults(handler=list_cards)

    g_add = gastown_sub.add_parser("add", help="Add a planning card")
    g_add.add_argument("--title", required=True)
    g_add.add_argument("--repo")
    g_add.add_argument("--lane", required=True, choices=LANE_ORDER)
    g_add.add_argument("--step", action="append", help="Step text (repeatable)")
    g_add.add_argument("--notes")
    g_add.set_defaults(handler=add_card)

    g_step_add = gastown_sub.add_parser("step-add", help="Add a step to a card")
    g_step_add.add_argument("card_id")
    g_step_add.add_argument("--text", required=True)
    g_step_add.set_defaults(handler=add_step)

    g_step_done = gastown_sub.add_parser("step-done", help="Mark step done")
    g_step_done.add_argument("card_id")
    g_step_done.add_argument("--step", required=True, type=int)
    g_step_done.set_defaults(handler=lambda args: set_step_done(args, done=True))

    g_step_open = gastown_sub.add_parser("step-open", help="Mark step open")
    g_step_open.add_argument("card_id")
    g_step_open.add_argument("--step", required=True, type=int)
    g_step_open.set_defaults(handler=lambda args: set_step_done(args, done=False))

    g_move = gastown_sub.add_parser("move", help="Move card to another lane")
    g_move.add_argument("card_id")
    g_move.add_argument("--lane", required=True, choices=LANE_ORDER)
    g_move.set_defaults(handler=move_card)

    g_status = gastown_sub.add_parser("status", help="Set card status")
    g_status.add_argument("card_id")
    g_status.add_argument("--status", required=True, choices=sorted(VALID_STATUSES))
    g_status.set_defaults(handler=set_card_status)

    g_block = gastown_sub.add_parser("block", help="Add card dependency")
    g_block.add_argument("card_id")
    g_block.add_argument("--on", dest="depends_on", required=True)
    g_block.set_defaults(handler=add_dependency)

    g_unblock = gastown_sub.add_parser("unblock", help="Remove card dependency")
    g_unblock.add_argument("card_id")
    g_unblock.add_argument("--on", dest="depends_on", required=True)
    g_unblock.set_defaults(handler=remove_dependency)

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
