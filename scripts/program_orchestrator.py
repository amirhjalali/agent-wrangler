#!/usr/bin/env python3
"""Agent Wrangler program orchestrator: team, loops, and impeccable-product roadmap."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import terminal_sentinel
import tmux_teams

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "impeccable_program.json"
REPORTS_DIR = ROOT / "reports"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ts_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def default_program() -> dict[str, Any]:
    return {
        "version": "1.0",
        "program_name": "Agent Wrangler Impeccable Program",
        "objective": "Ship an impeccable multi-session agent command center for daily and overnight operations.",
        "targets": {
            "min_readiness_score": 92,
            "max_attention": 1,
            "max_high_drift_projects": 1,
            "max_dirty_files": 120,
        },
        "team": [
            {
                "id": "wrangler-chief",
                "title": "Program Orchestrator",
                "mission": "Own roadmap, gating, and execution sequencing across all agent teams.",
                "kpi": "Readiness score >= 92 for 7 consecutive days.",
            },
            {
                "id": "fleet-warden",
                "title": "Runtime Reliability",
                "mission": "Keep fleet manager healthy and attention queue under strict control.",
                "kpi": "Attention count <= 1 and red panes <= 1.",
            },
            {
                "id": "repo-sheriff",
                "title": "Repo Hygiene",
                "mission": "Prevent drift and dirty-tree entropy across active projects.",
                "kpi": "High-drift projects <= 1 and dirty files <= 120.",
            },
            {
                "id": "ux-rider",
                "title": "Operator Experience",
                "mission": "Reduce command friction and improve response speed in manager views.",
                "kpi": "Primary flows completed in <= 2 commands.",
            },
            {
                "id": "qa-gunslinger",
                "title": "Quality and Safety",
                "mission": "Harden error handling and guardrails for long unattended runs.",
                "kpi": "No critical regressions from command upgrades.",
            },
            {
                "id": "automation-rancher",
                "title": "Loop Automation",
                "mission": "Turn manual checks into repeatable program loops and reports.",
                "kpi": "All loop reports generated on cadence with clear next actions.",
            },
        ],
        "loops": [
            {
                "id": "fleet-heartbeat",
                "owner": "fleet-warden",
                "cadence": "every 3 minutes",
                "goal": "Detect attention spikes and session health regressions quickly.",
                "entry_command": "agent-wrangler fleet watch --interval 3",
            },
            {
                "id": "drift-sweep",
                "owner": "repo-sheriff",
                "cadence": "every 30 minutes",
                "goal": "Keep dirty-tree growth bounded and visible.",
                "entry_command": "agent-wrangler drift --fleet --alert-dirty 25",
            },
            {
                "id": "hardening-loop",
                "owner": "qa-gunslinger",
                "cadence": "every 2 hours",
                "goal": "Fix top failure mode from current signals and close with validation.",
                "entry_command": "agent-wrangler program plan --write-report",
            },
            {
                "id": "delivery-loop",
                "owner": "wrangler-chief",
                "cadence": "daily",
                "goal": "Promote readiness gates and ship one measurable improvement.",
                "entry_command": "agent-wrangler program loop --iterations 1 --apply-safe --write-report",
            },
        ],
        "phases": [
            {
                "id": "phase-1",
                "name": "Operational Hardening",
                "definition_of_done": [
                    "fleet manager runs reliably for full workday",
                    "attention queue remains <= 1 for most cycles",
                    "no blocking pane-creation/layout regressions",
                ],
            },
            {
                "id": "phase-2",
                "name": "Impeccable UX",
                "definition_of_done": [
                    "primary flows fit in 1-2 commands",
                    "fleet and drift outputs are readable at a glance",
                    "operator can jump to any session quickly",
                ],
            },
            {
                "id": "phase-3",
                "name": "Autonomous Loops",
                "definition_of_done": [
                    "program loop emits actionable report each run",
                    "safe auto-remediations reduce manual interventions",
                    "overnight behavior is bounded and observable",
                ],
            },
            {
                "id": "phase-4",
                "name": "Impeccable Product",
                "definition_of_done": [
                    "readiness score >= 92 for 7 consecutive days",
                    "high drift remains <= 1 project",
                    "fleet attention remains <= 1 without constant manual babysitting",
                ],
            },
        ],
        "state": {
            "created_at": now_iso(),
            "last_run_at": None,
            "last_readiness_score": None,
            "last_stage": None,
            "readiness_streak": 0,
            "current_phase": 1,
            "phase_statuses": [],
            "history": [],
        },
    }


def normalize_program(raw: dict[str, Any] | None) -> dict[str, Any]:
    base = default_program()
    current = raw if isinstance(raw, dict) else {}
    merged = {
        **base,
        **current,
    }
    for key in ("targets", "state"):
        base_obj = base.get(key, {})
        cur_obj = current.get(key, {}) if isinstance(current.get(key), dict) else {}
        merged[key] = {
            **base_obj,
            **cur_obj,
        }
    for key in ("team", "loops", "phases"):
        if not isinstance(merged.get(key), list):
            merged[key] = list(base.get(key, []))
    return merged


def load_program() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return default_program()
    try:
        parsed = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return default_program()
    return normalize_program(parsed)


def save_program(program: dict[str, Any]) -> None:
    normalized = normalize_program(program)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(normalized, indent=2), encoding="utf-8")


def add_signal_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sessions", help="Comma-separated fleet sessions override")
    parser.add_argument("--pattern", help="Filter sessions by substring")
    parser.add_argument("--include-manager", action="store_true", help="Include fleet manager session")
    parser.add_argument("--capture-lines", type=int, default=80)
    parser.add_argument("--wait-attention-min", type=int, default=1)
    parser.add_argument("--alert-dirty", type=int, default=25)


def collect_signals(args: argparse.Namespace, store: dict[str, Any]) -> dict[str, Any]:
    tmux_teams.ensure_tmux()
    sessions = tmux_teams.resolve_fleet_sessions(
        store=store,
        explicit_csv=args.sessions,
        pattern=args.pattern,
        include_manager=args.include_manager,
    )
    rows = tmux_teams.fleet_health_rows(
        sessions=sessions,
        capture_lines=max(20, int(args.capture_lines)),
        wait_attention_min=max(0, int(args.wait_attention_min)),
        apply_colors=False,
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

    proj_map = tmux_teams.project_map()
    drift_rows: list[dict[str, Any]] = []
    for session in sessions:
        if not tmux_teams.session_exists(session):
            continue
        try:
            projects = tmux_teams.project_rows_for_session(session, proj_map)
        except ValueError:
            continue
        for item in projects:
            git = item.get("git")
            dirty = int(git.get("dirty") or 0) if isinstance(git, dict) else 0
            drift_rows.append(
                {
                    "session": session,
                    "project_id": str(item.get("project_id") or "-"),
                    "path": str(item.get("path") or "-"),
                    "dirty": dirty,
                    "branch": (str(git.get("branch") or "-") if isinstance(git, dict) else "-"),
                    "ahead": (int(git.get("ahead") or 0) if isinstance(git, dict) else 0),
                    "behind": (int(git.get("behind") or 0) if isinstance(git, dict) else 0),
                }
            )

    dirty_total = sum(int(item.get("dirty") or 0) for item in drift_rows)
    high_drift = [item for item in drift_rows if int(item.get("dirty") or 0) >= int(args.alert_dirty)]
    high_drift.sort(key=lambda item: int(item.get("dirty") or 0), reverse=True)

    fleet_cfg = store.get("fleet", {})
    manager_session = str(fleet_cfg.get("manager_session") or tmux_teams.DEFAULT_FLEET_MANAGER_SESSION)
    manager_running = bool(manager_session and tmux_teams.session_exists(manager_session))

    return {
        "generated_at": now_iso(),
        "sessions": sessions,
        "fleet_rows": rows,
        "fleet_totals": totals,
        "drift_rows": drift_rows,
        "dirty_total": dirty_total,
        "high_drift": high_drift,
        "manager_session": manager_session,
        "manager_running": manager_running,
    }


def compute_readiness(program: dict[str, Any], metrics: dict[str, Any]) -> tuple[int, str]:
    targets = program.get("targets", {})
    score = 100

    attention = int(metrics.get("fleet_totals", {}).get("attention") or 0)
    red = int(metrics.get("fleet_totals", {}).get("red") or 0)
    sessions = int(metrics.get("fleet_totals", {}).get("sessions") or 0)
    high_drift_count = len(metrics.get("high_drift", []))
    dirty_total = int(metrics.get("dirty_total") or 0)
    manager_running = bool(metrics.get("manager_running"))

    if sessions == 0:
        score = 0
    score -= min(45, attention * 10)
    score -= min(20, red * 8)
    score -= min(25, high_drift_count * 6)

    dirty_limit = int(targets.get("max_dirty_files", 120))
    if dirty_total > dirty_limit:
        overflow = dirty_total - dirty_limit
        score -= min(20, 5 + (overflow // 25) * 3)

    if not manager_running:
        score -= 10

    score = max(0, min(100, score))
    if score >= 92:
        stage = "impeccable-candidate"
    elif score >= 78:
        stage = "strong"
    elif score >= 60:
        stage = "progressing"
    else:
        stage = "unstable"
    return score, stage


def recent_loop_report_hours() -> float | None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(REPORTS_DIR.glob("program-loop-*.json"))
    if not files:
        return None
    latest = files[-1]
    age_seconds = time.time() - latest.stat().st_mtime
    return age_seconds / 3600.0


def build_gates(program: dict[str, Any], metrics: dict[str, Any], score: int) -> list[dict[str, Any]]:
    targets = program.get("targets", {})
    totals = metrics.get("fleet_totals", {})
    attention = int(totals.get("attention") or 0)
    red = int(totals.get("red") or 0)
    dirty_total = int(metrics.get("dirty_total") or 0)
    high_drift_count = len(metrics.get("high_drift", []))
    manager_running = bool(metrics.get("manager_running"))
    sessions = int(totals.get("sessions") or 0)
    recent_hours = recent_loop_report_hours()

    gates: list[dict[str, Any]] = []
    gates.append(
        {
            "id": "fleet-control",
            "name": "Fleet Control",
            "pass": sessions > 0 and manager_running and attention <= int(targets.get("max_attention", 1)),
            "detail": f"sessions={sessions} manager_running={manager_running} attention={attention}",
        }
    )
    gates.append(
        {
            "id": "runtime-health",
            "name": "Runtime Health",
            "pass": red <= 1,
            "detail": f"red={red}",
        }
    )
    gates.append(
        {
            "id": "repo-hygiene",
            "name": "Repo Hygiene",
            "pass": high_drift_count <= int(targets.get("max_high_drift_projects", 1))
            and dirty_total <= int(targets.get("max_dirty_files", 120)),
            "detail": f"high_drift={high_drift_count} dirty_total={dirty_total}",
        }
    )
    gates.append(
        {
            "id": "loop-cadence",
            "name": "Loop Cadence",
            "pass": (recent_hours is not None) and (recent_hours <= 24),
            "detail": "last_loop_report={val}".format(
                val=("{:.1f}h".format(recent_hours) if recent_hours is not None else "never")
            ),
        }
    )
    gates.append(
        {
            "id": "readiness",
            "name": "Readiness Score",
            "pass": score >= int(targets.get("min_readiness_score", 92)),
            "detail": f"score={score} target={targets.get('min_readiness_score', 92)}",
        }
    )
    return gates


def build_actions(program: dict[str, Any], metrics: dict[str, Any], gates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {item.get("id"): item for item in gates}
    actions: list[dict[str, Any]] = []
    totals = metrics.get("fleet_totals", {})
    high_drift = list(metrics.get("high_drift", []))
    manager_session = metrics.get("manager_session")

    if not by_id["fleet-control"]["pass"]:
        actions.append(
            {
                "priority": "P0",
                "owner": "fleet-warden",
                "title": "Stabilize fleet control surface",
                "command": "agent-wrangler fleet manager --replace --update-defaults",
                "success": "Fleet manager running with attention <= target.",
            }
        )

    if int(totals.get("attention") or 0) > 0:
        actions.append(
            {
                "priority": "P0",
                "owner": "fleet-warden",
                "title": "Triage attention hotspots",
                "command": "agent-wrangler fleet status",
                "success": "All critical red panes acknowledged with fix or stop action.",
            }
        )

    if high_drift:
        top = high_drift[0]
        actions.append(
            {
                "priority": "P1",
                "owner": "repo-sheriff",
                "title": f"Reduce drift in {top.get('project_id')}",
                "command": f"agent-wrangler drift --fleet --alert-dirty {int(top.get('dirty') or 25)}",
                "success": "High-drift project count reduced to target.",
            }
        )

    if not by_id["loop-cadence"]["pass"]:
        actions.append(
            {
                "priority": "P1",
                "owner": "automation-rancher",
                "title": "Re-start execution loop cadence",
                "command": "agent-wrangler program loop --iterations 1 --write-report",
                "success": "Fresh program-loop report generated and queued.",
            }
        )

    actions.append(
        {
            "priority": "P2",
            "owner": "ux-rider",
            "title": "Shorten operator workflow to one launcher command",
            "command": "agent-wrangler up --mode import --max-panes 10 --nav --manager --manager-replace",
            "success": "Operator can reach full command center from one command.",
        }
    )
    actions.append(
        {
            "priority": "P2",
            "owner": "qa-gunslinger",
            "title": "Harden reliability regressions with loop checks",
            "command": "agent-wrangler program status",
            "success": "No surprise regressions in readiness gates.",
        }
    )

    priority_rank = {"P0": 0, "P1": 1, "P2": 2}
    actions.sort(key=lambda item: (priority_rank.get(str(item.get("priority")), 9), str(item.get("title"))))
    return actions


def gate_map(gates: list[dict[str, Any]]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for item in gates:
        key = str(item.get("id") or "")
        if not key:
            continue
        out[key] = bool(item.get("pass"))
    return out


def readiness_streak(history: list[dict[str, Any]], target: int) -> int:
    streak = 0
    for item in reversed(history):
        try:
            score = int(item.get("score") or 0)
        except Exception:
            score = 0
        if score >= target:
            streak += 1
        else:
            break
    return streak


def evaluate_phases(program: dict[str, Any], metrics: dict[str, Any], gates: list[dict[str, Any]], streak: int) -> list[dict[str, Any]]:
    gm = gate_map(gates)
    totals = metrics.get("fleet_totals", {})
    attention = int(totals.get("attention") or 0)
    red = int(totals.get("red") or 0)
    sessions = int(totals.get("sessions") or 0)
    waiting = int(totals.get("waiting") or 0)
    target_score = int(program.get("targets", {}).get("min_readiness_score", 92))
    score_gate = bool(gm.get("readiness"))
    hygiene_gate = bool(gm.get("repo-hygiene"))
    loop_gate = bool(gm.get("loop-cadence"))
    fleet_gate = bool(gm.get("fleet-control"))
    runtime_gate = bool(gm.get("runtime-health"))

    statuses: list[dict[str, Any]] = []
    statuses.append(
        {
            "id": "phase-1",
            "name": "Operational Hardening",
            "pass": fleet_gate and runtime_gate and sessions > 0,
            "detail": f"fleet={fleet_gate} runtime={runtime_gate} sessions={sessions}",
        }
    )
    statuses.append(
        {
            "id": "phase-2",
            "name": "Impeccable UX",
            "pass": fleet_gate and runtime_gate and attention <= 2 and red <= 1 and waiting <= 3,
            "detail": f"attention={attention} red={red} waiting={waiting}",
        }
    )
    statuses.append(
        {
            "id": "phase-3",
            "name": "Autonomous Loops",
            "pass": loop_gate and metrics.get("manager_running") is True,
            "detail": f"loop_cadence={loop_gate} manager_running={metrics.get('manager_running')}",
        }
    )
    statuses.append(
        {
            "id": "phase-4",
            "name": "Impeccable Product",
            "pass": score_gate and hygiene_gate and streak >= 7,
            "detail": f"score_gate={score_gate} hygiene={hygiene_gate} streak={streak}/7 target={target_score}",
        }
    )
    return statuses


def highest_contiguous_phase(statuses: list[dict[str, Any]]) -> int:
    idx = 0
    for offset, item in enumerate(statuses, start=1):
        if bool(item.get("pass")):
            idx = offset
            continue
        break
    return idx


def persist_state(program: dict[str, Any], score: int, stage: str, gates: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    state = program.setdefault("state", {})
    state["last_run_at"] = now_iso()
    state["last_readiness_score"] = score
    state["last_stage"] = stage

    history = state.setdefault("history", [])
    if not isinstance(history, list):
        history = []
        state["history"] = history
    history.append(
        {
            "at": now_iso(),
            "score": score,
            "stage": stage,
            "gates": gate_map(gates),
        }
    )
    if len(history) > 200:
        del history[:-200]

    target = int(program.get("targets", {}).get("min_readiness_score", 92))
    streak = readiness_streak(history, target=target)
    state["readiness_streak"] = streak

    phase_statuses = evaluate_phases(program, metrics, gates, streak=streak)
    state["phase_statuses"] = phase_statuses
    reached = max(1, highest_contiguous_phase(phase_statuses))
    current = int(state.get("current_phase") or 1)
    state["current_phase"] = max(current, reached)


def print_team(program: dict[str, Any]) -> int:
    team = list(program.get("team", []))
    print(f"Team members: {len(team)}")
    print(f"{'ID':<18} {'TITLE':<24} KPI")
    for member in team:
        print(f"{str(member.get('id') or '-'):<18} {str(member.get('title') or '-'):<24} {str(member.get('kpi') or '-')}")
    return 0


def print_loops(program: dict[str, Any]) -> int:
    loops = list(program.get("loops", []))
    print(f"Loops: {len(loops)}")
    print(f"{'ID':<18} {'OWNER':<18} {'CADENCE':<16} GOAL")
    for item in loops:
        print(
            f"{str(item.get('id') or '-'):<18} {str(item.get('owner') or '-'):<18} "
            f"{str(item.get('cadence') or '-'):<16} {str(item.get('goal') or '-')}"
        )
    return 0


def print_status(
    program: dict[str, Any],
    metrics: dict[str, Any],
    score: int,
    stage: str,
    gates: list[dict[str, Any]],
    *,
    streak: int,
    phase_statuses: list[dict[str, Any]],
) -> None:
    totals = metrics.get("fleet_totals", {})
    print(f"[{metrics.get('generated_at')}] {program.get('program_name')}")
    print(f"Objective: {program.get('objective')}")
    print(
        "Readiness: {score}/100 ({stage}) streak={streak}  sessions={sessions} panes={panes} attention={attention} waiting={waiting} active={active}".format(
            score=score,
            stage=stage,
            streak=streak,
            sessions=totals.get("sessions", 0),
            panes=totals.get("panes", 0),
            attention=totals.get("attention", 0),
            waiting=totals.get("waiting", 0),
            active=totals.get("active", 0),
        )
    )
    print(
        "Drift: projects={projects} dirty_total={dirty_total} high_drift={high}".format(
            projects=len(metrics.get("drift_rows", [])),
            dirty_total=metrics.get("dirty_total", 0),
            high=len(metrics.get("high_drift", [])),
        )
    )
    print(f"Manager: {metrics.get('manager_session')} running={metrics.get('manager_running')}")
    print("")
    print(f"{'GATE':<18} {'PASS':<5} DETAIL")
    for gate in gates:
        state = "yes" if gate.get("pass") else "no"
        print(f"{str(gate.get('name') or '-'):<18} {state:<5} {str(gate.get('detail') or '-')}")

    print("")
    print(f"{'PHASE':<24} {'PASS':<5} DETAIL")
    for item in phase_statuses:
        state = "yes" if item.get("pass") else "no"
        print(f"{str(item.get('name') or '-'):<24} {state:<5} {str(item.get('detail') or '-')}")

    high_drift = list(metrics.get("high_drift", []))
    if high_drift:
        print("")
        print("Top high-drift projects:")
        for item in high_drift[:5]:
            print(
                "- {session}:{project} dirty={dirty} branch={branch}".format(
                    session=item.get("session"),
                    project=item.get("project_id"),
                    dirty=item.get("dirty"),
                    branch=item.get("branch"),
                )
            )


def markdown_report(
    *,
    program: dict[str, Any],
    metrics: dict[str, Any],
    score: int,
    stage: str,
    streak: int,
    gates: list[dict[str, Any]],
    phase_statuses: list[dict[str, Any]],
    actions: list[dict[str, Any]],
) -> str:
    totals = metrics.get("fleet_totals", {})
    lines: list[str] = []
    lines.append(f"# {program.get('program_name')} Report")
    lines.append("")
    lines.append(f"- Generated: {metrics.get('generated_at')}")
    lines.append(f"- Readiness: **{score}/100** ({stage}), streak={streak}")
    lines.append(
        "- Fleet: sessions={sessions} panes={panes} attention={attention} waiting={waiting} active={active}".format(
            sessions=totals.get("sessions", 0),
            panes=totals.get("panes", 0),
            attention=totals.get("attention", 0),
            waiting=totals.get("waiting", 0),
            active=totals.get("active", 0),
        )
    )
    lines.append(
        "- Drift: projects={projects} dirty_total={dirty} high_drift={high}".format(
            projects=len(metrics.get("drift_rows", [])),
            dirty=metrics.get("dirty_total", 0),
            high=len(metrics.get("high_drift", [])),
        )
    )
    lines.append(f"- Fleet manager: `{metrics.get('manager_session')}` running={metrics.get('manager_running')}")
    lines.append("")
    lines.append("## Gates")
    for gate in gates:
        mark = "PASS" if gate.get("pass") else "FAIL"
        lines.append(f"- [{mark}] {gate.get('name')}: {gate.get('detail')}")
    lines.append("")
    lines.append("## Phase Status")
    for item in phase_statuses:
        mark = "PASS" if item.get("pass") else "FAIL"
        lines.append(f"- [{mark}] {item.get('name')}: {item.get('detail')}")
    lines.append("")
    lines.append("## Action Queue")
    for action in actions:
        lines.append(
            "- {priority} | {owner} | {title}\n  - Command: `{command}`\n  - Success: {success}".format(
                priority=action.get("priority"),
                owner=action.get("owner"),
                title=action.get("title"),
                command=action.get("command"),
                success=action.get("success"),
            )
        )
    return "\n".join(lines).strip() + "\n"


def write_report_files(prefix: str, payload: dict[str, Any], markdown: str) -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = ts_stamp()
    json_path = REPORTS_DIR / f"{prefix}-{stamp}.json"
    md_path = REPORTS_DIR / f"{prefix}-{stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    return json_path, md_path


def run_init(args: argparse.Namespace) -> int:
    if CONFIG_PATH.exists() and not args.force:
        print(f"Program config exists: {CONFIG_PATH}")
        print("Use --force to recreate defaults.")
        return 0
    program = default_program()
    save_program(program)
    print(f"Initialized program config: {CONFIG_PATH}")
    return 0


def run_team(_: argparse.Namespace) -> int:
    return print_team(load_program())


def run_loops(_: argparse.Namespace) -> int:
    return print_loops(load_program())


def run_phases(args: argparse.Namespace) -> int:
    program = load_program()
    store = tmux_teams.load_store()
    metrics = collect_signals(args, store)
    score, stage = compute_readiness(program, metrics)
    gates = build_gates(program, metrics, score)
    state = program.get("state", {})
    history = state.get("history", []) if isinstance(state.get("history"), list) else []
    target = int(program.get("targets", {}).get("min_readiness_score", 92))
    pre_streak = readiness_streak(history, target=target)
    statuses = evaluate_phases(program, metrics, gates, streak=pre_streak)
    contiguous = highest_contiguous_phase(statuses)

    print(f"Current phase: {int(state.get('current_phase') or 1)}")
    print(f"Contiguous phase readiness: {contiguous}")
    print(f"Readiness score: {score} ({stage}) streak={pre_streak}")
    print("")
    print(f"{'PHASE':<8} {'NAME':<24} {'PASS':<5} DETAIL")
    for idx, item in enumerate(statuses, start=1):
        mark = "yes" if item.get("pass") else "no"
        print(f"{idx:<8} {str(item.get('name') or '-'):<24} {mark:<5} {str(item.get('detail') or '-')}")

    if args.refresh_state:
        persist_state(program, score, stage, gates, metrics)
        save_program(program)
        print("")
        print("State refreshed in config.")
    return 0


def run_promote(args: argparse.Namespace) -> int:
    program = load_program()
    store = tmux_teams.load_store()
    metrics = collect_signals(args, store)
    score, stage = compute_readiness(program, metrics)
    gates = build_gates(program, metrics, score)
    state = program.setdefault("state", {})
    history = state.get("history", []) if isinstance(state.get("history"), list) else []
    target = int(program.get("targets", {}).get("min_readiness_score", 92))
    pre_streak = readiness_streak(history, target=target)
    statuses = evaluate_phases(program, metrics, gates, streak=pre_streak)
    contiguous = highest_contiguous_phase(statuses)
    current = int(state.get("current_phase") or 1)
    requested = int(args.phase) if args.phase else None

    if requested is not None:
        if requested < 1 or requested > 4:
            raise ValueError("phase must be between 1 and 4")
        if not args.force and requested > contiguous:
            raise ValueError(
                f"Cannot promote to phase {requested}: readiness only supports phase {contiguous}. Use --force to override."
            )
        state["current_phase"] = requested
        persist_state(program, score, stage, gates, metrics)
        save_program(program)
        print(f"Set current phase to {requested}")
        return 0

    if contiguous > current:
        state["current_phase"] = contiguous
        persist_state(program, score, stage, gates, metrics)
        save_program(program)
        print(f"Promoted phase {current} -> {contiguous}")
    else:
        persist_state(program, score, stage, gates, metrics)
        save_program(program)
        print(f"No promotion. Current phase={current}, readiness supports phase={contiguous}.")
    return 0


def run_complete(args: argparse.Namespace) -> int:
    program = load_program()
    store = tmux_teams.load_store()
    metrics = collect_signals(args, store)
    score, stage = compute_readiness(program, metrics)
    gates = build_gates(program, metrics, score)
    state = program.get("state", {})
    history = state.get("history", []) if isinstance(state.get("history"), list) else []
    target = int(program.get("targets", {}).get("min_readiness_score", 92))
    pre_streak = readiness_streak(history, target=target)
    statuses = evaluate_phases(program, metrics, gates, streak=pre_streak)
    contiguous = highest_contiguous_phase(statuses)
    complete = contiguous >= 4 and all(bool(item.get("pass")) for item in statuses)

    print(f"Impeccable completion: {'yes' if complete else 'no'}")
    print(f"Contiguous phase readiness: {contiguous}/4")
    print(f"Readiness score={score} stage={stage} streak={pre_streak}")
    if complete:
        print("All phases satisfied.")
    else:
        failing = [item for item in statuses if not item.get("pass")]
        print("Blocking phases:")
        for item in failing:
            print(f"- {item.get('name')}: {item.get('detail')}")
    return 0


def run_status(args: argparse.Namespace) -> int:
    program = load_program()
    store = tmux_teams.load_store()
    metrics = collect_signals(args, store)
    score, stage = compute_readiness(program, metrics)
    gates = build_gates(program, metrics, score)
    state = program.get("state", {})
    history = state.get("history", []) if isinstance(state.get("history"), list) else []
    target = int(program.get("targets", {}).get("min_readiness_score", 92))
    pre_streak = readiness_streak(history, target=target)
    phase_statuses = evaluate_phases(program, metrics, gates, streak=pre_streak)
    print_status(program, metrics, score, stage, gates, streak=pre_streak, phase_statuses=phase_statuses)
    persist_state(program, score, stage, gates, metrics)
    save_program(program)
    return 0


def run_plan(args: argparse.Namespace) -> int:
    program = load_program()
    store = tmux_teams.load_store()
    metrics = collect_signals(args, store)
    score, stage = compute_readiness(program, metrics)
    gates = build_gates(program, metrics, score)
    state = program.get("state", {})
    history = state.get("history", []) if isinstance(state.get("history"), list) else []
    target = int(program.get("targets", {}).get("min_readiness_score", 92))
    pre_streak = readiness_streak(history, target=target)
    phase_statuses = evaluate_phases(program, metrics, gates, streak=pre_streak)
    actions = build_actions(program, metrics, gates)

    print_status(program, metrics, score, stage, gates, streak=pre_streak, phase_statuses=phase_statuses)
    print("")
    print("Execution actions:")
    for action in actions:
        print(f"- {action.get('priority')} {action.get('owner')}: {action.get('title')}")
        print(f"  cmd: {action.get('command')}")

    persist_state(program, score, stage, gates, metrics)
    save_program(program)

    if args.write_report:
        payload = {
            "generated_at": metrics.get("generated_at"),
            "program": {
                "name": program.get("program_name"),
                "objective": program.get("objective"),
            },
            "readiness": {
                "score": score,
                "stage": stage,
            },
            "metrics": metrics,
            "gates": gates,
            "actions": actions,
        }
        md = markdown_report(
            program=program,
            metrics=metrics,
            score=score,
            stage=stage,
            streak=pre_streak,
            gates=gates,
            phase_statuses=phase_statuses,
            actions=actions,
        )
        json_path, md_path = write_report_files("program-plan", payload, md)
        print("")
        print(f"Wrote plan report: {md_path}")
        print(f"Wrote plan data: {json_path}")
    return 0


def apply_guardrails(
    *,
    max_ai_sessions: int,
    wait_limit_minutes: int,
    apply: bool,
) -> list[dict[str, Any]]:
    snapshot, _ = terminal_sentinel.classify_sessions(source_filter="ghostty", include_idle=False)
    rec = terminal_sentinel.recommend_actions(
        snapshot=snapshot,
        max_ai_sessions=max_ai_sessions,
        wait_limit_minutes=wait_limit_minutes,
    )
    return terminal_sentinel.apply_actions(rec, dry_run=(not apply))


def run_safe_apply(metrics: dict[str, Any], args: argparse.Namespace) -> tuple[list[str], list[dict[str, Any]]]:
    actions: list[str] = []
    if not metrics.get("manager_running"):
        ns = argparse.Namespace(
            manager_session=metrics.get("manager_session"),
            window=tmux_teams.DEFAULT_FLEET_MANAGER_WINDOW,
            sessions=None,
            pattern=None,
            include_manager=False,
            capture_lines=80,
            wait_attention_min=1,
            interval=3,
            replace=False,
            no_colorize=False,
            focus=False,
            attach=False,
            update_defaults=False,
        )
        tmux_teams.run_fleet_manager(ns)
        actions.append(f"started manager session {metrics.get('manager_session')}")

    for session in list(metrics.get("sessions", [])):
        if not tmux_teams.session_exists(str(session)):
            continue
        tmux_teams.refresh_pane_health(
            session=str(session),
            capture_lines=max(20, int(args.capture_lines)),
            wait_attention_min=max(0, int(args.wait_attention_min)),
            apply_colors=True,
        )
    actions.append("repainted fleet pane health styles")

    guardrail_results = apply_guardrails(
        max_ai_sessions=max(1, int(args.max_ai_sessions)),
        wait_limit_minutes=max(1, int(args.kill_waiting_ai_after)),
        apply=bool(args.apply_guardrails),
    )
    if guardrail_results:
        mode = "APPLY" if args.apply_guardrails else "PLAN"
        actions.append(f"guardrails {mode}: {len(guardrail_results)} action(s)")
    else:
        actions.append("guardrails: no actions")

    return actions, guardrail_results


def run_loop(args: argparse.Namespace) -> int:
    program = load_program()
    loops = 0
    interval = max(5, int(args.interval))
    try:
        while True:
            store = tmux_teams.load_store()
            metrics = collect_signals(args, store)
            score, stage = compute_readiness(program, metrics)
            gates = build_gates(program, metrics, score)
            state = program.get("state", {})
            history = state.get("history", []) if isinstance(state.get("history"), list) else []
            target = int(program.get("targets", {}).get("min_readiness_score", 92))
            pre_streak = readiness_streak(history, target=target)
            phase_statuses = evaluate_phases(program, metrics, gates, streak=pre_streak)
            actions = build_actions(program, metrics, gates)

            apply_results: list[str] = []
            guardrail_results: list[dict[str, Any]] = []
            if args.apply_safe:
                apply_results, guardrail_results = run_safe_apply(metrics, args)

            print_status(
                program,
                metrics,
                score,
                stage,
                gates,
                streak=pre_streak,
                phase_statuses=phase_statuses,
            )
            print("")
            print("Loop actions:")
            for action in actions[:8]:
                print(f"- {action.get('priority')} {action.get('owner')}: {action.get('title')}")
            if apply_results:
                print("")
                print("Safe apply actions:")
                for line in apply_results:
                    print(f"- {line}")
            if guardrail_results:
                print("")
                print("Guardrail results:")
                for item in guardrail_results[:10]:
                    mode = "APPLY" if item.get("applied") else "PLAN"
                    print(
                        "- {mode} pid={pid} tty={tty} reason={reason} => {result}".format(
                            mode=mode,
                            pid=item.get("pid"),
                            tty=item.get("tty"),
                            reason=item.get("reason"),
                            result=item.get("result"),
                        )
                    )

            persist_state(program, score, stage, gates, metrics)
            save_program(program)

            if args.write_report:
                payload = {
                    "generated_at": metrics.get("generated_at"),
                    "loop_iteration": loops + 1,
                    "readiness": {"score": score, "stage": stage},
                    "metrics": metrics,
                    "gates": gates,
                    "phase_statuses": phase_statuses,
                    "actions": actions,
                    "apply_results": apply_results,
                    "guardrail_results": guardrail_results,
                }
                md = markdown_report(
                    program=program,
                    metrics=metrics,
                    score=score,
                    stage=stage,
                    streak=pre_streak,
                    gates=gates,
                    phase_statuses=phase_statuses,
                    actions=actions,
                )
                json_path, md_path = write_report_files("program-loop", payload, md)
                print("")
                print(f"Wrote loop report: {md_path}")
                print(f"Wrote loop data: {json_path}")

            loops += 1
            if args.iterations > 0 and loops >= args.iterations:
                break
            time.sleep(interval)
            print("")
            print("-" * 80)
            print("")
    except KeyboardInterrupt:
        return 0
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-wrangler program", description="Program orchestration for impeccable product delivery")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Initialize impeccable program config")
    init.add_argument("--force", action="store_true", help="Overwrite existing config with defaults")
    init.set_defaults(handler=run_init)

    team = sub.add_parser("team", help="Show configured team")
    team.set_defaults(handler=run_team)

    loops = sub.add_parser("loops", help="Show configured loops")
    loops.set_defaults(handler=run_loops)

    phases = sub.add_parser("phases", help="Show phase-by-phase readiness status")
    add_signal_args(phases)
    phases.add_argument("--refresh-state", action="store_true", help="Persist the computed phase status to config")
    phases.set_defaults(handler=run_phases)

    promote = sub.add_parser("promote", help="Promote current phase based on readiness")
    add_signal_args(promote)
    promote.add_argument("--phase", type=int, help="Explicit phase number to set (1-4)")
    promote.add_argument("--force", action="store_true", help="Allow manual promotion beyond readiness checks")
    promote.set_defaults(handler=run_promote)

    complete = sub.add_parser("complete", help="Check if all phases are fully satisfied")
    add_signal_args(complete)
    complete.set_defaults(handler=run_complete)

    status = sub.add_parser("status", help="Show current readiness gates and signals")
    add_signal_args(status)
    status.set_defaults(handler=run_status)

    plan = sub.add_parser("plan", help="Generate prioritized execution plan from live signals")
    add_signal_args(plan)
    plan.add_argument("--write-report", action="store_true", help="Write plan report to reports/")
    plan.set_defaults(handler=run_plan)

    loop = sub.add_parser("loop", help="Run iterative program loop with reports")
    add_signal_args(loop)
    loop.add_argument("--interval", type=int, default=600, help="Loop interval in seconds")
    loop.add_argument("--iterations", type=int, default=1, help="0 means infinite")
    loop.add_argument("--apply-safe", action="store_true", help="Apply safe non-destructive remediations")
    loop.add_argument("--max-ai-sessions", type=int, default=4, help="Guardrail max concurrent AI sessions")
    loop.add_argument("--kill-waiting-ai-after", type=int, default=120, help="Guardrail wait threshold in minutes")
    loop.add_argument(
        "--apply-guardrails",
        action="store_true",
        help="Actually stop sessions selected by guardrails (default is dry-run plan)",
    )
    loop.add_argument("--write-report", action="store_true", help="Write each loop iteration report")
    loop.set_defaults(handler=run_loop)

    daemon = sub.add_parser("daemon", help="Run continuous loop with sane defaults for unattended operation")
    add_signal_args(daemon)
    daemon.add_argument("--interval", type=int, default=600)
    daemon.add_argument("--apply-guardrails", action="store_true")
    daemon.add_argument("--max-ai-sessions", type=int, default=4)
    daemon.add_argument("--kill-waiting-ai-after", type=int, default=120)
    daemon.add_argument("--write-report", action="store_true", default=True)
    daemon.set_defaults(
        handler=lambda a: run_loop(
            argparse.Namespace(
                sessions=a.sessions,
                pattern=a.pattern,
                include_manager=a.include_manager,
                capture_lines=a.capture_lines,
                wait_attention_min=a.wait_attention_min,
                alert_dirty=a.alert_dirty,
                interval=a.interval,
                iterations=0,
                apply_safe=True,
                max_ai_sessions=a.max_ai_sessions,
                kill_waiting_ai_after=a.kill_waiting_ai_after,
                apply_guardrails=a.apply_guardrails,
                write_report=True if getattr(a, "write_report", True) else False,
            )
        )
    )

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
