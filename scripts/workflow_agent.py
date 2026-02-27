#!/usr/bin/env python3
"""Workflow control utility for multi-project development on one machine."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "projects.json"
REPORTS_DIR = ROOT / "reports"
SCAN_SKIP_DIRS = {
    ".git",
    "node_modules",
    ".next",
    "dist",
    "build",
    "coverage",
    "__pycache__",
    "Library",
    "Pictures",
    "Movies",
    "Music",
    "Applications",
    "Downloads",
    "Desktop",
    "Documents",
    "Public",
    ".Trash",
}
BUSINESS_HINTS = ("gabooja", "agentcy", "creator-studio", "content-studio")


def run_cmd(cmd: list[str], timeout: int = 10) -> tuple[int, str, str]:
    """Run a command and return (code, stdout, stderr)."""
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


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def parse_etime_to_seconds(raw: str) -> int:
    """Parse ps elapsed time formats: DD-HH:MM:SS, HH:MM:SS, MM:SS, SS."""
    raw = raw.strip()
    if not raw:
        return 0

    days = 0
    time_part = raw
    if "-" in raw:
        day_part, time_part = raw.split("-", 1)
        try:
            days = int(day_part)
        except ValueError:
            days = 0

    fields = time_part.split(":")
    try:
        nums = [int(f) for f in fields]
    except ValueError:
        return 0

    if len(nums) == 3:
        hours, minutes, seconds = nums
    elif len(nums) == 2:
        hours = 0
        minutes, seconds = nums
    elif len(nums) == 1:
        hours = 0
        minutes = 0
        seconds = nums[0]
    else:
        return 0

    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def fmt_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h"
    return f"{seconds // 86400}d"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def gather_processes() -> tuple[list[dict[str, Any]], str | None]:
    code, out, err = run_cmd(["ps", "-Ao", "pid=,etime=,command="], timeout=12)
    if code != 0:
        detail = err.strip() if err else "unknown error"
        return [], f"ps unavailable: {detail}"

    rows: list[dict[str, Any]] = []
    for line in out.splitlines():
        # pid etime command
        match = re.match(r"\s*(\d+)\s+(\S+)\s+(.*)$", line)
        if not match:
            continue

        pid = int(match.group(1))
        elapsed_raw = match.group(2)
        command = match.group(3)

        rows.append(
            {
                "pid": pid,
                "elapsed_raw": elapsed_raw,
                "elapsed_seconds": parse_etime_to_seconds(elapsed_raw),
                "command": command,
            }
        )
    return rows, None


def gather_listening_ports() -> tuple[dict[int, list[int]], str | None]:
    code, out, err = run_cmd(["lsof", "-nP", "-iTCP", "-sTCP:LISTEN"], timeout=12)
    if code != 0:
        detail = err.strip() if err else "unknown error"
        return {}, f"lsof unavailable: {detail}"

    pid_to_ports: dict[int, set[int]] = {}
    for line in out.splitlines()[1:]:
        pid_match = re.search(r"\s(\d+)\s", line)
        port_match = re.search(r":(\d+)\s+\(LISTEN\)$", line)
        if not pid_match or not port_match:
            continue

        pid = int(pid_match.group(1))
        port = int(port_match.group(1))
        pid_to_ports.setdefault(pid, set()).add(port)

    return {pid: sorted(list(ports)) for pid, ports in pid_to_ports.items()}, None


def gather_git_status(path: Path, default_branch: str) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "is_repo": False,
            "branch": None,
            "dirty_count": 0,
            "last_commit": None,
            "last_commit_age_days": None,
            "default_branch": default_branch,
        }

    if not (path / ".git").exists():
        return {
            "exists": True,
            "is_repo": False,
            "branch": None,
            "dirty_count": 0,
            "last_commit": None,
            "last_commit_age_days": None,
            "default_branch": default_branch,
        }

    _, branch_out, _ = run_cmd(["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"])
    _, dirty_out, _ = run_cmd(["git", "-C", str(path), "status", "--porcelain"])
    _, commit_ts_out, _ = run_cmd(["git", "-C", str(path), "log", "-1", "--format=%ct"])
    _, commit_msg_out, _ = run_cmd(["git", "-C", str(path), "log", "-1", "--format=%s"])

    branch = branch_out.strip() or None
    dirty_count = len([ln for ln in dirty_out.splitlines() if ln.strip()])

    commit_ts = commit_ts_out.strip()
    last_commit = None
    age_days = None
    if commit_ts.isdigit():
        dt = datetime.fromtimestamp(int(commit_ts), tz=timezone.utc)
        age_days = int((datetime.now(timezone.utc) - dt).total_seconds() // 86400)
        last_commit = {
            "timestamp": dt.replace(microsecond=0).isoformat(),
            "subject": commit_msg_out.strip(),
        }

    return {
        "exists": True,
        "is_repo": True,
        "branch": branch,
        "dirty_count": dirty_count,
        "last_commit": last_commit,
        "last_commit_age_days": age_days,
        "default_branch": default_branch,
    }


def score_project(project: dict[str, Any]) -> tuple[int, list[str]]:
    git = project["git"]
    running = project["running_processes"]

    score = 0
    reasons: list[str] = []

    if not git["exists"]:
        return 100, ["Path missing"]

    if git["is_repo"]:
        dirty = git["dirty_count"]
        if dirty > 0:
            score += 3
            reasons.append(f"{dirty} uncommitted changes")

        if dirty >= 20:
            score += 2
            reasons.append("Large dirty tree")

        branch = git["branch"]
        default_branch = git["default_branch"]
        if branch and default_branch and branch != default_branch:
            score += 1
            reasons.append(f"On non-default branch ({branch})")

        age_days = git["last_commit_age_days"]
        if age_days is not None and age_days > 14 and dirty > 0:
            score += 1
            reasons.append("Worktree dirty + stale commits")

    if running:
        score += 1
        reasons.append(f"{len(running)} active process(es)")

        long_running = [p for p in running if p["elapsed_seconds"] >= 4 * 3600]
        if long_running:
            score += 2
            reasons.append("Long-running process context")

        if git["dirty_count"] > 0:
            score += 1
            reasons.append("Running process on dirty tree")

    if project.get("group") == "business":
        score += 1

    return score, reasons


def query_linear_workspace(workspace_name: str, token: str) -> dict[str, Any]:
    graph_query = """
    query WorkflowAgentAssignedIssues {
      viewer {
        name
        assignedIssues(first: 30) {
          nodes {
            identifier
            title
            priority
            updatedAt
            url
            state {
              name
              type
            }
            team {
              key
              name
            }
            project {
              name
            }
          }
        }
      }
    }
    """

    payload = json.dumps({"query": graph_query}).encode("utf-8")

    headers_to_try = [
        {"Content-Type": "application/json", "Authorization": token},
        {"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    ]

    body = None
    for headers in headers_to_try:
        req = request.Request(
            "https://api.linear.app/graphql",
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=8) as resp:
                body = resp.read().decode("utf-8")
                break
        except error.HTTPError:
            continue
        except Exception as exc:
            return {
                "workspace": workspace_name,
                "ok": False,
                "error": str(exc),
                "issues": [],
            }

    if body is None:
        return {
            "workspace": workspace_name,
            "ok": False,
            "error": "Authentication failed",
            "issues": [],
        }

    try:
        decoded = json.loads(body)
    except json.JSONDecodeError:
        return {
            "workspace": workspace_name,
            "ok": False,
            "error": "Invalid JSON response",
            "issues": [],
        }

    if "errors" in decoded:
        return {
            "workspace": workspace_name,
            "ok": False,
            "error": decoded["errors"][0].get("message", "Unknown error"),
            "issues": [],
        }

    viewer = decoded.get("data", {}).get("viewer", {})
    nodes = viewer.get("assignedIssues", {}).get("nodes", [])

    open_issues: list[dict[str, Any]] = []
    for node in nodes:
        state_type = (node.get("state") or {}).get("type", "").lower()
        if state_type in {"completed", "canceled"}:
            continue
        open_issues.append(
            {
                "identifier": node.get("identifier"),
                "title": node.get("title"),
                "priority": node.get("priority"),
                "updated_at": node.get("updatedAt"),
                "url": node.get("url"),
                "state": (node.get("state") or {}).get("name"),
                "team": (node.get("team") or {}).get("key") or (node.get("team") or {}).get("name"),
                "project": (node.get("project") or {}).get("name"),
            }
        )

    # Priority first (higher number first), then most recent update
    open_issues.sort(
        key=lambda issue: (
            issue.get("priority") if isinstance(issue.get("priority"), int) else -1,
            issue.get("updated_at") or "",
        ),
        reverse=True,
    )

    return {
        "workspace": workspace_name,
        "ok": True,
        "viewer": viewer.get("name"),
        "issues": open_issues,
    }


def fetch_conductor_status() -> dict[str, Any] | None:
    urls = [
        "http://localhost:3847/api/conductor/status",
        "http://localhost:3847/status",
    ]
    for url in urls:
        try:
            with request.urlopen(url, timeout=1.5) as resp:
                raw = resp.read().decode("utf-8")
                if not raw.strip():
                    continue
                data = json.loads(raw)
                data["_source"] = url
                return data
        except Exception:
            continue
    return None


def build_snapshot(config: dict[str, Any], include_linear: bool) -> dict[str, Any]:
    processes, process_error = gather_processes()
    listening, listening_error = gather_listening_ports()
    warnings: list[str] = []
    if process_error:
        warnings.append(process_error)
    if listening_error:
        warnings.append(listening_error)

    projects_out: list[dict[str, Any]] = []
    for project in config.get("projects", []):
        path = Path(project["path"]).expanduser()
        git = gather_git_status(path, project.get("default_branch", "main"))

        matched_processes = [p for p in processes if project["path"] in p["command"]]
        for proc in matched_processes:
            proc["ports"] = listening.get(proc["pid"], [])

        project_out = {
            "id": project["id"],
            "name": project["name"],
            "group": project.get("group", "uncategorized"),
            "path": project["path"],
            "startup_command": project.get("startup_command"),
            "notes": project.get("notes"),
            "git": git,
            "running_processes": matched_processes,
        }

        score, reasons = score_project(project_out)
        project_out["focus_score"] = score
        project_out["focus_reasons"] = reasons
        projects_out.append(project_out)

    projects_out.sort(key=lambda p: p["focus_score"], reverse=True)

    linear_data: list[dict[str, Any]] = []
    if include_linear:
        for ws in config.get("linear_workspaces", []):
            env_name = ws.get("api_key_env")
            if not env_name:
                continue
            token = os.environ.get(env_name)
            if not token:
                linear_data.append(
                    {
                        "workspace": ws.get("name", env_name),
                        "ok": False,
                        "error": f"Missing env var: {env_name}",
                        "issues": [],
                    }
                )
                continue
            linear_data.append(query_linear_workspace(ws.get("name", env_name), token))

    conductor = fetch_conductor_status()

    dirty_projects = sum(1 for p in projects_out if p["git"]["dirty_count"] > 0)
    running_projects = sum(1 for p in projects_out if p["running_processes"])

    return {
        "generated_at": now_utc_iso(),
        "host_user": os.environ.get("USER", "unknown"),
        "collection_warnings": warnings,
        "totals": {
            "projects": len(projects_out),
            "dirty_projects": dirty_projects,
            "running_projects": running_projects,
        },
        "projects": projects_out,
        "linear": linear_data,
        "conductor": conductor,
    }


def snapshot_to_markdown(snapshot: dict[str, Any], top_n: int = 6) -> str:
    lines: list[str] = []
    lines.append(f"# Workflow Snapshot ({snapshot['generated_at']})")
    lines.append("")

    totals = snapshot["totals"]
    lines.append(f"- Projects tracked: **{totals['projects']}**")
    lines.append(f"- Dirty repos: **{totals['dirty_projects']}**")
    lines.append(f"- Projects with active processes: **{totals['running_projects']}**")
    warnings = snapshot.get("collection_warnings", [])
    if warnings:
        lines.append(f"- Collection warnings: **{len(warnings)}**")

    conductor = snapshot.get("conductor")
    if conductor:
        lines.append(f"- Conductor status source: `{conductor.get('_source')}`")

    if warnings:
        lines.append("")
        lines.append("## Collection Warnings")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")

    lines.append("")
    lines.append("## Focus Queue")
    lines.append("")
    lines.append("| Score | Project | Group | Branch | Dirty | Processes | Top Reason |")
    lines.append("|---:|---|---|---|---:|---:|---|")

    for project in snapshot["projects"][:top_n]:
        git = project["git"]
        reason = project["focus_reasons"][0] if project["focus_reasons"] else "No urgent signal"
        lines.append(
            "| {score} | {name} | {group} | {branch} | {dirty} | {procs} | {reason} |".format(
                score=project["focus_score"],
                name=project["name"],
                group=project["group"],
                branch=git["branch"] or "-",
                dirty=git["dirty_count"],
                procs=len(project["running_processes"]),
                reason=reason,
            )
        )

    lines.append("")
    lines.append("## Running Contexts")
    lines.append("")

    running_any = False
    for project in snapshot["projects"]:
        procs = project["running_processes"]
        if not procs:
            continue
        running_any = True
        lines.append(f"### {project['name']}")
        lines.append("")
        for proc in procs[:5]:
            short_cmd = proc["command"]
            if len(short_cmd) > 120:
                short_cmd = short_cmd[:117] + "..."
            ports = ",".join(str(port) for port in proc.get("ports", [])) or "-"
            lines.append(
                f"- pid `{proc['pid']}` | uptime `{fmt_duration(proc['elapsed_seconds'])}` | ports `{ports}` | `{short_cmd}`"
            )
        if len(procs) > 5:
            lines.append(f"- ... and {len(procs) - 5} more process(es)")
        lines.append("")

    if not running_any:
        lines.append("No tracked project processes detected.")
        lines.append("")

    lines.append("## Linear Signals")
    lines.append("")
    linear_rows = snapshot.get("linear", [])
    if not linear_rows:
        lines.append("Linear fetch not requested in this snapshot.")
        lines.append("")
    else:
        for ws in linear_rows:
            lines.append(f"### Workspace: {ws.get('workspace')}")
            if not ws.get("ok"):
                lines.append(f"- Error: `{ws.get('error')}`")
                lines.append("")
                continue

            issues = ws.get("issues", [])
            lines.append(f"- Open assigned issues: **{len(issues)}**")
            for issue in issues[:8]:
                lines.append(
                    f"- `{issue.get('identifier')}` [{issue.get('state')}] ({issue.get('team')}) {issue.get('title')}"
                )
            if len(issues) > 8:
                lines.append(f"- ... and {len(issues) - 8} more")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_snapshot(snapshot: dict[str, Any]) -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = REPORTS_DIR / f"snapshot-{ts}.json"
    md_path = REPORTS_DIR / f"snapshot-{ts}.md"

    json_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    md_path.write_text(snapshot_to_markdown(snapshot), encoding="utf-8")
    return json_path, md_path


def print_focus(snapshot: dict[str, Any], limit: int) -> int:
    print("Focus queue:")
    for idx, project in enumerate(snapshot["projects"][:limit], start=1):
        reason = project["focus_reasons"][0] if project["focus_reasons"] else "No urgent signal"
        git = project["git"]
        print(
            f"{idx}. {project['name']} [{project['group']}] score={project['focus_score']} "
            f"branch={git.get('branch') or '-'} dirty={git.get('dirty_count')} reason={reason}"
        )
    return 0


def get_project_by_id(config: dict[str, Any], project_id: str) -> dict[str, Any] | None:
    for project in config.get("projects", []):
        if project.get("id") == project_id:
            return project
    return None


def launch_project(config: dict[str, Any], project_id: str, dry_run: bool) -> int:
    project = get_project_by_id(config, project_id)
    if not project:
        print(f"Unknown project id: {project_id}", file=sys.stderr)
        return 1

    path = project["path"]
    cmd = [
        "open",
        "-na",
        "Ghostty.app",
        "--args",
        f"--working-directory={path}",
    ]

    print(f"Project: {project['name']}")
    print("Launch command:")
    print("  " + " ".join(cmd))

    startup = project.get("startup_command")
    if startup:
        print(f"After launch, run: {startup}")

    if dry_run:
        return 0

    code, _, err = run_cmd(cmd)
    if code != 0:
        print(f"Launch failed: {err}", file=sys.stderr)
        return 1
    return 0


def run_doctor(snapshot: dict[str, Any]) -> int:
    issues: list[str] = []

    totals = snapshot["totals"]
    if totals["running_projects"] > 5:
        issues.append(
            f"Too many active contexts ({totals['running_projects']}). Target <= 5 total, <= 3 business."
        )

    if totals["dirty_projects"] > 4:
        issues.append(
            f"Too many dirty repos ({totals['dirty_projects']}). Finish/commit or stash low-priority work."
        )

    for project in snapshot["projects"]:
        git = project["git"]
        if not git["exists"]:
            issues.append(f"Missing path for project {project['id']}: {project['path']}")

        if git["dirty_count"] >= 50:
            issues.append(
                f"Very large dirty tree in {project['name']} ({git['dirty_count']} files)."
            )

    if issues:
        print("Workflow doctor found issues:")
        for idx, issue in enumerate(issues, start=1):
            print(f"{idx}. {issue}")
        return 1

    print("Workflow doctor: no critical issues detected.")
    return 0


def slugify(raw: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")
    return slug or "repo"


def discover_git_repos(scan_root: Path, max_depth: int, include_hidden: bool) -> tuple[list[Path], list[str]]:
    repos: list[Path] = []
    errors: list[str] = []

    def on_error(err: OSError) -> None:
        errors.append(str(err))

    for current, dirs, _ in os.walk(scan_root, topdown=True, onerror=on_error):
        current_path = Path(current)
        try:
            rel = current_path.relative_to(scan_root)
        except ValueError:
            continue

        depth = 0 if str(rel) == "." else len(rel.parts)
        if depth > max_depth:
            dirs[:] = []
            continue

        if (current_path / ".git").is_dir():
            repos.append(current_path)
            dirs[:] = []
            continue

        filtered: list[str] = []
        for dirname in dirs:
            if dirname in SCAN_SKIP_DIRS:
                continue
            if not include_hidden and dirname.startswith("."):
                continue
            filtered.append(dirname)
        dirs[:] = filtered

    repos.sort()
    return repos, errors


def suggest_startup_command(repo_path: Path) -> str | None:
    package_json = repo_path / "package.json"
    if not package_json.exists():
        return None

    try:
        data = json.loads(package_json.read_text(encoding="utf-8"))
    except Exception:
        return None

    scripts = data.get("scripts", {}) if isinstance(data, dict) else {}
    if "dev" in scripts:
        return "npm run dev"
    if "start" in scripts:
        return "npm start"
    if "build" in scripts:
        return "npm run build"
    return None


def suggest_group(repo_path: Path) -> str:
    lowered = str(repo_path).lower()
    if any(hint in lowered for hint in BUSINESS_HINTS):
        return "business"
    return "personal"


def suggest_project_entry(repo_path: Path) -> dict[str, Any]:
    return {
        "id": slugify(repo_path.name),
        "name": repo_path.name,
        "group": suggest_group(repo_path),
        "path": str(repo_path),
        "default_branch": "main",
        "startup_command": suggest_startup_command(repo_path),
    }


def write_discovery_report(payload: dict[str, Any]) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = REPORTS_DIR / f"discovery-{ts}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def run_discover(
    config: dict[str, Any],
    scan_root: Path,
    max_depth: int,
    include_hidden: bool,
    show_all: bool,
    write_report: bool,
) -> int:
    repos, scan_errors = discover_git_repos(scan_root, max_depth=max_depth, include_hidden=include_hidden)
    tracked_paths = {str(Path(p["path"]).resolve()): p for p in config.get("projects", [])}

    rows: list[dict[str, Any]] = []
    for repo in repos:
        resolved = str(repo.resolve())
        tracked = resolved in tracked_paths
        git = gather_git_status(repo, "main")
        rows.append(
            {
                "path": resolved,
                "tracked": tracked,
                "branch": git.get("branch"),
                "dirty_count": git.get("dirty_count", 0),
                "suggestion": None if tracked else suggest_project_entry(repo),
            }
        )

    untracked = [row for row in rows if not row["tracked"]]

    print(
        f"Discovery scan root={scan_root} depth={max_depth} repos={len(rows)} "
        f"tracked={len(rows) - len(untracked)} untracked={len(untracked)}"
    )

    if scan_errors:
        print(f"Scan warnings: {len(scan_errors)} (first 5 shown)")
        for warning in scan_errors[:5]:
            print(f"- {warning}")

    output_rows = rows if show_all else untracked
    if not output_rows:
        print("No untracked repositories found.")
    else:
        print("Repositories:")
        for idx, row in enumerate(output_rows, start=1):
            label = "tracked" if row["tracked"] else "untracked"
            print(
                f"{idx}. [{label}] {row['path']} "
                f"(branch={row.get('branch') or '-'} dirty={row.get('dirty_count', 0)})"
            )
            suggestion = row.get("suggestion")
            if suggestion:
                startup = suggestion.get("startup_command")
                startup_text = startup if startup else "-"
                print(
                    f"   suggestion: id={suggestion['id']} group={suggestion['group']} startup={startup_text}"
                )

    if write_report:
        payload = {
            "generated_at": now_utc_iso(),
            "scan_root": str(scan_root),
            "max_depth": max_depth,
            "include_hidden": include_hidden,
            "scan_warnings": scan_errors,
            "repos": rows,
        }
        report_path = write_discovery_report(payload)
        print(f"Wrote discovery report: {report_path}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Workflow controller for multi-project setup")
    sub = parser.add_subparsers(dest="command", required=True)

    snap_parser = sub.add_parser("snapshot", help="Build and save a full snapshot")
    snap_parser.add_argument("--include-linear", action="store_true", help="Fetch Linear assigned issues")
    snap_parser.add_argument("--no-write", action="store_true", help="Do not write snapshot files")

    focus_parser = sub.add_parser("focus", help="Print prioritized focus list")
    focus_parser.add_argument("--include-linear", action="store_true", help="Fetch Linear assigned issues")
    focus_parser.add_argument("--limit", type=int, default=6, help="Number of projects to show")

    doctor_parser = sub.add_parser("doctor", help="Run quick health checks")
    doctor_parser.add_argument("--include-linear", action="store_true", help="Fetch Linear assigned issues")

    launch_parser = sub.add_parser("launch", help="Open a Ghostty window for one project")
    launch_parser.add_argument("project_id", help="Project id from config/projects.json")
    launch_parser.add_argument("--dry-run", action="store_true", help="Print command without launching")

    discover_parser = sub.add_parser("discover", help="Scan filesystem for git repos")
    discover_parser.add_argument(
        "--scan-root",
        default=str(Path.home()),
        help="Root directory to scan (default: home directory)",
    )
    discover_parser.add_argument("--max-depth", type=int, default=3, help="Maximum scan depth")
    discover_parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden directories in scan",
    )
    discover_parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show tracked and untracked repos (default: only untracked)",
    )
    discover_parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write discovery JSON report to reports/",
    )

    args = parser.parse_args()
    config = load_config()

    if args.command == "launch":
        return launch_project(config, args.project_id, args.dry_run)

    if args.command == "discover":
        return run_discover(
            config=config,
            scan_root=Path(args.scan_root).expanduser(),
            max_depth=args.max_depth,
            include_hidden=args.include_hidden,
            show_all=args.show_all,
            write_report=args.write_report,
        )

    include_linear = bool(getattr(args, "include_linear", False))
    snapshot = build_snapshot(config, include_linear=include_linear)

    if args.command == "snapshot":
        if not args.no_write:
            json_path, md_path = write_snapshot(snapshot)
            print(f"Wrote snapshot JSON: {json_path}")
            print(f"Wrote snapshot Markdown: {md_path}")
        print_focus(snapshot, limit=6)
        return 0

    if args.command == "focus":
        return print_focus(snapshot, limit=args.limit)

    if args.command == "doctor":
        return run_doctor(snapshot)

    return 0


if __name__ == "__main__":
    sys.exit(main())
