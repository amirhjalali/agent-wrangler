#!/usr/bin/env python3
"""Tmux team-grid orchestration for multi-repo agent sessions."""

from __future__ import annotations

import argparse
import json
import os
import re
import select
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

ROOT = Path(__file__).resolve().parents[1]
SELF_PATH = str(ROOT)  # Agent-wrangler's own repo — the cowboy, not a horse
CONFIG_PATH = ROOT / "config" / "team_grid.json"
PERSISTENCE_DIR = ROOT / ".state" / "persistence"

DEFAULT_SESSION = "amir-grid"
DEFAULT_LAYOUT = "auto"
PROJECTS_CONFIG = ROOT / "config" / "projects.json"
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

# --- Sound system (Phase 8) ---
SOUND_COOLDOWN: dict[str, float] = {}  # key → last play time
SOUND_COOLDOWN_SEC = 10  # minimum seconds between sounds per key


def _sounds_enabled() -> bool:
    """Check if sounds are enabled. Off by default."""
    if os.environ.get("AW_SOUNDS", "").strip() in ("1", "true", "yes"):
        return True
    try:
        store = load_store()
        return bool(store.get("sounds", False))
    except Exception:
        return False


def play_sound(name: str, volume: float = 0.5, key: str = "") -> None:
    """Play a macOS system sound non-blocking. Silently does nothing if unavailable."""
    if not _sounds_enabled():
        return
    if key:
        now = time.time()
        if now - SOUND_COOLDOWN.get(key, 0) < SOUND_COOLDOWN_SEC:
            return
        SOUND_COOLDOWN[key] = now
    path = Path(f"/System/Library/Sounds/{name}.aiff")
    if path.exists():
        try:
            subprocess.Popen(
                ["afplay", "-v", str(volume), str(path)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            pass


# --- Health history for sparklines (Phase 2) ---
_health_history: dict[str, list[int]] = {}  # project_id → last N health values
_prev_rail_health: dict[str, str] = {}  # project_id → previous health level
_prev_rail_costs: dict[str, float] = {}  # project_id → previous cost
_sparkle_countdown: dict[str, int] = {}  # project_id → frames remaining for ✦
_transition_state: dict[str, list[str]] = {}  # project_id → color transition queue
_campfire_frame: int = 0  # flickering campfire frame counter
_prev_activity_state: dict[str, dict[str, Any]] = {}  # project_id -> last logged state


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


def load_projects_config() -> dict[str, Any]:
    """Load projects.json. Returns empty config if file doesn't exist."""
    if not PROJECTS_CONFIG.exists():
        return {"projects": []}
    try:
        return json.loads(PROJECTS_CONFIG.read_text(encoding="utf-8"))
    except Exception:
        return {"projects": []}


def project_map() -> dict[str, dict[str, Any]]:
    config = load_projects_config()
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
    for project in load_projects_config().get("projects", []):
        if project.get("barn"):
            continue  # In the barn — skip unless explicitly requested
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
        # Try matching against known projects first
        project_id = infer_project_id_from_path(cwd, proj_map)
        if project_id:
            return project_id, cwd
        # No match — use directory basename as project ID (auto-discover)
        norm_cwd = os.path.abspath(cwd)
        home = os.path.expanduser("~")
        if norm_cwd == home or norm_cwd == "/":
            return None, cwd
        basename = os.path.basename(norm_cwd)
        if basename:
            return basename, cwd

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


def _build_session_indexes(
    snapshot: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Build TTY-keyed and path-keyed indexes from sentinel sessions.

    Path index maps normalized absolute paths to AI sessions (Ghostty tabs).
    When a tmux pane's TTY doesn't match any sentinel session, the path
    index lets us fall back to matching by project directory.
    """
    by_tty: dict[str, dict[str, Any]] = {}
    by_path: dict[str, dict[str, Any]] = {}
    for item in snapshot.get("sessions", []):
        tty = str(item.get("tty") or "")
        if tty:
            by_tty[tty] = item
        cwd = item.get("cwd")
        if cwd and item.get("kind") == "ai":
            norm = os.path.normpath(cwd)
            # First match wins (sessions are sorted by status priority)
            if norm not in by_path:
                by_path[norm] = item
    return by_tty, by_path


def _resolve_monitor(
    tty_short: str,
    pane_path: str,
    by_tty: dict[str, dict[str, Any]],
    by_path: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Find the best sentinel session for a pane: TTY match first, then path.

    Path fallback matches Ghostty AI sessions by directory. When a match comes
    from path (not TTY), it's flagged as 'path_match' so health detection knows
    to trust the sentinel's status instead of scanning the tmux pane scrollback.
    """
    monitor = by_tty.get(tty_short)
    if monitor and monitor.get("agent"):
        return monitor
    # Fallback: match Ghostty AI session by project directory
    if pane_path:
        norm = os.path.normpath(pane_path)
        path_match = by_path.get(norm)
        if path_match:
            result = dict(path_match)
            result["path_match"] = True
            return result
    return monitor or {}


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
    r"\s*\|.*?(\d+)k?/(\d+)k\s*\((\d+)%\)"
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


def detect_prompt_waiting(raw_text: str, agent: str) -> bool:
    """Detect if an AI agent is at its input prompt by scanning pane text.

    CPU-based detection doesn't work for AI tools because the thinking
    happens on remote servers — local CPU is near-zero even when active.
    Instead, look for the tool's prompt character in the last few lines.
    """
    if not raw_text or not agent:
        return False
    # Get last non-empty lines (skip status bar area at very bottom)
    lines = [ln for ln in raw_text.splitlines() if ln.strip()]
    if not lines:
        return False
    # Check last ~10 meaningful lines for prompt patterns
    tail = lines[-10:]
    for line in reversed(tail):
        stripped = line.strip()
        # Skip Claude Code status bar lines
        if any(k in stripped for k in ("Opus ", "Sonnet ", "Haiku ", "○○", "●●", "bypass permissions", "extra $")):
            continue
        # Skip separator lines
        if stripped and all(c in "─━═—-" for c in stripped):
            continue
        # Claude Code prompt: ❯ or > at the start of a line (possibly with spaces)
        if agent == "claude" and stripped in ("❯", ">", "❯ "):
            return True
        # Codex / Gemini CLI prompt
        if agent in ("codex", "gemini") and stripped in (">", "❯"):
            return True
        # If we hit a non-status, non-separator, non-prompt line, it's output
        return False
    return False


def pane_health_level(
    monitor: dict[str, Any],
    error_marker: str | None,
    wait_attention_min: int,
    prompt_waiting: bool = False,
) -> tuple[str, bool, str]:
    status = str(monitor.get("status") or "idle")
    agent = str(monitor.get("agent") or "")
    wait = monitor.get("waiting_minutes")
    is_path_match = bool(monitor.get("path_match"))

    if error_marker:
        return "red", True, f"error: {error_marker}"

    if agent and agent != "-":
        if is_path_match:
            # Agent matched by directory (e.g. Ghostty running Claude for this
            # project). Trust the sentinel's status — the tmux pane scrollback
            # belongs to a shell, not the agent.
            if status == "active":
                return "green", False, ""
            if status == "waiting":
                return "yellow", False, ""
            return "yellow", False, status
        else:
            # Agent matched by TTY — it's running in this pane directly.
            # Use scrollback-based detection: prompt visible = waiting,
            # no prompt = generating (CPU-based detection doesn't work
            # because AI tools think on remote servers).
            if prompt_waiting:
                return "yellow", False, ""
            else:
                return "green", False, ""

    if status == "background":
        return "yellow", False, "background"
    # No agent running — pane is idle at a shell prompt
    if status in {"active", "idle"}:
        return "yellow", False, "no agent"
    return "yellow", False, status


def style_for_level(level: str) -> tuple[str, str]:
    """Return (inactive_border_style, active_border_style) for a health level.

    Health is encoded via border color (hue). Active pane gets warm gold
    border — lit up like it's near the campfire.
    """
    if level == "red":
        return "fg=colour88", "fg=colour214,bold"
    if level == "yellow":
        return "fg=colour130", "fg=colour214,bold"
    return "fg=colour22", "fg=colour214,bold"


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
    # Border format: active marker (▶) + health dot + project + ranch status
    fmt = (
        "#{?pane_active,#[fg=colour214 bold]▶ ,  }"
        "#{?#{==:#{@health},RED},#[fg=colour196]● ,"
        "#{?#{==:#{@health},YELLOW},#[fg=colour220]● ,"
        "#[fg=colour34]● }}"
        "#[fg=colour250]#{@project_id}"
        " #[fg=colour240]· "
        "#{?#{==:#{@health},RED},#[fg=colour196]down#{?@health_reason,: #{@health_reason},},"
        "#{?#{==:#{@health},YELLOW},#[fg=colour220]at fence#{?@health_reason, #{@health_reason},},"
        "#[fg=colour34]grazing}}"
        "#[default]"
    )
    tmux(["set-option", "-w", "-t", target, "pane-border-format", fmt], timeout=5)
    # Status bar: ranch-branded left + herd tally right
    tmux(["set-option", "-t", session, "status-style", "bg=colour235,fg=colour250"], timeout=5)
    tmux(["set-option", "-t", session, "status-left",
          " #[fg=colour130 bold]AW#[default] #[fg=colour172]⟨#[default] "
          "#[fg=colour250]#{session_name}#[default] #[fg=colour172]⟩#[default] "], timeout=5)
    tmux(["set-option", "-t", session, "status-left-length", "30"], timeout=5)
    tmux(["set-option", "-t", session, "status-right",
          "#{?window_zoomed_flag,#[fg=colour208 bold] ◎ ZOOMED #[default],} "
          "#[fg=colour250]%H:%M#[default] "], timeout=5)
    tmux(["set-option", "-t", session, "status-right-length", "120"], timeout=5)


_WINDOW_FORMAT_APPLIED: set[str] = set()


def refresh_pane_health(
    session: str,
    capture_lines: int,
    wait_attention_min: int,
    apply_colors: bool = True,
) -> list[dict[str, Any]]:
    panes = list_panes(session)
    snapshot, _ = terminal_sentinel.classify_sessions(source_filter="all", include_idle=True)
    by_tty, by_path = _build_session_indexes(snapshot)
    rows: list[dict[str, Any]] = []

    # Window format only needs to be set once per session, not every refresh
    if apply_colors and session not in _WINDOW_FORMAT_APPLIED:
        set_window_orchestrator_format(session)
        _WINDOW_FORMAT_APPLIED.add(session)

    for pane in panes:
        tty_short = pane.pane_tty.split("/")[-1]
        monitor = _resolve_monitor(tty_short, pane.pane_path, by_tty, by_path)
        agent = str(monitor.get("agent") or "-")
        status = str(monitor.get("status") or "idle")
        wait = monitor.get("waiting_minutes")

        # Single capture per pane — use raw text for both health detection and status bar
        raw_text = capture_pane_raw(pane.pane_id, lines=capture_lines)
        pane_text = raw_text.lower()
        error_marker = detect_error_marker(pane_text)
        missing_command = detect_missing_command(pane_text)
        port_in_use = detect_port_in_use(pane_text)
        prompt_waiting = detect_prompt_waiting(raw_text, agent)
        level, needs_attention, reason = pane_health_level(
            monitor=monitor,
            error_marker=error_marker,
            wait_attention_min=wait_attention_min,
            prompt_waiting=prompt_waiting,
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

    # Update status bar with ranch-branded herd tally or per-project dots
    if apply_colors and rows:
        counts = {"green": 0, "yellow": 0, "red": 0}
        for row in rows:
            lev = str(row.get("health") or "green").lower()
            counts[lev] = counts.get(lev, 0) + 1
        total = sum(counts.values())

        if total > 6:
            # Compact herd tally for large grids
            tally_parts = [f"#[fg=colour172]⟨ {total} head"]
            if counts["green"]:
                tally_parts.append(f"#[fg=colour34]{counts['green']}●")
            if counts["yellow"]:
                tally_parts.append(f"#[fg=colour220]{counts['yellow']}●")
            if counts["red"]:
                tally_parts.append(f"#[fg=colour196]{counts['red']}●")
            tally_parts.append("#[fg=colour172]⟩")
            tabs_str = " ".join(tally_parts)
        else:
            # Per-project dots for small grids
            tab_parts = []
            for row in rows:
                pid = str(row.get("project_id") or "?")
                lev = str(row.get("health") or "").lower()
                if len(pid) > 14:
                    pid = pid[:13] + "~"
                dot_color = {"green": "colour34", "yellow": "colour220", "red": "colour196"}.get(lev, "colour250")
                tab_parts.append(f"#[fg={dot_color}]●#[fg=colour250] {pid}")
            tabs_str = " #[fg=colour240]│#[default] ".join(tab_parts)

        status_right = (
            f" {tabs_str} "
            "#{?window_zoomed_flag,#[fg=colour208 bold] ◎ ZOOMED #[default],}"
            " #[fg=colour250]%H:%M#[default] "
        )
        tmux(["set-option", "-t", session, "status-right", status_right], timeout=3)
        tmux(["set-option", "-t", session, "status-right-length", "120"], timeout=3)

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
    snapshot, _ = terminal_sentinel.classify_sessions(source_filter="all", include_idle=True)
    by_tty, by_path = _build_session_indexes(snapshot)
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        name, window_id, pane_id, tty, path = parts[0], parts[1], parts[2], parts[3], parts[4]
        if not name.startswith(HIDDEN_PREFIX):
            continue
        project_id = name[len(HIDDEN_PREFIX):]
        tty_short = tty.split("/")[-1]
        monitor = _resolve_monitor(tty_short, path, by_tty, by_path)
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
    if not sys.stdout.isatty():
        # Not a terminal (e.g. subprocess call) — skip attach silently
        return 0
    if os.environ.get("TMUX"):
        # Already inside tmux — switch to the target session instead of attaching
        proc = subprocess.run(
            ["tmux", "switch-client", "-t", session],
            stdin=subprocess.DEVNULL, check=False,
        )
        return int(proc.returncode)
    proc = subprocess.run(["tmux", "attach-session", "-t", session], check=False)
    return int(proc.returncode)


def print_panes(session: str, panes: list[TmuxPane]) -> None:
    snapshot, _ = terminal_sentinel.classify_sessions(source_filter="all", include_idle=True)
    by_tty, by_path = _build_session_indexes(snapshot)

    print(f"Session: {session}  panes={len(panes)}")
    print(
        f"{'IDX':<4} {'PANE':<6} {'PROJECT':<22} {'TITLE':<24} {'AGENT':<8} {'STATUS':<10} "
        f"{'CMD':<10} {'TTY':<10} PATH"
    )
    for pane in panes:
        tty_short = pane.pane_tty.split("/")[-1]
        monitor = _resolve_monitor(tty_short, pane.pane_path, by_tty, by_path)
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
        pane_send(pane.pane_id, f"echo '\\n  [{project_id}] ready'", enter=True)

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

        # Agent-wrangler is the cowboy, not a horse — never add self to grid
        if cwd and Path(cwd).resolve() == Path(SELF_PATH).resolve():
            continue
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

        # Auto-discovered terminal (not in projects.json) — register with cwd
        if project_id not in proj_map and cwd:
            project_overrides[project_id] = {"path": cwd, "name": project_id}

        mapped_project_id = project_id
        if preserve_duplicates:
            if len(project_ids) >= max(1, max_panes):
                continue
            seen = duplicate_counts.get(project_id, 0) + 1
            duplicate_counts[project_id] = seen
            if seen > 1:
                mapped_project_id = f"{project_id}__dup{seen}"
                base = proj_map.get(project_id, project_overrides.get(project_id, {}))
                project_overrides[mapped_project_id] = dict(base)
            project_ids.append(mapped_project_id)
        else:
            if project_id not in project_ids:
                if len(project_ids) >= max(1, max_panes):
                    continue
                project_ids.append(project_id)

        agent = str(session.get("agent") or "").strip().lower()
        if agent in {"claude", "codex", "gemini"} and mapped_project_id not in agent_by_project:
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


def _auto_register_projects(
    project_ids: list[str],
    project_overrides: dict[str, dict[str, Any]],
    proj_map: dict[str, dict[str, Any]],
) -> None:
    """Auto-add discovered terminals to projects.json so they're remembered."""
    new_projects: list[dict[str, Any]] = []
    for pid in project_ids:
        # Skip duplicates and already-known projects
        base_id = pid.split("__dup")[0]
        if base_id in proj_map:
            continue
        override = project_overrides.get(pid, {})
        path = str(override.get("path") or "")
        if not path:
            continue
        # Agent-wrangler is the cowboy, not a horse
        if Path(path).resolve() == Path(SELF_PATH).resolve():
            continue
        new_projects.append({
            "id": base_id,
            "name": base_id,
            "path": path,
            "group": "personal",
            "default_branch": "main",
            "startup_command": "",
        })

    if not new_projects:
        return

    # Deduplicate by id
    seen_ids: set[str] = set()
    unique: list[dict[str, Any]] = []
    for p in new_projects:
        if p["id"] not in seen_ids:
            seen_ids.add(p["id"])
            unique.append(p)

    try:
        config: dict[str, Any] = {}
        if PROJECTS_CONFIG.exists():
            config = json.loads(PROJECTS_CONFIG.read_text(encoding="utf-8"))
        existing_ids = {p["id"] for p in config.get("projects", [])}
        added = [p for p in unique if p["id"] not in existing_ids]
        if added:
            config.setdefault("projects", []).extend(added)
            PROJECTS_CONFIG.parent.mkdir(parents=True, exist_ok=True)
            PROJECTS_CONFIG.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            names = ", ".join(p["id"] for p in added)
            print(f"Auto-registered: {names}")
    except Exception:
        pass  # Non-critical — don't fail the import


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
        project_ids = choose_projects(None, limit=args.max_panes, group=None)
        agent_by_project = {}
        project_overrides = {}
        mapped = []
        unmatched = []
        print("No Ghostty terminals detected — starting from projects.json.")

    # Auto-register discovered terminals into projects.json
    _auto_register_projects(project_ids, project_overrides, proj_map)

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
                project_ids = choose_projects(args.projects, limit=args.max_panes, group=args.group)
                agent_by_project = None
                project_overrides = None
                if args.projects or args.group:
                    print("No Ghostty matches found; using configured project selection fallback.")
                else:
                    print("No Ghostty terminals detected — starting from projects.json.")
            else:
                _auto_register_projects(project_ids, project_overrides, proj_map)
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
        play_sound("Bottle", 0.35)

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


def _sparkline(history: list[int]) -> str:
    """Render a colored health sparkline from history values (0=red, 1=yellow, 2=green)."""
    bars = "▁▂▃▄▅▆▇█"
    color_map = {0: "\033[31m", 1: "\033[33m", 2: "\033[32m"}
    parts = []
    for val in history[-10:]:
        val = max(0, min(2, val))
        bar_idx = [0, 4, 7][val]  # red→▁, yellow→▅, green→█
        parts.append(f"{color_map[val]}{bars[bar_idx]}")
    return "".join(parts) + "\033[0m" if parts else ""


def _context_bar(pct: int, width: int = 20) -> str:
    """Render a mini context usage bar: [████░░░░] with threshold coloring."""
    filled = int(pct * width / 100)
    empty = width - filled
    color = "\033[31m" if pct >= 80 else "\033[33m" if pct >= 50 else "\033[32m"
    return f"{color}{'█' * filled}\033[2m{'░' * empty}\033[0m"


def _campfire_header(frame: int, counts: dict[str, int]) -> list[str]:
    """Render ranch board header with cowboy hat."""
    total = sum(counts.values())
    g = counts.get("green", 0)
    y = counts.get("yellow", 0)
    r = counts.get("red", 0)

    hat_color = 172  # amber
    lines = [
        f"\033[38;5;{hat_color}m      .~~~~`\\~~\\\033[0m",
        f"\033[38;5;{hat_color}m     ;       ~~ \\\033[0m",
        f"\033[38;5;{hat_color}m     |           ;\033[0m",
        f"\033[38;5;{hat_color}m ,--------,______|---.\033[0m",
        f"\033[38;5;{hat_color}m/          \\-----`    \\\033[0m",
        f"\033[38;5;{hat_color}m`.__________`-_______-'\033[0m",
        "",
        f" \033[38;5;130m\033[1mRANCH BOARD\033[0m",
        (
            f" {total} head · "
            f"\033[32m{g}●\033[0m "
            f"\033[33m{y}●\033[0m "
            f"\033[31m{r}●\033[0m"
        ),
    ]
    return lines


# --- Barn discovery cache ---
_barn_cache: list[dict[str, str]] = []
_barn_cache_time: float = 0.0
_BARN_CACHE_TTL = 60.0  # rescan ~/  every 60 seconds


def _discover_barn_repos(active_project_ids: set[str]) -> list[dict[str, str]]:
    """Scan ~/ for git repos not currently active in the grid. Cached."""
    global _barn_cache, _barn_cache_time
    import time as _t

    now = _t.time()
    if _barn_cache and (now - _barn_cache_time) < _BARN_CACHE_TTL:
        # Return cached list filtered against current active set
        return [r for r in _barn_cache if r["id"] not in active_project_ids]

    home = Path.home()
    repos: list[dict[str, str]] = []
    try:
        for entry in sorted(home.iterdir()):
            if entry.name.startswith(".") or not entry.is_dir():
                continue
            if str(entry) == SELF_PATH:
                continue  # exclude agent-wrangler itself
            if (entry / ".git").is_dir():
                repos.append({
                    "id": entry.name.replace(" ", "-").lower(),
                    "name": entry.name,
                    "path": str(entry),
                })
    except OSError:
        pass

    _barn_cache = repos
    _barn_cache_time = now
    return [r for r in repos if r["id"] not in active_project_ids]


def _graze_project(path: str) -> None:
    """Open a project in a new Ghostty tab and launch Claude in it."""
    escaped_path = path.replace("\\", "\\\\").replace('"', '\\"')
    script = (
        'tell application "Ghostty"\n'
        "    set cfg to new surface configuration\n"
        f'    set initial working directory of cfg to "{escaped_path}"\n'
        "    set t to new tab in front window with configuration cfg\n"
        '    input text "claude --dangerously-skip-permissions" to t\n'
        '    send key "enter" to t\n'
        "end tell"
    )
    try:
        subprocess.Popen(
            ["osascript", "-e", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        pass


def run_rail(args: argparse.Namespace) -> int:
    """Auto-refreshing ranch board status rail for a narrow tmux split."""
    import time as _time
    import tty
    import termios
    global _campfire_frame

    ensure_tmux()
    store = load_store()
    session = args.session or store.get("default_session") or DEFAULT_SESSION
    interval = max(1, int(args.interval))

    # Set stdin to raw mode for non-blocking keypress detection (barn graze)
    old_tty_settings = None
    if sys.stdin.isatty():
        try:
            old_tty_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            sys.stdout.write("\033[?1000h\033[?1006h")  # enable SGR mouse tracking
            sys.stdout.flush()
        except termios.error:
            pass

    def _restore_tty() -> None:
        try:
            sys.stdout.write("\033[?1000l\033[?1006l")  # disable mouse tracking
            sys.stdout.flush()
        except OSError:
            pass
        if old_tty_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty_settings)
            except termios.error:
                pass

    try:
        return _rail_loop(args, session, interval, _time)
    finally:
        _restore_tty()


def _rail_loop(args: argparse.Namespace, session: str, interval: int, _time: Any) -> int:
    """Inner rail loop (separated so tty restore happens in finally)."""
    global _campfire_frame

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
        _log_transitions(rows)

        lines: list[str] = []
        lines.append("\033[2J\033[H")  # clear screen

        # --- Campfire header with herd count ---
        counts: dict[str, int] = {"green": 0, "yellow": 0, "red": 0}
        waiting = 0
        total_cost = 0.0
        total_tokens_k = 0
        claude_count = 0

        for row in rows:
            health = str(row.get("health") or "green")
            counts[health] = counts.get(health, 0) + 1
            if str(row.get("status") or "idle") == "waiting":
                waiting += 1

        _campfire_frame += 1
        lines.extend(_campfire_header(_campfire_frame, counts))
        lines.append("\033[2m" + "─" * 34 + "\033[0m")

        # --- Ranch Status summary box ---
        total = sum(counts.values())
        g, y, r = counts.get("green", 0), counts.get("yellow", 0), counts.get("red", 0)
        ranch_terms = []
        if g:
            ranch_terms.append(f"\033[32m{g} grazing\033[0m")
        if y:
            ranch_terms.append(f"\033[33m{y} at fence\033[0m")
        if r:
            ranch_terms.append(f"\033[31m{r} down\033[0m")
        lines.append(f" {' · '.join(ranch_terms)}")
        lines.append("\033[2m" + "─" * 34 + "\033[0m")

        # --- Per-pane entries with sparklines ---
        health_val = {"green": 2, "yellow": 1, "red": 0}
        for row in rows:
            health = str(row.get("health") or "green")
            project = str(row.get("project_id") or row.get("pane_title") or "?")
            agent = str(row.get("agent") or row.get("ai_tool") or "")
            status = str(row.get("status") or "idle")
            if len(project) > 16:
                project = project[:15] + "~"

            # Update health history
            h_val = health_val.get(health, 2)
            hist = _health_history.setdefault(project, [])
            hist.append(h_val)
            if len(hist) > 10:
                _health_history[project] = hist[-10:]

            # Check for state change sparkle
            prev = _prev_rail_health.get(project)
            if prev and prev != health:
                _sparkle_countdown[project] = 2
                # Play sound on health transitions
                if health == "red":
                    play_sound("Basso", 0.4, key=f"red-{project}")
                elif health == "green" and prev == "red":
                    play_sound("Bottle", 0.3, key=f"green-{project}")
            _prev_rail_health[project] = health

            sparkle = ""
            if _sparkle_countdown.get(project, 0) > 0:
                sparkle = " \033[38;5;220m✦\033[0m"
                _sparkle_countdown[project] -= 1

            dot_color = {"green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m"}.get(health, "\033[0m")
            spark = _sparkline(_health_history.get(project, []))

            line = f" {dot_color}●\033[0m{sparkle} {project:<16}"
            if agent and agent != "-":
                line += f" \033[2m{agent:<8}\033[0m"

            lines.append(line)

            # Sparkline on its own line (compact, below the pane entry)
            if len(_health_history.get(project, [])) > 1:
                lines.append(f"   {spark}")

            # Claude Code stats with context bar
            cc = row.get("cc_stats")
            if cc:
                claude_count += 1
                ctx = cc.get("context_pct")
                cost = cc.get("cost")

                if ctx is not None:
                    bar = _context_bar(ctx)
                    lines.append(f"   ctx {bar} {ctx}%")

                if cost is not None:
                    # Flash gold when cost increases
                    prev_cost = _prev_rail_costs.get(project, 0.0)
                    cost_color = "\033[38;5;214m" if cost > prev_cost else "\033[2m"
                    _prev_rail_costs[project] = cost
                    lines.append(f"   {cost_color}${cost:.2f}\033[0m")
                    total_cost += cost
                total_tokens_k += cc.get("tokens_k", 0)

        # --- Totals ---
        lines.append("\033[2m" + "─" * 34 + "\033[0m")
        if claude_count > 0:
            lines.append(
                f" \033[36m{claude_count} claude\033[0m  "
                f"\033[2m{total_tokens_k}k tok  ${total_cost:.2f}\033[0m"
            )

        # --- Hidden panes ---
        hidden = list_hidden_panes(session)
        if hidden:
            lines.append(f" \033[2m{len(hidden)} hidden\033[0m")
            for h in hidden:
                hp = str(h.get("project_id") or "?")
                if len(hp) > 16:
                    hp = hp[:15] + "~"
                ha = str(h.get("agent") or "")
                hs = str(h.get("status") or "")
                agent_label = f"  {ha}" if ha and ha != "-" else ""
                lines.append(f" \033[2m○ {hp:<16}{agent_label}  {hs}\033[0m")

        # --- Last roundup timestamp ---
        now_str = datetime.now().strftime("%H:%M:%S")
        lines.append(f"\n\033[2m  last roundup: {now_str}\033[0m")

        # --- Barn: discovered repos not in the grid ---
        active_ids = {
            str(row.get("project_id") or row.get("pane_title") or "")
            for row in rows
        }
        # Also include hidden pane IDs
        for h in (hidden if hidden else []):
            hid = str(h.get("project_id") or "")
            if hid:
                active_ids.add(hid)

        barn_repos = _discover_barn_repos(active_ids)
        barn_item_y: dict[int, int] = {}  # terminal Y coord → barn index (0-based)
        if barn_repos:
            lines.append("\033[2m" + "─" * 34 + "\033[0m")
            lines.append(f" \033[38;5;130mIN THE BARN ({len(barn_repos)})\033[0m")
            # Show up to 9 numbered entries
            for idx, repo in enumerate(barn_repos[:9], 1):
                name = repo["name"]
                if len(name) > 22:
                    name = name[:21] + "~"
                barn_item_y[len(lines) + 1] = idx - 1  # Y = lines index + 1
                lines.append(f" \033[2m[{idx}] {name}\033[0m")
            if len(barn_repos) > 9:
                lines.append(f" \033[2m    +{len(barn_repos) - 9} more\033[0m")
            lines.append(f" \033[2m click or press 1-{min(len(barn_repos), 9)} to graze\033[0m")

        print("\n".join(lines), flush=True)

        # --- Non-blocking keypress + mouse click detection for barn graze ---
        try:
            deadline = _time.time() + interval
            while _time.time() < deadline:
                remaining = deadline - _time.time()
                if remaining <= 0:
                    break
                ready, _, _ = select.select([sys.stdin], [], [], min(remaining, 0.2))
                if ready:
                    key = sys.stdin.read(1)
                    if key == '\033':
                        # Read rest of escape sequence (mouse click or other)
                        buf = '\033'
                        while len(buf) < 32:
                            r2, _, _ = select.select([sys.stdin], [], [], 0.05)
                            if not r2:
                                break
                            ch = sys.stdin.read(1)
                            buf += ch
                            if ch in ('M', 'm') and '[<' in buf:
                                break  # SGR mouse sequence complete
                        # Parse SGR mouse press: \033[<button;x;yM
                        if len(buf) > 5 and buf[1:3] == '[<' and buf[-1] == 'M':
                            parts = buf[3:-1].split(';')
                            if len(parts) == 3:
                                try:
                                    btn, _mx, my = int(parts[0]), int(parts[1]), int(parts[2])
                                    if btn == 0 and my in barn_item_y:
                                        chosen = barn_repos[barn_item_y[my]]
                                        _graze_project(chosen["path"])
                                        global _barn_cache_time
                                        _barn_cache_time = 0.0
                                        break
                                except (ValueError, IndexError):
                                    pass
                    elif key.isdigit() and 1 <= int(key) <= min(len(barn_repos) if barn_repos else 0, 9):
                        chosen = barn_repos[int(key) - 1]
                        _graze_project(chosen["path"])
                        _barn_cache_time = 0.0
                        break
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
            _log_transitions(rows)
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
            print(f"Warning: failed to create manager window '{window}': {err.strip()}")
            return 1

        # Split right pane for status rail (~25% width)
        rail_script = ROOT / "scripts" / "agent_wrangler.py"
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
    snapshot, _ = terminal_sentinel.classify_sessions(source_filter="all", include_idle=True)
    by_tty, by_path = _build_session_indexes(snapshot)
    pane_rows: list[dict[str, Any]] = []
    for pane in panes:
        tty_short = pane.pane_tty.split("/")[-1]
        monitor = _resolve_monitor(tty_short, pane.pane_path, by_tty, by_path)
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
        if agent in {"claude", "codex", "gemini"}:
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
    ensure_tmux()
    store = load_store()

    session = args.session or store.get("default_session") or DEFAULT_SESSION
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


def _build_context_menu_cmd(session: str) -> list[str]:
    """Build a tmux display-menu command for right-click pane management."""
    aw = str(ROOT / "scripts" / "agent-wrangler")
    aw_py = str(ROOT / "scripts" / "agent_wrangler.py")
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
    summary_script = str(ROOT / "scripts" / "agent_wrangler.py")
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
            tmux(["unbind-key", "-n", key], timeout=5)
        print("Removed no-prefix Alt navigation bindings (pane + window).")
        return 0

    for key, cmd in all_bindings:
        tmux(["bind-key", "-n", key, *cmd], timeout=5)

    # Double-click to zoom/unzoom a pane
    tmux(["bind-key", "-n", "DoubleClick1Pane", "resize-pane", "-Z"], timeout=5)

    # Right-click context menu for pane management
    menu_cmd = _build_context_menu_cmd(session)
    tmux(["bind-key", "-n", "MouseDown3Pane", *menu_cmd], timeout=5)

    # Source Ghostty-optimized tmux config if available
    tmux_conf = ROOT / "config" / "tmux.conf"
    if tmux_conf.exists():
        tmux(["source-file", str(tmux_conf)], timeout=5)

    print("Enabled navigation bindings.")
    print("Mouse: click select | double-click zoom | right-click menu | scroll browse")
    print("Zoomed: Option+n (next) | Option+p (prev) | Option+z (unzoom)")
    print("Windows: Option+m (manager) | Option+g (grid) | Option+[ / ]")
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
    play_sound("Morse", 0.3, key=f"agent-{pane.pane_id}")
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

    play_sound("Submarine", 0.3)
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
    config = load_projects_config()
    group = args.group.lower() if args.group else None
    print(f"{'ID':<22} {'GROUP':<10} PATH")
    for project in config.get("projects", []):
        pg = str(project.get("group", ""))
        if group and pg.lower() != group:
            continue
        print(f"{project['id']:<22} {pg:<10} {project.get('path', '')}")
    return 0


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


# ---------------------------------------------------------------------------
# Activity log — append-only JSONL for graze / barn discovery
# ---------------------------------------------------------------------------

ACTIVITY_LOG_PATH = ROOT / ".state" / "activity.jsonl"
ACTIVITY_MAX_BYTES = 5 * 1024 * 1024  # 5 MB, then rotate


def _append_activity(entries: list[dict[str, Any]]) -> None:
    """Append activity entries to the JSONL log.

    Each entry is written as a single JSON line with an auto-added ``ts``
    field (ISO-8601 UTC) when one is not already present.  The log file is
    rotated to ``*.jsonl.old`` once it exceeds *ACTIVITY_MAX_BYTES*.

    All errors are silently swallowed — this is a non-critical feature and
    must never interfere with normal operation.
    """
    try:
        ACTIVITY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Rotate if the file is too large.
        if ACTIVITY_LOG_PATH.exists() and ACTIVITY_LOG_PATH.stat().st_size > ACTIVITY_MAX_BYTES:
            rotated = ACTIVITY_LOG_PATH.with_suffix(".jsonl.old")
            ACTIVITY_LOG_PATH.replace(rotated)

        now = datetime.now(timezone.utc).isoformat()
        with ACTIVITY_LOG_PATH.open("a", encoding="utf-8") as fh:
            for entry in entries:
                if "ts" not in entry:
                    entry = {**entry, "ts": now}
                fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
    except Exception:
        pass


def _read_activity(*, since_minutes: float = 0, limit: int = 500) -> list[dict[str, Any]]:
    """Read activity entries from the JSONL log.

    Parameters
    ----------
    since_minutes:
        When > 0, only entries whose ``ts`` field is within the last
        *since_minutes* minutes are returned.
    limit:
        Maximum number of entries to return (most recent first after
        filtering).

    Returns an empty list on any error.
    """
    try:
        if not ACTIVITY_LOG_PATH.exists():
            return []

        lines = ACTIVITY_LOG_PATH.read_text(encoding="utf-8").splitlines()
        results: list[dict[str, Any]] = []

        if since_minutes > 0:
            cutoff = datetime.now(timezone.utc).timestamp() - since_minutes * 60
            for line in lines:
                if not line.strip():
                    continue
                entry = json.loads(line)
                ts_str = entry.get("ts", "")
                if ts_str:
                    try:
                        entry_ts = datetime.fromisoformat(ts_str).timestamp()
                    except (ValueError, TypeError):
                        continue
                    if entry_ts >= cutoff:
                        results.append(entry)
                # entries without ts are skipped when filtering by time
        else:
            for line in lines:
                if not line.strip():
                    continue
                results.append(json.loads(line))

        # Return the most recent entries, respecting the limit.
        if len(results) > limit:
            results = results[-limit:]
        return results
    except Exception:
        return []


def _log_transitions(rows: list[dict[str, Any]]) -> None:
    """Log health/status transitions to the activity log.

    Compares each row against ``_prev_activity_state`` and emits an entry
    when the health or status value has changed (or when the project is
    observed for the first time).  The ``ts`` field is intentionally
    omitted so that ``_append_activity`` adds it automatically.
    """
    entries: list[dict[str, Any]] = []
    for row in rows:
        project = str(row.get("project_id") or row.get("pane_title") or "")
        if not project or project == "-":
            continue

        health = str(row.get("health") or "green")
        status = str(row.get("status") or "idle")
        agent = str(row.get("agent") or "-")
        reason = str(row.get("reason") or "")
        cc = row.get("cc_stats") or {}

        prev = _prev_activity_state.get(project)

        if prev is None:
            event = "first_seen"
        elif prev.get("health") != health:
            event = f"health_{prev.get('health')}_to_{health}"
        elif prev.get("status") != status:
            event = f"status_{prev.get('status')}_to_{status}"
        else:
            # No change — skip.
            continue

        entry: dict[str, Any] = {
            "project": project,
            "event": event,
            "health": health,
            "status": status,
            "agent": agent,
        }
        if reason:
            entry["reason"] = reason
        cost = cc.get("cost")
        if cost is not None:
            entry["cost"] = cost
        context_pct = cc.get("context_pct")
        if context_pct is not None:
            entry["context_pct"] = context_pct

        entries.append(entry)
        _prev_activity_state[project] = {"health": health, "status": status}

    if entries:
        _append_activity(entries)


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
    # Agent-wrangler is the cowboy, not a horse
    if path == Path(SELF_PATH).resolve():
        print("Agent Wrangler is the cowboy, not a horse. It manages the grid from the manager window.")
        return 1

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


def run_remove(args: argparse.Namespace) -> int:
    """Remove a project from config and optionally kill its grid pane."""
    proj_id = args.project
    store = load_store()
    session = args.session or store.get("default_session") or DEFAULT_SESSION

    # Remove from projects.json
    removed_from_config = False
    if PROJECTS_CONFIG.exists():
        config = json.loads(PROJECTS_CONFIG.read_text(encoding="utf-8"))
        projects = config.get("projects", [])
        before = len(projects)
        config["projects"] = [p for p in projects if p.get("id") != proj_id]
        if len(config["projects"]) < before:
            PROJECTS_CONFIG.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            removed_from_config = True
            print(f"Removed '{proj_id}' from {PROJECTS_CONFIG}")

    if not removed_from_config:
        print(f"Project '{proj_id}' not found in config.")

    # Kill the pane in the running grid if it exists
    if session_exists(session):
        try:
            pane = pane_target(session, proj_id)
            tmux(["kill-pane", "-t", pane.pane_id], timeout=5)
            apply_layout(session, "tiled")
            set_window_orchestrator_format(session)
            print(f"Killed pane for '{proj_id}' in session '{session}'")
        except (ValueError, RuntimeError):
            if removed_from_config:
                print(f"No running pane for '{proj_id}' (will be excluded on next start)")

    return 0


def _set_barn_flag(proj_id: str, barn: bool) -> bool:
    """Set or clear the barn flag for a project in projects.json. Returns True if found."""
    if not PROJECTS_CONFIG.exists():
        return False
    config = json.loads(PROJECTS_CONFIG.read_text(encoding="utf-8"))
    found = False
    for p in config.get("projects", []):
        if p.get("id") == proj_id:
            if barn:
                p["barn"] = True
            else:
                p.pop("barn", None)
            found = True
            break
    if found:
        PROJECTS_CONFIG.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return found


def _resolve_project_id(token: str, session: str) -> tuple[str, str | None]:
    """Resolve a token (project id, pane id, index) to (project_id, pane_id).

    If the token is a pane ID like %5, look up the pane and return its project_id.
    """
    pane_id = None
    if session_exists(session):
        try:
            pane = pane_target(session, token)
            pane_id = pane.pane_id
            if pane.project_id and pane.project_id != "-":
                return pane.project_id, pane_id
        except (ValueError, RuntimeError):
            pass
    return token, pane_id


def run_barn(args: argparse.Namespace) -> int:
    """Send a project to the barn — remove from grid, keep in config."""
    store = load_store()
    session = args.session or store.get("default_session") or DEFAULT_SESSION
    proj_id, pane_id = _resolve_project_id(args.project, session)

    if not _set_barn_flag(proj_id, barn=True):
        print(f"Project '{proj_id}' not found in config.")
        return 1

    print(f"Sent '{proj_id}' to the barn.")

    # Kill the pane in the running grid
    if pane_id:
        try:
            tmux(["kill-pane", "-t", pane_id], timeout=5)
            apply_layout(session, "tiled")
            set_window_orchestrator_format(session)
            print(f"Removed pane from grid.")
        except (ValueError, RuntimeError):
            pass

    return 0


def run_unbarn(args: argparse.Namespace) -> int:
    """Let a project out of the barn — add back to grid."""
    proj_id = args.project
    store = load_store()
    session = args.session or store.get("default_session") or DEFAULT_SESSION

    if not _set_barn_flag(proj_id, barn=False):
        print(f"Project '{proj_id}' not found in config.")
        return 1

    print(f"Let '{proj_id}' out of the barn.")

    # Hot-add to running grid
    if session_exists(session):
        proj_map = project_map()
        proj = proj_map.get(proj_id)
        path = str(proj.get("path", "")) if proj else ""
        if path:
            split_for_path(session, path)
            apply_layout(session, "tiled")
            panes = list_panes(session)
            if panes:
                last_pane = panes[-1]
                pane_set_project_id(last_pane.pane_id, proj_id)
                tmux(["select-pane", "-t", last_pane.pane_id, "-T", proj_id], timeout=5)
            set_window_orchestrator_format(session)
            print(f"Added pane to grid.")
    else:
        print(f"Session not running. Will appear on next start.")

    return 0


def run_barn_list(args: argparse.Namespace) -> int:
    """List projects in the barn."""
    config = load_projects_config()
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
        store = load_store()
        session = store.get("default_session") or DEFAULT_SESSION
        if session_exists(session):
            rows = refresh_pane_health(session, capture_lines=40, wait_attention_min=5, apply_colors=False)
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
