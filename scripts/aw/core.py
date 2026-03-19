#!/usr/bin/env python3
"""Core library for Agent Wrangler — tmux grid, health, pane ops, activity log."""

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

# terminal_sentinel lives in the parent scripts/ directory
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import terminal_sentinel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
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
_campfire_frame = [0]  # flickering campfire frame counter (list for cross-module mutability)
_prev_activity_state: dict[str, dict[str, Any]] = {}  # project_id -> last logged state
_last_active_time: dict[str, float] = {}  # project_id -> timestamp when last seen green
_STALL_THRESHOLD_MIN = 10  # minutes of waiting after being active = stalled


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


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tmux pane operations
# ---------------------------------------------------------------------------

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


# -- Claude Code status bar parser ------------------------------------------
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


# ---------------------------------------------------------------------------
# Grid and layout
# ---------------------------------------------------------------------------

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


# -- Hide / Show pane toggling ----------------------------------------------
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
        code2, counts_str, _ = run(
            ["git", "-C", path, "rev-list", "--left-right", "--count", f"{upstream.strip()}...HEAD"],
            timeout=5,
        )
        if code2 == 0:
            parts = counts_str.strip().split()
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


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------

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


_ACTIVE_TIMES_PATH = ROOT / ".state" / "active_times.json"


def _load_active_times() -> dict[str, float]:
    try:
        if _ACTIVE_TIMES_PATH.exists():
            return json.loads(_ACTIVE_TIMES_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_active_times(data: dict[str, float]) -> None:
    try:
        _ACTIVE_TIMES_PATH.parent.mkdir(parents=True, exist_ok=True)
        _ACTIVE_TIMES_PATH.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
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


# ---------------------------------------------------------------------------
# Barn helpers
# ---------------------------------------------------------------------------

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
