# Phase 2: Monolith Split + README + Auto-Recovery

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split the 4400-line monolith into focused modules, add a proper README, and add auto-recovery for stalled agents.

**Architecture:** Extract agent_wrangler.py into a package (`scripts/aw/`) with core, rail, and commands modules. The entry point stays at `scripts/agent_wrangler.py` and becomes a thin dispatcher. Auto-recovery hooks into the existing stall detection in the rail loop.

**Tech Stack:** Python 3.10+ stdlib only. No new dependencies.

---

### Task 1: Create the `aw/` package with core module

Extract shared utilities, constants, data types, tmux wrappers, config/store, project resolution, pane operations, health detection, and grid operations into `scripts/aw/core.py`.

**Files:**
- Create: `scripts/aw/__init__.py`
- Create: `scripts/aw/core.py`
- Modify: `scripts/agent_wrangler.py`

**Step 1: Create the package directory**

```bash
mkdir -p scripts/aw
```

**Step 2: Create `scripts/aw/__init__.py`**

Empty file:
```python
```

**Step 3: Create `scripts/aw/core.py`**

Move these sections from `agent_wrangler.py` into `scripts/aw/core.py`:

1. All imports (lines 1-19, minus `import terminal_sentinel` — adjust to `from . import` nothing yet, sentinel stays separate)
2. Module constants: ROOT through HOOK_EVENTS (lines 23-47)
3. Sound system: SOUND_COOLDOWN through play_sound (lines 49-84)
4. Health history state dicts (lines 87-96): `_health_history`, `_prev_rail_health`, `_prev_rail_costs`, `_sparkle_countdown`, `_transition_state`, `_campfire_frame`, `_prev_activity_state`, `_last_active_time`, `_STALL_THRESHOLD_MIN`
5. TmuxPane dataclass (lines 100-111)
6. Utility functions: `now_iso` through `session_exists` (lines 112-397)
7. Tmux pane listing and monitoring: `list_tmux_sessions` through `_resolve_monitor` (lines 400-515)
8. Pane capture and options: `capture_pane_text` through `batch_set_pane_options` (lines 521-545)
9. Claude status parser: `_CC_MODEL_RE` through `parse_claude_status` (lines 548-594)
10. Error/health detection: `detect_error_marker` through `pane_health_level` (lines 597-695)
11. Style and format: `style_for_level` through `set_window_orchestrator_format` (lines 701-748)
12. `_WINDOW_FORMAT_APPLIED` set and `refresh_pane_health` (lines 750-873)
13. Layout and pane management: `apply_layout` through `pane_ctrl_c` (lines 878-984)
14. Hide/show: `HIDDEN_PREFIX` through `list_hidden_panes` (lines 989-1054)
15. Session/printing: `attach_session` through `backfill_pane_project_ids` (lines 1060-1109)
16. Health summary and git: `session_health_summary` through `project_rows_for_session` (lines 1115-1242)
17. Grid creation and import: `create_grid_session` through `_auto_register_projects` (lines 1248-1464)
18. Notifications: `NOTIFY_STATE_PATH` through `check_and_notify` (lines 3197-3355)
19. Active times persistence: `_load_active_times`, `_save_active_times` (lines 3227-3242)
20. Activity log: `ACTIVITY_LOG_PATH` through `_log_transitions` (lines 3356-3487)
21. Barn helpers: `_set_barn_flag`, `_resolve_project_id` (lines 3670-3707)

The file should start with:
```python
#!/usr/bin/env python3
"""Core library for Agent Wrangler — tmux grid, health, pane ops, activity log."""

from __future__ import annotations
```

And import `terminal_sentinel` using a sys.path adjustment:
```python
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import terminal_sentinel
```

**Step 4: Update `agent_wrangler.py` to import from `aw.core`**

Replace the moved code with imports. The entry point becomes:
```python
#!/usr/bin/env python3
"""Tmux team-grid orchestration for multi-repo agent sessions."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure aw package is importable
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from aw.core import *  # noqa: F403 — re-export everything for backward compat
from aw.core import (
    ROOT, SELF_PATH, CONFIG_PATH, DEFAULT_SESSION, DEFAULT_LAYOUT,
    PROJECTS_CONFIG, LAYOUT_CHOICES, DEFAULT_LIMIT,
    _campfire_frame, _health_history, _prev_rail_health, _prev_rail_costs,
    _sparkle_countdown, _prev_activity_state, _last_active_time,
    _STALL_THRESHOLD_MIN,
)
```

Then keep all the `run_*` functions, `register_subparser`, and `main` in `agent_wrangler.py`.

**Step 5: Verify**

```bash
python3 scripts/agent_wrangler.py teams status 2>&1 | head -5
python3 scripts/agent_wrangler.py teams briefing --since 1 2>&1
```

Both should work as before.

**Step 6: Commit**

```bash
git add scripts/aw/ scripts/agent_wrangler.py
git commit -m "refactor: extract core library into scripts/aw/core.py"
```

---

### Task 2: Extract rail module

Move rail rendering, barn discovery, sparklines, stall tracking, and the rail loop into `scripts/aw/rail.py`.

**Files:**
- Create: `scripts/aw/rail.py`
- Modify: `scripts/agent_wrangler.py`

**Step 1: Create `scripts/aw/rail.py`**

Move these functions from `agent_wrangler.py`:
- `_sparkline`
- `_context_bar`
- `_campfire_header`
- `_discover_barn_repos` and barn cache variables
- `_graze_project`
- `run_rail`
- `_rail_loop`

The file imports from `aw.core`:
```python
from aw.core import (
    ROOT, SELF_PATH, DEFAULT_SESSION,
    ensure_tmux, load_store, session_exists,
    refresh_pane_health, list_hidden_panes,
    _health_history, _prev_rail_health, _prev_rail_costs,
    _sparkle_countdown, _campfire_frame,
    _last_active_time, _STALL_THRESHOLD_MIN,
    _log_transitions, _append_activity,
    _load_active_times, _save_active_times,
    play_sound,
)
```

NOTE: Since `_campfire_frame` is a module-level int that gets modified with `global`, you'll need to handle this by making it a mutable container (list with one element) or by keeping it as a module-level var in rail.py and updating it there. The simplest approach: just redeclare `_campfire_frame = 0` in rail.py since it's only used in the rail.

**Step 2: Update `agent_wrangler.py`**

Replace the moved functions with:
```python
from aw.rail import run_rail, _rail_loop  # noqa: F401
```

The `run_rail` reference in `register_subparser` should still work since it's imported.

**Step 3: Verify**

```bash
python3 scripts/agent_wrangler.py teams rail --interval 3 &
sleep 8 && kill %1
# Should see rail output without errors
```

**Step 4: Commit**

```bash
git add scripts/aw/rail.py scripts/agent_wrangler.py
git commit -m "refactor: extract rail rendering into scripts/aw/rail.py"
```

---

### Task 3: README

Write a proper README.md with install, quickstart, features, architecture, and command reference.

**Files:**
- Create: `README.md`

**Step 1: Write README.md**

Content should cover:
- **What it is**: One-paragraph description
- **Quick Start**: 3 steps (clone, run start, walk away)
- **Features**: bullet list of key capabilities (auto-discovery, health detection, manager mode, barn system, walk-away mode with briefing, auto-pilot agents)
- **Commands**: table of all commands with one-line descriptions
- **Architecture**: brief description of the file structure (agent_wrangler.py entry point, aw/ package, terminal_sentinel.py, welcome_banner.sh)
- **Navigation**: keyboard shortcuts table
- **Configuration**: brief description of projects.json and team_grid.json
- **Requirements**: Python 3.10+, tmux, macOS (Ghostty preferred)

Keep it concise. No badges, no screenshots. Just the useful stuff.

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with quickstart, commands, and architecture"
```

---

### Task 4: Auto-recovery for stalled agents

When the rail detects an agent has been stalled (yellow, idle > 20 minutes), auto-send Ctrl-C and restart it. This is opt-in via env var `AW_AUTO_RECOVER=1` or config flag.

**Files:**
- Modify: `scripts/aw/core.py` (add recovery helpers)
- Modify: `scripts/aw/rail.py` (trigger recovery in rail loop)
- Modify: `scripts/agent_wrangler.py` (add `--auto-recover` to start command)

**Step 1: Add recovery config check to core**

Add to `scripts/aw/core.py`:

```python
_AUTO_RECOVER_THRESHOLD_MIN = 20  # minutes idle before auto-recovery


def _auto_recover_enabled() -> bool:
    """Check if auto-recovery is enabled. Off by default."""
    if os.environ.get("AW_AUTO_RECOVER", "").strip() in ("1", "true", "yes"):
        return True
    try:
        store = load_store()
        return bool(store.get("auto_recover", False))
    except Exception:
        return False
```

**Step 2: Add recovery action to core**

Add to `scripts/aw/core.py`:

```python
_recovery_cooldown: dict[str, float] = {}  # project -> last recovery timestamp
_RECOVERY_COOLDOWN_SEC = 300  # 5 min between recovery attempts per project


def attempt_recovery(session: str, project_id: str, pane_id: str) -> bool:
    """Send Ctrl-C + restart agent for a stalled pane. Returns True if attempted."""
    now = time.time()
    last = _recovery_cooldown.get(project_id, 0)
    if now - last < _RECOVERY_COOLDOWN_SEC:
        return False  # Too soon since last recovery attempt

    _recovery_cooldown[project_id] = now

    # Send Ctrl-C to kill stuck process
    try:
        pane_ctrl_c(pane_id)
    except ValueError:
        return False

    # Wait briefly for the process to die
    time.sleep(0.5)

    # Restart Claude in auto-pilot mode
    try:
        pane_send(pane_id, "claude --dangerously-skip-permissions", enter=True)
    except ValueError:
        return False

    # Log the recovery
    _append_activity([{
        "project": project_id,
        "event": "auto_recovered",
        "health": "yellow",
        "status": "stalled",
    }])

    return True
```

**Step 3: Hook recovery into rail loop**

In `scripts/aw/rail.py`, in `_rail_loop`, in the stall detection section where we check `idle_min >= _STALL_THRESHOLD_MIN`, add auto-recovery:

After the stall label is built (when `idle_min >= _AUTO_RECOVER_THRESHOLD_MIN`):

```python
            # Auto-recovery: restart stalled agents
            if (health == "yellow" and project in _last_active_time
                    and agent and agent != "-"
                    and _auto_recover_enabled()):
                idle_min_val = (time.time() - _last_active_time[project]) / 60.0
                if idle_min_val >= _AUTO_RECOVER_THRESHOLD_MIN:
                    pane_id = str(row.get("pane_id") or "")
                    if pane_id and attempt_recovery(session, project, pane_id):
                        line += f" \033[33m⟳ recovering\033[0m"
                        _last_active_time[project] = time.time()  # reset stall timer
```

**Step 4: Show recovery events in briefing**

In `run_briefing` (in `agent_wrangler.py`), recovery events (`auto_recovered`) will already show up in the timeline since `_log_transitions` logs them via `_append_activity`. No code changes needed — they'll appear as timeline entries.

**Step 5: Verify**

```bash
python3 -c "
import sys; sys.path.insert(0, 'scripts')
from aw.core import _auto_recover_enabled, attempt_recovery
print('Functions importable:', True)
print('Auto-recover enabled:', _auto_recover_enabled())
"
```

Expected: Functions importable, auto-recover disabled (default off).

**Step 6: Commit**

```bash
git add scripts/aw/core.py scripts/aw/rail.py scripts/agent_wrangler.py
git commit -m "feat: auto-recovery for stalled agents (opt-in via AW_AUTO_RECOVER=1)"
```

---

## Verification

After all tasks:

1. `python3 scripts/agent_wrangler.py teams status` — should work
2. `python3 scripts/agent_wrangler.py teams briefing --since 1` — should work
3. `wc -l scripts/agent_wrangler.py` — should be ~2500 lines (down from 4400)
4. `wc -l scripts/aw/core.py scripts/aw/rail.py` — should be ~1200 + ~400
5. `cat README.md` — should exist and be useful

## Non-goals

- Full test suite (separate effort)
- Splitting commands.py further (diminishing returns)
- Auto-recovery for non-Claude agents (future — needs per-agent restart logic)
