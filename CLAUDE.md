# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Agent Wrangler is a machine-level control layer for managing many repos, terminals, and task queues without losing context. It orchestrates tmux sessions, monitors Ghostty terminals, and enforces phase-gated delivery toward an "impeccable product" readiness score. The metaphor: engineers as "agent wranglers" steering autonomous AI agents, not writing code directly.

## Running Commands

All commands run from the repo root (`/Users/amirjalali/agent-wrangler`).

**Primary CLI** (bash router that dispatches to Python modules):
```bash
./scripts/agent-wrangler <command> [args]
```

**Key commands:**
- `./scripts/agent-wrangler start` - Full startup: welcome banner + Ghostty import + manager (Claude Code + status rail) + grid navigator + nav bindings
- `./scripts/agent-wrangler grid` - Standalone grid navigator (curses pane browser)
- `./scripts/agent-wrangler ops` - Interactive operator console (numbered menu)
- `./scripts/agent-wrangler rail` - Compact auto-refreshing status rail (for narrow splits)
- `./scripts/agent-wrangler status` - Pane health overview
- `./scripts/agent-wrangler program status` - Readiness score + phase gates

**Shell aliases** (thin wrappers in `scripts/`):
- `wrangler` -> `agent-wrangler`
- `hq` -> `agent-wrangler up`
- `teams` -> `tmux_teams.py`
- `cc` -> `command_center.py` (legacy)
- `workflow` -> `workflow_agent.py`
- `termwatch` -> `terminal_sentinel.py`

**No build step, no test suite.** Pure Python 3.10+ and Bash. No package manager or virtual environment needed. All dependencies are stdlib (`subprocess`, `json`, `argparse`, `curses`, `pathlib`).

## Architecture

### UI Model (tmux windows)

After `agent-wrangler start`, the tmux session has three window types:

- **Manager window** (`Option+m`): Claude Code session (left ~75%) + auto-refreshing status rail (right ~25%). This is the primary interface for communicating with the user and orchestrating agents.
- **Grid navigator window** (`Option+g`): Curses-based pane browser with health coloring, j/k navigation, Enter to jump into a pane, c/x to launch claude/codex. Shows session stats header (agent count, output volume, context usage from periodic `/usage` polls).
- **Teams window** (`Option+t`): The actual project panes in a tiled layout. Each pane is a project repo with optional AI agent running.

### Layer Model

```
Layer 4  program_orchestrator.py   Phase-gated delivery (team roles, loops, readiness gates)
Layer 3  command_center.py         Gastown planning board + Ant Farm runtime (legacy curses UI deprecated)
Layer 2  tmux_teams.py             Tmux grid control (panes, sessions, fleet, health coloring, manager, rail)
        grid_navigator.py         Curses pane browser (standalone, imports from tmux_teams)
        session_stats.py          Cheap + periodic context stats collection
Layer 1  terminal_sentinel.py      Process/TTY monitoring, AI tool classification
Layer 0  Ghostty / tmux            Terminal substrate
```

### Command Routing

`agent-wrangler` (bash) routes subcommands:
- `start` -> welcome banner + `teams up` with manager + grid + nav flags
- `grid` -> `grid_navigator.py` directly
- `up`, `status`, `paint`, `watch`, `fleet`, `nav`, `rail`, etc. -> `command_center.py teams ...` -> delegates to `tmux_teams.py`
- `program ...` -> `program_orchestrator.py` directly
- `ops`, `gastown`, `antfarm` -> `command_center.py` directly

### Key Modules

**`tmux_teams.py`** - The core engine. Manages tmux sessions, panes, health detection, fleet orchestration, persistence, profiles, hooks, and Ghostty-to-tmux import. The `run_manager` function creates a Claude Code + status rail split window. The `run_rail` function provides the compact auto-refreshing sidebar. Uses `@dataclass TmuxPane` as its primary data structure.

**`grid_navigator.py`** - Standalone curses pane browser. Shows pane list with health coloring (green/yellow/red), stats header with agent counts and context usage. Supports jump-to-pane, launch agent, send command, and switch-to-manager keybindings.

**`session_stats.py`** - Two-tier stats collection. Cheap stats (scrollback size, uptime) gathered every refresh cycle. Periodic `/usage` polls sent to one idle Claude session every ~5 minutes. Stores in `.state/session_stats.json`.

**`command_center.py`** - Two subsystems: *Gastown* (card-based planning with lanes: now/next/week/later) and *Ant Farm* (runtime session monitor). The old curses UI is deprecated in favor of the manager + grid setup.

**`program_orchestrator.py`** - Delivery roadmap with 4 phases, 6 team roles, 4 loop cadences, and 5 readiness gates. Computes a 0-100 readiness score. Phase 4 completion requires 92+ score for 7 consecutive days.

**`terminal_sentinel.py`** - Parses `ps` output to classify terminal sessions as active/waiting/idle. Detects AI tools (claude, codex, aider, gemini) from process commands. Powers overnight guardrails.

**`workflow_agent.py`** - Snapshot generation, focus ranking, health checks (doctor), repo discovery, and optional Linear GraphQL integration.

### Configuration Files (all in `config/`)

- **`projects.json`** - Project registry. Each entry: `id`, `name`, `group` (business/personal), `path`, `default_branch`, `startup_command`. Two groups with WIP limits (business: 3, personal: 2).
- **`team_grid.json`** - Tmux session config, fleet membership, workspace profiles (each profile sets `max_panes`), persistence settings.
- **`command_center.json`** - Gastown cards and Ant Farm source settings.
- **`impeccable_program.json`** - Phase definitions, team roles, loop cadences, readiness history.

### State and Reports

- **`.state/`** - Runtime persistence (tmux resurrect snapshots, session stats)
- **`reports/`** - Generated JSON/Markdown snapshots with ISO8601 timestamps

## Code Conventions

- Python modules use `argparse` for CLI, standalone imports (no package structure), and `Path(__file__).resolve().parents[1]` for repo root resolution
- Type hints via `from __future__ import annotations`
- `@dataclass` for structured data (`TmuxPane`, `Proc`)
- Subprocess calls wrapped with timeout handling (10-30 sec defaults)
- Dry-run modes (`--dry-run`) on destructive commands
- Environment variables: `AW_MAX_PANES` (override max panes), `LINEAR_API_KEY_GABOOJA`, `LINEAR_API_KEY_AMIRHJALALI`
- Health signals use consistent color coding: green (ok), yellow (attention), red (failure)
- AI tool detection markers: `claude`, `codex`, `aider`, `chatgpt`, `gemini`
- Navigation bindings: `Option+Arrow` (panes), `Option+[/]` (windows), `Option+m/g/t` (manager/grid/teams)
