# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Agent Wrangler is a tmux-based command center for managing teams of AI coding agents (Claude Code, Codex, Aider, Gemini) across multiple repos. It orchestrates tmux sessions, monitors terminals, and provides health-aware grid views with idle detection.

## Running Commands

All commands run from the repo root.

**Primary CLI** (bash router that dispatches to Python modules):
```bash
./scripts/agent-wrangler <command> [args]
```

**Key commands:**
- `./scripts/agent-wrangler start` - Full startup: welcome banner + Ghostty import + manager (Claude Code + status rail) + grid (project panes) + nav bindings
- `./scripts/agent-wrangler ops` - Interactive operator console (numbered menu)
- `./scripts/agent-wrangler rail` - Compact auto-refreshing status rail (for narrow splits)
- `./scripts/agent-wrangler status` - Pane health overview
- `./scripts/agent-wrangler program status` - Readiness score + phase gates
- `./scripts/agent-wrangler init` - Interactive project setup (scan repos, create config)
- `./scripts/agent-wrangler add .` - Add current directory to config + running grid
- `./scripts/agent-wrangler summary <pane>` - Show recent pane output (or press `Option+s`)
- `./scripts/agent-wrangler exit` - Kill session (or press `Option+q`)

**Shell aliases** (thin wrappers in `scripts/`):
- `wrangler` -> `agent-wrangler`
- `hq` -> `agent-wrangler up`
- `teams` -> `tmux_teams.py`
- `cc` -> `command_center.py` (legacy)
- `workflow` -> `workflow_agent.py`
- `termwatch` -> `terminal_sentinel.py`

**No build step, no test suite.** Pure Python 3.7+ and Bash. No package manager or virtual environment needed. All dependencies are stdlib (`subprocess`, `json`, `argparse`, `curses`, `pathlib`).

## Architecture

### UI Model (tmux windows)

After `agent-wrangler start`, the tmux session has three window types:

- **Manager window** (`Option+m`): Claude Code session (left ~75%) + auto-refreshing status rail (right ~25%). Primary interface for orchestrating agents. Claude Code's built-in status bar shows model, context usage, and session info at the bottom of each pane.
- **Grid window** (`Option+g`): Tiled project panes. Each pane is a project repo with optional AI agent running. Active pane highlighted with bright white border + `▶` marker + dimmed inactive panes. Health shown via symbols: `●` green, `⚑` yellow, `✖` red. Zoom with `Option+z`, jump by number with `Option+j`.

### Layer Model

```
Layer 3  program_orchestrator.py   Phase-gated delivery (team roles, loops, readiness gates)
Layer 2  command_center.py         Ant Farm runtime monitor + operator console
        tmux_teams.py             Tmux grid control (panes, sessions, health coloring, manager, rail)
Layer 1  terminal_sentinel.py      Process/TTY monitoring, AI tool classification
Layer 0  Ghostty / tmux            Terminal substrate
```

### Command Routing

`agent-wrangler` (bash) routes subcommands:
- `start` -> welcome banner + `teams up` with manager + grid + nav flags
- `up`, `status`, `paint`, `watch`, `nav`, `rail`, `exit`, etc. -> `command_center.py teams ...` -> delegates to `tmux_teams.py`
- `program ...` -> `program_orchestrator.py` directly
- `ops`, `antfarm` -> `command_center.py` directly

### Key Modules

**`tmux_teams.py`** - The core engine. Manages tmux sessions, panes, health detection, persistence, profiles, hooks, and Ghostty-to-tmux import. The `run_manager` function creates a Claude Code + status rail split window. The `run_rail` function provides the compact auto-refreshing sidebar. Uses `@dataclass TmuxPane` as its primary data structure.

**`command_center.py`** - Ant Farm (runtime session monitor for Ghostty terminals) + interactive operator console (`ops`). Thin dispatcher that delegates grid operations to `tmux_teams.py`.

**`program_orchestrator.py`** - Delivery roadmap with 4 phases, 6 team roles, 4 loop cadences, and 5 readiness gates. Computes a 0-100 readiness score. Phase 4 completion requires 92+ score for 7 consecutive days.

**`terminal_sentinel.py`** - Parses `ps` output to classify terminal sessions as active/waiting/idle. Detects AI tools (claude, codex, aider, gemini) from process commands. Powers overnight guardrails.

**`workflow_agent.py`** - Snapshot generation, focus ranking, health checks (doctor), and repo discovery.

### Configuration Files (all in `config/`)

- **`projects.json`** - Project registry. Each entry: `id`, `name`, `group` (business/personal), `path`, `default_branch`, `startup_command`. Two groups with WIP limits (business: 3, personal: 2).
- **`team_grid.json`** - Tmux session config, workspace profiles (each profile sets `max_panes`), persistence settings.
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
- Environment variables: `AW_MAX_PANES` (override max panes)
- Health signals use consistent color coding: green (ok), yellow (attention), red (failure)
- AI tool detection markers: `claude`, `codex`, `aider`, `chatgpt`, `gemini`
- Navigation bindings: `Option+Arrow` (panes), `Option+[/]` (windows), `Option+m/g` (manager/grid), `Option+z` (zoom), `Option+j` (jump by number), `Option+q` (exit)
- Mouse: click pane to select, scroll to browse output
