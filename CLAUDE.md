# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Agent Wrangler is a tmux-based control layer for managing teams of AI coding agents (Claude Code, Codex, Aider, Gemini) across multiple repos from one terminal. One command to start, zero config required — it auto-discovers your running Ghostty terminals and builds a health-aware grid.

## Running Commands

All commands run from the repo root.

**Primary CLI** (bash router → Python engine):
```bash
./scripts/agent-wrangler <command> [args]
```

**Key commands:**
- `./scripts/agent-wrangler start` — Full startup: welcome banner + Ghostty import + manager + grid + nav bindings
- `./scripts/agent-wrangler status` — Pane health overview
- `./scripts/agent-wrangler ops` — Interactive operator console
- `./scripts/agent-wrangler rail` — Compact auto-refreshing status rail
- `./scripts/agent-wrangler init` — Interactive project setup (scan repos, create config)
- `./scripts/agent-wrangler add .` — Add current directory to config + running grid
- `./scripts/agent-wrangler summary <pane>` — Show recent pane output
- `./scripts/agent-wrangler briefing` — Show what happened while you were away
- `./scripts/agent-wrangler dispatch` — Send a prompt to multiple agents at once
- `./scripts/agent-wrangler barn-list` — Show projects: grazing vs in the barn
- `./scripts/agent-wrangler exit` — Kill session

**No build step, no test suite.** Pure Python 3.10+ and Bash. No package manager or virtual environment needed. All dependencies are stdlib.

## Architecture

### Files

```
scripts/agent-wrangler       Bash router — translates subcommands, calls agent_wrangler.py
scripts/agent_wrangler.py    Command handlers — all run_* functions, argparse, main()
scripts/aw/core.py           Core library — tmux wrappers, health detection, grid ops, activity log
scripts/aw/rail.py           Rail rendering — auto-refreshing status display, barn discovery, stall tracking
scripts/terminal_sentinel.py Process monitoring — classifies terminals, detects AI tools
scripts/welcome_banner.sh    Startup banner
```

### UI Model (tmux windows)

After `agent-wrangler start`, the tmux session has two window types:

- **Manager window** (`Option+m`): Claude Code session (left ~75%) + auto-refreshing status rail (right ~25%). Primary orchestration interface.
- **Grid window** (`Option+g`): Tiled project panes. Each pane is a project repo with optional AI agent. Health shown via border symbols: `●` green (active), `⚑` yellow (waiting), `✖` red (problem).

### Command Routing

`agent-wrangler` (bash) routes all subcommands to `agent_wrangler.py`:
- `start` → welcome banner + `teams up` with manager + grid + nav flags
- `exit` → `teams exit`
- Everything else → `teams <subcommand>`

### Key Design Patterns

- **Auto-discovery**: Ghostty terminals are discovered via process scanning. No config needed — terminals not in `projects.json` are imported using their directory basename and auto-registered.
- **Health detection**: Scrollback-based — captures pane text and looks for the agent's prompt character (`❯` for Claude Code, `>` for Codex/Gemini, etc.). Green = generating output, yellow = at prompt waiting for input. CPU-based detection doesn't work because AI tools think on remote servers.
- **Zoomed navigation**: `Option+z` zooms a pane to fullscreen. `Option+n`/`Option+p` cycle panes while staying zoomed — each pane feels like a full terminal window.
- **Notifications**: Optional (off by default). Desktop alerts via bundled macOS `.app` with debounce (2 consecutive checks) and cooldown (120s).
- **Activity log**: State transitions logged to `.state/activity.jsonl` during rail refresh. Powers briefing command and auto-recovery.

### Configuration Files (all in `config/`)

- **`projects.json`** — Optional project registry. Auto-created by `init` or auto-populated by discovery. Each entry: `id`, `name`, `path`, `startup_command`.
- **`team_grid.json`** — Tmux session config, workspace profiles (`max_panes`), notification toggle.

### State

- **`.state/`** — Runtime persistence (tmux resurrect snapshots, session stats)
- **`reports/`** — Generated snapshots

## Code Conventions

- Python modules use `argparse` for CLI, standalone imports, `Path(__file__).resolve().parents[1]` for repo root
- Type hints via `from __future__ import annotations`
- `@dataclass` for structured data (`TmuxPane`, `Proc`)
- Subprocess calls with timeout handling (10-30 sec defaults)
- Dry-run modes (`--dry-run`) on destructive commands
- Environment variables: `AW_MAX_PANES` (override max panes), `AW_NOTIFY` (enable notifications)
- Health colors: green (active), yellow (waiting), red (problem)
- AI tool markers: `claude`, `codex`, `aider`, `chatgpt`, `gemini`
- Navigation: `Option+Arrow` (panes), `Option+[/]` (windows), `Option+m/g` (manager/grid), `Option+z` (zoom), `Option+n/p` (cycle zoomed), `Option+j` (jump), `Option+q` (exit)
- Mouse: click select, double-click zoom, right-click context menu, scroll browse

## Manager Mode

When running as the manager session (the left pane of the manager window), you can orchestrate all project agents using the CLI:

### Quick Reference
- `./scripts/agent-wrangler status` — See all pane health
- `./scripts/agent-wrangler summary <project>` — Recent output from a pane
- `./scripts/agent-wrangler capture <project> --lines 50` — Raw pane text
- `./scripts/agent-wrangler agent <project> claude` — Start Claude in a pane
- `./scripts/agent-wrangler send <project> --command "..."` — Send text to a pane
- `./scripts/agent-wrangler stop <project>` — Send Ctrl-C
- `./scripts/agent-wrangler restart <project>` — Restart with startup command
- `./scripts/agent-wrangler briefing` — See all state changes since you left
- `./scripts/agent-wrangler dispatch` — Send a prompt to multiple agents at once

## Walk-Away Mode

Agent Wrangler logs state transitions to `.state/activity.jsonl` while the rail is running. When you come back:

- `./scripts/agent-wrangler briefing` — See all state changes, errors, costs since you left
- `./scripts/agent-wrangler briefing --since 120` — Look back 2 hours

Stall detection shows idle time next to waiting agents in the rail. After 10 minutes idle, time is shown in red.

### Auto-Recovery (opt-in)

Set `AW_AUTO_RECOVER=1` or add `"auto_recover": true` to `team_grid.json`. Agents stalled for >20 minutes get automatically restarted with `claude --dangerously-skip-permissions`. 5-minute cooldown per project.

### Managed Projects
@/Users/amirjalali/creator-studio/CLAUDE.md
@/Users/amirjalali/gabooja-labs/CLAUDE.md
@/Users/amirjalali/gabooja-knowledge-base/CLAUDE.md
