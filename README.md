# Agent Wrangler

Tmux-based control layer for managing teams of AI coding agents across multiple repos from one terminal.

## Quick Start

```bash
git clone https://github.com/AmirhJalworker/agent-wrangler.git
cd agent-wrangler
./scripts/agent-wrangler start
```

Agent Wrangler auto-discovers your running Ghostty terminals, imports them into a tmux grid with health-aware borders, and opens a manager window. No config file needed.

## What It Does

- **Auto-discovery** -- scans running Ghostty terminals by process tree, matches each to a project by working directory, imports unknown terminals by directory name. Zero config required.
- **Health monitoring** -- scrollback-based detection of agent state. Green = generating output, yellow = waiting at prompt, red = error or stalled. Health shown in pane borders, status rail, and status bar.
- **Manager mode** -- dedicated window with Claude Code (left 75%) and a live-refreshing status rail (right 25%). Orchestrate all agents from one place.
- **Barn system** -- temporarily remove projects from the grid without losing config. Barn a project when you don't need it, unbarn to bring it back.
- **Walk-away briefing** -- run `briefing` after stepping away to see what changed: which agents finished, which errored, git drift across all panes.
- **Auto-pilot agents** -- start Claude Code, Codex, Aider, or Gemini in any pane. Send commands, stop agents, capture output, restart with one command.

## Commands

All commands are invoked as `./scripts/agent-wrangler <command>`.

| Command | Description |
|---------|-------------|
| `start` | Launch grid + manager + nav bindings (full startup) |
| `status` | One-shot pane health overview |
| `add [path]` | Add a project to config and running grid |
| `remove <project>` | Remove a project from config and kill its pane |
| `summary <pane>` | Show recent output summary from a pane |
| `briefing` | Show what happened while you were away |
| `ops` | Interactive operator console |
| `agent <pane> <tool>` | Start an AI agent (claude, codex, aider, gemini) in a pane |
| `stop <pane>` | Send Ctrl-C to a pane |
| `restart <pane>` | Restart pane with its startup command |
| `send <pane> --command "..."` | Send a command to a pane |
| `capture <pane> --lines N` | Capture raw pane output |
| `doctor` | Diagnose broken or waiting agent panes |
| `drift` | Show git drift for pane projects |
| `barn <project>` | Send a project to the barn (remove from grid, keep in config) |
| `unbarn <project>` | Bring a project back from the barn into the grid |
| `barn-list` | List projects: grazing (active) vs in the barn |
| `hide <pane>` | Hide a pane (agent keeps running in background) |
| `show <pane>` | Restore a hidden pane to the grid |
| `hidden` | List hidden panes |
| `exit` | Kill the Agent Wrangler session |
| `init` | Interactive project setup (scan repos, create config) |

## Navigation

All shortcuts use the Option key (requires `macos-option-as-alt = true` in Ghostty config).

| Shortcut | Action |
|----------|--------|
| `Option+Arrow` | Move between panes |
| `Option+m` | Switch to manager window |
| `Option+g` | Switch to grid window |
| `Option+z` | Zoom current pane to fullscreen (toggle) |
| `Option+n` / `Option+p` | Next / previous pane (works while zoomed) |
| `Option+j` | Jump to pane by number |
| `Option+[` / `Option+]` | Previous / next window |
| `Option+q` | Exit Agent Wrangler |
| Click | Select a pane |
| Double-click | Zoom pane to fullscreen |
| Right-click | Context menu |
| Scroll | Browse pane output |

## Architecture

```
scripts/agent-wrangler         Bash router -- translates subcommands, calls agent_wrangler.py
scripts/agent_wrangler.py      Engine -- grid management, health detection, manager, rail, ops, nav
scripts/terminal_sentinel.py   Process monitoring -- classifies terminals, detects AI tools
scripts/welcome_banner.sh      Startup banner animation
config/                        projects.json (project registry) + team_grid.json (session config)
.state/                        Runtime persistence (tmux snapshots, session stats)
```

The bash router handles startup sequencing and subcommand translation. All core logic lives in `agent_wrangler.py`, which uses `argparse` for CLI dispatch and `subprocess` for tmux control. `terminal_sentinel.py` provides Ghostty process discovery and AI tool classification.

## Configuration

**`config/projects.json`** -- Project registry. Each entry has `id`, `name`, `path`, and `startup_command`. Auto-created by discovery or manually via `init`. Optional: Agent Wrangler works without it by auto-importing Ghostty terminals.

**`config/team_grid.json`** -- Session config with workspace profiles (`max_panes`), notification toggle, and persistence settings.

Run `./scripts/agent-wrangler init` for interactive setup: scans your home directory for git repos and lets you pick which ones to include.

Environment variables:

| Variable | Description |
|----------|-------------|
| `AW_MAX_PANES` | Override max panes in grid (default: 10) |
| `AW_NOTIFY` | Enable desktop notifications (`1` to enable) |
| `AW_SOUNDS` | Enable sound effects (`1` to enable) |

## Requirements

- Python 3.10+
- tmux
- macOS (uses macOS-specific features for notifications and sounds)
- Ghostty terminal (recommended; set `macos-option-as-alt = true` in config)

No pip install, no virtual environment, no build step. Pure stdlib Python and Bash.

## License

MIT
