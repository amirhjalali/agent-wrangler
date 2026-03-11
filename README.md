```
   █████   ██████  ███████ ███    ██ ████████            ▄███▄
  ██   ██ ██       ██      ████   ██    ██             ▄███████▄
  ███████ ██   ███ █████   ██ ██  ██    ██       ▄▀▀▀▀███████████▀▀▀▀▄
  ██   ██ ██    ██ ██      ██  ██ ██    ██        ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
  ██   ██  ██████  ███████ ██   ████    ██

  ██     ██ ██████   █████  ███    ██  ██████  ██      ███████ ██████
  ██     ██ ██   ██ ██   ██ ████   ██ ██       ██      ██      ██   ██
  ██  █  ██ ██████  ███████ ██ ██  ██ ██   ███ ██      █████   ██████
  ██ ███ ██ ██   ██ ██   ██ ██  ██ ██ ██    ██ ██      ██      ██   ██
   ███ ███  ██   ██ ██   ██ ██   ████  ██████  ███████ ███████ ██   ██
```

**Manage teams of AI coding agents from one terminal.**

Agent Wrangler gives you a tmux grid where each pane is a project with an AI agent (Claude Code, Codex, Aider, Gemini). Color-coded borders show you which agents are working, waiting, or broken. One command to start, keyboard shortcuts to navigate.

```
┌──── ● my-webapp ────────┬──── ⚑ api-server ────────┐
│                         │                           │
│  $ claude               │  $ codex                  │
│  > implementing auth    │  (waiting 3m)             │
│                         │                           │
├──── ● side-project ─────┼──── ✖ data-pipeline ──────┤
│                         │                           │
│  $ aider                │  (idle 45m)               │
│  > refactoring tests    │                           │
│                         │                           │
└─────────────────────────┴───────────────────────────┘
  ● green = working    ⚑ yellow = needs attention    ✖ red = problem
```

---

## Install

```bash
brew install tmux

git clone https://github.com/YOUR_USER/agent-wrangler.git
cd agent-wrangler
```

Python 3.7+ required. No pip install, no venv, no build step — just stdlib.

## Setup

The fastest way — auto-scan your home directory for git repos:

```bash
./scripts/agent-wrangler init
```

This finds all repos in `~/`, lets you pick which ones to include, and creates your config.

Or do it manually: `cp config/projects.example.json config/projects.json` and edit the paths.

## Start

```bash
./scripts/agent-wrangler start
```

This does everything:
1. Imports your running Ghostty/terminal tabs into tmux (matches by working directory)
2. Creates a **grid window** with one pane per project
3. Creates a **manager window** with Claude Code + a live status rail
4. Enables keyboard navigation (Option+Arrow, Option+m/g, etc.)

## Navigate

| Shortcut | What it does |
|---|---|
| `Option+g` | Switch to grid (your agent panes) |
| `Option+m` | Switch to manager (Claude Code + status rail) |
| `Option+Arrow` | Move between panes |
| `Option+z` | Zoom current pane to fullscreen (toggle) |
| `Option+j` | Jump to pane by number |
| `Option+[` / `]` | Previous / next window |
| `Option+q` | Exit Agent Wrangler |
| **Mouse click** | Select a pane |
| **Mouse scroll** | Scroll pane output |

## Control agents

Launch an AI agent in any pane:

```bash
./scripts/agent-wrangler agent my-webapp claude
./scripts/agent-wrangler agent api-server codex
```

Send commands, stop agents, or grab output:

```bash
./scripts/agent-wrangler send my-webapp --command "git status"
./scripts/agent-wrangler stop my-webapp
./scripts/agent-wrangler capture my-webapp --lines 40
```

## Hide and show

Don't need a pane right now? Hide it. The agent keeps running in the background:

```bash
./scripts/agent-wrangler hide my-webapp
./scripts/agent-wrangler show my-webapp
./scripts/agent-wrangler hidden            # list what's hidden
```

Or use `h` in the grid navigator (`./scripts/agent-wrangler grid`).

## Add projects on the fly

Add the current directory to your grid without editing config:

```bash
cd ~/projects/new-thing
./scripts/agent-wrangler add .
```

Or specify a path and name:

```bash
./scripts/agent-wrangler add ~/projects/cool-app --name cool-app
```

If the grid is running, the pane appears immediately.

## Pane summary

See what an agent has been doing:

```bash
./scripts/agent-wrangler summary my-webapp
```

Or press `Option+s` on any pane for a popup summary.

## Monitor

The **status bar** at the bottom of the grid shows aggregate health, agent count, context %, and cost at a glance.

The **status rail** (right side of manager window) auto-refreshes and shows:
- Health dot per pane (green/yellow/red)
- Agent type (claude, codex, aider, gemini)
- Context window % and cost for Claude Code sessions
- Hidden panes (dimmed)
- Desktop notifications when a pane goes red or recovers (macOS)

For a wider view:

```bash
./scripts/agent-wrangler watch             # live health table with CTX% and COST columns
./scripts/agent-wrangler status            # one-shot health overview
./scripts/agent-wrangler grid              # curses pane browser with actions
```

## Grid navigator

Open with `./scripts/agent-wrangler grid`. A curses UI for quick pane management:

| Key | Action |
|---|---|
| `j`/`k` | Navigate up/down |
| `Enter` | Jump to pane |
| `h` | Hide/show toggle |
| `c` | Launch Claude Code |
| `x` | Launch Codex |
| `s` | Send a command |
| `K` | Send Ctrl-C |
| `q` | Quit |

## How it works

Agent Wrangler is pure Bash + Python stdlib. No frameworks, no dependencies.

- **`tmux_teams.py`** — Core engine. Creates sessions, manages panes, detects health, paints borders, runs the rail.
- **`terminal_sentinel.py`** — Inspects process trees (`ps`) to classify terminals as active/waiting/idle and detect AI tools.
- **`grid_navigator.py`** — Curses pane browser with health coloring and direct actions.
- **`session_stats.py`** — Scrapes Claude Code's status bar for context %, tokens, and cost.

```
tmux_teams.py             Grid control, health coloring, manager, rail
grid_navigator.py         Curses pane browser
session_stats.py          Context stats collection
terminal_sentinel.py      Process monitoring, AI tool classification
────────────────────────────────────────────────────────
Ghostty / tmux            Terminal substrate
```

## Environment variables

| Variable | Description |
|---|---|
| `AW_MAX_PANES` | Max panes in grid (default: 10, or set via profile) |

## License

MIT
