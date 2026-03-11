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

# Agent Wrangler

A tmux-based command center for managing teams of AI coding agents across multiple repos.

Run Claude Code, Codex, Aider, or Gemini sessions side-by-side in a health-monitored grid. See which agents are active, idle, or need attention — at a glance.

## What it does

- **Grid view** — Tile your AI agent sessions in a tmux grid with color-coded health borders (green/yellow/red)
- **Manager window** — Claude Code + auto-refreshing status rail showing all panes
- **Idle detection** — Classifies each terminal as active, waiting, idle, or background by inspecting process trees
- **Agent detection** — Automatically identifies running AI tools (claude, codex, aider, gemini)
- **Fleet management** — Monitor multiple tmux sessions from one place
- **Terminal import** — Import existing Ghostty tabs into the tmux grid without losing context
- **Navigation** — Option+Arrow panes, Option+m/g windows, Option+z zoom, Option+j jump

## Requirements

- macOS (tested on Ghostty, works with any terminal)
- Python 3.7+
- tmux

```bash
brew install tmux
brew install fzf   # optional, enables fuzzy-find in fleet jump
```

No pip install, no virtual environment, no build step. Pure Python stdlib + Bash.

## Quick start

```bash
git clone https://github.com/YOUR_USER/agent-wrangler.git
cd agent-wrangler

# Start everything: grid + manager + nav bindings
./scripts/agent-wrangler start

# Or run the interactive operator console
./scripts/agent-wrangler ops
```

## How it works

After `agent-wrangler start`, you get two tmux windows:

**Manager** (`Option+m`) — Claude Code on the left (~75%), auto-refreshing status rail on the right (~25%). The rail shows per-pane health dots, agent type, and status.

**Grid** (`Option+g`) — Tiled panes, one per project. Each pane border shows health: `●` green (ok), `⚑` yellow (attention), `✖` red (problem). Active pane gets a bright white border with `▶` marker.

```
┌──────────────────────┬──────────────────────┐
│ ● my-webapp          │ ⚑ api-server         │
│                      │                      │
│  $ claude            │  $ codex --yolo      │
│  > working on auth   │  (waiting 3m)        │
│                      │                      │
├──────────────────────┼──────────────────────┤
│ ● side-project       │ ✖ data-pipeline      │
│                      │                      │
│  $ aider             │  (idle 45m)          │
│  > refactoring       │                      │
│                      │                      │
└──────────────────────┴──────────────────────┘
```

## Commands

### Core

```bash
./scripts/agent-wrangler start          # Full startup: grid + manager + nav
./scripts/agent-wrangler ops            # Interactive operator menu
./scripts/agent-wrangler status         # Pane health overview
./scripts/agent-wrangler watch          # Live-updating health table
./scripts/agent-wrangler grid           # Curses pane browser
```

### Agent control

```bash
./scripts/agent-wrangler agent my-webapp claude       # Launch Claude in a pane
./scripts/agent-wrangler agent api-server codex       # Launch Codex in a pane
./scripts/agent-wrangler send my-webapp --command "git status"
./scripts/agent-wrangler stop my-webapp               # Send Ctrl-C
./scripts/agent-wrangler capture my-webapp --lines 40  # Grab scrollback
```

### Grid management

```bash
./scripts/agent-wrangler paint          # Color pane borders by health
./scripts/agent-wrangler manager        # Start manager window
./scripts/agent-wrangler rail           # Compact status sidebar
./scripts/agent-wrangler import         # Import Ghostty tabs into tmux grid
./scripts/agent-wrangler nav            # Enable Option+Arrow navigation
```

### Fleet (multi-session)

```bash
./scripts/agent-wrangler fleet set --sessions my-grid
./scripts/agent-wrangler fleet status
./scripts/agent-wrangler fleet watch --interval 3
./scripts/agent-wrangler fleet jump --fzf    # Fuzzy-find across sessions
./scripts/agent-wrangler fleet popup         # Quick triage popup
```

### Workflow

```bash
./scripts/agent-wrangler drift               # Repo cleanliness report
./scripts/agent-wrangler doctor              # Health checks
./scripts/agent-wrangler persistence save    # Save session layout
./scripts/agent-wrangler persistence restore # Restore session layout
./scripts/agent-wrangler profile list        # List workspace profiles
```

## Navigation keybindings

| Key | Action |
|---|---|
| `Option+Arrow` | Move between panes |
| `Option+[` / `Option+]` | Previous/next window |
| `Option+m` | Jump to manager window |
| `Option+g` | Jump to grid window |
| `Option+z` | Zoom/unzoom current pane |
| `Option+j` | Jump to pane by number |
| `Option+1..9` | Jump to window by number |

## Configuration

### `config/projects.json`

Register your projects. Each entry needs an `id`, `path`, and optionally a `startup_command` and `group`:

```json
{
  "groups": {
    "work": { "max_active": 3 },
    "personal": { "max_active": 2 }
  },
  "projects": [
    {
      "id": "my-webapp",
      "name": "My Web App",
      "group": "work",
      "path": "~/projects/my-webapp",
      "default_branch": "main",
      "startup_command": "npm run dev"
    }
  ]
}
```

### `config/team_grid.json`

Tmux session layout, fleet membership, profiles, and persistence settings.

### Environment variables

| Variable | Description |
|---|---|
| `AW_MAX_PANES` | Override maximum panes in grid (default from profile) |

## Architecture

```
Layer 4  program_orchestrator.py   Phase-gated delivery (team roles, readiness gates)
Layer 3  command_center.py         Planning board + runtime monitor
Layer 2  tmux_teams.py             Grid control, health coloring, manager, rail
         grid_navigator.py         Curses pane browser
         session_stats.py          Context stats collection
Layer 1  terminal_sentinel.py      Process/TTY monitoring, AI tool classification
Layer 0  Ghostty / tmux            Terminal substrate
```

**tmux_teams.py** is the core engine — sessions, panes, health detection, fleet, persistence, Ghostty import.

**terminal_sentinel.py** parses `ps` output to classify terminals and detect AI tools.

**grid_navigator.py** is a curses pane browser with health coloring and agent launch shortcuts.

**session_stats.py** collects scrollback size, uptime, and periodic `/usage` polls from idle Claude sessions.

## Grid navigator keys

| Key | Action |
|---|---|
| `j`/`k` or arrows | Navigate panes |
| `Enter` | Jump to pane |
| `c` | Launch Claude in pane |
| `x` | Launch Codex in pane |
| `s` | Send command to pane |
| `K` | Send Ctrl-C to pane |
| `q` | Quit |

## License

MIT
