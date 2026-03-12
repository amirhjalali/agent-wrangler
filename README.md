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

Agent Wrangler gives you a tmux grid where each pane is a project with an AI agent (Claude Code, Codex, Aider, Gemini). Color-coded borders show which agents are working, waiting, or broken. One command to start, zero config required.

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

git clone https://github.com/AmirhJalworker/agent-wrangler.git
cd agent-wrangler
```

Python 3.10+ required. No pip install, no venv, no build step — just stdlib.

## Start

```bash
./scripts/agent-wrangler start
```

That's it. Agent Wrangler auto-discovers your running Ghostty terminals, imports them into a tmux grid, and sets up a manager window with Claude Code + a live status rail. No config file needed.

If you want more control, create a project list:

```bash
./scripts/agent-wrangler init
```

This scans your home directory for git repos and lets you pick which ones to include.

## Navigate

| Shortcut | What it does |
|---|---|
| `Option+g` | Switch to grid (your agent panes) |
| `Option+m` | Switch to manager (Claude Code + status rail) |
| `Option+Arrow` | Move between panes |
| `Option+z` | Zoom current pane to fullscreen (toggle) |
| `Option+n` / `p` | Next / previous pane (works while zoomed) |
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

## Add projects on the fly

```bash
cd ~/projects/new-thing
./scripts/agent-wrangler add .
```

Or specify a path:

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
- Desktop notifications when a pane goes red (macOS, opt-in)

Other monitoring commands:

```bash
./scripts/agent-wrangler watch             # live health table
./scripts/agent-wrangler status            # one-shot health overview
```

## Hide and show

Don't need a pane right now? Hide it — the agent keeps running in the background:

```bash
./scripts/agent-wrangler hide my-webapp
./scripts/agent-wrangler show my-webapp
./scripts/agent-wrangler hidden            # list what's hidden
```

## How it works

Agent Wrangler is pure Bash + Python stdlib. No frameworks, no dependencies.

```
agent-wrangler           Bash router
agent_wrangler.py        Grid control, health coloring, manager, rail, ops
terminal_sentinel.py     Process monitoring, AI tool classification
───────────────────────────────────────────────────
Ghostty / tmux           Terminal substrate
```

**Auto-discovery**: On startup, Agent Wrangler scans running Ghostty terminals via process trees. Each terminal is matched to a project by its working directory. Unmatched terminals are imported using their directory name — no config file required.

**Health detection**: Scrollback-based. Each pane's text is captured and checked for the agent's prompt character (`❯` for Claude Code, `aider>` for Aider). If the prompt is visible, the agent is waiting for input (yellow). If there's streaming output, it's working (green). This works because AI tools think on remote servers — local CPU tells you nothing.

**Notifications**: Optional desktop alerts (macOS) when a pane goes red. Off by default — enable with `"notifications": true` in `config/team_grid.json` or `AW_NOTIFY=1`.

## Environment variables

| Variable | Description |
|---|---|
| `AW_MAX_PANES` | Max panes in grid (default: 10, or set via profile) |
| `AW_NOTIFY` | Enable desktop notifications (`1` to enable) |

## License

MIT
