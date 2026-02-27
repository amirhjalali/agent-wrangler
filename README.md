# AmirWorkflow

A machine-level control layer for managing many repos, terminals, and Linear queues without losing context.

## Why this exists

Your current setup has strong building blocks (`agentcy`, `gabooja-agents/linear-agent`) but no single operator layer for:

1. Prioritizing what should be active now.
2. Keeping terminal contexts bounded.
3. Unifying business and personal work streams.
4. Preventing stale long-running contexts.

## What was observed on this machine (2026-02-25)

- Multiple active dev servers were detected (`3006`, `3009`, `5000`, `8080`, `3847`).
- `gabooja-agents` had a very large dirty tree (185 files).
- `creator-studio`, `argumend`, and `gabooja-website` also had uncommitted changes.
- You already have a runnable Linear conductor in `gabooja-agents/linear-agent`.
- There is no current terminal/window orchestration config (`tmux`, `zellij`, `aerospace`, `yabai`, Ghostty config).

## Operating model

Use this as your default daily flow:

1. Start day with one command: `scripts/daily-start.sh`
2. Work from the top 3 focus items only.
3. Keep max active contexts:
   - business: 3
   - personal: 2
4. Launch a project context with: `scripts/workflow launch <project_id>`
5. End day by running: `scripts/workflow snapshot`

## Commands

From `/Users/amirjalali/AmirWorkflow`:

```bash
# Full snapshot + report files in reports/
./scripts/workflow snapshot

# Include both Linear workspaces (if env vars are set)
./scripts/workflow snapshot --include-linear

# Quick ranked queue
./scripts/workflow focus --limit 6

# Health checks
./scripts/workflow doctor

# Open a Ghostty context for one project
./scripts/workflow launch creator-studio

# Print launch command only
./scripts/workflow launch creator-studio --dry-run

# Discover repos not yet in projects.json
./scripts/workflow discover --scan-root /Users/amirjalali --max-depth 3

# Discovery with JSON report output
./scripts/workflow discover --scan-root /Users/amirjalali --max-depth 3 --write-report
```

## Agent Wrangler UI (Gastown + Ant Farm)

The main UI is now a lightweight command center:

```bash
# One-shot combined view
./scripts/agent-wrangler dashboard

# Fullscreen command-center UI (recommended)
./scripts/agent-wrangler ui

# Live combined view
./scripts/agent-wrangler watch --interval 10

# Planning board lanes and cards
./scripts/agent-wrangler gastown list --open-only --verbose
./scripts/agent-wrangler gastown add --title "Stabilize creator deploy" --repo creator-studio --lane now --step "Audit dirty files" --step "Split into 2 PRs" --step "Run tests and deploy"
./scripts/agent-wrangler gastown move GAS-001 --lane next
./scripts/agent-wrangler gastown step-done GAS-001 --step 1
./scripts/agent-wrangler gastown block GAS-002 --on GAS-001

# Runtime monitor and guardrails
./scripts/agent-wrangler antfarm status --source ghostty
./scripts/agent-wrangler antfarm overnight --iterations 1
./scripts/agent-wrangler antfarm overnight --apply --iterations 1

# Quick shortcut list
./scripts/agent-wrangler palette
```

`agent-wrangler` and `termwatch` classify mixed agent sessions (`claude`, `codex`, `aider`, `gemini`) in the same Ghostty tab set.

`agent-wrangler ui` keys:
- `tab`: switch between `Overview` and `Session Admin` pages
- `1`: go to `Overview` page
- `2`: go to `Session Admin` page
- `q`: quit
- `r`: manual refresh
- `k`: kill oldest waiting session (overview) / kill selected session (admin)
- `up/down`: move selection in admin page
- `g` or `enter`: open Ghostty monitor window for selected tty
- `i`: copy selected tty inspect command to clipboard
- `:`: admin command mode (`help`, `kill`, `inspect`, `open`, `plan`, `apply`)
- `o`: run overnight guard in dry-run mode once
- `a`: run overnight guard and apply actions once

## Tmux Teams Grid (individual pane control)

Use this when you want a real operator grid similar to "team sessions":

```bash
# Simplest launcher (one command)
./scripts/hq --rebuild --mode import --max-panes 10
./scripts/hq --rebuild --mode import --max-panes 10 --nav --manager --manager-replace

# Single command entrypoint (recommended)
./scripts/agent-wrangler up --rebuild --mode import --max-panes 10

# Manager/orchestrator screen + health coloring
./scripts/agent-wrangler manager --replace
./scripts/agent-wrangler paint
./scripts/agent-wrangler watch --interval 3

# Fleet orchestrator (manager-over-managers across tmux sessions)
./scripts/agent-wrangler fleet list
./scripts/agent-wrangler fleet set --sessions amir-grid,gabooja-grid
./scripts/agent-wrangler fleet status
./scripts/agent-wrangler fleet watch --interval 3
./scripts/agent-wrangler fleet manager --replace --update-defaults
./scripts/agent-wrangler fleet focus amir-grid

# Repo drift view (AOE-style diff/dirty awareness)
./scripts/agent-wrangler drift
./scripts/agent-wrangler drift --fleet --alert-dirty 25

# Program orchestration (team + loops + readiness gates)
./scripts/agent-wrangler program init
./scripts/agent-wrangler program team
./scripts/agent-wrangler program loops
./scripts/agent-wrangler program status
./scripts/agent-wrangler program plan --write-report
./scripts/agent-wrangler program loop --iterations 1 --apply-safe --write-report

# Fast pane navigation without tmux prefix
./scripts/agent-wrangler nav

# Import your current Ghostty sessions into a dynamic tmux grid (recommended first move)
./scripts/agent-wrangler import --max-panes 10 --layout auto
./scripts/agent-wrangler import --dry-run --max-panes 10 --layout auto

# Optional: also run startup commands and re-open detected agents
./scripts/agent-wrangler import --max-panes 10 --layout auto --startup --agent

# Build a 4-pane tmux grid from selected repos
./scripts/agent-wrangler bootstrap --projects creator-studio,gabooja-agents,agentcy,argumend --layout tiled

# Attach to the grid
./scripts/agent-wrangler attach

# View pane status with agent/waiting signals
./scripts/agent-wrangler status

# Send command to a pane (index, pane id, or title/project id)
./scripts/agent-wrangler send 0 --command "codex"
./scripts/agent-wrangler send gabooja-agents --command "claude"

# Agent shortcut per pane
./scripts/agent-wrangler agent gabooja-agents claude
./scripts/agent-wrangler agent creator-studio codex --flags "--yolo"
./scripts/agent-wrangler agent creator-studio codex -- --help

# Control panes individually
./scripts/agent-wrangler stop gabooja-agents
./scripts/agent-wrangler restart gabooja-agents
./scripts/agent-wrangler shell gabooja-agents
./scripts/agent-wrangler capture gabooja-agents --lines 40
./scripts/agent-wrangler kill gabooja-agents
```

Notes:
- `hq` is a thin shortcut for `agent-wrangler up`.
- `cc` and `teams` are still available as legacy aliases.
- `fleet manager` creates a dedicated HQ tmux session that monitors all managed sessions in real time.
- `fleet set` lets you pin exactly which tmux sessions count as your operating universe.
- `drift --fleet` gives a fast per-project branch/dirty summary across active sessions.
- `program` gives a deep execution system with explicit team roles, loops, and readiness gates toward an impeccable product target.
- You do not need to reset Ghostty first. Import into tmux, verify control, then close old tabs gradually.
- `import` maps live Ghostty sessions to repos by process cwd/path and can carry detected agent type (`claude`/`codex`) when enabled.
- `import` is safe by default: startup commands and agent relaunch are disabled unless you pass `--startup` and/or `--agent`.
- `teams up` gives one command behavior: reuse existing grid if present, or build from import/bootstrap and attach.
- `teams paint/watch/manager` sets pane borders to green/yellow/red based on attention signals (waiting/error markers/activity).
- In tmux, standard pane navigation is `Ctrl-b` + arrows. `teams nav` enables `Option+Arrow` no-prefix movement.
- `bootstrap` can auto-run each repo's `startup_command`. Use `--no-startup` to disable.
- Add `--agent codex` or `--agent claude` during `bootstrap` to launch the same agent command in every pane.

## Lightweight terminal UI (Ghostty monitor)

The monitor is intentionally terminal-first (no heavy web app):

```bash
# One-shot status (active vs waiting vs idle)
./scripts/termwatch summary --source ghostty --no-idle

# Live updating terminal view
./scripts/termwatch watch --source ghostty --no-idle --interval 10

# Overnight guardrails (dry-run by default)
./scripts/termwatch overnight --source ghostty --interval 300 --max-ai-sessions 4 --kill-waiting-ai-after 120

# Apply guardrails (sends SIGTERM to selected sessions)
./scripts/termwatch overnight --source ghostty --apply --interval 300 --max-ai-sessions 4 --kill-waiting-ai-after 120
```

Convenience wrapper:

```bash
./scripts/overnight-guard.sh
```

## Linear setup for dual-workspace visibility

This workflow expects these env vars if you want cross-workspace task signals:

- `LINEAR_API_KEY_GABOOJA`
- `LINEAR_API_KEY_AMIRHJALALI`

The script will call the Linear GraphQL API and pull assigned, non-completed issues per workspace.

## Current project registry

Edit `/Users/amirjalali/AmirWorkflow/config/projects.json` to keep this accurate.

Best practice:

1. Keep full repo inventory in `projects.json` and use focus scoring to choose the active subset.
2. Set one default startup command per repo.
3. Keep business/personal grouping strict.
4. Run `discover` weekly and add untracked repos.

## What to improve next

1. Route both Linear workspaces into one operator queue issue label set (`domain:business`, `domain:personal`, `focus:today`).
2. Patch hardcoded `/Users/gabooja/...` defaults in `gabooja-agents/linear-agent` so your local paths are first-class.
3. Add a scheduler (launchd or Codex automation) to run `snapshot --include-linear` every hour and alert on context drift.
4. Optionally install a terminal multiplexer (tmux or zellij) to move from many windows to a few stable sessions.

## Files in this folder

- `config/projects.json`: project inventory, grouping, startup commands.
- `config/command_center.json`: Gastown planning cards + Ant Farm defaults.
- `scripts/workflow_agent.py`: snapshot, focus ranking, doctor checks, discovery, Ghostty launcher.
- `scripts/workflow`: short wrapper around `workflow_agent.py`.
- `scripts/command_center.py`: unified `Gastown` planning + `Ant Farm` runtime command center.
- `scripts/agent-wrangler`: primary CLI for the command center + tmux teams.
- `scripts/wrangler`: short alias for `agent-wrangler`.
- `scripts/cc`: legacy wrapper for `command_center.py`.
- `scripts/tmux_teams.py`: tmux team-grid orchestrator used by `agent-wrangler`/`cc teams`.
- `scripts/program_orchestrator.py`: impeccable-product program engine (team, loops, gates, reports).
- `scripts/teams`: direct wrapper for `tmux_teams.py`.
- `scripts/terminal_sentinel.py`: session classification for Ghostty terminals (`active`/`waiting`/`idle`).
- `scripts/termwatch`: wrapper for `terminal_sentinel.py`.
- `scripts/overnight-guard.sh`: default unattended guardrail mode.
- `scripts/daily-start.sh`: one-command morning routine.
- `reports/`: generated snapshot JSON + Markdown reports.
- `research/sources.md`: links used during research.
