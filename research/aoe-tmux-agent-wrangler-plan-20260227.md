# Agent Wrangler: AOE + tmux Fusion Plan (2026-02-27)

## Objective
Build a lightweight but powerful terminal command center that combines:
- AOE-style repo/session orchestration ergonomics
- tmux-native speed and durability
- your manager-over-managers workflow for many active agent sessions

## External Research Highlights

### AOE patterns worth borrowing
1. Command ergonomics with one CLI for full lifecycle (`init`, `start`, `status`, `stop`, `connect`).
2. Per-repo configuration and startup hooks for repeatable behavior.
3. Worktree-native branching flow for isolated issue execution.
4. Built-in diff/drift visibility from the orchestrator UI.

### tmux ecosystem patterns worth borrowing
1. tmux native control primitives (`display-popup`, `set-hook`, `choose-tree`) to keep control in-terminal.
2. Session jumping and fuzzy navigation (`sesh`, `tmux-fzf`) for fast movement.
3. Declarative session templates (`tmuxp`) to codify known layouts.
4. Session persistence and restore (`tmux-resurrect`, `tmux-continuum`) for overnight/long-running operations.

## Product Direction

### Layer A: Team Grid (already implemented)
- Dynamic pane import from Ghostty.
- Agent and attention detection.
- Pane-level control commands.

### Layer B: Fleet Orchestrator (implemented in this pass)
- Cross-session rollup (`fleet status`).
- Live multi-session monitor (`fleet watch`).
- Dedicated HQ manager session (`fleet manager`).
- Fast jump between sessions (`fleet focus`).
- Persistent fleet selection (`fleet set` / `fleet clear`).

### Layer C: Drift Intelligence (implemented in this pass)
- AOE-style repo drift summary for pane projects (`drift`).
- Fleet-wide drift checks with alert thresholds (`drift --fleet --alert-dirty`).

### Layer D: Optional next upgrades
1. Add popup controls and choose-tree shortcuts inside manager views.
2. Add optional tmuxp export/import for named playbooks.
3. Add optional resurrect/continuum bootstrap script.
4. Add event hooks (on session start/stop/fail) to run custom commands.

## Why this design stays lightweight
- No web backend required.
- Uses tmux + Python CLI only.
- Reuses existing `terminal_sentinel` and `command_center` logic.
- Keeps existing aliases (`cc`, `teams`, `hq`) for zero migration pain.

## Immediate Operating Flow
1. Build or import your main grid:
   - `./scripts/agent-wrangler up --rebuild --mode import --max-panes 10 --nav --manager --manager-replace`
2. Define managed fleet once:
   - `./scripts/agent-wrangler fleet set --sessions amir-grid`
3. Run HQ manager:
   - `./scripts/agent-wrangler fleet manager --replace --update-defaults`
4. Check code/repo drift before delegation:
   - `./scripts/agent-wrangler drift --fleet --alert-dirty 25`

## Success Criteria
- One command opens your control surface.
- One manager session can oversee multiple work sessions.
- Attention hotspots and repo drift are visible in under 5 seconds.
- Long-running overnight sessions remain observable and recoverable.
