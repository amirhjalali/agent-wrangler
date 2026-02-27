# Agent Wrangler Multi-Agent Field Study (2026-02-27)

## Objective
Design an operator-grade command center for 8-12 concurrent AI terminal sessions (Claude + Codex mixed), with:
- one-command startup,
- low-friction navigation,
- clear "needs attention" signaling,
- overnight-safe operation,
- reliable recovery after crashes/reboots.

## Local Baseline (this machine)
- Fleet now: `sessions=1 panes=7 attention=4 waiting=5 active=1`
- Dominant red reason: command/runtime failures (`zsh: command not found`)
- Drift pressure: `dirty_files=209` (mostly `gabooja-agents`)
- Readiness gate: unstable (`20/100`)

Interpretation: the core tmux control layer works, but fidelity and operating discipline are the current constraints.

## Research Squad Outputs

### 1) Orchestrator Agent (session control patterns)
Findings:
- tmux has native primitives you can build a manager around:
  - `choose-tree` (interactive jump across sessions/windows/panes)
  - `display-popup` (ephemeral control views)
  - `set-hook` and `command-error` hooks for event-driven actions
- tmux control mode supports machine parsing and subscriptions, which is the right base for a future richer manager process.

Implication for Agent Wrangler:
- Keep manager in tmux, but make it event-assisted (hooks + control mode), not only polling.

### 2) Resilience Agent (save/restore)
Findings:
- `tmux-resurrect` restores panes, directories, and commands.
- `tmux-continuum` adds periodic autosave and auto-restore on reboot/login.

Implication:
- Add first-class `agent-wrangler persistence ...` commands and optional plugin integration.

### 3) Navigation Agent (speed of movement)
Findings:
- tmux-native jump/popup flows are fast and stable.
- session managers like `sesh` emphasize fuzzy launch/switch and wildcard startup defaults.

Implication:
- Keep your no-prefix hotkeys and add fuzzy selector mode (`fzf` if installed) for pane/session jump.

### 4) Import-Fidelity Agent (Ghostty -> tmux mapping)
Findings:
- Ghostty CLI/man pages expose launch actions but not a complete "live tab/session API".
- Reliable import requires process-level inference (`ps`/`tty`/cwd/command lineage), which you already do via `terminal_sentinel`.

Implication:
- Keep process-based import and preserve one-pane-per-source-session as an explicit mode.

### 5) Product Model Agent (AOE + CLI architecture)
Findings:
- AOE style command taxonomy (`add/list/status/session/group/profile/worktree`) is clean and scales.
- Profile/workspace separation is key for business vs personal context.

Implication:
- Introduce profile boundaries in Agent Wrangler (`gabooja`, `personal`) for defaults, limits, guardrails.

### 6) Operations Agent (overnight + token discipline)
Findings:
- Best pattern is guardrails on wait-time, active session caps, and loop cadence.
- Board/session-log artifacts reduce morning context loss and wasted token loops.

Implication:
- Add per-pane log streams and a morning summary command that points directly to overnight incidents.

## Option Matrix

| Option | Strengths | Gaps vs your vision | Recommendation |
|---|---|---|---|
| Keep raw tmux only | Fast, minimal deps | Too manual, no policy layer | Not enough |
| Move to zellij | Nice UX, modern feel | Migration cost, less existing tooling here | Not now |
| Adopt AOE as-is | Good command model/worktrees | Missing your custom multi-view manager | Borrow concepts only |
| Agent Wrangler (current) | Already integrated with your repos and monitor data | Needs import fidelity, persistence, profile model | Best base to keep building |

## Concrete Gaps vs Target UX
1. Import currently can collapse multiple Ghostty tabs in same repo into one pane.
2. Manager is mostly poll-driven; missing hook/event acceleration.
3. No official persistence workflow yet.
4. No first-class profile mode for `gabooja` vs `personal`.
5. No dedicated overnight log bundle and morning digest command.

## Implemented in this sprint
1. Added duplicate-preserving import mode:
   - `--preserve-duplicates` in `teams import` and `teams up`.
   - This keeps one pane per matched Ghostty session.
2. Wired default startup to preserve duplicates:
   - `./scripts/agent-wrangler start` now includes `--preserve-duplicates`.
3. Updated docs/examples to use duplicate-preserving startup/import patterns.

## Next Build Plan (ranked)

### P0 (immediate)
1. `doctor agents`:
   - auto-diagnose common red-pane causes (`command not found`, missing binaries, wrong cwd).
2. `fleet jump --fzf`:
   - optional fuzzy jump path for session/pane targeting.
3. `manager` right rail:
   - show top N failing panes + direct fix actions.

### P1 (next)
1. `persistence` namespace:
   - `enable`, `disable`, `save`, `restore`, `status`.
2. `profile` namespace:
   - `profile use gabooja|personal`
   - separate default sessions, caps, guardrails.
3. Overnight artifacts:
   - per-pane logs + `overnight summary`.

### P2 (later)
1. Event-driven manager:
   - tmux hooks + control mode subscriptions for lower-latency updates.
2. Worktree-native pane creation:
   - task-specific isolated branches/worktrees.
3. Optional declarative export:
   - tmuxp-compatible session export/import for portability.

## Recommended Operating Commands (now)

```bash
cd /Users/amirjalali/agent-wrangler

# exact carry-over from Ghostty mental model
./scripts/agent-wrangler import --dry-run --preserve-duplicates --max-panes 12

# one-command startup (now preserves duplicates by default)
./scripts/agent-wrangler start

# live manager + drift checks
./scripts/agent-wrangler fleet manager --replace --update-defaults
./scripts/agent-wrangler fleet status
./scripts/agent-wrangler drift --fleet --alert-dirty 25
```

## Sources
- tmux manual (`choose-tree`, `display-popup`, hooks): https://man7.org/linux/man-pages/man1/tmux.1.html
- tmux control mode/subscriptions: https://github.com/tmux/tmux/wiki/Control-Mode
- tmux-resurrect: https://github.com/tmux-plugins/tmux-resurrect
- tmux-continuum: https://github.com/tmux-plugins/tmux-continuum
- AoE docs + CLI model: https://www.agent-of-empires.com/docs/cli/reference.html
- AoE repository: https://github.com/pab1it0/agent-of-empires
- sesh (session manager patterns): https://github.com/joshmedeski/sesh
- tmuxp (declarative session configs): https://github.com/tmux-python/tmuxp
- Ghostty man page (CLI/action scope): https://man.archlinux.org/man/ghostty.1.en
