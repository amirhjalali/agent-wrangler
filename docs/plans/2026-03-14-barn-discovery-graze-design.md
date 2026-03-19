# Barn Discovery + Click-to-Graze

**Date**: 2026-03-14
**Status**: Approved

## Summary

Auto-discover all git repos on the machine, show non-active ones in the barn section at the bottom of the ranch board (status rail), and let the user press a number key to "send to graze" — opening the repo in a new Ghostty window that gets auto-imported into the grid.

## Design

### Barn Section in Rail

Below the existing rail content (after "last roundup" timestamp), a new barn section lists repos found in `~/` that have a `.git/` directory but are NOT currently in the active grid:

```
──────────────────────────────────
 IN THE BARN (12)
 [1] agentcy
 [2] argumend
 [3] campalborz.org
 [4] gabooja-agents
 [5] gabooja-client
 ...
 press 1-9 to send to graze
──────────────────────────────────
```

Each entry is numbered 1-9. Pressing a number key opens that repo in a new Ghostty window. The sentinel auto-discovers it on the next scan cycle (~5s), imports it into the grid, and it disappears from the barn list.

### Repo Discovery

- Scan `~/` for top-level directories containing `.git/`
- Exclude dotfiles, non-directories, and the agent-wrangler repo itself
- Cache the scan result; refresh every 60 seconds (not every 5s rail refresh)
- Filter out repos that already have an active pane in the grid
- Sort alphabetically

### Click-to-Graze Mechanism

Pressing a number key (1-9) in the rail pane:
- Looks up the corresponding barn entry
- Runs `open -na Ghostty --args --working-directory=<path>`
- The sentinel discovers the new Ghostty terminal on its next scan
- The terminal gets imported into the tmux grid
- The project moves from barn to active in the rail

The rail already runs in a `while True` loop. Add non-blocking stdin reads (like the welcome banner does) to detect keypresses between refresh cycles.

### Stale Pane Auto-Cleanup

Each rail refresh checks active panes for staleness:
- If the pane's shell process has exited (tmux reports dead pane)
- If the Ghostty source terminal for a path-matched project is no longer running

Stale panes are auto-removed from the grid. The project returns to the barn list on the next refresh.

## Files to Change

- `scripts/agent_wrangler.py` — `run_rail()`: add barn section rendering, repo discovery cache, keypress handling, stale pane detection
- No changes needed to bash router or other files

## Non-Goals

- Deep recursive repo scanning (only `~/` top-level)
- Mouse click support (number keys only for now)
- Barn pagination beyond 9 entries (show first 9, mention overflow count)
