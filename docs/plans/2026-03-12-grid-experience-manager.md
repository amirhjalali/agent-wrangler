# Grid Experience + Manager Awareness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform Agent Wrangler into a mouse-driven, Ghostty-native workspace where you click/double-click panes in a grid, right-click for management actions, and the manager Claude has full awareness of all projects.

**Architecture:** Add mouse bindings (double-click zoom, right-click context menu) to the existing tmux nav setup. Configure Ghostty+tmux for optimal rendering. Set up the manager Claude with cross-project CLAUDE.md imports. Remove auto-agent-launch from start (already done in bash router).

**Tech Stack:** Python 3.10+ stdlib, tmux display-menu, Ghostty config, CLAUDE.md @import

---

### Task 1: Ghostty + tmux Optimized Config

**Files:**
- Create: `config/tmux.conf`
- Modify: `scripts/agent_wrangler.py` (run_nav function, ~line 2439)

**Step 1: Create tmux.conf with Ghostty-optimized settings**

Create `config/tmux.conf`:

```conf
# Agent Wrangler tmux config — optimized for Ghostty
# Source with: tmux source-file config/tmux.conf

# True color + modern terminal features
set -g default-terminal "tmux-256color"
set -as terminal-features ",xterm-ghostty:RGB"
set -as terminal-features ",xterm-ghostty:usstyle"
set -as terminal-features ",xterm-ghostty:strikethrough"
set -as terminal-features ",xterm-ghostty:sync"
set -as terminal-features ",xterm-ghostty:clipboard"

# Fast escape (Ghostty is fast, no need for delay)
set -sg escape-time 10

# Mouse
set -g mouse on

# Focus events (Ghostty supports these)
set -g focus-events on

# OSC 52 clipboard
set -s set-clipboard on

# Passthrough for advanced escape sequences
set -g allow-passthrough on
```

**Step 2: Source tmux.conf during nav setup**

In `run_nav()` after the mouse settings (~line 2439), add:

```python
# Source Ghostty-optimized tmux config if available
tmux_conf = ROOT / "config" / "tmux.conf"
if tmux_conf.exists():
    tmux(["source-file", str(tmux_conf)], timeout=5)
```

**Step 3: Commit**

```bash
git add config/tmux.conf scripts/agent_wrangler.py
git commit -m "feat: add Ghostty-optimized tmux config"
```

---

### Task 2: Double-Click to Zoom

**Files:**
- Modify: `scripts/agent_wrangler.py` (run_nav function, ~line 2436)

**Step 1: Add double-click zoom binding**

In `run_nav()`, after the `for key, cmd in all_bindings` loop and before the mouse settings, add the double-click binding:

```python
# Double-click to zoom/unzoom a pane
tmux(["bind-key", "-n", "DoubleClick1Pane", "resize-pane", "-Z"], timeout=5)
```

**Step 2: Update the nav print output**

Update the mouse print line to mention double-click:

```python
print("Mouse: click select | double-click zoom | right-click menu | scroll browse")
```

**Step 3: Commit**

```bash
git add scripts/agent_wrangler.py
git commit -m "feat: double-click pane to zoom fullscreen"
```

---

### Task 3: Right-Click Context Menu

**Files:**
- Modify: `scripts/agent_wrangler.py` (run_nav function + new helper function)

**Step 1: Add context menu helper function**

Add this function before `run_nav()` (around line 2390):

```python
def _build_context_menu_cmd(session: str) -> list[str]:
    """Build a tmux display-menu command for right-click pane management."""
    aw = str(ROOT / "scripts" / "agent-wrangler")
    menu_items = [
        # label, key, command
        ("Zoom", "z", "resize-pane -Z"),
        ("View Output", "o", f"display-popup -E -w 80 -h 30 "
         f"'python3 {ROOT}/scripts/agent_wrangler.py teams summary "
         f"#{{pane_title}} --session {session} --lines 50; read'"),
        ("Send Command...", "c", f"command-prompt -p 'command:' "
         f"\"run-shell '{aw} send #{{pane_title}} --command \\\"%%\\\"'\""),
        ("", "", ""),  # separator
        ("Start Claude", "1", f"run-shell '{aw} agent #{{pane_title}} claude'"),
        ("Start Codex", "2", f"run-shell '{aw} agent #{{pane_title}} codex'"),
        ("Start Aider", "3", f"run-shell '{aw} agent #{{pane_title}} aider'"),
        ("", "", ""),  # separator
        ("Restart", "r", f"run-shell '{aw} restart #{{pane_title}}'"),
        ("Stop (Ctrl-C)", "s", f"run-shell '{aw} stop #{{pane_title}}'"),
        ("Kill", "k", f"run-shell '{aw} kill #{{pane_title}}'"),
    ]
    args = ["display-menu", "-T", "#[bold]#{pane_title}", "-x", "R", "-y", "S"]
    for label, key, cmd in menu_items:
        if not label:
            args.append("")  # separator
        else:
            args.extend([label, key, cmd])
    return args
```

**Step 2: Bind right-click to the context menu in run_nav()**

After the double-click binding, add:

```python
# Right-click context menu for pane management
menu_cmd = _build_context_menu_cmd(session)
tmux(["bind-key", "-n", "MouseDown3Pane", *menu_cmd], timeout=5)
```

**Step 3: Commit**

```bash
git add scripts/agent_wrangler.py
git commit -m "feat: right-click context menu for pane management"
```

---

### Task 4: Remove Auto-Agent Launch + Clean Startup

**Files:**
- Modify: `scripts/agent-wrangler` (already done — verify)
- Modify: `scripts/welcome_banner.sh` (already done — verify)
- Modify: `scripts/agent_wrangler.py` (create_grid_session, ~line 1139)

**Step 1: Verify --agent was removed from start command**

The bash router should NOT have `--agent` in the start case. Already done — just verify.

**Step 2: Improve pane ready message**

In `create_grid_session()` at line 1140, replace the echo with a cleaner message:

```python
pane_send(pane.pane_id, f"echo '\\n  [{project_id}] ● ready — right-click for options'", enter=True)
```

**Step 3: Commit**

```bash
git add scripts/agent_wrangler.py
git commit -m "chore: clean up pane ready message"
```

---

### Task 5: Manager CLAUDE.md with Cross-Project Context

**Files:**
- Modify: `CLAUDE.md` (add @import for managed projects + manager instructions)

**Step 1: Add cross-project imports and manager section to CLAUDE.md**

Append to the end of CLAUDE.md:

```markdown
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

### Managed Projects
@/Users/amirjalali/creator-studio/CLAUDE.md
@/Users/amirjalali/gabooja-labs/CLAUDE.md
@/Users/amirjalali/gabooja-knowledge-base/CLAUDE.md
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "feat: add manager mode + cross-project context to CLAUDE.md"
```

---

### Task 6: Status Bar with Clickable Project Tabs

**Files:**
- Modify: `scripts/agent_wrangler.py` (refresh_pane_health status bar section, ~line 720)

**Step 1: Replace aggregate stats with per-project tabs in the status bar**

In `refresh_pane_health()`, replace the status-right generation (the block starting around line 738) with per-project clickable tabs:

```python
    # Build per-project tab status bar
    if apply_colors and rows:
        tab_parts = []
        for row in rows:
            pid = str(row.get("project_id") or "?")
            lev = str(row.get("health") or "").lower()
            if len(pid) > 14:
                pid = pid[:13] + "~"
            dot_color = {"green": "colour34", "yellow": "colour220", "red": "colour196"}.get(lev, "colour250")
            tab_parts.append(f"#[fg={dot_color}]●#[fg=colour250] {pid}")
        tabs_str = " #[fg=colour240]│#[default] ".join(tab_parts)
        status_right = (
            f" {tabs_str} "
            "#{?window_zoomed_flag,#[fg=colour214 bold] ZOOM #[default],}"
            " #[fg=colour250]%H:%M#[default] "
        )
        tmux(["set-option", "-t", session, "status-right", status_right], timeout=3)
        tmux(["set-option", "-t", session, "status-right-length", "120"], timeout=3)
```

**Step 2: Commit**

```bash
git add scripts/agent_wrangler.py
git commit -m "feat: per-project health tabs in status bar"
```

---

### Task 7: Ghostty Config for Agent Wrangler

**Files:**
- Modify: `README.md` (add Ghostty setup section)

**Step 1: Add recommended Ghostty settings to README**

Add a section after "Install" in README.md:

```markdown
## Ghostty Setup (Recommended)

Add to your Ghostty config (`~/.config/ghostty/config` or `~/Library/Application Support/com.mitchellh.ghostty/config`):

```
macos-option-as-alt = true
mouse-hide-while-typing = true
window-padding-x = 4
window-padding-y = 4
window-padding-balance = true
```

`macos-option-as-alt` is required for Option+key navigation to work.
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add Ghostty config recommendations to README"
```

---

### Task 8: Final Integration Test

**Step 1: Kill any existing session and restart**

```bash
tmux kill-server 2>/dev/null; sleep 1
./scripts/agent-wrangler start
```

**Step 2: Verify checklist**

- [ ] Grid shows panes with project names, no agents auto-launched
- [ ] Click a pane to select it
- [ ] Double-click a pane to zoom fullscreen
- [ ] Double-click again to unzoom
- [ ] Right-click a pane shows context menu
- [ ] Context menu "Start Claude" launches Claude in the pane
- [ ] Context menu "View Output" shows popup
- [ ] Manager window has Claude Code + status rail
- [ ] Status bar shows per-project health tabs
- [ ] Option+m / Option+g switches between manager and grid

**Step 3: Final commit**

```bash
git add -A
git commit -m "chore: integration test pass — grid experience complete"
git push
```
