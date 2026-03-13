# Cowboy Tech Ranch UI — Visual Overhaul Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform Agent Wrangler from functional-but-static into a living, breathing cowboy tech ranch. Campfire warmth animation on startup, animated status rail with sparklines and cattle metaphors, branded pane borders, themed context menus, sound effects, and a cinematic startup roundup sequence.

**Aesthetic:** Warm rust/amber palette. Campfire embers. Ranch metaphors (head of cattle, grazing, at fence, roundup). Professional but with personality. Everything feels alive without being distracting.

**Tech Stack:** Pure bash + Python 3.10+ stdlib. No external dependencies. ANSI 256-color + truecolor. macOS `afplay` for sounds. tmux format strings for borders/status bar.

**Palette:** Ember dark (colour 52) → rust (130) → burnt orange (166) → amber (172/208) → gold (214/220) → bright yellow (226). Grayscale smoke: 232-240. Green health: 34. Active border: 255.

---

## Phase 1: Campfire Welcome Banner

**Files:**
- Rewrite: `scripts/welcome_banner.sh`

### Task 1: Animated Campfire Warmth Banner

Replace the static banner with a center-outward warmth-spreading animation. The ASCII art starts near-invisible (colour 232), then embers catch from the center of the cowboy hat, spreading outward through the rust/amber gradient until the full art glows warm.

**Implementation:**

**Step 1:** Restructure `welcome_banner.sh` with animation framework:

```bash
#!/usr/bin/env bash
# Agent Wrangler — Campfire Welcome Banner
# Warmth spreads from the center outward, like embers catching

set -euo pipefail

# --- Skip conditions ---
# Skip animation if not interactive, piped, or AW_SKIP_ANIM is set
if [[ ! -t 1 ]] || [[ -n "${AW_SKIP_ANIM:-}" ]]; then
    # Fast path: print final state instantly
    printf '\033[38;5;130m'
    cat <<'BANNER'
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
BANNER
    printf '\033[0m\n'
    printf '\033[2m  Steering agents. Wrangling terminals. Shipping code.\033[0m\n'
    printf '\033[2m  ⌥m manager  ⌥g grid  ⌥z zoom  ⌥n/p cycle  ⌥j jump\033[0m\n'
    exit 0
fi
```

**Step 2:** Add the warmth-spread animation engine. This is the core algorithm:

1. Store the ASCII art as an array of lines
2. Find the center point (the cowboy hat, approximately row 3, col 55 — center of the `▄███▄` hat)
3. Pre-compute Manhattan distance from center for every character position (multiply y by 3 to compensate for character aspect ratio)
4. For each of ~20 frames (total ~1.5 seconds at 15fps):
   - Compute the warmth radius using ease-out-cubic: `radius = ease(frame/total) * max_distance`
   - For each non-space character:
     - If distance > radius: print in colour 232 (near-invisible)
     - If distance <= radius: map `(radius - distance)` to the warm gradient (232 → 236 → 94 → 130 → 166 → 172 → 208)
     - Characters at the "frontier" (distance ≈ radius) get a bright flash: colour 220 (gold) for 1 frame, then settle to their final warm tone
5. Use `printf '\033[H'` (cursor home) between frames, NOT `\033[2J` (clear screen) — this prevents flicker
6. Buffer each frame as a single string, write+flush once per frame
7. Hide cursor during animation (`\033[?25l`), restore after (`\033[?25h`)
8. Trap EXIT/INT/TERM to always restore cursor

**Step 3:** Add skip detection using non-blocking read:

```bash
old_stty=$(stty -g)
stty -echo -icanon min 0 time 0
# In animation loop: if read -rsn1 -t 0 key; then break; fi
# After loop: stty "$old_stty"
```

Any keypress during animation jumps to the final warm state instantly.

**Step 4:** After the art is fully warm, fade in the tagline and nav hints:

- Tagline appears with a brief typewriter effect (each character at 30ms intervals)
- Nav hints fade from dim (colour 236) to normal dim (standard `\033[2m`) over 5 frames
- Total post-art animation: ~0.5 seconds

**Step 5:** Add campfire sound effect:

```bash
# Subtle startup sound (background, won't block)
if [[ -f /System/Library/Sounds/Blow.aiff ]]; then
    afplay -v 0.3 /System/Library/Sounds/Blow.aiff &
    disown
fi
```

Play `Blow.aiff` (breathy whoosh, 1.4s) at low volume at the start of the animation — it complements the warmth spreading.

**Step 6:** After animation completes, add a gentle hat pulse. The cowboy hat characters (`▄███▄` area) cycle through 130→172→208→172→130 once over ~0.5 seconds. This is the "embers settling" moment before the script exits.

**Acceptance Criteria:**
- [x] Banner animates warmth from center outward in ~1.5 seconds
- [x] Cowboy hat area is the origin point of the warmth spread
- [x] Any keypress skips to final state
- [x] `AW_SKIP_ANIM=1` skips animation entirely
- [x] Non-interactive terminals (pipes, CI) get the static fast path
- [x] Cursor is always restored on exit (even on Ctrl-C)
- [x] Subtle sound plays on startup (non-blocking)
- [x] Hat pulses gently once after the main animation

---

## Phase 2: Status Rail Overhaul

**Files:**
- Modify: `scripts/agent_wrangler.py` — `run_rail()` function (lines 1610-1711)

### Task 2: Polished Health Indicators

Update the rail's per-pane display from plain text to a structured, aligned format with proper health dots.

**Step 1:** Replace the current dot rendering. Currently all states use `●`. Change to:
- Green active: `\033[32m●\033[0m` (already correct)
- Yellow waiting: `\033[33m●\033[0m` (change from `⚑` to filled `●` for consistency)
- Red error: `\033[31m●\033[0m` (change from `✖` to filled `●` for consistency)
- The symbols (⚑, ✖) should still appear in the pane borders (Phase 4), but the rail uses uniform dots for clean alignment

**Step 2:** Add a state-change sparkle. When a pane's health changes between rail refreshes, show a brief `✦` (U+2726) next to the dot for 2 refresh cycles, then fade it out. Track previous health states in a dict.

**Step 3:** Improve alignment. Use fixed-width columns:
```
● project-name      claude  active   52% $0.45
● other-project     aider   waiting  ──  ──
```

Pad project names to 18 chars, agent to 8, status to 8, context to 5, cost to 6.

**Acceptance Criteria:**
- [x] All health states use filled `●` with proper colors
- [x] State changes show `✦` sparkle for 2 cycles
- [x] Columns are aligned with fixed widths
- [x] Hidden panes still show as `○` in dim

### Task 3: Live Dashboard — Sparklines and Activity

Add sparklines and richer data to the rail.

**Step 1:** Add a health history buffer. Store the last 10 health check results per pane as numeric values (green=2, yellow=1, red=0). On each refresh cycle, append the current state.

**Step 2:** Render a mini sparkline next to each pane using the block characters `▁▂▃▄▅▆▇█`:
- Map health values: red(0)→`▁`, yellow(1)→`▄`, green(2)→`█`
- Color the sparkline: each bar gets the color of its health level
- Show last 10 checks: `\033[32m█\033[32m█\033[33m▄\033[32m█\033[32m█\033[32m█\033[31m▁\033[32m█\033[33m▄\033[32m█\033[0m`

**Step 3:** Add a Claude context progress bar. For panes running Claude Code, render a mini bar:
```
ctx [████████░░░░░░░░░░░░] 42%
```
- Use `█` (filled) and `░` (light shade) characters
- Color by threshold: green < 50%, yellow 50-79%, red >= 80%
- Width: 20 characters

**Step 4:** Add a cost ticker. Show per-pane cost with a subtle animation — when cost increases, briefly flash the number in colour 214 (gold) before settling to dim.

**Step 5:** Add the "herd status" summary header:

```
╭─ Ranch Status ────────────────╮
│ 🐄 6 head │ 4 grazing │ 1 at fence │ 1 down │
╰───────────────────────────────╯
```

Use box-drawing characters (rounded corners: `╭╮╰╯─│`). Map health states to ranch terms:
- Green active → "grazing" (working, producing)
- Yellow waiting → "at fence" (idle, waiting for input)
- Red error → "down" (needs attention)

The cattle emoji `🐄` is optional — if it causes tmux width issues, use `[*]` instead. Test in Ghostty first.

**Acceptance Criteria:**
- [x] Each pane shows a 10-check sparkline with colored bars
- [x] Claude panes show a context usage progress bar
- [x] Cost values flash gold briefly when they increase
- [x] "Ranch Status" header with cattle count and rounded box
- [x] Sparklines update every refresh cycle

### Task 4: Animated Ranch Board Header

Add visual flair to the rail header.

**Step 1:** Replace the plain "STATUS RAIL" header with a small campfire ASCII art that flickers:

```
    )
   ) \
  / ) (
  \(_)/    RANCH BOARD
  _|__|_   ─────────────
 |      |  6 head · 4🟢 1🟡 1🔴
```

The campfire flames (`)(\/`) cycle through 3 frames of slightly different arrangements on each rail refresh, creating a subtle flicker effect. Use colours 166 (burnt orange), 208 (fire orange), 214 (gold), 220 (bright yellow) for the flames, cycling which characters get which color each frame.

**Step 2:** Add a "last roundup" timestamp at the bottom of the rail:
```
\033[2m  last roundup: 14:32:07\033[0m
```

This shows when health was last checked, in dim text.

**Step 3:** Add color transitions for pane entries. Instead of snapping from green to yellow, interpolate through amber (colour 172) for 1-2 refresh cycles:
- Green → Yellow transition: show colour 34 → 172 → 220 over 2 cycles
- Yellow → Red transition: show colour 220 → 208 → 196 over 2 cycles
- Any → Green recovery: show colour 34 immediately (recovery should feel instant and reassuring)

**Acceptance Criteria:**
- [x] Campfire ASCII art flickers subtly in the header
- [x] "RANCH BOARD" replaces "STATUS RAIL"
- [x] Last roundup timestamp shown at bottom
- [x] Health color transitions animate over 2 cycles instead of snapping

---

## Phase 3: Startup Roundup Sequence

**Files:**
- Modify: `scripts/agent_wrangler.py` — `run_up()` function (lines 1500-1586)

### Task 5: Roundup Discovery Animation

After the banner, show each pane being discovered with a lasso-roping animation.

**Step 1:** After grid creation but before attaching, add a roundup display. For each project pane that was created or imported:

```
  Rounding up the herd...

  ◦ ─ ─ ● creator-studio        wrangled ✓
  ◦ ─ ─ ● gabooja-labs           wrangled ✓
  ◦ ─ ─ ● gabooja-knowledge-base wrangled ✓
  ◦ ─ ● agent-wrangler           wrangled ✓
```

**Animation per line (0.3s each):**
1. Show `◦` (dim, hollow dot) — the pane is being discovered
2. Animate the lasso: `◦ ─` then `◦ ─ ─` then `◦ ─ ─ ●` — the rope extends and catches the project
3. Project name appears in rust (colour 130)
4. `wrangled ✓` appears in green (colour 34)

**Step 2:** The dots connecting to the project name are coloured 240 (dim gray), creating a visual "rope" from the wrangler to the project.

**Step 3:** After all projects are wrangled, show a summary line:
```
  \033[32m✓\033[0m All 6 head accounted for. Let's ride.\033[0m
```

**Step 4:** Play a completion sound:
```python
subprocess.Popen(['afplay', '-v', '0.4', '/System/Library/Sounds/Bottle.aiff'],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
```

`Bottle.aiff` (cork pop) at medium volume — a satisfying "done" sound.

**Step 5:** Add `AW_SKIP_ANIM` check. If set, skip the per-line animation and just print the final state of all lines at once.

**Acceptance Criteria:**
- [x] Each project appears with a lasso rope animation
- [x] Projects show in rust color, confirmation in green
- [x] Summary line after all projects wrangled
- [x] Cork pop sound on completion
- [x] Skippable with `AW_SKIP_ANIM=1`
- [x] Total roundup time: ~0.3s × N projects (scales with project count)

---

## Phase 4: Ranch-Branded Pane Borders

**Files:**
- Modify: `scripts/agent_wrangler.py` — `set_window_orchestrator_format()` (lines 650-681) and `style_for_level()` (lines 637-647)

### Task 6: Branded Border Format

Update the tmux pane border format string to include ranch character.

**Step 1:** Update the border format from:
```
▶ ● project-name (reason)
```
to:
```
▶ ● project-name · grazing
```

The status word replaces the technical reason in most cases:
- Green active: "grazing" in colour 34
- Yellow waiting: "at fence" in colour 220
- Yellow waiting > 3min: "at fence 3m" with the time in colour 220
- Red error: the original error reason stays (it's useful), but prefix with "down:" in colour 196

**Step 2:** Update border colors to use warm tones for inactive borders:
- Green inactive: `fg=colour22` → keep (dark green is fine, earthy)
- Yellow inactive: `fg=colour136` → `fg=colour130` (rust — warmer, fits ranch theme)
- Red inactive: `fg=colour88` → keep (dark red is dramatic and appropriate)
- Active border: `fg=colour214,bold` (warm gold instead of cold white — makes the active pane feel "lit up" like it's near the campfire)

**Step 3:** Update the status bar tabs (built in the health refresh loop) to use the same ranch terminology. Instead of just dots, show:
```
● creator-studio │ ⚑ gabooja-labs │ ...
```
Change to:
```
● creator-studio │ ● gabooja-labs │ ...
```
All use filled dots `●` with health colors (matching the rail change in Task 2).

**Acceptance Criteria:**
- [x] Pane borders show ranch-themed status words
- [x] Active border uses warm gold instead of cold white
- [x] Yellow inactive borders use rust color
- [x] Status bar tabs use uniform `●` dots

---

## Phase 5: Themed Context Menu

**Files:**
- Modify: `scripts/agent_wrangler.py` — `_build_context_menu_cmd()` function

### Task 7: Ranch-Themed Right-Click Menu

Update the context menu labels to use cowboy language while keeping them clear.

**Step 1:** Change menu labels:

| Current | Ranch Themed |
|---------|-------------|
| Zoom | Zoom In |
| View Output | Check Output |
| Send Command... | Send Command... |
| Start Claude | Saddle Up Claude |
| Start Codex | Saddle Up Codex |
| Start Aider | Saddle Up Aider |
| Restart | Round Up Again |
| Stop (Ctrl-C) | Whoa (Ctrl-C) |
| Kill | Put Down |

**Step 2:** Add a branded header line. The first item in `display-menu` is the pane title. Enhance it:
- Current: `#{pane_title}`
- New: `🤠 #{pane_title}` or `⟨ #{pane_title} ⟩` (test emoji support in tmux display-menu first — if emoji breaks width, use the bracket variant)

**Step 3:** Add a health indicator in the menu header showing current state:
```
⟨ creator-studio · ● grazing ⟩
```

**Acceptance Criteria:**
- [x] Menu labels use cowboy terminology
- [x] Header shows project name with health status
- [x] All menu actions still work correctly (only labels change)

---

## Phase 6: Ops Console Glow-Up

**Files:**
- Modify: `scripts/agent_wrangler.py` — `run_ops()` function (lines 3066-3120)

### Task 8: Ranch Operations Board

Transform the plain numbered menu into a visually branded console.

**Step 1:** Add a box-drawn header with mini cowboy hat:

```
╭─ Ranch Operations ──────────────────╮
│     /\                              │
│    /  \   Agent Wrangler v1.0       │
│   /    \  6 head · 4🟢 1🟡 1🔴       │
│  '──────'                           │
╰─────────────────────────────────────╯
```

Use colours: 130 (rust) for the hat, 250 (light gray) for the box borders, default for text inside.

**Step 2:** Show a quick health summary before the menu options. Call `refresh_pane_health()` to get current state, then display:
```
  Herd:  ● creator-studio  ● gabooja-labs  ⚑ knowledge-base
```

One line, compact, using colored dots.

**Step 3:** Style the menu with ANSI colors:
```
  \033[38;5;130m 1.\033[0m  Start all (import + grid + manager)
  \033[38;5;130m 2.\033[0m  Attach grid session
  \033[38;5;130m 3.\033[0m  Show herd status
  \033[38;5;130m 4.\033[0m  Focus pane by name
  \033[38;5;130m 5.\033[0m  Send command to pane
  \033[38;5;130m 6.\033[0m  Saddle up agent
  \033[38;5;130m 7.\033[0m  Whoa (stop pane)
  \033[38;5;130m 8.\033[0m  Open manager window
  \033[38;5;130m 9.\033[0m  Doctor (check attention)
```

Number labels in rust, descriptions in default color.

**Step 4:** Style the prompt: `\033[38;5;208mranch>\033[0m ` (amber prompt instead of plain `ops> `)

**Acceptance Criteria:**
- [x] Box-drawn header with hat ASCII and health summary
- [x] Menu numbers in rust color
- [x] Amber-colored `ranch>` prompt
- [x] Health summary line before menu options

---

## Phase 7: Grid Window Status Bar Brand

**Files:**
- Modify: `scripts/agent_wrangler.py` — `set_window_orchestrator_format()` (status bar section)

### Task 9: Ranch-Branded Status Bar

Update the tmux session status bar to show a ranch-branded herd tally.

**Step 1:** Update the left side of the status bar:
- Current: `#[fg=colour39 bold]AW #[fg=colour130]#{session_name}`
- New: `#[fg=colour130 bold]AW #[fg=colour172]⟨ #[fg=colour250]#{session_name} #[fg=colour172]⟩`

Replace the cyan `AW` with rust bold, add warm amber angle brackets around the session name.

**Step 2:** Update the dynamic status-right tabs to include a herd tally:
```
#[fg=colour172]⟨ 6 head #[fg=colour34]4● #[fg=colour220]1● #[fg=colour196]1● #[fg=colour172]⟩  %H:%M
```

This replaces the per-project dot list with a compact tally when there are many panes (> 6). When <= 6 panes, keep the per-project dots.

**Step 3:** When a pane is zoomed, update the ZOOM indicator:
- Current: `#[fg=colour214 bold]ZOOM`
- New: `#[fg=colour208 bold]🔭 ZOOMED` or `#[fg=colour208 bold]◎ ZOOMED` (test emoji first, use `◎` fallback)

**Acceptance Criteria:**
- [x] Status bar uses warm rust/amber branding
- [x] Herd tally shows compact count for large grids
- [x] Zoom indicator updated with ranch flair
- [x] Time display preserved

---

## Phase 8: Sound Effects

**Files:**
- Modify: `scripts/agent_wrangler.py` — add `play_sound()` helper, integrate into key functions

### Task 10: Campfire Sound System

Add optional, non-blocking sound effects for key events.

**Step 1:** Add a sound helper function near the top of `agent_wrangler.py`:

```python
def play_sound(name: str, volume: float = 0.5) -> None:
    """Play a macOS system sound non-blocking. Silently does nothing if unavailable."""
    if not SOUNDS_ENABLED:
        return
    path = Path(f'/System/Library/Sounds/{name}.aiff')
    if path.exists():
        try:
            subprocess.Popen(
                ['afplay', '-v', str(volume), str(path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            pass
```

**Step 2:** Add a `SOUNDS_ENABLED` flag, controlled by:
- `AW_SOUNDS=1` environment variable (opt-in, off by default)
- `"sounds": true` in `config/team_grid.json`

**Step 3:** Wire up sounds to events:

| Event | Sound | Volume | Rationale |
|-------|-------|--------|-----------|
| Agent started in pane | `Morse.aiff` | 0.3 | Subtle morse tap — "transmission started" |
| Agent finished/returned to prompt | `Tink.aiff` | 0.3 | Light tap — "done" |
| Health turned red | `Basso.aiff` | 0.4 | Deep warning tone |
| Health recovered to green | `Bottle.aiff` | 0.3 | Cork pop — celebration |
| Session startup complete | `Blow.aiff` | 0.3 | (Already in banner) |
| Session exit | `Submarine.aiff` | 0.3 | Fading sonar — "going dark" |

**Step 4:** Respect debounce — sounds should not fire more often than every 10 seconds per pane. Use the existing notification cooldown pattern.

**Acceptance Criteria:**
- [x] `play_sound()` helper is non-blocking and fail-safe
- [x] Sounds are off by default, opt-in via env var or config
- [x] Each event triggers the appropriate sound at low volume
- [x] Debounce prevents sound spam
- [x] Works silently on non-macOS (no errors, just no sound)

---

## Implementation Order

The phases are designed to be independent and can be done in any order, but this sequence minimizes conflicts:

1. **Phase 1** (Banner) — standalone bash file, zero risk to other features
2. **Phase 8** (Sounds) — adds `play_sound()` helper that other phases can use
3. **Phase 2** (Rail) — biggest visual impact, self-contained in `run_rail()`
4. **Phase 4** (Pane Borders) — updates format strings, small blast radius
5. **Phase 7** (Status Bar) — updates status bar config, adjacent to Phase 4
6. **Phase 5** (Context Menu) — label changes only, very safe
7. **Phase 6** (Ops Console) — cosmetic changes to `run_ops()`
8. **Phase 3** (Roundup Sequence) — integrates with startup flow, do last

Each phase should be committed separately so individual features can be reverted if the user doesn't like them.

---

## Testing Approach

No test suite exists. Verify each phase manually:

1. **Banner:** Run `bash scripts/welcome_banner.sh` directly. Test skip with `AW_SKIP_ANIM=1 bash scripts/welcome_banner.sh`. Test Ctrl-C cleanup.
2. **Rail:** Run `./scripts/agent-wrangler rail` and watch for visual correctness. Check alignment at different terminal widths.
3. **Borders:** Run `./scripts/agent-wrangler start` and inspect pane borders visually. Check all three health states.
4. **Menu:** Right-click a pane in the grid window. Verify all menu items work.
5. **Sounds:** Set `AW_SOUNDS=1` and trigger events. Verify volume is subtle, not jarring.
6. **Startup:** Run `./scripts/agent-wrangler start` end-to-end. Verify the full sequence: banner → roundup → grid → manager.

---

## Rollback Strategy

Every feature is cosmetic — no data model changes, no config format changes, no API changes. If any feature doesn't feel right:

- Banner: Revert `welcome_banner.sh` to previous commit
- Rail: The old rendering is a simple format change within `run_rail()`
- Borders: Revert format strings in `set_window_orchestrator_format()`
- Sounds: Set `AW_SOUNDS=0` or remove the `play_sound()` calls
- Everything: `git checkout HEAD~N -- scripts/` reverts all script changes
