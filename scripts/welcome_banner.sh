#!/usr/bin/env bash
# welcome_banner.sh - Campfire warmth animated welcome banner for Agent Wrangler
# Warmth spreads from the cowboy hat outward, like embers catching fire.

set -euo pipefail

# --- ASCII art lines (no leading blank line) ---
ART=(
"   █████   ██████  ███████ ███    ██ ████████            ▄███▄        "
"  ██   ██ ██       ██      ████   ██    ██             ▄███████▄      "
"  ███████ ██   ███ █████   ██ ██  ██    ██       ▄▀▀▀▀███████████▀▀▀▀▄"
"  ██   ██ ██    ██ ██      ██  ██ ██    ██        ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀"
"  ██   ██  ██████  ███████ ██   ████    ██                            "
"                                                                      "
"  ██     ██ ██████   █████  ███    ██  ██████  ██      ███████ ██████ "
"  ██     ██ ██   ██ ██   ██ ████   ██ ██       ██      ██      ██   ██"
"  ██  █  ██ ██████  ███████ ██ ██  ██ ██   ███ ██      █████   ██████ "
"  ██ ███ ██ ██   ██ ██   ██ ██  ██ ██ ██    ██ ██      ██      ██   ██"
"   ███ ███  ██   ██ ██   ██ ██   ████  ██████  ███████ ███████ ██   ██"
)

TAGLINE="  Steering agents. Wrangling terminals. Shipping code."
NAV_HINTS="  ⌥m manager  ⌥g grid  ⌥z zoom  ⌥n/p cycle  ⌥j jump"

# --- Warm gradient: cold/dark → rust → amber → gold ---
# 256-color codes from near-invisible to warm campfire glow
GRADIENT=(232 233 234 236 238 240 95 130 166 172 208 214 220)
GRADIENT_LEN=${#GRADIENT[@]}
FRONTIER_COLOR=229   # bright yellow-white flash at the spreading edge
FINAL_COLOR=130      # rust — the resting warm state

# --- Center of the cowboy hat (the origin of warmth) ---
# Row 0 is "AGENT" line; the hat ▄███▄ is at row 0, ~col 56
CENTER_ROW=1
CENTER_COL=56

# Total animation frames and timing
TOTAL_FRAMES=22
FRAME_DELAY=0.055  # ~18fps, total ~1.2 seconds

# --- Helper: print the final (fully warm) state ---
print_final() {
    printf '\033[38;5;%dm' "$FINAL_COLOR"
    for line in "${ART[@]}"; do
        printf '%s\n' "$line"
    done
    printf '\033[0m\n'
    printf '\033[2m%s\033[0m\n' "$TAGLINE"
    printf '\033[2m%s\033[0m\n\n' "$NAV_HINTS"
}

# --- Fast path: skip animation ---
if [[ ! -t 1 ]] || [[ -n "${AW_SKIP_ANIM:-}" ]]; then
    print_final
    exit 0
fi

# --- Cleanup: always restore cursor and terminal state ---
OLD_STTY=""
cleanup() {
    printf '\033[?25h\033[0m'  # show cursor, reset colors
    [[ -n "$OLD_STTY" ]] && stty "$OLD_STTY" 2>/dev/null
}
trap cleanup EXIT INT TERM

# --- Play subtle startup sound (non-blocking) ---
if [[ -f /System/Library/Sounds/Blow.aiff ]]; then
    afplay -v 0.25 /System/Library/Sounds/Blow.aiff &
    disown 2>/dev/null
fi

# --- Pre-compute art dimensions ---
HEIGHT=${#ART[@]}
WIDTH=0
for line in "${ART[@]}"; do
    (( ${#line} > WIDTH )) && WIDTH=${#line}
done

# --- Pre-compute max distance for radius scaling ---
MAX_DIST=0
for ((y = 0; y < HEIGHT; y++)); do
    for ((x = 0; x < WIDTH; x++)); do
        dy=$(( (y - CENTER_ROW) * 3 ))  # aspect ratio compensation
        dx=$(( x - CENTER_COL ))
        # Manhattan distance (fast, no bc needed)
        dist=$(( ${dy#-} + ${dx#-} ))
        (( dist > MAX_DIST )) && MAX_DIST=$dist
    done
done

# --- Set up non-blocking keypress detection for skip ---
OLD_STTY=$(stty -g 2>/dev/null || true)
stty -echo -icanon min 0 time 0 2>/dev/null || true

# --- Hide cursor ---
printf '\033[?25l'

# --- Print blank lines to reserve space ---
printf '\n%.0s' $(seq 1 $((HEIGHT + 3)))

# --- Animation loop: warmth spreading from center ---
SKIPPED=0
for ((frame = 0; frame <= TOTAL_FRAMES; frame++)); do
    # Check for keypress to skip
    if read -rsn1 -t 0 _key 2>/dev/null; then
        SKIPPED=1
        break
    fi

    # Ease-out cubic: fast start, slow finish
    # Approximate with integer math: radius = (frame^2 * (3*total - 2*frame)) * max / total^3
    # Simplified: use a lookup that grows fast then slows
    if (( TOTAL_FRAMES > 0 )); then
        # Linear with slight overshoot to ensure full coverage
        radius=$(( (frame * (MAX_DIST + 20)) / TOTAL_FRAMES ))
    fi

    # Move cursor to start of art area
    printf '\033[%dA' "$((HEIGHT + 3))"

    buf=""
    for ((y = 0; y < HEIGHT; y++)); do
        line="${ART[$y]}"
        line_buf=""
        last_color=""
        for ((x = 0; x < ${#line}; x++)); do
            ch="${line:$x:1}"

            if [[ "$ch" == " " ]]; then
                line_buf+=" "
                last_color=""
                continue
            fi

            dy=$(( (y - CENTER_ROW) * 3 ))
            dx=$(( x - CENTER_COL ))
            dist=$(( ${dy#-} + ${dx#-} ))

            if (( dist > radius )); then
                # Outside warmth radius: near-invisible
                color=232
            elif (( dist > radius - 3 )); then
                # Frontier: bright flash
                color=$FRONTIER_COLOR
            else
                # Inside warmth: map depth to gradient
                depth=$(( radius - dist ))
                if (( MAX_DIST > 0 )); then
                    idx=$(( depth * (GRADIENT_LEN - 1) / (MAX_DIST + 1) ))
                    (( idx >= GRADIENT_LEN )) && idx=$((GRADIENT_LEN - 1))
                else
                    idx=$((GRADIENT_LEN - 1))
                fi
                color=${GRADIENT[$idx]}
            fi

            # Only emit color escape when color changes (much faster)
            if [[ "$color" != "$last_color" ]]; then
                line_buf+="\033[38;5;${color}m"
                last_color="$color"
            fi
            line_buf+="$ch"
        done
        buf+="${line_buf}\033[0m\n"
    done
    # Blank lines for tagline/hints area (empty during art animation)
    buf+="\n\n\n"

    printf '%b' "$buf"
    sleep "$FRAME_DELAY"
done

# --- Restore terminal for typing ---
stty "$OLD_STTY" 2>/dev/null || true
OLD_STTY=""

# --- Final state: full warm art ---
printf '\033[%dA' "$((HEIGHT + 3))"
print_final

# --- Gentle hat pulse: embers settling ---
# Pulse the cowboy hat area (rows 0-1, cols ~52-66) through warm tones
if (( SKIPPED == 0 )); then
    PULSE_COLORS=(172 208 214 208 172 130)
    for pc in "${PULSE_COLORS[@]}"; do
        # Check for skip
        if read -rsn1 -t 0 _key 2>/dev/null; then
            break
        fi
        # Move to hat position (row 1, col 53 — the ▄███▄ area)
        printf '\033[%dA' "$((HEIGHT + 2))"  # go to row 0
        # Print row 0 with hat highlighted
        line="${ART[0]}"
        printf '\033[38;5;%dm%s' "$FINAL_COLOR" "${line:0:52}"
        printf '\033[38;5;%dm%s' "$pc" "${line:52:14}"
        printf '\033[38;5;%dm%s\033[0m\n' "$FINAL_COLOR" "${line:66}"
        # Print row 1 with hat highlighted
        line="${ART[1]}"
        printf '\033[38;5;%dm%s' "$FINAL_COLOR" "${line:0:52}"
        printf '\033[38;5;%dm%s' "$pc" "${line:52:16}"
        printf '\033[38;5;%dm%s\033[0m\n' "$FINAL_COLOR" "${line:68}"
        # Move cursor back down to bottom
        printf '\033[%dB' "$((HEIGHT))"
        sleep 0.06
    done
fi

# --- Show cursor ---
printf '\033[?25h'
