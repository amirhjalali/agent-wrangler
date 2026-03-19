#!/usr/bin/env bash
# welcome_banner.sh - Animated welcome banner for Agent Wrangler
# Line-by-line reveal with a cowboy hat tip.

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

FINAL_COLOR=130  # rust

# --- Hat shift helper: move hat portion by offset chars (+ right, − left) ---
hat_shift() {
    local line="$1" split="$2" offset="$3"
    local text="${line:0:$split}"
    local hat="${line:$split}"
    if (( offset > 0 )); then
        local pad=""
        for ((s=0; s<offset; s++)); do pad+=" "; done
        printf '%s' "${text}${pad}${hat}"
    elif (( offset < 0 )); then
        local abs=$(( -offset ))
        printf '%s' "${text}${hat:$abs}"
    else
        printf '%s' "$line"
    fi
}

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
    return 0
}
trap cleanup EXIT INT TERM

# --- Play subtle startup sound (non-blocking) ---
if [[ -f /System/Library/Sounds/Blow.aiff ]]; then
    afplay -v 0.25 /System/Library/Sounds/Blow.aiff &
    disown 2>/dev/null
fi

HEIGHT=${#ART[@]}

# --- Set up non-blocking keypress detection for skip ---
OLD_STTY=$(stty -g 2>/dev/null || true)
stty -echo -icanon min 0 time 0 2>/dev/null || true

printf '\033[?25l'  # hide cursor

# --- Phase 1: Line-by-line reveal ---
SKIPPED=0
printf '\033[38;5;%dm' "$FINAL_COLOR"
for ((i = 0; i < HEIGHT; i++)); do
    if read -rsn1 -t 0 _key 2>/dev/null; then
        SKIPPED=1
        for ((j = i; j < HEIGHT; j++)); do
            printf '%s\n' "${ART[$j]}"
        done
        break
    fi
    printf '%s\n' "${ART[$i]}"
    sleep 0.04
done
printf '\033[0m\n'
printf '\033[2m%s\033[0m\n' "$TAGLINE"
printf '\033[2m%s\033[0m\n\n' "$NAV_HINTS"

# --- Phase 2: Hat sway — slow horse-riding strut ---
# The cowboy hat sways gently left and right like a rider on a slow horse.
if (( SKIPPED == 0 )); then
    sleep 0.3  # pause before the strut begins

    # Smooth sine-wave sway offsets (2 full cycles)
    SWAY=(1 2 1 0 -1 -2 -1 0 1 2 1 0 -1 -2 -1 0)

    # Column where hat region starts per row (rows 0–3 carry hat art)
    S0=45; S1=42; S2=42; S3=42

    # Cursor math: art(HEIGHT) + blank + tagline + hints + blank = HEIGHT+4
    UP=$((HEIGHT + 4))

    for offset in "${SWAY[@]}"; do
        if read -rsn1 -t 0 _key 2>/dev/null; then
            # Snap to rest position on skip
            printf '\033[%dA' "$UP"
            printf '\033[38;5;%dm' "$FINAL_COLOR"
            for ((r=0; r<4; r++)); do printf '%s\033[K\n' "${ART[$r]}"; done
            printf '\033[0m\033[%dB' "$((UP - 4))"
            break
        fi

        printf '\033[%dA' "$UP"
        printf '\033[38;5;%dm' "$FINAL_COLOR"
        printf '%s\033[K\n' "$(hat_shift "${ART[0]}" $S0 "$offset")"
        printf '%s\033[K\n' "$(hat_shift "${ART[1]}" $S1 "$offset")"
        printf '%s\033[K\n' "$(hat_shift "${ART[2]}" $S2 "$offset")"
        printf '%s\033[K\n' "$(hat_shift "${ART[3]}" $S3 "$offset")"
        printf '\033[0m\033[%dB' "$((UP - 4))"
        sleep 0.13
    done
fi

# --- Restore terminal ---
stty "$OLD_STTY" 2>/dev/null || true
OLD_STTY=""
printf '\033[?25h'  # show cursor
