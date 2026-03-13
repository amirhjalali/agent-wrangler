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

# --- Hat tip helper: shift hat portion of a line right ---
hat_tip() {
    local line="$1" split="$2" shift_n="$3"
    local text="${line:0:$split}"
    local hat="${line:$split}"
    local pad=""
    for ((s=0; s<shift_n; s++)); do pad+=" "; done
    local shifted="${text}${pad}${hat}"
    # Trim to original length
    printf '%s' "${shifted:0:${#line}}"
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

# --- Phase 2: Hat tip animation ---
# The cowboy hat tips right (like a greeting) then settles back.
if (( SKIPPED == 0 )); then
    sleep 0.25  # brief pause before the tip

    # Column where hat area begins on each row
    SPLIT0=45  # row 0: after AGENT text
    SPLIT1=42  # row 1: after A text

    # Pre-generate tipped lines
    ROW0_T1=$(hat_tip "${ART[0]}" $SPLIT0 1)
    ROW0_T2=$(hat_tip "${ART[0]}" $SPLIT0 2)
    ROW1_T1=$(hat_tip "${ART[1]}" $SPLIT1 1)

    # Cursor math: go from bottom to row 0 of art
    # Art (HEIGHT lines) + 1 reset newline + 1 tagline + 1 hints + 1 extra newline = HEIGHT+4
    UP=$((HEIGHT + 4))

    for tip_frame in 1 2 3 4; do
        if read -rsn1 -t 0 _key 2>/dev/null; then
            # Snap hat back to final position on skip
            printf '\033[%dA' "$UP"
            printf '\033[38;5;%dm%s\033[0m\n' "$FINAL_COLOR" "${ART[0]}"
            printf '\033[38;5;%dm%s\033[0m\n' "$FINAL_COLOR" "${ART[1]}"
            printf '\033[%dB' "$((UP - 2))"
            break
        fi

        printf '\033[%dA' "$UP"
        case $tip_frame in
            1)  # Start tipping right — crown shifts 1
                printf '\033[38;5;%dm%s\033[0m\n' "$FINAL_COLOR" "$ROW0_T1"
                printf '\033[38;5;%dm%s\033[0m\n' "$FINAL_COLOR" "${ART[1]}"
                ;;
            2)  # Full tip — crown shifts 2, body shifts 1
                printf '\033[38;5;%dm%s\033[0m\n' "$FINAL_COLOR" "$ROW0_T2"
                printf '\033[38;5;%dm%s\033[0m\n' "$FINAL_COLOR" "$ROW1_T1"
                ;;
            3)  # Returning — crown shifts 1
                printf '\033[38;5;%dm%s\033[0m\n' "$FINAL_COLOR" "$ROW0_T1"
                printf '\033[38;5;%dm%s\033[0m\n' "$FINAL_COLOR" "${ART[1]}"
                ;;
            4)  # Settled — back to original
                printf '\033[38;5;%dm%s\033[0m\n' "$FINAL_COLOR" "${ART[0]}"
                printf '\033[38;5;%dm%s\033[0m\n' "$FINAL_COLOR" "${ART[1]}"
                ;;
        esac
        printf '\033[%dB' "$((UP - 2))"
        sleep 0.1
    done
fi

# --- Restore terminal ---
stty "$OLD_STTY" 2>/dev/null || true
OLD_STTY=""
printf '\033[?25h'  # show cursor
