#!/usr/bin/env bash
# welcome_banner.sh - Cowboy-themed welcome banner for Agent Wrangler

RUST='\033[38;5;130m'
DIM='\033[2m'
RESET='\033[0m'

printf "${RUST}"
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
printf "${RESET}"
printf "${DIM}  Steering agents. Wrangling terminals. Shipping code.${RESET}\n"
printf "${DIM}  ⌥m manager  ⌥g grid  ⌥z zoom  ⌥n/p cycle  ⌥j jump${RESET}\n\n"
sleep 1
