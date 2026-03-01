#!/usr/bin/env bash
# welcome_banner.sh - Cowboy-themed welcome banner for Agent Wrangler

CYAN='\033[0;36m'
DIM='\033[2m'
RESET='\033[0m'

printf "${CYAN}"
cat <<'BANNER'

      .-.
     (o.o)
      |=|          _                    _
     __|__        / \   __ _  ___ _ __ | |_
   //.=|=.\\     / _ \ / _` |/ _ \ '_ \| __|
  // .=|=. \\   / ___ \ (_| |  __/ | | | |_
  \\ .=|=. // /_/   \_\__, |\___|_| |_|\__|
   \\(_=_)//          |___/
    (:| |:)
     || ||  __        __                    _
     () ()  \ \      / / __ __ _ _ __   ___| | ___ _ __
     || ||   \ \ /\ / / '__/ _` | '_ \ / _` | / _ \ '__|
     || ||    \ V  V /| | | (_| | | | | (_| | |  __/ |
    ==' '==    \_/\_/ |_|  \__,_|_| |_|\__, |_|\___|_|
                                        |___/
BANNER
printf "${RESET}"
printf "${DIM}        Steering agents. Wrangling terminals. Shipping code.${RESET}\n\n"
