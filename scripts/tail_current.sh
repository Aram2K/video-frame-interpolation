#!/bin/bash
# Follow the log of my most recent job (Ctrl-C to stop). Pass -n to just print the last lines.
R=${MVFI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}
USER=${USER:-$(id -un)}
J=$(squeue -u "$USER" -h -t RUNNING -o "%i" | head -1)
LOG=$([ -n "$J" ] && ls -t "$R"/logs/*_"$J".log 2>/dev/null | head -1)
[ -z "$LOG" ] && LOG=$(ls -t "$R"/logs/vfi_*.log | head -1)
echo "== $LOG"
if [ "${1:-}" = "-n" ]; then tr '\r' '\n' < "$LOG" | grep -vE "^\s*$" | tail -20; else tail -f "$LOG" | tr '\r' '\n'; fi
