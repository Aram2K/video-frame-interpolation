#!/bin/bash
# One-shot overview of the movie_vfi pipeline: queue, GPU, progress, finished runs, failures.
#   scripts/status.sh            # once
#   watch -n 30 scripts/status.sh   # refresh every 30 s (Ctrl-C to stop)
# Light read-only commands only: safe to run on the login node.
R=${MVFI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}
SHOT=${SHOT:-test01}
USER=${USER:-$(id -un)}          # not set when run as: ssh host 'script'
PY=$(command -v python3 || echo ${MVFI_CONDA_BASE:-$HOME/miniconda3}/bin/python3)

echo "=============== $(date '+%H:%M:%S')  movie_vfi status (shot: $SHOT)"
echo "--- my jobs in the queue"
squeue -u "$USER" -o "%.8i %.22j %.9T %.10M %.11L %.20R" | sed 's/^/  /'
echo "  (State: R=running, PD=pending. Reason 'Dependency' = waiting for the job before it.)"

echo "--- GPU of the running job (from the job's own 5-second log, no extra load)"
for id in $(squeue -u "$USER" -h -t RUNNING -o "%i"); do
  csv=$(ls -t "$R"/logs/gpu_usage_*"${id}"*.csv "$R"/output/*/*/gpu_usage_*"${id}"*.csv 2>/dev/null | head -1)
  name=$(squeue -h -j "$id" -o "%j")
  if [ -n "$csv" ]; then
    printf "  job %s (%s): %s\n" "$id" "$name" "$(tail -1 "$csv")"
    awk -F', ' 'NR>1{gsub(/ MiB/,"",$3); if($3+0>m)m=$3+0} END{if(m)printf "    peak so far: %.1f GiB\n", m/1024}' "$csv"
  else
    printf "  job %s (%s): no GPU log yet\n" "$id" "$name"
  fi
done
[ -z "$(squeue -u "$USER" -h -t RUNNING -o '%i')" ] && echo "  (nothing running)"

echo "--- progress of the running job (last lines of its log)"
for id in $(squeue -u "$USER" -h -t RUNNING -o "%i"); do
  log=$(ls -t "$R"/logs/*_"${id}".log 2>/dev/null | head -1)
  [ -n "$log" ] && { echo "  $(basename "$log"):"; tr '\r' '\n' < "$log" | grep -vE "^\s*$" | tail -3 | cut -c1-150 | sed 's/^/    /'; }
done

echo "--- finished runs for $SHOT (frames written / expected)"
for rep in "$R"/output/$SHOT/*/*/logs/run_report.json "$R"/output/$SHOT/*/*/*/logs/run_report.json; do
  [ -e "$rep" ] || continue
  d=$(dirname "$(dirname "$rep")")
  "$PY" - "$rep" "$d" <<'PY' 2>/dev/null
import json,sys,glob,os
rep=json.load(open(sys.argv[1])); d=sys.argv[2]
n=len(glob.glob(os.path.join(d,"frame_*.tif")))
t=rep.get("timing_s",{}).get("generation_and_write",0)/60
g=(rep.get("gpu") or {}).get("max_memory_allocated_gib")
print(f"  {os.path.relpath(d, os.path.dirname(os.path.dirname(os.path.dirname(d))))}: {n}/{rep['output_frames']} frames, {t:.1f} min, peak {g:.1f} GiB" if g else
      f"  {os.path.relpath(d, os.path.dirname(os.path.dirname(os.path.dirname(d))))}: {n}/{rep['output_frames']} frames, {t:.1f} min")
PY
done
echo "--- in-progress output folders (no report yet)"
for d in "$R"/output/$SHOT/*/*/ "$R"/output/$SHOT/*/*/*/; do
  [ -d "$d" ] || continue
  [ -e "$d/logs/run_report.json" ] && continue
  n=$(ls "$d"frame_*.tif 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] && echo "  $(basename "$(dirname "$d")")/$(basename "$d"): $n frames so far"
done

echo "--- jobs that ended in the last 12 h (FAILED/CANCELLED first)"
sacct -u "$USER" -S "$(date -d '12 hours ago' +%Y-%m-%dT%H:%M)" -X -n -o JobID%10,JobName%24,State%14,Elapsed,MaxRSS 2>/dev/null \
  | grep -E "vfi_" | grep -vE "RUNNING|PENDING" | sort -k3 | tail -14 | sed 's/^/  /'
echo "=============== jobs on the gpu partition (all users): $(squeue -p gpu -h | wc -l)"
