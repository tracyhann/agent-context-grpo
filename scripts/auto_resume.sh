#!/bin/bash
# Crash supervisor for the verl run: if the trainer dies inside acg_persist,
# rotate the log and exec-relaunch (verl resume_mode=auto continues from the
# newest checkpoint). Survives Claude session restarts (plain setsid process).
# Guards: max 25 auto-resumes (raised: crashes are a known RAM-cap cause, not a mystery), 10-min backoff, stops if no checkpoint exists.
cd /DATA/tracy/agentic-context-grpo
TAG=${1:-verl_ccpo_alfworld}
# The ESTIMATOR must be passed too. It used to be hardcoded to `ccpo` in the
# relaunch line below while only the tag was parameterised, so the GRPO baseline
# silently became a CCPO run at its first auto-resume (step 8) and every step
# after that was the wrong arm. Default it from the tag so a mismatch is loud.
case "$TAG" in *grpo*) EST=${2:-grpo} ;; *) EST=${2:-ccpo} ;; esac
echo "$(date '+%m-%d %H:%M') supervisor up: TAG=$TAG EST=$EST" >> experiments/08-27/logs/auto_resume.log
N=0
while [ $N -lt 25 ]; do
  sleep 300
  docker ps --format '{{.Names}}' | grep -q '^acg_persist$' || { echo "$(date '+%m-%d %H:%M') persist container gone — supervisor exiting" >> experiments/08-27/logs/auto_resume.log; exit 1; }
  if docker exec acg_persist pgrep -f main_ppo > /dev/null 2>&1; then continue; fi
  # trainer is down — only resume if we have a checkpoint to stand on
  [ -f experiments/08-27/results/$TAG/latest_checkpointed_iteration.txt ] || { echo "$(date '+%m-%d %H:%M') trainer dead, no checkpoint — not resuming" >> experiments/08-27/logs/auto_resume.log; exit 1; }
  N=$((N+1))
  last=$(cat experiments/08-27/results/$TAG/latest_checkpointed_iteration.txt)
  echo "$(date '+%m-%d %H:%M') trainer dead (tail: $(tail -1 experiments/08-27/logs/$TAG.log | cut -c1-80)) — auto-resume #$N from step $last" >> experiments/08-27/logs/auto_resume.log
  mv experiments/08-27/logs/$TAG.log experiments/08-27/logs/$TAG.crash$N.log 2>/dev/null
  bash experiments/08-27/run_exec_ccpo.sh "$EST" "$TAG" >> experiments/08-27/logs/auto_resume.log 2>&1
  sleep 600   # backoff: let spin-up finish before re-checking
done
echo "$(date '+%m-%d %H:%M') 6 auto-resumes exhausted — human needed" >> experiments/08-27/logs/auto_resume.log
