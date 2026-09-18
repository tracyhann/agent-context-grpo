#!/bin/bash
# Preserve one checkpoint that a rotating trainer (max_actor_ckpt_to_keep) would delete.
# Waits until <ckpt_dir>/latest_checkpointed_iteration.txt reports >= <step> (the trainer
# writes it only AFTER the save completes, so the copy is never taken mid-write), then
# hard-links global_step_<step> to <name>. A hard-link copy costs no disk until the
# trainer rmtree()s its own copy, and survives that deletion.
#
# Usage: snapshot_ckpt.sh <ckpt_dir> <step> <name>      e.g. ... 100 step100-budget
set -u
C=$1; S=$2; N=$3
while :; do
  it=$(cat "$C/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
  if [ "${it:-0}" -ge "$S" ] 2>/dev/null; then
    if [ -d "$C/global_step_$S" ]; then
      cp -al "$C/global_step_$S" "$C/$N" && echo "$(date +%F' '%T) snapshot $C/global_step_$S -> $N" && exit 0
    fi
    echo "$(date +%F' '%T) MISSED: iteration $it but global_step_$S already gone"; exit 1
  fi
  sleep 120
done
