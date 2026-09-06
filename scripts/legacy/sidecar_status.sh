#!/bin/bash
# Sidecar health, usable from inside acg_persist (NetworkMode=host, so
# 127.0.0.1:810K is the same loopback the sidecar containers publish on).
# Starting/stopping works from inside too: acg_persist carries
# /var/run/docker.sock and a bind-mounted host docker binary.
for K in 1 2 3 4 5; do
  R=$(curl -s -m 2 "http://127.0.0.1:810$K/v1/models" 2>/dev/null)
  if [ -n "$R" ]; then
    echo "side$K  UP    serving: $(echo "$R" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)"
  else
    echo "side$K  DOWN"
  fi
done
S=/DATA/tracy/agentic-context-grpo/experiments/08-27/results
for f in "$S"/*/sidecar_state.json; do
  [ -f "$f" ] && echo "state: $f -> $(cat "$f")"
done
