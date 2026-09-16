#!/bin/bash
# patches/verl-agent/ is the source of truth for every file we modify; verl-agent/
# is the runnable checkout (upstream + our overlay). Edit under patches/, then run
# this. Never edit verl-agent/ directly -- it is gitignored and gets overwritten.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cp -r "$ROOT/patches/verl-agent/." "$ROOT/verl-agent/"
echo "synced $(cd "$ROOT/patches/verl-agent" && find . -type f | wc -l) files -> verl-agent/"
