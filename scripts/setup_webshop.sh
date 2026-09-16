#!/usr/bin/env bash
# Provision the separate WebShop runtime inside this project.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$ROOT/scripts/setup_webshop.py" "$@"
