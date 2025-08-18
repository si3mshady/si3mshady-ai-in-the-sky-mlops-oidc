#!/usr/bin/env bash
set -euo pipefail

cmd="${1:-serve}"

if [[ "$cmd" == "serve" ]]; then
  exec python /opt/ml/code/app.py
else
  exec "$@"
fi

