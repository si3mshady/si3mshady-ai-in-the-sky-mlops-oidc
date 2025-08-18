#!/usr/bin/env bash
set -euo pipefail
CMD="${1:-serve}"
if [ "$CMD" = "serve" ]; then
  exec python /opt/ml/code/app.py
else
  exec "$@"
fi

