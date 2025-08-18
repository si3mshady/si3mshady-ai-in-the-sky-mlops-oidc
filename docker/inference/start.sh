#!/usr/bin/env bash
set -euo pipefail
cmd="${1:-serve}"
shift || true

# SageMaker runs:  docker run <image> serve
# Locally you can also do: docker run <image> serve
if [[ "$cmd" == "serve" ]]; then
  exec python /opt/ml/code/app.py
fi

# Anything else: exec directly (useful for debugging shells)
exec "$cmd" "$@"

