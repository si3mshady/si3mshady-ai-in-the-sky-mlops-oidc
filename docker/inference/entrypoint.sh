#!/usr/bin/env bash
set -euo pipefail

cmd="${1:-serve}"

if [[ "$cmd" == "serve" ]]; then
  # Start FastAPI on 8080 for SageMaker
  exec python -m uvicorn serve:app --host 0.0.0.0 --port 8080
fi

# Allow custom commands if explicitly provided
exec "$@"

