#!/usr/bin/env bash
set -euo pipefail

CMD="${1:-serve}"

if [[ "$CMD" == "serve" ]]; then
  # start FastAPI + show access logs so you SEE inference logs in CWL
  exec uvicorn serve:app --host 0.0.0.0 --port 8080 --log-level info
fi

exec "$@"

