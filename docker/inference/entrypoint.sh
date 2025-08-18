#!/usr/bin/env bash
set -e
if [ "$1" = "serve" ] || [ -z "$1" ]; then
  exec python /opt/ml/code/serve.py
else
  exec "$@"
fi

