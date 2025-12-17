#!/usr/bin/env sh
set -e

echo "[entrypoint] Checking DVC metadata..."

if [ -d ".dvc" ] || [ -f "dvc.yaml" ]; then
  echo "[entrypoint] Running dvc pull..."
  uv run dvc pull -v
else
  echo "[entrypoint] No DVC metadata found, skipping dvc pull."
fi

exec "$@"