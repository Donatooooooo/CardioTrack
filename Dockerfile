FROM python:3.11.9-slim-bookworm

# avoid creating unnecessary .pyc, buffers, and pip caches
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# install curl and certificates needed to install uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# create a non-root user for added security
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /cardioTrack

# install uv as user
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# copy the project files and do uv sync
COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project

# copy the rest of the files needed for inference
COPY --chown=user . .

# entrypoint
RUN printf '%s\n' \
  '#!/usr/bin/env sh' \
  'set -e' \
  '' \
  'echo "[entrypoint] Checking DVC metadata..."' \
  'if [ -d ".dvc" ] || [ -f "dvc.yaml" ]; then' \
  '  echo "[entrypoint] Running dvc pull..."' \
  '  uv run dvc pull -v' \
  'else' \
  '  echo "[entrypoint] No DVC metadata found, skipping dvc pull."' \
  'fi' \
  '' \
  'exec "$@"' \
  > /home/user/entrypoint.sh \
  && chmod +x /home/user/entrypoint.sh

EXPOSE 7860

ENTRYPOINT ["/home/user/entrypoint.sh"]

CMD ["uv", "run", "uvicorn", "predicting_outcomes_in_heart_failure.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
