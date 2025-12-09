FROM python:3.11.9-slim-bookworm

# avoid creating unnecessary .pyc, buffers, and pip caches
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# install curl and certificates needed to install uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
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
COPY --chown=user predicting_outcomes_in_heart_failure ./predicting_outcomes_in_heart_failure
COPY --chown=user models/nosex/random_forest.joblib ./models/nosex/random_forest.joblib
COPY --chown=user reports/nosex/random_forest/cv_parameters.json ./reports/nosex/random_forest/cv_parameters.json
COPY --chown=user data/interim/preprocess_artifacts/scaler.joblib ./data/interim/preprocess_artifacts/scaler.joblib
COPY --chown=user metrics/test/nosex/random_forest.json ./metrics/test/nosex/random_forest.json
COPY --chown=user README.md ./README.md
COPY --chown=user models/README.md ./models/README.md
COPY --chown=user data/README.md ./data/README.md


EXPOSE 7860

CMD ["uv", "run", "uvicorn", "predicting_outcomes_in_heart_failure.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
