# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

RUN groupadd --gid 10001 pydsge \
    && useradd --no-log-init --uid 10001 --gid 10001 --create-home pydsge

FROM base AS builder

RUN python -m venv /opt/venv

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install .

FROM builder AS test

COPY tests ./tests
COPY configs ./configs
COPY scripts ./scripts
COPY dynare_comprobation ./dynare_comprobation

RUN python -m pip install ".[dev]"

USER 10001:10001
CMD ["python", "-m", "pytest", "-q"]

FROM base AS runtime

LABEL org.opencontainers.image.title="PyDSGEforge" \
      org.opencontainers.image.description="Pure-Python DSGE modeling and Bayesian estimation toolkit" \
      org.opencontainers.image.source="https://github.com/pablo-reyes8/PyDSGEforge" \
      org.opencontainers.image.licenses="MIT"

COPY --from=builder /opt/venv /opt/venv
COPY configs ./configs

USER 10001:10001
CMD ["python", "-c", "from src import DSGE; print('PyDSGEforge runtime ready')"]
