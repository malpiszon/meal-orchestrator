FROM python:3.13-slim AS builder

RUN apt-get update && apt-get install --no-install-recommends -y git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY .git ./.git
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir build && python -m build --wheel


FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /app/dist/*.whl ./
RUN pip install --no-cache-dir ./*.whl && rm ./*.whl

ENTRYPOINT ["meal-orchestrator"]
CMD ["--help"]
