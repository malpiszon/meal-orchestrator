FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
COPY prompts ./prompts

RUN pip install --no-cache-dir .

ENTRYPOINT ["meal-orchestrator"]
CMD ["--config", "config/app.example.yaml", "--users", "config/users.example.yaml", "--dry-run"]
