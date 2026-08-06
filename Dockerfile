FROM python:3.13-slim AS builder

WORKDIR /app

# Version is resolved from git tags outside the build context (see release.yml)
# and passed in here, so the image doesn't need git or the .git directory.
ARG VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=$VERSION

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir build && python -m build --wheel


FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /app/dist/*.whl ./
RUN pip install --no-cache-dir ./*.whl && rm ./*.whl

# Common to all users and version-controlled, unlike the per-user prompts/*.local.md
# files (gitignored, bind-mounted at runtime — see run-local.sh).
COPY prompts/app.md ./prompts/app.md

ENTRYPOINT ["meal-orchestrator"]
CMD ["--help"]
