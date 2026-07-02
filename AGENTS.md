# AGENTS.md

Read `README.md` for the current architecture and design notes; the code is
the source of truth.

Prefer the simplest solution consistent with the existing design.

Implement only what is required for the current task.

Avoid speculative abstractions and future-proofing unless explicitly requested.

After completing a task, check whether `README.md` still accurately describes
the project. Update it if the change is user-visible; skip it for internal
changes that don't affect the described behaviour or usage.

Before pushing, run the CI checks locally and fix any failures:

```
.venv/bin/ruff check .
.venv/bin/pytest
```
