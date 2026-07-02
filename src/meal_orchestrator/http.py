from __future__ import annotations

import urllib.error
import urllib.request


def post_json(url: str, *, headers: dict[str, str], body: bytes, timeout_seconds: int) -> bytes:
    """POST body to url and return the raw response bytes.

    Wraps HTTPError to append the response body text to the exception message,
    since urllib discards it by default and it's often the most useful part
    of an API error (e.g. validation details).
    """
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise urllib.error.HTTPError(
            exc.url, exc.code, f"{exc.reason} — {detail}", exc.headers, None
        ) from exc
