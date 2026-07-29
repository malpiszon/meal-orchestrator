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
    return _urlopen(req, timeout_seconds)


def get_json(url: str, *, headers: dict[str, str], timeout_seconds: int) -> bytes:
    """GET url and return the raw response bytes, with the same HTTPError handling as post_json."""
    req = urllib.request.Request(url, headers=headers, method="GET")
    return _urlopen(req, timeout_seconds)


def _urlopen(req: urllib.request.Request, timeout_seconds: int) -> bytes:
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        error = urllib.error.HTTPError(
            exc.url, exc.code, f"{exc.reason} — {detail}", exc.headers, None
        )
        error.response_body = detail
        raise error from exc
