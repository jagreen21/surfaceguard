"""HTTPS with a trust store that works inside an app bundle.

The python.org framework builds do not read the system keychain, and a frozen
bundle has no system Python to fall back on. Every outbound request in this app —
the updater, the release manifest, the Node fetch at build time — goes through
here so TLS verification is never accidentally skipped.
"""

from __future__ import annotations

import ssl
import urllib.error
import urllib.request
from functools import lru_cache

USER_AGENT = "SurfaceGuard"


@lru_cache(maxsize=1)
def ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    try:
        import certifi

        ctx.load_verify_locations(cafile=certifi.where())
    except Exception:
        # Fall back to whatever the platform gives us rather than disabling
        # verification: an unverified update channel is worse than no updates.
        pass
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx


def request(
    url: str,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
    method: str = "GET",
) -> tuple[int, bytes, dict[str, str]]:
    """Fetch a URL. Returns (status, body, response headers); never raises on 4xx/5xx."""
    req = urllib.request.Request(url, method=method)
    req.add_header("User-Agent", USER_AGENT)
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ssl_context()) as r:
            return r.status, r.read(), {k.lower(): v for k, v in r.headers.items()}
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), {k.lower(): v for k, v in exc.headers.items()}


def get(url: str, headers: dict[str, str] | None = None, timeout: float = 60.0) -> bytes:
    """Fetch a URL, raising with a readable message on any non-2xx."""
    status, body, _ = request(url, headers, timeout)
    if not 200 <= status < 300:
        raise urllib.error.URLError(f"{url} returned HTTP {status}")
    return body
