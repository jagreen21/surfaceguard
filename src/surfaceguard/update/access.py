"""The GitHub side of updates.

The release repository is public, so no credential is needed and none is stored: a
token that silently expires would stop updates with no visible symptom, and the
signature — not the repository's privacy — is what actually stops a bad update
installing. Whoever controls the host still cannot make this app run code they
wrote.

A token is still honoured if one is present, for a private fork or to lift the
unauthenticated rate limit. When there is one, its health is checked daily and
GitHub's own expiry header warns before it lapses rather than after.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from ..bridge.credentials import get_update_token
from ..net import request

API = "https://api.github.com"

# Warn this far ahead of the expiry GitHub reports, so there is time to rotate.
WARN_WITHIN_DAYS = 14
CHECK_EVERY_S = 24 * 3600


@dataclass
class TokenHealth:
    ok: bool = False
    checked_at: float = 0.0
    reason: str = ""
    remedy: str = ""
    expires_at: datetime | None = None

    @property
    def days_left(self) -> float | None:
        if self.expires_at is None:
            return None
        return (self.expires_at - datetime.now(timezone.utc)).total_seconds() / 86_400

    @property
    def expiring_soon(self) -> bool:
        days = self.days_left
        return days is not None and 0 <= days <= WARN_WITHIN_DAYS

    def summary(self) -> str:
        if not self.ok:
            return self.reason
        days = self.days_left
        if days is None:
            return "Updates are working"
        if days < 0:
            return "The update access token has expired"
        return f"Update access is working ({days:.0f} days until the token expires)"


def _headers(token: str | None) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _parse_expiry(headers: dict[str, str]) -> datetime | None:
    """GitHub reports fine-grained token expiry on every authenticated response."""
    raw = headers.get("github-authentication-token-expiration", "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S %Z", "%Y-%m-%d %H:%M:%S UTC", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def check_token(repo: str, token: str | None = None) -> TokenHealth:
    """Confirm the token can still read the release repository, and for how long."""
    token = token or get_update_token()
    now = time.time()

    status, body, headers = request(f"{API}/repos/{repo}", _headers(token), timeout=20.0)
    if not token:
        # No credential is expected: the release repository is public.
        if status == 200:
            return TokenHealth(True, now, "", "")
        if status in (401, 403, 404):
            return TokenHealth(
                False, now,
                "Updates cannot reach the release page",
                f"{repo} is not publicly readable. Add an access token in Settings, "
                "or make the repository public.",
            )
        return TokenHealth(
            False, now,
            f"GitHub returned an error while checking for updates ({status})",
            "This is usually temporary. It will try again later.",
        )
    expires = _parse_expiry(headers)

    if status == 200:
        health = TokenHealth(True, now, "", "", expires)
        if health.days_left is not None and health.days_left < 0:
            return TokenHealth(
                False, now, "The update access token has expired",
                "Create a new token on GitHub and paste it into Settings.", expires,
            )
        return health
    if status == 401:
        return TokenHealth(
            False, now, "The update access token is no longer valid",
            "It was revoked or has expired. Create a new one and paste it into Settings.",
            expires,
        )
    if status in (403, 404):
        return TokenHealth(
            False, now, "The update access token cannot read the update repository",
            f"Give the token read access to {repo}, or create a new one.", expires,
        )
    return TokenHealth(
        False, now, f"GitHub returned an error while checking for updates ({status})",
        "This is usually temporary. It will try again later.", expires,
    )


def latest_release(repo: str, token: str | None = None) -> dict:
    status, body, _ = request(f"{API}/repos/{repo}/releases/latest",
                              _headers(token or get_update_token()), timeout=30.0)
    if status == 404:
        return {}          # nothing published yet is not an error
    if status != 200:
        raise RuntimeError(f"GitHub returned HTTP {status} when looking for the latest release")
    return json.loads(body)


def download_asset(repo: str, asset_id: int, token: str | None = None) -> bytes:
    """Assets are fetched through the API, which works with or without a token."""
    headers = {**_headers(token or get_update_token()), "Accept": "application/octet-stream"}
    status, body, _ = request(
        f"{API}/repos/{repo}/releases/assets/{asset_id}", headers, timeout=600.0
    )
    if status != 200:
        raise RuntimeError(f"Downloading the update failed (HTTP {status})")
    return body


def find_asset(release: dict, name: str) -> dict | None:
    return next((a for a in release.get("assets", []) if a.get("name") == name), None)
