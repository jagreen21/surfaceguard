"""Secrets in the macOS Keychain, never in the app's own files.

Access goes through ``/usr/bin/security`` on purpose. The Keychain grants access
per *accessing binary*, and an ad-hoc signature changes on every build — so if the
app read the Keychain directly, every OTA update would re-prompt for permission.
Apple's own signed tool is a stable accessor, so the grant survives updates.

Values never pass through logs, exceptions or the activity log.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

SERVICE = "SurfaceGuard"
ACCOUNT_EUFY = "eufy-account"
ACCOUNT_UPDATE_TOKEN = "update-token"


class KeychainError(RuntimeError):
    """Keychain refused, in words worth showing the user."""


def _security(args: list[str], secret: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["/usr/bin/security", *args],
        input=secret, capture_output=True, text=True, check=False,
    )


def set_secret(account: str, secret: str, label: str = "") -> None:
    """Store (or replace) one secret. ``-U`` updates rather than duplicating."""
    result = _security([
        "add-generic-password",
        "-a", account, "-s", SERVICE,
        "-l", label or f"{SERVICE} ({account})",
        "-U", "-w", secret,
    ])
    if result.returncode != 0:
        raise KeychainError(
            "macOS would not save this to the Keychain. "
            + (result.stderr.strip() or "Try again, or check Keychain Access.")
        )


def get_secret(account: str) -> str | None:
    result = _security(["find-generic-password", "-a", account, "-s", SERVICE, "-w"])
    if result.returncode != 0:
        return None
    return result.stdout.rstrip("\n") or None


def delete_secret(account: str) -> bool:
    return _security(["delete-generic-password", "-a", account, "-s", SERVICE]).returncode == 0


@dataclass
class EufyAccount:
    """The Eufy sign-in the bridge needs. The password lives only in the Keychain."""

    username: str
    country: str = "US"
    language: str = "en"

    @property
    def configured(self) -> bool:
        return bool(self.username and self.password)

    @property
    def password(self) -> str | None:
        return get_secret(ACCOUNT_EUFY)

    def save_password(self, password: str) -> None:
        set_secret(ACCOUNT_EUFY, password, label="Surface Guard — Eufy account")

    @staticmethod
    def forget_password() -> bool:
        return delete_secret(ACCOUNT_EUFY)


def get_update_token() -> str | None:
    return get_secret(ACCOUNT_UPDATE_TOKEN)


def set_update_token(token: str) -> None:
    set_secret(ACCOUNT_UPDATE_TOKEN, token, label="Surface Guard — update access token")
