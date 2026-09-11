"""What a release looks like, and how its authenticity is established.

The download is verified twice over: a SHA-256 that must match the manifest, and
an Ed25519 signature over that manifest made with a key that never leaves the
build machine. The public half is compiled into the app. So whoever controls the
release host — GitHub, a bucket, a compromised account — still cannot make this
app run code they wrote.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

SIGNATURE_FIELD = "signature"


@dataclass
class Release:
    """One published build."""

    version: str
    sha256: str
    size: int
    notes: str = ""
    asset_name: str = ""
    asset_id: int = 0
    url: str = ""
    min_macos: str = ""
    published_at: str = ""

    def signing_payload(self) -> bytes:
        """The exact bytes that get signed: the manifest minus the signature itself.

        Sorted keys and compact separators, so signer and verifier cannot disagree
        about whitespace.
        """
        body = {k: v for k, v in asdict(self).items() if k != SIGNATURE_FIELD}
        return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()

    def to_json(self, signature_hex: str) -> str:
        payload = asdict(self)
        payload[SIGNATURE_FIELD] = signature_hex
        return json.dumps(payload, sort_keys=True, indent=2)

    @staticmethod
    def from_json(text: str) -> tuple["Release", str]:
        raw = json.loads(text)
        signature = str(raw.pop(SIGNATURE_FIELD, ""))
        known = {k: v for k, v in raw.items() if k in Release.__dataclass_fields__}
        return Release(**known), signature


def verify(release: Release, signature_hex: str, public_key_hex: str) -> tuple[bool, str]:
    """Check the manifest signature. Returns (ok, reason)."""
    if not signature_hex:
        return False, "the update manifest is not signed"
    if not public_key_hex:
        return False, "this build has no update signing key compiled in"
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError:
        return False, "the signature checker is missing from this install"
    try:
        key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
        key.verify(bytes.fromhex(signature_hex), release.signing_payload())
    except InvalidSignature:
        return False, "the update was not signed by the expected key"
    except (ValueError, TypeError) as exc:
        return False, f"the update signature is malformed ({exc})"
    return True, ""


def is_newer(candidate: str, current: str) -> bool:
    """Compare dotted versions numerically, ignoring a leading 'v'."""
    return _parts(candidate) > _parts(current)


def _parts(version: str) -> tuple[int, ...]:
    cleaned = version.strip().lstrip("vV").split("+")[0].split("-")[0]
    out = []
    for chunk in cleaned.split("."):
        try:
            out.append(int(chunk))
        except ValueError:
            out.append(0)
    return tuple(out or [0])
