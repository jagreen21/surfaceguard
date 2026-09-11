"""Signing in to Eufy on its own, for Settings.

First-run setup does not use this: the sign-in is part of
:class:`~surfaceguard.ui.onboarding.OnboardingDialog` so that a new user meets one
flow with one step counter. This is the same flow stopped after a camera has been
chosen, for changing accounts later without rescanning the room.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from ..bridge.credentials import EufyAccount
from .onboarding import COUNTRIES, OnboardingDialog, Page

__all__ = ["EufySignInDialog", "COUNTRIES"]


class EufySignInDialog(OnboardingDialog):
    """Credentials, any challenge, and the camera list — then stop."""

    def __init__(self, account: EufyAccount | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent, allow_demo=False, account=account, sign_in_only=True)
        self.setWindowTitle("Connect your camera")
        self.setMinimumSize(600, 460)

    @property
    def stack_page(self) -> Page:
        return self.page
