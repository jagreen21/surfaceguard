"""A procedural room the app can be developed and tested against with no hardware.

This is not a toy stub: it renders a wide, feature-rich panorama and returns real
crops of it under a virtual pan/tilt, with the backlash, noise and jitter that
make registration necessary in the first place. Every geometry, gate and UI path
can therefore be exercised — and regression-tested — before the E30 arrives.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from .base import Capabilities, CameraSource, Frame, PetEvent

MAP_W, MAP_H = 2400, 760
PX_PER_DEGREE = 12.0  # so the 200 deg pan range spans the map width


def _text(canvas, s, org, scale, colour, thickness=1):
    cv2.putText(canvas, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)


def render_room(seed: int = 7) -> np.ndarray:
    """Draw a wide synthetic room with enough texture for ORB to lock onto."""
    rng = np.random.default_rng(seed)
    room = np.full((MAP_H, MAP_W, 3), 176, np.uint8)

    # Wall/floor split, plus a grain so featureless regions still yield corners.
    cv2.rectangle(room, (0, 470), (MAP_W, MAP_H), (150, 154, 158), -1)
    grain = rng.normal(0, 9, (MAP_H, MAP_W, 1)).repeat(3, axis=2)
    room = np.clip(room.astype(np.float32) + grain, 0, 255).astype(np.uint8)

    # Furniture: rectangles with edge detail and labels. The labels matter —
    # text is a dense, distinctive feature source, like real room clutter.
    pieces = [
        ("CUPBOARDS", (60, 120), (430, 300), (120, 132, 146)),
        ("COUNTER", (150, 430), (700, 505), (196, 190, 176)),
        ("FRIDGE", (760, 210), (930, 520), (206, 208, 210)),
        ("TABLE", (1050, 455), (1480, 540), (150, 122, 96)),
        ("WINDOW", (1120, 140), (1420, 330), (214, 222, 228)),
        ("SHELF A", (1610, 250), (1900, 300), (142, 116, 92)),
        ("SHELF B", (1610, 360), (1900, 410), (142, 116, 92)),
        ("SOFA", (1950, 430), (2330, 560), (110, 118, 142)),
        ("HALL SHELF", (2210, 230), (2360, 300), (138, 112, 90)),
    ]
    for label, tl, br, colour in pieces:
        cv2.rectangle(room, tl, br, colour, -1)
        cv2.rectangle(room, tl, br, (70, 72, 76), 2)
        _text(room, label, (tl[0] + 8, tl[1] + 26), 0.55, (44, 46, 50), 2)
        for i in range(tl[0] + 18, br[0] - 10, 46):  # panel lines / slats
            cv2.line(room, (i, tl[1] + 4), (i, br[1] - 4), (86, 88, 92), 1)

    # Wall art and sockets: small high-contrast corners spread across the map.
    for _ in range(70):
        x = int(rng.integers(20, MAP_W - 40))
        y = int(rng.integers(30, 440))
        w = int(rng.integers(10, 34))
        cv2.rectangle(room, (x, y), (x + w, y + int(rng.integers(10, 34))),
                      tuple(int(v) for v in rng.integers(40, 230, 3)), -1)

    # A degree ruler along the top, so screenshots are self-documenting.
    for deg in range(-90, 91, 10):
        x = int(MAP_W / 2 + deg * PX_PER_DEGREE)
        if 0 <= x < MAP_W:
            cv2.line(room, (x, 0), (x, 18 if deg % 30 else 30), (60, 62, 66), 1)
            if deg % 30 == 0:
                _text(room, f"{deg:+d}", (x - 16, 48), 0.5, (60, 62, 66), 1)
    return room


def _draw_cat(canvas: np.ndarray, cx: float, cy: float, height: float) -> None:
    """A crude but correctly-proportioned cat, drawn standing at (cx, cy)."""
    h = max(8.0, height)
    w = h * 1.45
    x1, y1, x2, y2 = int(cx - w / 2), int(cy - h), int(cx + w / 2), int(cy)
    body = (58, 58, 62)
    cv2.ellipse(canvas, ((x1 + x2) // 2, int(y2 - h * 0.34)),
                (int(w * 0.44), int(h * 0.30)), 0, 0, 360, body, -1)
    head_r = int(h * 0.24)
    hx, hy = int(x2 - w * 0.16), int(y1 + head_r)
    cv2.circle(canvas, (hx, hy), head_r, body, -1)
    for dx in (-head_r, head_r):  # ears
        cv2.drawContours(canvas, [np.array([
            [hx + int(dx * 0.7), hy - head_r + 2],
            [hx + int(dx * 1.0), hy - int(head_r * 1.7)],
            [hx + int(dx * 0.1), hy - head_r - 2]])], -1, body, -1)
    for dx in (-0.30, -0.05, 0.22, 0.38):  # legs
        lx = int((x1 + x2) / 2 + w * dx)
        cv2.line(canvas, (lx, int(y2 - h * 0.26)), (lx, int(y2)), body, max(2, int(h * 0.07)))
    cv2.line(canvas, (x1 + 2, int(y2 - h * 0.5)), (x1 - int(w * 0.2), int(y2 - h * 0.9)),
             body, max(2, int(h * 0.07)))


def _draw_person(canvas: np.ndarray, cx: float, cy: float, height: float) -> None:
    """A blocky standing figure, for exercising the no_person gate."""
    h = max(20.0, height)
    w = h * 0.34
    colour = (96, 132, 170)
    x1, y1, x2, y2 = int(cx - w / 2), int(cy - h), int(cx + w / 2), int(cy)
    cv2.rectangle(canvas, (x1, int(y1 + h * 0.20)), (x2, int(y2 - h * 0.42)), colour, -1)
    cv2.circle(canvas, (int(cx), int(y1 + h * 0.11)), int(h * 0.11), colour, -1)
    for dx in (-0.22, 0.22):  # legs
        lx = int(cx + w * dx)
        cv2.line(canvas, (lx, int(y2 - h * 0.42)), (lx, y2), colour, max(3, int(h * 0.07)))


class SyntheticCamera(CameraSource):
    """Virtual pan/tilt camera over :func:`render_room`.

    ``backlash_px`` reproduces the reason registration exists: asking for the same
    angle twice does not return the same pixels.
    """

    def __init__(
        self,
        frame_size: tuple[int, int] = (960, 540),
        fps: float = 12.0,
        pan: float = 0.0,
        tilt: float = 0.0,
        backlash_px: float = 14.0,
        noise: float = 4.0,
        latency_s: float = 0.28,
        seed: int = 7,
    ) -> None:
        super().__init__()
        self.room = render_room(seed)
        self.frame_w, self.frame_h = frame_size
        self.fps = fps
        self.pan, self.tilt = pan, tilt
        self.backlash_px = backlash_px
        self.noise = noise
        self.latency_s = latency_s
        self._rng = np.random.default_rng(seed + 1)
        self._offset = np.zeros(2)
        self._running = False
        self._last_read = 0.0
        # Subjects in map coordinates: (x, y, height).
        self.cat: tuple[float, float, float] | None = None
        self.person: tuple[float, float, float] | None = None
        self._last_boxes: list[tuple[str, float, float, float, float]] = []

    # ------------------------------------------------------------- lifecycle

    def start(self) -> None:
        self._running = True
        self._reseat()

    def stop(self) -> None:
        self._running = False

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            name="Synthetic room",
            model="synthetic",
            has_ptz=True,
            reports_angles=True,
            has_speaker=True,
            emits_pet_events=True,
            pan_range=(-95.0, 95.0),
            tilt_range=(-20.0, 20.0),
            notes=["Simulated source: crops are translations, not true reprojection."],
        )

    # ------------------------------------------------------------------ read

    def read(self, timeout: float = 2.0) -> Frame | None:
        if not self._running:
            return None
        interval = 1.0 / self.fps
        wait = interval - (time.monotonic() - self._last_read)
        if wait > 0:
            time.sleep(min(wait, timeout))
        self._last_read = time.monotonic()

        x0, y0 = self._window_origin()
        crop = self.room[y0:y0 + self.frame_h, x0:x0 + self.frame_w].copy()
        self._last_boxes = []
        subjects = (("cat", self.cat, _draw_cat), ("person", self.person, _draw_person))
        for label, placement, draw in subjects:
            if placement is None:
                continue
            mx, my, mh = placement
            fx, fy = mx - x0, my - y0
            draw(crop, fx, fy, mh)
            half = mh * (0.72 if label == "cat" else 0.17)
            self._last_boxes.append((label, fx - half, fy - mh, fx + half, fy))
        if self.noise > 0:
            crop = np.clip(
                crop.astype(np.float32) + self._rng.normal(0, self.noise, crop.shape), 0, 255
            ).astype(np.uint8)

        now = time.monotonic()
        return Frame(
            image=crop,
            seq=self._next_seq(),
            ts_received=now,
            ts_capture=now - self.latency_s,
            pan=self.pan,
            tilt=self.tilt,
        )

    # ------------------------------------------------------------------- ptz

    def move_to(self, pan: float, tilt: float = 0.0, settle_s: float = 0.0) -> bool:
        lo, hi = self.capabilities.pan_range or (-90.0, 90.0)
        tlo, thi = self.capabilities.tilt_range or (-20.0, 20.0)
        self.pan = float(np.clip(pan, lo, hi))
        self.tilt = float(np.clip(tilt, tlo, thi))
        self._reseat()
        if settle_s:
            time.sleep(settle_s)
        return True

    def current_angles(self) -> tuple[float | None, float | None]:
        return (self.pan, self.tilt)

    def play_sound_on_camera(self, name: str = "default") -> bool:
        return True

    def inject_pet_event(self) -> None:
        self._publish_pet_event(PetEvent(time.monotonic(), device="synthetic"))

    # -------------------------------------------------------------- internals

    def place_cat(self, x: float, y: float, height: float = 90.0) -> None:
        """Put a cat at a map position, standing with its paws at ``y``."""
        self.cat = (float(x), float(y), float(height))

    def clear_cat(self) -> None:
        self.cat = None

    def place_person(self, x: float, y: float, height: float = 300.0) -> None:
        self.person = (float(x), float(y), float(height))

    def clear_person(self) -> None:
        self.person = None

    def ground_truth_boxes(self) -> list[tuple[str, float, float, float, float]]:
        """What the last :meth:`read` actually drew, in that frame's pixels.

        Used by SyntheticDetector, which perturbs these rather than trying to
        threshold them back out of the image.
        """
        return list(self._last_boxes)

    def _reseat(self) -> None:
        """New random sub-pixel-to-tens-of-pixels offset: mechanical backlash."""
        if self.backlash_px > 0:
            self._offset = self._rng.normal(0, self.backlash_px / 2.0, 2)

    def _window_origin(self) -> tuple[int, int]:
        cx = MAP_W / 2 + self.pan * PX_PER_DEGREE + self._offset[0]
        cy = MAP_H / 2 + self.tilt * PX_PER_DEGREE + self._offset[1]
        x0 = int(np.clip(cx - self.frame_w / 2, 0, MAP_W - self.frame_w))
        y0 = int(np.clip(cy - self.frame_h / 2, 0, MAP_H - self.frame_h))
        return x0, y0

    def true_frame_to_map(self) -> np.ndarray:
        """Ground truth, for tests that need to score registration accuracy."""
        x0, y0 = self._window_origin()
        return np.array([[1.0, 0.0, x0], [0.0, 1.0, y0], [0.0, 0.0, 1.0]])
