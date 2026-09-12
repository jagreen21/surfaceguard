"""Where things live on disk, and the settings file.

Surfaces are stored in map coordinates (D1), so a saved configuration survives the
camera being nudged, restarted, or re-registered — it is tied to the room, not to a
frame.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..geometry.surface import BlindSpot, Deterrent, ScheduleWindow, Surface, Tuning

APP_NAME = "SurfaceGuard"


def support_dir() -> Path:
    base = Path.home() / "Library" / "Application Support" / APP_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def write_json_atomic(path: Path, payload: dict) -> None:
    """Write via a temp file + rename, so a crash mid-save cannot corrupt config."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@dataclass
class Preferences:
    """Everything the app remembers between launches."""

    surfaces: list[Surface] = field(default_factory=list)
    model_path: str | None = None
    target_fps: float = 8.0
    keep_thumbnails_days: int = 14
    save_thumbnails: bool = True
    launch_at_login: bool = False
    prefer_camera_speaker: bool = True
    master_volume: float = 0.6
    scan_weights: dict[str, float] = field(default_factory=dict)
    camera: dict = field(default_factory=dict)
    # P2P workarounds for cameras whose key exchange fails with the
    # defaults. Changing either needs the camera reconnected.
    p2p_local_only: bool = False
    p2p_embedded_pkcs1: bool = True
    room_name: str = "Kitchen"
    detection_sensitivity: str = "Balanced"
    custom_sounds: dict[str, str] = field(default_factory=dict)
    show_protected_zones: bool = True
    show_detection_boxes: bool = True
    show_surface_labels: bool = True

    # --- weekly review ------------------------------------------------------
    # ``review_enabled`` is the user's standing answer to "ask me to grade you".
    # ``review_declines`` counts consecutive "not now"s so the invitation can
    # stop asking on its own rather than becoming something to dismiss by reflex.
    review_enabled: bool = True
    last_review_at: float = 0.0
    last_review_offered_at: float = 0.0
    review_declines: int = 0
    # Keep pictures for the review even when the user has turned pictures off
    # everywhere else; they are deleted the moment the review is finished.
    review_pictures_only: bool = False

    # --------------------------------------------------------------- load/save

    @staticmethod
    def path() -> Path:
        return support_dir() / "preferences.json"

    @classmethod
    def load(cls, path: Path | None = None) -> "Preferences":
        p = path or cls.path()
        if not p.exists():
            return cls()
        try:
            raw = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            # A corrupt settings file must not stop the app from starting; the
            # user loses preferences, not the ability to launch.
            return cls()
        prefs = cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__ and k != "surfaces"})
        prefs.surfaces = [surface_from_dict(d) for d in raw.get("surfaces", [])]
        return prefs

    def save(self, path: Path | None = None) -> Path:
        p = path or self.path()
        payload = {
            k: v for k, v in asdict(self).items() if k != "surfaces"
        }
        payload["surfaces"] = [surface_to_dict(s) for s in self.surfaces]
        write_json_atomic(p, payload)
        return p

    # ----------------------------------------------------------------- helpers

    def surface_by_id(self, sid: str) -> Surface | None:
        return next((s for s in self.surfaces if s.id == sid), None)

    def weight_for(self, sid: str) -> float:
        return float(self.scan_weights.get(sid, 1.0))


def surface_to_dict(s: Surface) -> dict:
    return {
        "id": s.id,
        "name": s.name,
        "enabled": s.enabled,
        "polygon": np.asarray(s.polygon, float).round(2).tolist(),
        "plane_quad": None if s.plane_quad is None else np.asarray(s.plane_quad, float).round(2).tolist(),
        "height_samples": [round(v, 4) for v in s.height_samples],
        "deterrent": asdict(s.deterrent),
        "schedule": asdict(s.schedule),
        "tuning": asdict(s.tuning),
    }


def surface_from_dict(d: dict) -> Surface:
    sched = d.get("schedule") or {}
    if "days" in sched:
        sched = {**sched, "days": tuple(sched["days"])}
    return Surface(
        name=d.get("name", "Surface"),
        polygon=np.asarray(d["polygon"], float),
        id=d.get("id") or Surface(name="x", polygon=np.zeros((3, 2))).id,
        enabled=bool(d.get("enabled", True)),
        plane_quad=None if d.get("plane_quad") is None else np.asarray(d["plane_quad"], float),
        height_samples=[float(v) for v in (d.get("height_samples") or [])],
        deterrent=Deterrent(**(d.get("deterrent") or {})),
        schedule=ScheduleWindow(**sched) if sched else ScheduleWindow(),
        tuning=_tuning_from_dict(d.get("tuning")),
    )


def _tuning_from_dict(d: dict | None) -> Tuning:
    d = d or {}
    return Tuning(
        blind_spots=[BlindSpot(**b) for b in (d.get("blind_spots") or [])],
        min_score=None if d.get("min_score") is None else float(d["min_score"]),
        scale_tolerance=(None if d.get("scale_tolerance") is None
                         else float(d["scale_tolerance"])),
    )
