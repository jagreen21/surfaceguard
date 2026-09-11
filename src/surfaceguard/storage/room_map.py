"""Persisting the stitched room map and its keyframes."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..camera.panorama import MapKeyframe, RoomMap
from .preferences import support_dir, write_json_atomic


def map_dir() -> Path:
    d = support_dir() / "room_map"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_room_map(room: RoomMap, directory: Path | None = None) -> Path:
    d = directory or map_dir()
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "canvas.png"), room.canvas)
    meta = {"built_at": room.built_at, "source_name": room.source_name, "keyframes": []}
    for kf in room.keyframes:
        cv2.imwrite(str(d / f"{kf.id}.jpg"), kf.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        meta["keyframes"].append({
            "id": kf.id,
            "pan": kf.pan,
            "tilt": kf.tilt,
            "full_to_map": np.asarray(kf.full_to_map, float).tolist(),
            "features": kf.features,
        })
    write_json_atomic(d / "map.json", meta)
    return d


def load_room_map(directory: Path | None = None) -> RoomMap | None:
    d = directory or map_dir()
    meta_path, canvas_path = d / "map.json", d / "canvas.png"
    if not (meta_path.exists() and canvas_path.exists()):
        return None
    canvas = cv2.imread(str(canvas_path))
    if canvas is None:
        return None
    meta = json.loads(meta_path.read_text())
    keyframes: list[MapKeyframe] = []
    for entry in meta.get("keyframes", []):
        image = cv2.imread(str(d / f"{entry['id']}.jpg"))
        if image is None:
            continue  # a missing keyframe degrades the map, it does not break it
        keyframes.append(MapKeyframe(
            id=entry["id"],
            pan=entry.get("pan"),
            tilt=entry.get("tilt"),
            full_to_map=np.asarray(entry["full_to_map"], float),
            image=image,
            features=int(entry.get("features", 0)),
        ))
    if not keyframes:
        return None
    return RoomMap(
        canvas=canvas,
        keyframes=keyframes,
        built_at=float(meta.get("built_at", 0.0)),
        source_name=meta.get("source_name", ""),
    )
