"""Local, privacy-conscious activity history.

Every event records which gates passed, so a "Not a cat" correction becomes a
regression case rather than just a tally (§9 of the design doc). Thumbnails are
optional and expire on a timer.

Events also carry the paw point in *map* coordinates and, for the ones the weekly
review is likely to ask about, a short strip of frames. Both exist for the same
reason: a still photograph of the moment is exactly the wrong evidence for the
cases the review selects, because those are the ambiguous ones. Motion resolves
most of them in under a second, and map coordinates are what makes "this same
spot again" a question the app can answer at all.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .preferences import support_dir

# Review stills are wider than the 320 px activity thumbnail: the whole point of
# the cases the review picks is that they are hard to call, and a postage stamp
# makes the user guess rather than look.
REVIEW_WIDTH = 640

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL    NOT NULL,
    surface_id   TEXT    NOT NULL,
    surface_name TEXT    NOT NULL,
    fired        INTEGER NOT NULL,
    reason       TEXT    NOT NULL DEFAULT '',
    gates        TEXT    NOT NULL DEFAULT '[]',
    score        REAL,
    box          TEXT,
    latency_ms   REAL,
    thumbnail    TEXT,
    feedback     TEXT
);
CREATE INDEX IF NOT EXISTS events_ts ON events (ts DESC);
"""

# Added after the first release of the schema. Applied with ALTER TABLE rather
# than a rewrite, so an existing history survives the upgrade intact.
ADDED_COLUMNS = {
    "map_x": "REAL",
    "map_y": "REAL",
    "strip": "TEXT",
    "reviewed_at": "REAL",
    "review_batch": "INTEGER",
}

FIELDS = ("id, ts, surface_id, surface_name, fired, reason, gates, score, box,"
          " latency_ms, thumbnail, feedback, map_x, map_y, strip, reviewed_at, review_batch")

# The vocabulary a verdict can take. "correct" and "not_a_cat" predate the weekly
# review; the rest let a correction name which gate was wrong instead of just
# saying the app was.
VERDICTS = ("correct", "not_on_surface", "person", "not_a_cat", "missed", "unsure")


@dataclass
class Event:
    id: int
    ts: float
    surface_id: str
    surface_name: str
    fired: bool
    reason: str
    gates: list[dict]
    score: float | None
    box: tuple[float, float, float, float] | None
    latency_ms: float | None
    thumbnail: str | None
    feedback: str | None
    map_point: tuple[float, float] | None = None
    strip: list[str] = field(default_factory=list)
    reviewed_at: float | None = None
    review_batch: int | None = None

    @property
    def when(self) -> str:
        return time.strftime("%a %H:%M:%S", time.localtime(self.ts))

    @property
    def long_when(self) -> str:
        return time.strftime("%A, %-I:%M %p", time.localtime(self.ts))

    @property
    def minute_of_day(self) -> int:
        lt = time.localtime(self.ts)
        return lt.tm_hour * 60 + lt.tm_min

    def gate(self, name: str) -> dict | None:
        return next((g for g in self.gates if g.get("name") == name), None)

    @property
    def blocked_by(self) -> list[str]:
        return [g["name"] for g in self.gates
                if g.get("required", True) and g.get("status") != "pass"]

    @property
    def degraded(self) -> list[str]:
        return [g["name"] for g in self.gates
                if not g.get("required", True) and g.get("status") != "pass"]


class ActivityLog:
    def __init__(self, directory: Path | None = None) -> None:
        self.dir = directory or support_dir()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.thumb_dir = self.dir / "thumbnails"
        self.thumb_dir.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.dir / "activity.sqlite3", check_same_thread=False)
        self._db.executescript(SCHEMA)
        self._migrate()
        self._db.commit()

    def _migrate(self) -> None:
        have = {row[1] for row in self._db.execute("PRAGMA table_info(events)")}
        for name, kind in ADDED_COLUMNS.items():
            if name not in have:
                self._db.execute(f"ALTER TABLE events ADD COLUMN {name} {kind}")

    def close(self) -> None:
        self._db.close()

    # ------------------------------------------------------------------ writes

    def record(
        self,
        surface_id: str,
        surface_name: str,
        fired: bool,
        reason: str = "",
        gates: list[dict] | None = None,
        score: float | None = None,
        box: tuple[float, float, float, float] | None = None,
        latency_ms: float | None = None,
        frame: np.ndarray | None = None,
        save_thumbnail: bool = True,
        map_point: tuple[float, float] | None = None,
        strip_frames: list[np.ndarray] | None = None,
    ) -> int:
        ts = time.time()
        thumb: str | None = None
        if frame is not None and save_thumbnail:
            thumb = self._write_thumbnail(ts, frame, box)
        strip: list[str] = []
        if strip_frames and save_thumbnail:
            strip = self._write_strip(ts, strip_frames, box)
        cur = self._db.execute(
            "INSERT INTO events (ts, surface_id, surface_name, fired, reason, gates, score, box,"
            " latency_ms, thumbnail, map_x, map_y, strip)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (ts, surface_id, surface_name, int(fired), reason, json.dumps(gates or []),
             score, json.dumps(list(box)) if box else None, latency_ms, thumb,
             None if map_point is None else float(map_point[0]),
             None if map_point is None else float(map_point[1]),
             json.dumps(strip) if strip else None),
        )
        self._db.commit()
        return int(cur.lastrowid or 0)

    def set_feedback(self, event_id: int, feedback: str, batch: int | None = None) -> None:
        """``feedback`` is one of :data:`VERDICTS`."""
        self._db.execute(
            "UPDATE events SET feedback = ?, reviewed_at = ?, review_batch = ? WHERE id = ?",
            (feedback, time.time(), batch, event_id),
        )
        self._db.commit()

    def next_batch_id(self) -> int:
        row = self._db.execute("SELECT MAX(review_batch) FROM events").fetchone()
        return int((row[0] or 0) + 1)

    # ------------------------------------------------------------------- reads

    def recent(self, limit: int = 50, fired_only: bool = False) -> list[Event]:
        sql = f"SELECT {FIELDS} FROM events"
        if fired_only:
            sql += " WHERE fired = 1"
        sql += " ORDER BY ts DESC LIMIT ?"
        return [_to_event(r) for r in self._db.execute(sql, (limit,)).fetchall()]

    def since(self, ts: float, limit: int = 2000) -> list[Event]:
        """Everything logged since ``ts``, oldest first — the review's raw material."""
        rows = self._db.execute(
            f"SELECT {FIELDS} FROM events WHERE ts >= ? ORDER BY ts ASC LIMIT ?", (ts, limit)
        ).fetchall()
        return [_to_event(r) for r in rows]

    def by_ids(self, ids: list[int]) -> list[Event]:
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        rows = self._db.execute(
            f"SELECT {FIELDS} FROM events WHERE id IN ({marks})", [int(i) for i in ids]
        ).fetchall()
        by_id = {e.id: e for e in (_to_event(r) for r in rows)}
        return [by_id[i] for i in ids if i in by_id]

    def verdict_counts(self, since_s: float = 7 * 86_400.0) -> dict[str, int]:
        """How the user graded the app over a window, for the accuracy trend."""
        since = time.time() - since_s
        rows = self._db.execute(
            "SELECT feedback, COUNT(*) FROM events WHERE feedback IS NOT NULL AND ts >= ?"
            " GROUP BY feedback", (since,)
        ).fetchall()
        return {str(r[0]): int(r[1]) for r in rows}

    def counts(self, since_s: float = 86_400.0) -> dict[str, int]:
        since = time.time() - since_s
        row = self._db.execute(
            "SELECT COUNT(*), SUM(fired), SUM(feedback = 'not_a_cat') FROM events WHERE ts >= ?",
            (since,),
        ).fetchone()
        return {"events": row[0] or 0, "fired": row[1] or 0, "false_positives": row[2] or 0}

    # -------------------------------------------------------------- retention

    def prune(self, keep_days: int) -> int:
        """Delete thumbnails past the retention window. Rows are kept; images are not."""
        cutoff = time.time() - keep_days * 86_400
        rows = self._db.execute(
            "SELECT id, thumbnail, strip FROM events WHERE ts < ?"
            " AND (thumbnail IS NOT NULL OR strip IS NOT NULL)", (cutoff,)
        ).fetchall()
        for event_id, name, strip in rows:
            for filename in [name, *(json.loads(strip) if strip else [])]:
                if filename:
                    (self.thumb_dir / filename).unlink(missing_ok=True)
            self._db.execute(
                "UPDATE events SET thumbnail = NULL, strip = NULL WHERE id = ?", (event_id,)
            )
        self._db.commit()
        return len(rows)

    def forget_all_thumbnails(self) -> int:
        return self.prune(keep_days=0)

    def forget_pictures_for(self, event_ids: list[int]) -> int:
        """Delete the pictures for specific events — the end of a review, when the
        user only agreed to keep them for the review itself."""
        if not event_ids:
            return 0
        marks = ",".join("?" * len(event_ids))
        rows = self._db.execute(
            f"SELECT id, thumbnail, strip FROM events WHERE id IN ({marks})",
            [int(i) for i in event_ids],
        ).fetchall()
        gone = 0
        for event_id, thumb, strip in rows:
            for name in [thumb, *(json.loads(strip) if strip else [])]:
                if name:
                    (self.thumb_dir / name).unlink(missing_ok=True)
                    gone += 1
            self._db.execute(
                "UPDATE events SET thumbnail = NULL, strip = NULL WHERE id = ?", (event_id,)
            )
        self._db.commit()
        return gone

    def _write_thumbnail(
        self, ts: float, frame: np.ndarray, box: tuple[float, float, float, float] | None
    ) -> str:
        img = frame.copy()
        if box:
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (60, 200, 250), 2)
        h, w = img.shape[:2]
        scale = 320.0 / max(1, w)
        if scale < 1.0:
            img = cv2.resize(img, (320, max(1, int(h * scale))))
        name = f"{int(ts * 1000)}.jpg"
        cv2.imwrite(str(self.thumb_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return name

    def _write_strip(
        self,
        ts: float,
        frames: list[np.ndarray],
        box: tuple[float, float, float, float] | None,
    ) -> list[str]:
        """A few frames around the moment, wide enough to judge from.

        Only the last frame gets the detection box drawn on it — the box belongs
        to that frame, and stamping it onto neighbours would be inventing evidence.
        """
        names: list[str] = []
        stem = int(ts * 1000)
        for i, frame in enumerate(frames):
            if frame is None or frame.size == 0:
                continue
            img = frame.copy()
            if box is not None and i == len(frames) - 1:
                x1, y1, x2, y2 = (int(v) for v in box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (60, 200, 250), 2)
            h, w = img.shape[:2]
            scale = REVIEW_WIDTH / max(1, w)
            if scale < 1.0:
                img = cv2.resize(img, (REVIEW_WIDTH, max(1, int(h * scale))))
            name = f"{stem}_r{i}.jpg"
            cv2.imwrite(str(self.thumb_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
            names.append(name)
        return names


def _to_event(r: tuple) -> Event:
    return Event(
        id=r[0], ts=r[1], surface_id=r[2], surface_name=r[3], fired=bool(r[4]),
        reason=r[5], gates=json.loads(r[6] or "[]"), score=r[7],
        box=tuple(json.loads(r[8])) if r[8] else None,
        latency_ms=r[9], thumbnail=r[10], feedback=r[11],
        map_point=(r[12], r[13]) if r[12] is not None and r[13] is not None else None,
        strip=json.loads(r[14]) if r[14] else [],
        reviewed_at=r[15], review_batch=r[16],
    )
