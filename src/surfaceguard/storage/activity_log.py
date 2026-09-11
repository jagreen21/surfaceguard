"""Local, privacy-conscious activity history.

Every event records which gates passed, so a "Not a cat" correction becomes a
regression case rather than just a tally (§9 of the design doc). Thumbnails are
optional and expire on a timer.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .preferences import support_dir

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
    latency_ms: float | None
    thumbnail: str | None
    feedback: str | None

    @property
    def when(self) -> str:
        return time.strftime("%a %H:%M:%S", time.localtime(self.ts))


class ActivityLog:
    def __init__(self, directory: Path | None = None) -> None:
        self.dir = directory or support_dir()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.thumb_dir = self.dir / "thumbnails"
        self.thumb_dir.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.dir / "activity.sqlite3", check_same_thread=False)
        self._db.executescript(SCHEMA)
        self._db.commit()

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
    ) -> int:
        ts = time.time()
        thumb: str | None = None
        if frame is not None and save_thumbnail:
            thumb = self._write_thumbnail(ts, frame, box)
        cur = self._db.execute(
            "INSERT INTO events (ts, surface_id, surface_name, fired, reason, gates, score, box,"
            " latency_ms, thumbnail) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (ts, surface_id, surface_name, int(fired), reason, json.dumps(gates or []),
             score, json.dumps(list(box)) if box else None, latency_ms, thumb),
        )
        self._db.commit()
        return int(cur.lastrowid or 0)

    def set_feedback(self, event_id: int, feedback: str) -> None:
        """``feedback`` is 'correct' or 'not_a_cat'."""
        self._db.execute("UPDATE events SET feedback = ? WHERE id = ?", (feedback, event_id))
        self._db.commit()

    # ------------------------------------------------------------------- reads

    def recent(self, limit: int = 50, fired_only: bool = False) -> list[Event]:
        sql = ("SELECT id, ts, surface_id, surface_name, fired, reason, gates, score,"
               " latency_ms, thumbnail, feedback FROM events")
        if fired_only:
            sql += " WHERE fired = 1"
        sql += " ORDER BY ts DESC LIMIT ?"
        rows = self._db.execute(sql, (limit,)).fetchall()
        return [
            Event(r[0], r[1], r[2], r[3], bool(r[4]), r[5], json.loads(r[6]),
                  r[7], r[8], r[9], r[10])
            for r in rows
        ]

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
            "SELECT id, thumbnail FROM events WHERE thumbnail IS NOT NULL AND ts < ?", (cutoff,)
        ).fetchall()
        for event_id, name in rows:
            (self.thumb_dir / name).unlink(missing_ok=True)
            self._db.execute("UPDATE events SET thumbnail = NULL WHERE id = ?", (event_id,))
        self._db.commit()
        return len(rows)

    def forget_all_thumbnails(self) -> int:
        return self.prune(keep_days=0)

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
