"""
alerts.py — Beep alerting + anomaly logging (CSV and/or SQLite) for the
FloorPulse idle-monitoring pipeline.

Usage in your app.py:

    from alerts import AlertManager

    alert_mgr = AlertManager(
        csv_path="idle_violations.csv",
        sqlite_path="floorpulse.db",   # set to None to skip DB logging
        cooldown_sec=10,                # min seconds between repeat beeps for the SAME track
    )

    # Inside your per-track idle-detection loop, whenever a track is idle:
    if track_is_idle:
        alert_mgr.raise_alert(
            track_id=tid,
            idle_seconds=idle_duration,
            camera="Cam-1",             # optional, useful once you have multiple feeds
        )
"""

import csv
import os
import sqlite3
import threading
import time
import winsound  # Windows-only, built into the standard library — no install needed


class AlertManager:
    def __init__(self, csv_path="idle_violations.csv", sqlite_path="floorpulse.db",
                 cooldown_sec=10, beep_freq=1500, beep_duration_ms=350):
        self.csv_path = csv_path
        self.sqlite_path = sqlite_path
        self.cooldown_sec = cooldown_sec
        self.beep_freq = beep_freq
        self.beep_duration_ms = beep_duration_ms

        self._last_alert_time = {}      # track_id -> last time we beeped/logged for it
        self._lock = threading.Lock()   # guards _last_alert_time across frames

        self._init_csv()
        if self.sqlite_path:
            self._init_sqlite()

    # ------------------------------------------------------------------
    # Storage setup
    # ------------------------------------------------------------------
    def _init_csv(self):
        file_exists = os.path.isfile(self.csv_path)
        if not file_exists:
            with open(self.csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "track_id", "camera", "idle_seconds", "status"])

    def _init_sqlite(self):
        conn = sqlite3.connect(self.sqlite_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS idle_violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                track_id TEXT NOT NULL,
                camera TEXT,
                idle_seconds INTEGER NOT NULL,
                status TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Beep — runs in its own thread so it NEVER blocks the Streamlit
    # render loop or your video processing (winsound.Beep is blocking
    # by default).
    # ------------------------------------------------------------------
    def _beep_async(self):
        def _do_beep():
            try:
                winsound.Beep(self.beep_freq, self.beep_duration_ms)
            except RuntimeError:
                # Beep() can occasionally raise if called from a thread with
                # no default sound device context — fail silently rather than
                # crash the pipeline over an alert sound.
                pass
        threading.Thread(target=_do_beep, daemon=True).start()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log_csv(self, ts, track_id, camera, idle_seconds, status):
        with open(self.csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ts, track_id, camera, idle_seconds, status])

    def _log_sqlite(self, ts, track_id, camera, idle_seconds, status):
        conn = sqlite3.connect(self.sqlite_path)
        conn.execute(
            "INSERT INTO idle_violations (timestamp, track_id, camera, idle_seconds, status) "
            "VALUES (?, ?, ?, ?, ?)",
            (ts, track_id, camera, idle_seconds, status),
        )
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Public entry point — call this every frame for every idle track.
    # Internally rate-limited per track_id so you get ONE beep + ONE log
    # row per cooldown window, not one per frame.
    # ------------------------------------------------------------------
    def raise_alert(self, track_id, idle_seconds, camera="default", status="IDLE_VIOLATION"):
        now = time.time()
        with self._lock:
            last = self._last_alert_time.get(track_id, 0)
            if now - last < self.cooldown_sec:
                return False  # still in cooldown for this track, skip
            self._last_alert_time[track_id] = now

        ts = time.strftime("%Y-%m-%d %H:%M:%S")

        self._beep_async()
        self._log_csv(ts, track_id, camera, idle_seconds, status)
        if self.sqlite_path:
            self._log_sqlite(ts, track_id, camera, idle_seconds, status)

        return True

    def clear_track(self, track_id):
        """Call this when a track goes back to ACTIVE or leaves the frame,
        so a future idle period starts a fresh cooldown instead of being
        suppressed by an old timestamp."""
        with self._lock:
            self._last_alert_time.pop(track_id, None)