"""Shares the Polar coach's live state with the dashboard.

polar_coach.py calls StatusWriter.update() every 5 s; it writes polar_status.json in the
project root. serve_dashboard.py reads that file for GET /api/polar-status, and the
dashboard panel (frontend/assets/polar-coach-panel.js) polls that route.

A file (instead of a network call) keeps the two processes independent: the dashboard
works without the coach running, and the coach works without the dashboard.
"""
import json
import os
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATUS_PATH = ROOT / "polar_status.json"
HISTORY_MINUTES = 10   # how much trend the dashboard charts show
STALE_AFTER_S = 15     # no update for this long -> the panel shows "not updating"

FATIGUE_KEYS = ("status", "level", "label", "cause", "hr", "rmssd", "sdnn",
                "hr_pct", "rmssd_pct", "sdnn_pct", "score", "occurrence")
POSTURE_KEYS = ("status", "level", "label", "cause", "forward_deg", "side_deg",
                "movement_g", "still_s", "occurrence")


def _pick(state, keys):
    out = {}
    for k in keys:
        v = state.get(k)
        out[k] = round(v, 3) if isinstance(v, float) else v
    return out


class StatusWriter:
    def __init__(self, path=STATUS_PATH, source="live"):
        self.path = Path(path)
        self.source = source  # "live", "simulation" or "replay"
        self.history = deque()
        self.last_message = None

    def spoken(self, decision, minute):
        """Remember the last spoken decision (shown in the panel)."""
        self.last_message = {
            "track": decision.track, "level": decision.level, "level_label": decision.level_label,
            "cause": decision.cause_label, "message": decision.message,
            "action": decision.action, "goal": decision.goal,
            "minute": round(minute, 1), "spoken_at": time.time(),
        }

    def update(self, minute, fatigue_state, posture_state):
        ok_f = fatigue_state.get("status") == "ok"
        ok_p = posture_state.get("status") == "ok"
        self.history.append({
            "m": round(minute, 2),
            "score": round(fatigue_state["score"], 3) if ok_f else None,
            "f_level": fatigue_state.get("level") if ok_f else None,
            "forward": round(posture_state["forward_deg"], 1) if ok_p else None,
            "p_level": posture_state.get("level") if ok_p else None,
        })
        while self.history and minute - self.history[0]["m"] > HISTORY_MINUTES:
            self.history.popleft()
        self.write({
            "source": self.source,
            "updated_at": time.time(),
            "minute": round(minute, 2),
            "fatigue": _pick(fatigue_state, FATIGUE_KEYS),
            "posture": _pick(posture_state, POSTURE_KEYS),
            "last_message": self.last_message,
            "history": list(self.history),
        })

    def write(self, data):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        os.replace(tmp, self.path)  # atomic: the server never reads a half-written file


def read_status(path=STATUS_PATH):
    """What /api/polar-status returns."""
    path = Path(path)
    if not path.exists():
        return {"available": False, "reason": "Polar coach is not running (python backend/polar_coach.py)"}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"available": False, "reason": "status file is being written -- try again"}
    age = time.time() - data.get("updated_at", 0)
    data.update(available=True, age_s=round(age, 1), stale=age > STALE_AFTER_S)
    return data
