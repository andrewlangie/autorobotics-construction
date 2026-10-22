"""Tests for the dashboard status file: python -m unittest discover -s tests -v"""
import contextlib
import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import polar_coach  # noqa: E402
from polar_simulator import simulated_events  # noqa: E402
from polar_status import HISTORY_MINUTES, StatusWriter, read_status  # noqa: E402


class StatusTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "polar_status.json"

    def test_missing_file_means_not_running(self):
        self.assertFalse(read_status(self.path)["available"])

    def test_write_read_and_history_window(self):
        w = StatusWriter(self.path, source="simulation")
        ok_f = {"status": "ok", "level": 2, "label": "Medium Fatigue", "score": 0.4, "hr": 80.0}
        ok_p = {"status": "ok", "level": 1, "label": "Good Posture", "forward_deg": 5.0, "side_deg": 1.0}
        for i in range(0, 15 * 12):  # 15 minutes, every 5 s
            w.update(i / 12, ok_f, ok_p)
        s = read_status(self.path)
        self.assertTrue(s["available"])
        self.assertFalse(s["stale"])
        self.assertEqual(s["source"], "simulation")
        self.assertEqual(s["fatigue"]["label"], "Medium Fatigue")
        span = s["history"][-1]["m"] - s["history"][0]["m"]
        self.assertLessEqual(span, HISTORY_MINUTES)  # only the last 10 minutes are kept

    def test_old_file_is_stale(self):
        StatusWriter(self.path).write({"updated_at": time.time() - 60, "history": []})
        self.assertTrue(read_status(self.path)["stale"])

    def test_coach_run_writes_last_message(self):
        w = StatusWriter(self.path, source="simulation")
        polar_coach.LOG_PATH = self.path.with_name("log.csv")
        with contextlib.redirect_stdout(io.StringIO()):
            polar_coach.run(simulated_events(seed=7), speed=0, dry_run=True, status=w)
        data = json.loads(self.path.read_text())
        self.assertIsNotNone(data["last_message"])
        self.assertIn("goal", data["last_message"])
        self.assertTrue(data["history"])


if __name__ == "__main__":
    unittest.main()
