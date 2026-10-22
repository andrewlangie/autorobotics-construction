"""Tests for the rule accuracy evaluator: python -m unittest discover -s tests -v"""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import evaluate_rules as ev  # noqa: E402
import polar_simulator  # noqa: E402


class ParsingTests(unittest.TestCase):
    def test_times(self):
        self.assertEqual(ev.parse_time("5:30"), 330)
        self.assertEqual(ev.parse_time("1:00:00"), 3600)
        self.assertEqual(ev.parse_time("90"), 90)

    def test_levels(self):
        self.assertEqual(ev.parse_level("posture", "Critical"), 3)
        self.assertEqual(ev.parse_level("fatigue", "very_high"), 4)
        self.assertEqual(ev.parse_level("fatigue", "2"), 2)
        with self.assertRaises(ValueError):
            ev.parse_level("posture", "terrible")

    def test_label_lookup_last_one_wins(self):
        labels = [(0, 100, "posture", 1, None), (50, 60, "posture", 3, "forward_bending")]
        self.assertEqual(ev.label_at(labels, "posture", 10), (1, None))
        self.assertEqual(ev.label_at(labels, "posture", 55), (3, "forward_bending"))
        self.assertIsNone(ev.label_at(labels, "fatigue", 10))
        self.assertIsNone(ev.label_at(labels, "posture", 100))  # end is exclusive


class SimulatedRecordingTests(unittest.TestCase):
    """End to end on the simulated shift (its labels come from the simulator script)."""

    @classmethod
    def setUpClass(cls):
        cls.folder = Path(tempfile.mkdtemp())
        with contextlib.redirect_stdout(io.StringIO()):
            polar_simulator.save(cls.folder)
        cls.rows = ev.compare([("sim", ev.replay_events(cls.folder),
                                ev.load_labels(cls.folder / "labels.csv"))])

    def test_both_tracks_compared(self):
        tracks = {r["track"] for r in self.rows}
        self.assertEqual(tracks, {"posture", "fatigue"})

    def test_posture_rules_score_well_on_clear_data(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = ev.report(self.rows, "posture")
        self.assertGreater(result["accuracy"], 0.85)

    def test_wrong_labels_score_badly(self):
        wrong = [dict(r, true_level=3) for r in self.rows if r["track"] == "posture"]
        with contextlib.redirect_stdout(io.StringIO()):
            result = ev.report(wrong, "posture")
        self.assertLess(result["accuracy"], 0.3)


if __name__ == "__main__":
    unittest.main()
