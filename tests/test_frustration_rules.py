"""Tests for the frustration rule table: python -m unittest tests.test_frustration_rules -v"""
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import frustration_rules as R  # noqa: E402

CALM = {"eda_pct": 0, "scr_pct": 0, "hr_pct": 0, "rmssd_pct": 0, "sdnn_pct": 0}
LOW_SIG = {"eda_pct": 12, "scr_pct": 12, "hr_pct": 11, "rmssd_pct": -12, "sdnn_pct": -12}
MEDIUM_SIG = {"eda_pct": 22, "scr_pct": 27, "hr_pct": 16, "rmssd_pct": -21, "sdnn_pct": -21}
HIGH_SIG = {"eda_pct": 35, "scr_pct": 45, "hr_pct": 22, "rmssd_pct": -31, "sdnn_pct": -31}


class SheetRowTests(unittest.TestCase):
    """One test per sheet row group: signals + context exactly as written."""

    def check(self, signals, context, level, cause):
        r = R.evaluate(signals, context)
        self.assertEqual((r.level, r.cause), (level, cause), (signals, context))

    def test_rework(self):
        self.check(LOW_SIG, {"repeated_failures": 2}, R.LOW, "rework")
        self.check(MEDIUM_SIG, {"repeated_failures": 5}, R.MEDIUM, "rework")
        self.check(HIGH_SIG, {"repeated_failures": 6}, R.HIGH, "rework")

    def test_schedule_pressure(self):
        self.check(LOW_SIG, {"delay_pct": 7}, R.LOW, "schedule_pressure")
        self.check(MEDIUM_SIG, {"delay_pct": 15}, R.MEDIUM, "schedule_pressure")
        self.check(HIGH_SIG, {"delay_pct": 25}, R.HIGH, "schedule_pressure")

    def test_equipment(self):
        self.check(LOW_SIG, {"interruptions": 1}, R.LOW, "equipment")
        self.check(MEDIUM_SIG, {"interruptions": 3}, R.MEDIUM, "equipment")
        self.check(HIGH_SIG, {"interruptions": 4}, R.HIGH, "equipment")

    def test_material_shortage(self):
        self.check(LOW_SIG, {"waiting_min": 8}, R.LOW, "material_shortage")
        self.check(MEDIUM_SIG, {"waiting_min": 20}, R.MEDIUM, "material_shortage")
        self.check(HIGH_SIG, {"waiting_min": 31}, R.HIGH, "material_shortage")

    def test_poor_communication(self):
        self.check(LOW_SIG, {"clarification_requests": 1}, R.LOW, "poor_communication")
        self.check(MEDIUM_SIG, {"clarification_requests": 2}, R.MEDIUM, "poor_communication")
        self.check(HIGH_SIG, {"clarification_requests": 4}, R.HIGH, "poor_communication")


class StrictnessTests(unittest.TestCase):
    def test_every_condition_is_required(self):
        # Medium Rework with one signal just short (SDNN -14 instead of < -15)
        sig = dict(MEDIUM_SIG, sdnn_pct=-14)
        r = R.evaluate(sig, {"repeated_failures": 4})
        self.assertNotEqual(r.level, R.MEDIUM)

    def test_ranges_use_both_ends(self):
        # Medium Rework is failures 4-5; 6 with Medium signals is not in any row
        r = R.evaluate(MEDIUM_SIG, {"repeated_failures": 6})
        self.assertIsNone(r.cause)
        self.assertEqual(r.level, R.LOW)

    def test_missing_input_is_not_met(self):
        self.assertIsNone(R.evaluate(HIGH_SIG).cause)  # no context -> no row matches
        sig = {k: v for k, v in HIGH_SIG.items() if k not in ("rmssd_pct", "sdnn_pct")}
        r = R.evaluate(sig, {"repeated_failures": 6})
        self.assertNotEqual(r.level, R.HIGH)  # High Rework needs RMSSD and SDNN

    def test_nan_is_missing(self):
        r = R.evaluate(dict(HIGH_SIG, eda_pct=float("nan")), {"repeated_failures": 6})
        self.assertNotEqual(r.level, R.HIGH)

    def test_nothing_matched_is_low(self):
        r = R.evaluate(CALM, {})
        self.assertEqual((r.level, r.label, r.cause), (R.LOW, "Low Frustration", None))


class SeverityBandTests(unittest.TestCase):
    def test_bands(self):
        self.assertEqual(R.score_to_level(0.39), R.LOW)
        self.assertEqual(R.score_to_level(0.40), R.MEDIUM)
        self.assertEqual(R.score_to_level(0.69), R.MEDIUM)
        self.assertEqual(R.score_to_level(0.70), R.HIGH)
        self.assertEqual(R.score_to_level(1.00), R.HIGH)


class CoachIntegrationTests(unittest.TestCase):
    def test_prediction_feeds_the_three_in_a_row_rule(self):
        import speak_biomarker_labels as sbl
        coach = sbl.VoiceCoach()
        t = datetime(2026, 11, 17, 9, 0)
        high = (HIGH_SIG, {"repeated_failures": 6})
        medium = (MEDIUM_SIG, {"repeated_failures": 4})
        spoken = []
        for minute, (sig, ctx) in enumerate([high, high, medium, high, high, high]):
            pred = R.as_prediction(R.evaluate(sig, ctx))
            key, _ = coach.evaluate({"predictions": {"frustration": pred}}, t + timedelta(minutes=minute))
            if key in sbl.FRUSTRATION_ALERTS:
                spoken.append(minute)
        self.assertEqual(spoken, [5])  # Medium at minute 2 reset the count

    def test_not_ready(self):
        self.assertFalse(R.as_prediction(None)["ready"])


if __name__ == "__main__":
    unittest.main()
