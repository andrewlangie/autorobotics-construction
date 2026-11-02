"""Assignment rule: feedback only after 3 CONSECUTIVE High predictions; Low/Medium resets.
python -m unittest discover -s tests -v"""
import contextlib
import io
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from notify_policy import ConsecutiveHighRule, ThirdTimeRule  # noqa: E402
import speak_biomarker_labels as sbl  # noqa: E402

H, M, L = "high", "medium", "low"


def fire_points(labels):
    """Index of every prediction that triggers feedback."""
    rule = ConsecutiveHighRule()
    return [i for i, lab in enumerate(labels)
            if rule.update(None if lab is None else lab == H)]


class ConsecutiveHighRuleTests(unittest.TestCase):
    def test_third_high_in_a_row_triggers(self):
        self.assertEqual(fire_points([H, H, H]), [2])

    def test_two_highs_never_trigger(self):
        self.assertEqual(fire_points([H, H, L, H, H, M, H, H]), [])

    def test_medium_resets_the_count(self):
        self.assertEqual(fire_points([H, H, M, H, H, H]), [5])

    def test_low_resets_the_count(self):
        self.assertEqual(fire_points([H, H, L, H, H, H]), [5])

    def test_medium_alone_never_triggers(self):
        self.assertEqual(fire_points([M] * 30), [])

    def test_count_starts_again_after_feedback(self):
        self.assertEqual(fire_points([H] * 9), [2, 5, 8])

    def test_missing_prediction_neither_counts_nor_resets(self):
        self.assertEqual(fire_points([H, None, H, None, H]), [4])

    def test_times_fired_tracks_one_unbroken_run(self):
        rule = ConsecutiveHighRule()
        for _ in range(6):
            rule.update(True)
        self.assertEqual(rule.times_fired, 2)
        rule.update(False)
        self.assertEqual((rule.count, rule.times_fired), (0, 0))


class PostureEpisodeRuleTests(unittest.TestCase):
    """Posture (Team 2) still uses the older episode rule."""

    def test_third_episode_is_spoken(self):
        rule, spoken = ThirdTimeRule(reset_s=60), []
        ep = [True] * 6 + [False] * 13
        for i, high in enumerate(ep * 3):
            rule.observe(i * 5, high)
            if high and rule.allow(i * 5):
                spoken.append(i * 5)
        self.assertTrue(spoken)
        self.assertGreaterEqual(spoken[0], 2 * 19 * 5)


def run_coach(labels):
    """Feed one frustration prediction per minute; return {minute: alert_key}."""
    coach, start, out = sbl.VoiceCoach(), datetime(2026, 1, 1), {}
    pred = lambda label: {"label": label, "confidence": 0.9, "ready": True}  # noqa: E731
    for m, lab in enumerate(labels):
        sample = {"pulse_rate": 85, "temp": 33.0,
                  "predictions": {"frustration": pred(lab), "mental_demand": pred(L)}}
        key, _ = coach.evaluate(sample, start + timedelta(minutes=m))
        if key:
            out[m] = key
    return out


class FrustrationCoachTests(unittest.TestCase):
    def test_spoken_only_on_third_consecutive_high(self):
        self.assertEqual(run_coach([L, H, H, H]), {3: "checkin"})

    def test_medium_resets_and_is_never_spoken(self):
        alerts = run_coach([H, H, M, H, H, M, M, M, H, H, L])
        self.assertEqual(alerts, {})

    def test_continuing_high_escalates(self):
        alerts = run_coach([H] * 9)
        self.assertEqual(alerts, {2: "checkin", 5: "microbreak", 8: "required_break"})

    def test_recovery_after_feedback(self):
        alerts = run_coach([H, H, H] + [L] * 5)
        self.assertEqual(alerts, {2: "checkin", 7: "recovery"})

    def test_low_confidence_is_ignored(self):
        coach, start = sbl.VoiceCoach(), datetime(2026, 1, 1)
        keys = []
        for m, conf in enumerate([0.9, 0.9, 0.3, 0.9]):
            sample = {"predictions": {"frustration": {"label": H, "confidence": conf, "ready": True}}}
            keys.append(coach.evaluate(sample, start + timedelta(minutes=m))[0])
        self.assertEqual(keys, [None, None, None, "checkin"])   # 0.3 skipped, not a reset

    def test_simulation(self):
        spoken = []
        with contextlib.redirect_stdout(io.StringIO()) as out:
            coach = sbl.VoiceCoach()
            for sample, now in sbl.simulated_samples(0):
                key, _ = coach.evaluate(sample, now)
                if key:
                    spoken.append((int((now - coach.s.shift_start).total_seconds() // 60), key))
        frustration = [(m, k) for m, k in spoken if k in sbl.FRUSTRATION_ALERTS]
        self.assertEqual(frustration, [(20, "checkin"), (23, "microbreak"), (26, "required_break")])
        self.assertIn((31, "recovery"), spoken)
        del out


if __name__ == "__main__":
    unittest.main()
