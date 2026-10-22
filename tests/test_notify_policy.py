"""Team rule: only High/Very High is spoken, on the 3rd High occurrence.
python -m unittest discover -s tests -v"""
import contextlib
import io
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from notify_policy import ThirdTimeRule  # noqa: E402
import speak_biomarker_labels as sbl  # noqa: E402


def run_rule(levels_high, step=5, reset_s=60):
    """levels_high: list of bools, one per `step` seconds. Returns times an alert was allowed."""
    rule, spoken = ThirdTimeRule(reset_s=reset_s), []
    for i, high in enumerate(levels_high):
        t = i * step
        rule.observe(t, high)
        if high and rule.allow(t):
            spoken.append(t)
    return rule, spoken


class ThirdTimeRuleTests(unittest.TestCase):
    def test_third_episode_is_spoken(self):
        high, low = [True] * 6, [False] * 13           # 30 s high, 65 s low (ends the episode)
        _, spoken = run_rule(high + low + high + low + high)
        self.assertTrue(spoken)
        self.assertGreaterEqual(spoken[0], 2 * (6 + 13) * 5)   # only during the 3rd episode

    def test_one_long_high_stretch_counts_once(self):
        rule, spoken = run_rule([True] * 200)
        self.assertEqual(spoken, [])
        self.assertEqual(rule.count, 1)

    def test_short_dip_does_not_start_a_new_occurrence(self):
        rule, spoken = run_rule(([True] * 6 + [False] * 4) * 5)   # 20 s dips < 60 s reset
        self.assertEqual(spoken, [])
        self.assertEqual(rule.count, 1)

    def test_count_starts_again_after_speaking(self):
        ep = [True] * 6 + [False] * 13
        rule, spoken = run_rule(ep * 6)
        third, sixth = 2 * 19 * 5, 5 * 19 * 5
        self.assertTrue(any(third <= t < third + 30 for t in spoken))
        self.assertTrue(any(sixth <= t < sixth + 30 for t in spoken))
        self.assertFalse(any(third + 95 <= t < sixth for t in spoken))   # 4th and 5th silent


class StressCoachRuleTests(unittest.TestCase):
    def test_stress_simulation_speaks_on_third_high_episode(self):
        coach, spoken = sbl.VoiceCoach(), []
        with contextlib.redirect_stdout(io.StringIO()) as out:
            for sample, now in sbl.simulated_samples(0):
                key, _ = coach.evaluate(sample, now)
                if key:
                    spoken.append((key, int((now - coach.s.shift_start).total_seconds() // 60)))
        stress = [m for k, m in spoken if k in sbl.STRESS_ALERTS]
        self.assertTrue(stress)
        self.assertGreaterEqual(min(stress), 28)                     # not on High #1 (min 12) or #2 (min 20)
        self.assertIn("stress High occurrence 2/3: tracked", out.getvalue())
        self.assertNotIn("focus", [k for k, _ in spoken])            # 1st MD-high episode only: silent

    def test_medium_stress_is_never_spoken(self):
        coach = sbl.VoiceCoach()
        start = datetime(2026, 1, 1)
        pred = lambda label: {"label": label, "confidence": 0.9, "ready": True}  # noqa: E731
        keys = []
        for m in range(60):
            sample = {"pulse_rate": 90, "temp": 33.0,
                      "predictions": {"frustration": pred("medium"), "mental_demand": pred("low")}}
            keys.append(coach.evaluate(sample, start + timedelta(minutes=m))[0])
        self.assertFalse(set(keys) & sbl.STRESS_ALERTS)


if __name__ == "__main__":
    unittest.main()
