"""Tests for the signal maths: python -m unittest discover -s tests -v"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from hrv import RRWindow, clean_rr, pct_change, rmssd, sdnn  # noqa: E402
from posture_math import PostureTracker  # noqa: E402


class HrvTests(unittest.TestCase):
    def test_rmssd_and_sdnn_by_hand(self):
        rr = [800, 810, 790, 820]
        # successive differences 10, -20, 30 -> sqrt((100 + 400 + 900) / 3)
        self.assertAlmostEqual(rmssd(rr), math.sqrt(1400 / 3), places=6)
        mean = sum(rr) / 4
        self.assertAlmostEqual(sdnn(rr), math.sqrt(sum((x - mean) ** 2 for x in rr) / 3), places=6)

    def test_clean_rr_drops_glitches_and_jumps(self):
        cleaned = list(clean_rr([800, 805, 150, 810, 1600, 800, 2500]))
        self.assertEqual(cleaned, [800, 805, 810, 800])

    def test_window_needs_enough_beats_and_drops_old_ones(self):
        w = RRWindow(window_s=60)
        t = 0.0
        for _ in range(10):
            t += 0.8
            w.add(t, 800)
        self.assertIsNone(w.metrics())  # fewer than 20 beats
        for _ in range(200):
            t += 0.8
            w.add(t, 800)
        m = w.metrics()
        self.assertAlmostEqual(m.hr, 75.0, places=3)
        self.assertLessEqual(len(w.beats), 76)  # only the last 60 s are kept

    def test_pct_change(self):
        self.assertAlmostEqual(pct_change(110, 100), 10.0)
        self.assertAlmostEqual(pct_change(30, 40), -25.0)


def feed(tracker, seconds, gravity, start=0.0, rate=25):
    t = start
    for _ in range(int(seconds * rate)):
        tracker.update(t, *gravity)
        t += 1 / rate
    return t


def bent(forward_deg=0.0, side_deg=0.0):
    """Gravity on a chest strap whose X axis points up, Z out of the chest."""
    f, s = math.radians(forward_deg), math.radians(side_deg)
    return (math.cos(f) * math.cos(s), math.sin(s), math.sin(f) * math.cos(s))


class PostureMathTests(unittest.TestCase):
    def test_calibrates_then_measures_forward_bend(self):
        p = PostureTracker()
        t = feed(p, 12, bent())
        self.assertTrue(p.calibrated)
        feed(p, 10, bent(forward_deg=40), start=t)
        forward, side = p.angles()
        self.assertAlmostEqual(forward, 40, delta=1)
        self.assertLess(side, 1)

    def test_side_bend_does_not_look_like_forward_bend(self):
        p = PostureTracker()
        t = feed(p, 12, bent())
        feed(p, 10, bent(side_deg=15), start=t)
        forward, side = p.angles()
        self.assertAlmostEqual(side, 15, delta=1)
        self.assertLess(forward, 1)

    def test_strap_worn_upside_down_still_works(self):
        p = PostureTracker()
        flip = lambda g: (-g[0], g[1], g[2])  # noqa: E731
        t = feed(p, 12, flip(bent()))
        feed(p, 10, flip(bent(forward_deg=30)), start=t)
        self.assertAlmostEqual(p.angles()[0], 30, delta=1)

    def test_movement_is_low_when_still(self):
        p = PostureTracker()
        feed(p, 70, bent())
        self.assertTrue(p.movement_window_full())
        self.assertLess(p.movement(), 0.001)


if __name__ == "__main__":
    unittest.main()
