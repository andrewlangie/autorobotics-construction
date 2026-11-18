"""Tests for the frustration pipeline (Empatica feed -> % change -> rules -> voice coach).
python -m unittest discover -s tests -v"""
import contextlib
import io
import math
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import biomarker_runtime  # noqa: E402
import frustration_math as fm  # noqa: E402
import frustration_rules as fr  # noqa: E402
import speak_biomarker_labels as sbl  # noqa: E402

H, M, L = "high", "medium", "low"

# Signals/context that satisfy EVERY condition of one cause at one level.
RULE_CASES = {
    ("rework", fr.LOW): ({"eda_pct": 12, "scr_pct": 12, "hr_pct": 6, "rmssd_pct": -12, "sdnn_pct": -12},
                         {"repeated_failures": 2}),
    ("rework", fr.MEDIUM): ({"eda_pct": 22, "scr_pct": 30, "hr_pct": 12, "rmssd_pct": -18, "sdnn_pct": -18},
                            {"repeated_failures": 4}),
    ("rework", fr.HIGH): ({"eda_pct": 35, "scr_pct": 45, "hr_pct": 20, "rmssd_pct": -30, "sdnn_pct": -30},
                          {"repeated_failures": 6}),
    ("schedule_pressure", fr.LOW): ({"eda_pct": 12, "hr_pct": 12, "rmssd_pct": -12, "sdnn_pct": -12},
                                    {"delay_pct": 7}),
    ("schedule_pressure", fr.MEDIUM): ({"eda_pct": 18, "hr_pct": 17, "rmssd_pct": -22, "sdnn_pct": -22},
                                       {"delay_pct": 15}),
    ("schedule_pressure", fr.HIGH): ({"eda_pct": 28, "hr_pct": 22, "rmssd_pct": -32, "sdnn_pct": -32},
                                     {"delay_pct": 25}),
    ("equipment", fr.LOW): ({"eda_pct": 12, "scr_pct": 12, "hr_pct": 6}, {"interruptions": 1}),
    ("equipment", fr.MEDIUM): ({"eda_pct": 22, "scr_pct": 30, "hr_pct": 12}, {"interruptions": 2}),
    ("equipment", fr.HIGH): ({"eda_pct": 35, "scr_pct": 45, "hr_pct": 20, "rmssd_pct": -30, "sdnn_pct": -30},
                             {"interruptions": 4}),
    ("material_shortage", fr.LOW): ({"eda_pct": 7, "hr_pct": 6, "rmssd_pct": -12}, {"waiting_min": 7}),
    ("material_shortage", fr.MEDIUM): ({"eda_pct": 18, "hr_pct": 12, "rmssd_pct": -18, "sdnn_pct": -18},
                                       {"waiting_min": 15}),
    ("material_shortage", fr.HIGH): ({"eda_pct": 28, "hr_pct": 18, "rmssd_pct": -28, "sdnn_pct": -28},
                                     {"waiting_min": 40}),
    ("poor_communication", fr.LOW): ({"eda_pct": 12, "hr_pct": 6}, {"clarification_requests": 1}),
    ("poor_communication", fr.MEDIUM): ({"eda_pct": 22, "hr_pct": 12, "rmssd_pct": -18, "sdnn_pct": -18},
                                        {"clarification_requests": 2}),
    ("poor_communication", fr.HIGH): ({"eda_pct": 35, "scr_pct": 45, "hr_pct": 20, "rmssd_pct": -30,
                                       "sdnn_pct": -30}, {"clarification_requests": 4}),
}


class PercentChangeTests(unittest.TestCase):
    def test_formula(self):
        self.assertAlmostEqual(fm.percent_change(3.2, 2.5), 28.0)
        self.assertAlmostEqual(fm.percent_change(31, 38), -18.421052, places=5)

    def test_unusable_values_give_none(self):
        self.assertIsNone(fm.percent_change(None, 2.0))
        self.assertIsNone(fm.percent_change(2.0, None))
        self.assertIsNone(fm.percent_change(2.0, 0))            # no divide by zero
        self.assertIsNone(fm.percent_change(2.0, -1))
        self.assertIsNone(fm.percent_change(float("nan"), 2.0))
        self.assertIsNone(fm.percent_change(True, 2.0))

    def test_compute_signals_names_and_gaps(self):
        out = fm.compute_signals({"eda": 3.0, "hr": 84}, {"eda": 2.5, "scr": 4.0, "hr": 75})
        self.assertEqual(set(out), {"eda_pct", "scr_pct", "hr_pct", "rmssd_pct", "sdnn_pct"})
        self.assertAlmostEqual(out["eda_pct"], 20.0)
        self.assertAlmostEqual(out["hr_pct"], 12.0)
        self.assertIsNone(out["scr_pct"])                       # no current value
        self.assertIsNone(out["rmssd_pct"])


class BaselineTrackerTests(unittest.TestCase):
    def test_ready_after_n_samples_then_frozen(self):
        b = fm.BaselineTracker(minutes=3, keys=("eda", "hr"))
        for eda in (1.0, 2.0, 3.0):
            self.assertFalse(b.ready)
            b.add({"eda": eda, "hr": 70})
        self.assertTrue(b.ready)
        b.add({"eda": 100, "hr": 200})                          # ignored once ready
        self.assertEqual(b.values(), {"eda": 2.0, "hr": 70.0})

    def test_missing_values_are_skipped_but_minute_still_counts(self):
        b = fm.BaselineTracker(minutes=3, keys=("eda", "scr"))
        b.add({"eda": 2.0})
        b.add({"eda": 4.0, "scr": float("nan")})
        b.add(None)
        self.assertTrue(b.ready)
        self.assertEqual(b.values(), {"eda": 3.0, "scr": None})


class RuleTests(unittest.TestCase):
    def test_every_cause_and_level(self):
        for (cause, level), (signals, context) in RULE_CASES.items():
            with self.subTest(cause=cause, level=level):
                result = fr.evaluate(signals, context)
                self.assertEqual((result.cause, result.level), (cause, level))

    def test_nothing_matches_means_low_without_cause(self):
        result = fr.evaluate({"eda_pct": 1, "hr_pct": 0}, {})
        self.assertEqual((result.level, result.cause), (fr.LOW, None))

    def test_strong_signals_alone_are_not_enough(self):
        signals, _ = RULE_CASES[("rework", fr.HIGH)]
        result = fr.evaluate(signals, {"repeated_failures": 0})   # context condition missing
        self.assertIsNone(result.cause)
        self.assertEqual(result.level, fr.LOW)

    def test_missing_signal_means_condition_not_met(self):
        signals, context = RULE_CASES[("rework", fr.HIGH)]
        for bad in (None, float("nan")):
            result = fr.evaluate(dict(signals, scr_pct=bad), context)
            self.assertNotEqual(result.level, fr.HIGH)

    def test_thresholds_are_strict(self):
        signals, context = RULE_CASES[("rework", fr.HIGH)]
        self.assertEqual(fr.evaluate(dict(signals, eda_pct=30), context).level, fr.LOW)   # not > 30
        self.assertEqual(fr.evaluate(dict(signals, eda_pct=30.1), context).level, fr.HIGH)

    def test_context_ranges_include_both_ends(self):
        signals, _ = RULE_CASES[("rework", fr.MEDIUM)]
        for failures in (4, 5):
            self.assertEqual(fr.evaluate(signals, {"repeated_failures": failures}).level, fr.MEDIUM)
        self.assertNotEqual(fr.evaluate(signals, {"repeated_failures": 6}).level, fr.MEDIUM)

    def test_tie_goes_to_the_first_cause_in_sheet_order(self):
        signals, _ = RULE_CASES[("rework", fr.HIGH)]
        result = fr.evaluate(signals, {"repeated_failures": 6, "interruptions": 4,
                                       "clarification_requests": 4})
        self.assertEqual((result.cause, result.level), ("rework", fr.HIGH))

    def test_higher_level_beats_sheet_order(self):
        signals, _ = RULE_CASES[("poor_communication", fr.HIGH)]
        result = fr.evaluate(signals, {"repeated_failures": 2, "clarification_requests": 4})
        self.assertEqual((result.cause, result.level), ("poor_communication", fr.HIGH))

    def test_severity_bands(self):
        self.assertEqual(fr.score_to_level(0.39), fr.LOW)
        self.assertEqual(fr.score_to_level(0.40), fr.MEDIUM)
        self.assertEqual(fr.score_to_level(0.69), fr.MEDIUM)
        self.assertEqual(fr.score_to_level(0.70), fr.HIGH)
        self.assertEqual(fr.score_to_level(1.0), fr.HIGH)

    def test_prediction_shape_matches_what_the_coach_reads(self):
        pred = fr.as_prediction(fr.evaluate(*RULE_CASES[("equipment", fr.HIGH)]))
        self.assertEqual(pred["label"], "high")
        self.assertIsNone(pred["confidence"])
        self.assertTrue(pred["ready"])
        self.assertEqual((pred["cause"], pred["source"]), ("equipment", "rules"))
        self.assertFalse(fr.as_prediction(None)["ready"])

    def test_evaluate_raw_goes_from_readings_to_result(self):
        base = {"eda": 2.0, "scr": 4.0, "hr": 70, "rmssd": 40, "sdnn": 50}
        cur = {"eda": 2.8, "scr": 6.0, "hr": 84, "rmssd": 28, "sdnn": 35}   # +40, +50, +20, -30, -30
        result = fm.evaluate_raw(cur, base, {"repeated_failures": 6})
        self.assertEqual((result.cause, result.level), ("rework", fr.HIGH))


class FakeModel:
    def predict(self, X):
        return ["low"]


class RuntimeTests(unittest.TestCase):
    """The live predictor: rules only, Empatica feed only."""

    BASE = {"participant_id": "P1", "eda": 2.0, "scr": 4.0, "hr": 70, "rmssd": 40, "sdnn": 50,
            "temp": 33.0, "pulse_rate": 70}
    STRESSED = {"eda": 2.8, "scr": 6.0, "hr": 84, "rmssd": 28, "sdnn": 35}

    def setUp(self):
        self.model_dir = Path(tempfile.mkdtemp())
        with mock.patch.object(biomarker_runtime.joblib, "load", return_value=FakeModel()) as load:
            self.predictor = biomarker_runtime.LiveBiomarkerPredictor(self.model_dir)
        self.load = load
        self.start = datetime(2026, 1, 1, 10, 0)

    def step(self, minute, **fields):
        sample = dict(self.BASE, updated_at=(self.start + timedelta(minutes=minute)).isoformat(), **fields)
        return self.predictor.update(sample)

    def finish_baseline(self):
        for m in range(3):
            self.step(m)

    def test_only_the_mental_demand_model_is_loaded(self):
        self.assertEqual([c.args[0].name for c in self.load.call_args_list],
                         ["mental_demand_model.joblib"])
        self.assertFalse(hasattr(self.predictor, "frustration_model"))

    def test_not_ready_while_measuring_baseline(self):
        for m in range(3):
            pred = self.step(m)["frustration"]
            self.assertFalse(pred["ready"])
            self.assertIsNone(pred["label"])
            self.assertIn(f"{m + 1}/3", pred["reason"])
        self.assertTrue(self.step(3)["frustration"]["ready"])   # 4th minute is the first evaluated

    def test_baseline_minutes_never_look_like_calm(self):
        self.assertIsNone(coach_label_of(self.step(0)["frustration"]))

    def test_rules_prediction_with_cause_once_ready(self):
        self.finish_baseline()
        pred = self.step(3, repeated_failures=6, **self.STRESSED)["frustration"]
        self.assertEqual((pred["label"], pred["cause"], pred["source"]), ("high", "rework", "rules"))
        self.assertIsNone(pred["confidence"])
        self.assertAlmostEqual(pred["signals"]["eda_pct"], 40.0)
        self.assertAlmostEqual(pred["signals"]["rmssd_pct"], -30.0)
        self.assertEqual(pred["context"]["repeated_failures"], 6)

    def test_without_context_the_rules_stay_low(self):
        self.finish_baseline()
        pred = self.step(3, **self.STRESSED)["frustration"]
        self.assertEqual(pred["label"], "low")
        self.assertIsNone(pred["cause"])

    def test_pulse_rate_is_the_fallback_for_hr(self):
        for m in range(3):
            self.predictor.update({"participant_id": "P1", "eda": 2.0, "pulse_rate": 70,
                                   "updated_at": (self.start + timedelta(minutes=m)).isoformat()})
        pred = self.predictor.update({"participant_id": "P1", "eda": 2.0, "pulse_rate": 84,
                                      "updated_at": (self.start + timedelta(minutes=3)).isoformat()})
        self.assertAlmostEqual(pred["frustration"]["signals"]["hr_pct"], 20.0)

    def test_minute_without_signals_is_not_ready(self):
        self.finish_baseline()
        empty = {"participant_id": "P1", "updated_at": (self.start + timedelta(minutes=3)).isoformat()}
        pred = self.predictor.update(empty)["frustration"]
        self.assertFalse(pred["ready"])
        self.assertIsNone(pred["label"])

    def test_new_participant_starts_a_new_baseline(self):
        self.finish_baseline()
        other = dict(self.BASE, participant_id="P2",
                     updated_at=(self.start + timedelta(minutes=3)).isoformat())
        self.assertFalse(self.predictor.update(other)["frustration"]["ready"])

    def test_long_gap_starts_a_new_baseline(self):
        self.finish_baseline()
        self.assertFalse(self.step(30)["frustration"]["ready"])

    def test_mental_demand_still_uses_the_ml_model(self):
        md = self.step(0)["mental_demand"]
        self.assertEqual((md["label"], md["ready"]), ("low", True))


def coach_label_of(pred):
    return sbl.coach_label(pred, 0.5)


def run_coach(preds):
    """Feed one frustration prediction per minute; return {minute: (alert_key, cause)}."""
    coach, start, out = sbl.VoiceCoach(), datetime(2026, 1, 1), {}
    for m, pred in enumerate(preds):
        sample = {"pulse_rate": 85, "temp": 33.0,
                  "predictions": {"frustration": pred,
                                  "mental_demand": {"label": L, "confidence": 0.9, "ready": True}}}
        key, ctx = coach.evaluate(sample, start + timedelta(minutes=m))
        if key:
            out[m] = (key, ctx["cause"])
    return out


def rules_pred(label, cause=None):
    return {"label": label, "confidence": None, "ready": True, "cause": cause, "source": "rules"}


class CoachWithRulesTests(unittest.TestCase):
    def test_none_confidence_is_accepted(self):
        self.assertEqual(sbl.coach_label(rules_pred(H), 0.5), H)

    def test_not_ready_rules_prediction_is_ignored(self):
        self.assertIsNone(sbl.coach_label({"label": None, "confidence": None, "ready": False}, 0.5))

    def test_feedback_on_third_consecutive_high_carries_the_cause(self):
        alerts = run_coach([rules_pred(H, "rework")] * 3)
        self.assertEqual(alerts, {2: ("checkin", "rework")})

    def test_cause_can_change_between_tiers(self):
        preds = [rules_pred(H, "equipment")] * 3 + [rules_pred(H, "material_shortage")] * 3
        self.assertEqual(run_coach(preds), {2: ("checkin", "equipment"),
                                            5: ("microbreak", "material_shortage")})

    def test_medium_and_low_reset_and_are_not_spoken(self):
        preds = [rules_pred(H, "rework"), rules_pred(H, "rework"), rules_pred(M, "rework"),
                 rules_pred(H, "rework"), rules_pred(H, "rework"), rules_pred(L)]
        self.assertEqual(run_coach(preds), {})

    def test_baseline_minutes_do_not_reset_a_high_run(self):
        not_ready = {"label": None, "confidence": None, "ready": False}
        preds = [rules_pred(H, "rework"), rules_pred(H, "rework"), not_ready, rules_pred(H, "rework")]
        self.assertEqual(run_coach(preds), {3: ("checkin", "rework")})


class CauseMessageTests(unittest.TestCase):
    def test_every_cause_has_every_tier(self):
        for cause in fr.CAUSES:
            for tier in sbl.FRUSTRATION_ALERTS:
                with self.subTest(cause=cause, tier=tier):
                    options = sbl.CAUSE_MESSAGES["en"][cause][tier]
                    self.assertTrue(options)

    def test_messages_format_with_name_and_minutes(self):
        for cause, tiers in sbl.CAUSE_MESSAGES["en"].items():
            for tier, options in tiers.items():
                for text in options:
                    out = text.format(name=", Carlos", minutes=6, cause=cause)
                    self.assertNotIn("{", out)

    def test_cause_names_match_the_rules(self):
        self.assertEqual(set(sbl.CAUSE_MESSAGES["en"]), set(fr.CAUSES))

    def test_every_tier_has_a_generic_fallback(self):
        for tier in sbl.FRUSTRATION_ALERTS:
            self.assertTrue(sbl.COACH_MESSAGES["en"][tier])
            self.assertTrue(sbl.COACH_MESSAGES["es"][tier])


class CoachRunTests(unittest.TestCase):
    """The coach loop end to end, with no audio."""

    def setUp(self):
        self.log = Path(tempfile.mkdtemp()) / "voice_alert_log.csv"
        patcher = mock.patch.object(sbl, "COACH_LOG_PATH", self.log)
        patcher.start()
        self.addCleanup(patcher.stop)

    def args(self, **kw):
        base = dict(simulate=True, coach=True, speed=0, lang="en", worker_name="", vary=False,
                    dry_run=True, endpoint="", interval=1, debug=False)
        base.update(kw)
        return types.SimpleNamespace(**base)

    def test_simulation_speaks_cause_specific_feedback(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            sbl.run_coach(self.args())
        text = out.getvalue()
        self.assertIn("Redoing work is frustrating", text)             # min 20, rework check-in
        self.assertIn("repeated rework", text)                         # min 23, rework micro-break
        self.assertIn("escalate the material shortage", text)          # min 26, required break

    def test_spanish_falls_back_to_generic_wording(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            sbl.run_coach(self.args(lang="es"))
        self.assertNotIn("Redoing work", out.getvalue())
        self.assertIn("[speaking", out.getvalue())

    def test_log_has_cause_column(self):
        with contextlib.redirect_stdout(io.StringIO()):
            sbl.run_coach(self.args())
        lines = self.log.read_text(encoding="utf-8").splitlines()
        self.assertTrue(lines[0].endswith(",message,cause"))
        causes = [line.rsplit(",", 1)[1] for line in lines[1:]]
        self.assertIn("rework", causes)
        self.assertIn("material_shortage", causes)

    def test_simulated_predictions_use_rules_shape(self):
        with contextlib.redirect_stdout(io.StringIO()):
            preds = [s["predictions"]["frustration"] for s, _ in sbl.simulated_samples(0)]
        self.assertTrue(all(p["confidence"] is None for p in preds))
        high = [p for p in preds if p["label"] == H]
        self.assertTrue(high and all(p.get("cause") for p in high))
        self.assertTrue(all(p.get("cause") in fr.CAUSES for p in high))


if __name__ == "__main__":
    unittest.main()