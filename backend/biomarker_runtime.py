import json
from collections import deque
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import frustration_math
import frustration_rules

# Frustration is RULES-ONLY and uses the Empatica wristband live feed only
# (no Polar H10, no frustration ML model). Every signal is a percent change from a
# personal baseline taken from the first BASELINE_MINUTES minute-samples of the session.
BASELINE_MINUTES = 3
FRUSTRATION_SIGNALS = ("eda", "scr", "hr", "rmssd", "sdnn")

# Non-sensor context the rules need. TEAM-DECISION: nothing fills these yet, so they
# default to 0 and the rules cannot go above Low until a source is wired in.
CONTEXT_KEYS = (
    "repeated_failures",
    "delay_pct",
    "interruptions",
    "waiting_min",
    "clarification_requests",
)


def _num(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) else v


class LiveBiomarkerPredictor:
    def __init__(self, model_dir, max_gap_minutes=5):
        model_dir = Path(model_dir)
        # Only the mental-demand model is loaded; frustration comes from the rules.
        self.mental_demand_model = joblib.load(model_dir / "mental_demand_model.joblib")
        metadata_path = model_dir / "model_metadata.json"
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        self.max_gap_minutes = max_gap_minutes
        self.reset()

    def reset(self):
        self.history = deque(maxlen=10)
        self.last_participant_id = None
        self.last_timestamp = None
        self.session_minute_index = -1
        self.baseline = frustration_math.BaselineTracker(
            minutes=BASELINE_MINUTES, keys=FRUSTRATION_SIGNALS
        )

    def _parse_timestamp(self, sample):
        for key in ("eda_time", "temp_time", "pulse_time", "updated_at"):
            value = sample.get(key)
            if not value:
                continue
            try:
                return pd.to_datetime(value)
            except Exception:
                continue
        return pd.Timestamp(datetime.now())

    def _maybe_reset(self, participant_id, current_ts):
        if self.last_participant_id is None:
            return

        if participant_id != self.last_participant_id:
            self.reset()
            return

        if self.last_timestamp is not None:
            gap_minutes = abs((current_ts - self.last_timestamp).total_seconds()) / 60.0
            if gap_minutes > self.max_gap_minutes:
                self.reset()

    def _predict_with_confidence(self, model, features_df):
        label = model.predict(features_df)[0]
        confidence = None
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(features_df)[0]
                confidence = float(np.max(probs))
            except Exception:
                confidence = None
        return label, confidence

    def _frustration_prediction(self, sample):
        """Rules-only frustration from the Empatica feed (frustration_rules.py)."""
        current = {
            "eda": _num(sample.get("eda")),
            "scr": _num(sample.get("scr")),
            "rmssd": _num(sample.get("rmssd")),
            "sdnn": _num(sample.get("sdnn")),
        }
        hr = _num(sample.get("hr"))
        current["hr"] = hr if hr is not None else _num(sample.get("pulse_rate"))

        was_ready = self.baseline.ready
        self.baseline.add(current)  # no-op once the baseline is complete
        if not was_ready:
            # The sample that completes the baseline is not itself evaluated.
            return {
                "label": None,
                "confidence": None,
                "ready": False,
                "source": "rules",
                "reason": f"Measuring baseline ({self.baseline.count}/{BASELINE_MINUTES} minutes).",
            }

        signals = frustration_math.compute_signals(current, self.baseline.values())
        if all(v is None for v in signals.values()):
            # A minute with no usable data must not count as a calm Low.
            return {
                "label": None,
                "confidence": None,
                "ready": False,
                "source": "rules",
                "reason": "No usable Empatica signals this minute.",
            }

        context = {k: _num(sample.get(k)) or 0 for k in CONTEXT_KEYS}
        result = frustration_rules.evaluate(signals, context)
        prediction = frustration_rules.as_prediction(result)
        prediction.update({"reason": None, "signals": signals, "context": context})
        return prediction

    def update(self, sample):
        participant_id = sample.get("participant_id")
        current_ts = self._parse_timestamp(sample)

        self._maybe_reset(participant_id, current_ts)
        self.last_participant_id = participant_id
        self.last_timestamp = current_ts
        self.session_minute_index += 1

        current_point = {
            "eda": float(sample["eda"]) if sample.get("eda") is not None else np.nan,
            "temp": float(sample["temp"]) if sample.get("temp") is not None else np.nan,
            "pulse_rate": float(sample["pulse_rate"]) if sample.get("pulse_rate") is not None else np.nan,
            "minute_index": self.session_minute_index,
            "timestamp": str(current_ts),
        }
        self.history.append(current_point)

        response = {
            "session_minute_index": self.session_minute_index,
            "history_size": len(self.history),
            "model_notes": [
                "Frustration uses rules only (frustration_rules.py) on the Empatica feed: "
                "EDA, SCR, HR, RMSSD and SDNN as percent change from a personal baseline.",
                f"The first {BASELINE_MINUTES} minutes of a session set the baseline; frustration is not predicted until then.",
                "Mental demand uses the ML model on EDA + temperature + session minute index.",
            ],
            "frustration": self._frustration_prediction(sample),
            "mental_demand": {
                "label": None,
                "confidence": None,
                "ready": False,
                "reason": "Need at least 1 minute-level sample for the mental-demand model.",
            },
        }

        md_features = pd.DataFrame(
            [
                {
                    "eda": current_point["eda"],
                    "temp": current_point["temp"],
                    "minute_index": current_point["minute_index"],
                }
            ]
        )
        md_label, md_conf = self._predict_with_confidence(self.mental_demand_model, md_features)
        response["mental_demand"] = {
            "label": md_label,
            "confidence": md_conf,
            "ready": True,
            "reason": None,
        }

        return response
