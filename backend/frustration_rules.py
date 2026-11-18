"""Frustration decision rules (rule side only -- the % change calculation lives elsewhere).

Transcribed exactly from the "DBR Frustration" sheet:
- cause thresholds (Rework, Schedule Pressure, Equipment, Material Shortage, Poor Communication)
- Frustration Severity Output bands (0.00-0.39 Low, 0.40-0.69 Medium, 0.70-1.00 High)

A cause/level matches only when EVERY condition the sheet lists for it holds, including the
non-sensor one (failures, delay, interruptions, waiting time, clarification requests).
A missing input means that condition is not met.

INPUT (from the calculation step): percent change from baseline,
    X% = (X_current - X_baseline) / X_baseline * 100
    signals = {"eda_pct": 22.0, "scr_pct": 30.0, "hr_pct": 12.0, "rmssd_pct": -18.0, "sdnn_pct": -16.0}
CONTEXT (not measured by the sensors):
    context = {"repeated_failures": 4, "delay_pct": 12, "interruptions": 2,
               "waiting_min": 15, "clarification_requests": 2}

OUTPUT: evaluate(signals, context) -> FrustrationResult(level 1/2/3, label, cause, cause_label)
Feed it into the existing 3-consecutive-High rule with as_prediction().
"""
import math
from dataclasses import dataclass

LOW, MEDIUM, HIGH = 1, 2, 3
LEVEL_LABELS = {LOW: "Low Frustration", MEDIUM: "Medium Frustration", HIGH: "High Frustration"}
LEVEL_NAMES = {LOW: "low", MEDIUM: "medium", HIGH: "high"}

# ---- Thresholds per cause and level (as written in the sheet) -------------------------
# (input key, op, value):  ">" / "<" strict,  "==" exact,  "range" = (low, high) inclusive
CAUSES = {
    "rework": {  # Rework / Repeated Task Failure
        LOW: [("eda_pct", ">", 10), ("scr_pct", ">", 10), ("hr_pct", ">", 5),
              ("rmssd_pct", "<", -10), ("sdnn_pct", "<", -10), ("repeated_failures", "range", (2, 3))],
        MEDIUM: [("eda_pct", ">", 20), ("scr_pct", ">", 25), ("hr_pct", ">", 10),
                 ("rmssd_pct", "<", -15), ("sdnn_pct", "<", -15), ("repeated_failures", "range", (4, 5))],
        HIGH: [("eda_pct", ">", 30), ("scr_pct", ">", 40), ("hr_pct", ">", 15),
               ("rmssd_pct", "<", -25), ("sdnn_pct", "<", -25), ("repeated_failures", ">", 5)],
    },
    "schedule_pressure": {  # Schedule Pressure
        LOW: [("eda_pct", ">", 10), ("hr_pct", ">", 10), ("rmssd_pct", "<", -10),
              ("sdnn_pct", "<", -10), ("delay_pct", "range", (5, 10))],
        MEDIUM: [("eda_pct", ">", 15), ("hr_pct", ">", 15), ("rmssd_pct", "<", -20),
                 ("sdnn_pct", "<", -20), ("delay_pct", "range", (10, 20))],
        HIGH: [("eda_pct", ">", 25), ("hr_pct", ">", 20), ("rmssd_pct", "<", -30),
               ("sdnn_pct", "<", -30), ("delay_pct", ">", 20)],
    },
    "equipment": {  # Equipment / Tool Problems
        LOW: [("eda_pct", ">", 10), ("scr_pct", ">", 10), ("hr_pct", ">", 5),
              ("interruptions", "==", 1)],
        MEDIUM: [("eda_pct", ">", 20), ("scr_pct", ">", 25), ("hr_pct", ">", 10),
                 ("interruptions", "range", (2, 3))],
        HIGH: [("eda_pct", ">", 30), ("scr_pct", ">", 40), ("hr_pct", ">", 15),
               ("rmssd_pct", "<", -25), ("sdnn_pct", "<", -25), ("interruptions", ">", 3)],
    },
    "material_shortage": {  # Material Shortage / Waiting Time
        LOW: [("eda_pct", ">", 5), ("hr_pct", ">", 5), ("rmssd_pct", "<", -10),
              ("waiting_min", "range", (5, 10))],
        MEDIUM: [("eda_pct", ">", 15), ("hr_pct", ">", 10), ("rmssd_pct", "<", -15),
                 ("sdnn_pct", "<", -15), ("waiting_min", "range", (10, 30))],
        HIGH: [("eda_pct", ">", 25), ("hr_pct", ">", 15), ("rmssd_pct", "<", -25),
               ("sdnn_pct", "<", -25), ("waiting_min", ">", 30)],
    },
    "poor_communication": {  # Poor Communication / Unclear Instructions
        LOW: [("eda_pct", ">", 10), ("hr_pct", ">", 5), ("clarification_requests", "==", 1)],
        MEDIUM: [("eda_pct", ">", 20), ("hr_pct", ">", 10), ("rmssd_pct", "<", -15),
                 ("sdnn_pct", "<", -15), ("clarification_requests", "range", (2, 3))],
        HIGH: [("eda_pct", ">", 30), ("scr_pct", ">", 40), ("hr_pct", ">", 15),
               ("rmssd_pct", "<", -25), ("sdnn_pct", "<", -25), ("clarification_requests", ">", 3)],
    },
}
CAUSE_ORDER = list(CAUSES)  # sheet order; used only if two causes match at the same level
CAUSE_LABELS = {
    "rework": "Rework / Repeated Task Failure",
    "schedule_pressure": "Schedule Pressure",
    "equipment": "Equipment / Tool Problems",
    "material_shortage": "Material Shortage / Waiting Time",
    "poor_communication": "Poor Communication / Unclear Instructions",
    None: "No cause matched",
}

# ---- Frustration Severity Output (sheet) ------------------------------------------------
SCORE_BANDS = [(0.70, HIGH), (0.40, MEDIUM), (0.0, LOW)]  # 0.70-1.00 / 0.40-0.69 / 0.00-0.39


@dataclass
class FrustrationResult:
    level: int        # 1 Low, 2 Medium, 3 High
    label: str        # "High Frustration"
    cause: str = None
    cause_label: str = "No cause matched"


def _missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def _check(value, op, limit):
    if _missing(value):
        return False  # the sheet needs every condition; no data -> not met
    if op == ">":
        return value > limit
    if op == "<":
        return value < limit
    if op == "==":
        return value == limit
    if op == "range":
        return limit[0] <= value <= limit[1]
    raise ValueError(op)


def level_matches(cause, level, values):
    return all(_check(values.get(key), op, limit) for key, op, limit in CAUSES[cause][level])


def cause_level(cause, values):
    """Highest level of this cause whose conditions ALL hold; 0 if none."""
    for level in (HIGH, MEDIUM, LOW):
        if level_matches(cause, level, values):
            return level
    return 0


def score_to_level(score):
    """Frustration Severity Output: 0..1 score -> Low / Medium / High."""
    for floor, level in SCORE_BANDS:
        if score >= floor:
            return level
    return LOW


def evaluate(signals, context=None):
    values = dict(signals or {})
    values.update(context or {})
    levels = {c: cause_level(c, values) for c in CAUSE_ORDER}
    best = max(CAUSE_ORDER, key=lambda c: (levels[c], -CAUSE_ORDER.index(c)))
    if levels[best] == 0:
        # nothing in the sheet matched -> Low (the assignment has only Low / Medium / High)
        return FrustrationResult(LOW, LEVEL_LABELS[LOW])
    return FrustrationResult(levels[best], LEVEL_LABELS[levels[best]], best, CAUSE_LABELS[best])


def as_prediction(result):
    """Shape a result like the ML prediction the voice coach already reads
    (speak_biomarker_labels.coach_label), so the 3-consecutive-High rule works unchanged."""
    if result is None:
        return {"label": None, "confidence": None, "ready": False}
    return {"label": LEVEL_NAMES[result.level], "confidence": None, "ready": True,
            "cause": result.cause, "cause_label": result.cause_label, "source": "rules"}