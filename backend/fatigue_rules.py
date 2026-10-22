"""Fatigue decision table, transcribed from:
- "Complete Fatigue Prediction Framework Using Polar H10" (levels, formulas, cause patterns)
- "Feedback System - Fatigue" sheet (messages, closed-loop actions, durations, goals)

Values marked TEAM-DECISION are not in the sheets -- they are our defaults, change them here.
"""

# ---- Baseline -------------------------------------------------------------
BASELINE_S = 180  # TEAM-DECISION: first 3 minutes of the shift (after 60 s of beats) = baseline
SMOOTHING_S = 60  # TEAM-DECISION: % changes are averaged over ~1 min before deciding a level

# ---- General fatigue level (% change from baseline) ------------------------
# Level fires when AT LEAST 2 OF THE 3 conditions hold (TEAM-DECISION: the sheet
# lists all three; requiring all three almost never happens with real data).
LEVEL_THRESHOLDS = {  # level: (HR% above, RMSSD% below, SDNN% below)
    4: (25, -40, -35),  # Very High
    3: (15, -35, -30),  # High
    2: (10, -25, -20),  # Medium
    1: (5, -15, -15),   # Low
}
CONDITIONS_NEEDED = 2
LEVEL_LABELS = {0: "No fatigue", 1: "Low Fatigue", 2: "Medium Fatigue", 3: "High Fatigue", 4: "Very High Fatigue"}

# ---- Duration gate (sheet "Duration" column) -------------------------------
# A level must hold this many seconds before we speak. Level 4 also fires at once on
# "rapid deterioration" = level jumped by RAPID_JUMP or more within RAPID_WINDOW_S.
DURATION_S = {2: 30, 3: 60, 4: 120}
RAPID_JUMP = 2
RAPID_WINDOW_S = 60

# ---- Fatigue score (framework "optional") ----------------------------------
# Each part is scaled 0..1 against the Very High threshold (TEAM-DECISION).
SCORE_WEIGHTS = (0.4, 0.3, 0.3)  # HR, RMSSD, SDNN
SCORE_SCALE = (25, -40, -35)

# ---- Cause patterns (framework arrows -> numbers, TEAM-DECISION) ------------
# HR:  slight_up 5..10 %,  up 10..15 %,  up2 > 15 %
# HRV: down -15..-25 %,  down2 -25..-40 %,  down3 < -40 %
HR_BANDS = [("up2", 15), ("up", 10), ("slight_up", 5)]
HRV_BANDS = [("down3", -40), ("down2", -25), ("down", -15)]

CAUSE_PATTERNS = {  # cause: (HR, RMSSD, SDNN). Order = tie-break order.
    "cardiovascular": ("up", "down2", "down2"),
    "prolonged_exertion": ("up2", "down2", "down2"),
    "heat": ("up2", "down", "down"),
    "sleep": ("slight_up", "down3", "down2"),
    "long_work": ("up", "down2", "down2"),  # same as cardiovascular -> needs the tie-break below
}
LONG_WORK_AFTER_MIN = 240  # TEAM-DECISION: same pattern + 4 h on shift -> "long_work"
HEAT_SKIN_TEMP_C = 35.0    # TEAM-DECISION: if a skin temperature is supplied and above this -> "heat"
MIN_MATCHES = 2            # a cause needs at least 2 of its 3 arrows to match

CAUSE_LABELS = {
    "cardiovascular": "Cardiovascular Fatigue",
    "prolonged_exertion": "Prolonged Physical Exertion",
    "heat": "Heat-Related Fatigue",
    "sleep": "Sleep-Related Fatigue",
    "long_work": "Long Work Duration",
    None: "General fatigue",
}

# ---- Feedback (sheet: Feedback, Closed-Loop Action, Feedback Goal) ----------
FEEDBACK = {
    (2, "cardiovascular"): ("Take several slow breaths and monitor your effort.",
                            "Encourage pacing reduction", "Improve autonomic recovery"),
    (2, "prolonged_exertion"): ("Reduce work intensity briefly if safe.",
                                "Lower physical workload", "Prevent fatigue accumulation"),
    (2, "heat"): ("Hydrate when possible and reduce exertion.",
                  "Hydration reminder", "Prevent heat-related deterioration"),
    (2, "sleep"): ("Increase awareness and focus on safe work practices.",
                   "Increase monitoring frequency", "Maintain alertness"),
    (2, "long_work"): ("Take a short recovery moment if safe.",
                       "Schedule micro-break", "Reduce cumulative fatigue"),
    (3, "cardiovascular"): ("High fatigue detected. Slow your pace and recover.",
                            "Reduce workload intensity", "Restore physiological balance"),
    (3, "prolonged_exertion"): ("Take a short rest break if safe.",
                                "Initiate recovery break", "Prevent performance deterioration"),
    (3, "heat"): ("Move to a cooler area and hydrate if available.",
                  "Reduce heat exposure", "Reduce heat strain"),
    (3, "sleep"): ("Increase caution and avoid rushing.",
                   "Reduce task complexity", "Reduce error likelihood"),
    (3, "long_work"): ("Take a recovery break before continuing.",
                       "Pause task temporarily", "Improve recovery"),
    (4, "cardiovascular"): ("Critical fatigue detected. Stop and recover immediately.",
                            "Suspend physical activity", "Prevent physiological overexertion"),
    (4, "prolonged_exertion"): ("Recovery is required before continuing work.",
                                "Mandatory break recommendation", "Prevent exhaustion"),
    (4, "heat"): ("Stop work and move to a recovery area.",
                  "Remove worker from heat source", "Prevent heat illness"),
    (4, "sleep"): ("Critical fatigue detected. Increase recovery before continuing.",
                   "Delay safety-critical tasks", "Prevent fatigue-related incidents"),
    (4, "long_work"): ("Immediate recovery break recommended.",
                       "Trigger fatigue management protocol", "Restore physiological recovery"),
    # TEAM-DECISION: the sheet has no line for "fatigue but no cause pattern matched"
    (2, None): ("Fatigue is building. Ease your pace if safe.",
                "Encourage pacing reduction", "Prevent fatigue accumulation"),
    (3, None): ("High fatigue detected. Take a short rest break if safe.",
                "Initiate recovery break", "Restore physiological balance"),
    (4, None): ("Critical fatigue detected. Stop and recover before continuing.",
                "Suspend physical activity", "Prevent exhaustion"),
}
