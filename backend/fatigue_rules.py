"""Fatigue decision table, transcribed from:
- "Complete Fatigue Prediction Framework Using Polar H10" (levels, formulas, cause patterns)
- "Feedback System - Fatigue" sheet (messages, closed-loop actions, durations, goals)

Values marked TEAM-DECISION are not in the sheets -- they are our defaults, change them here.
"""

# ---- Baseline -------------------------------------------------------------
BASELINE_S = 180  # TEAM-DECISION: first 3 minutes of the shift (after 60 s of beats) = baseline
SMOOTHING_S = 60  # TEAM-DECISION: % changes are averaged over ~1 min before deciding a level

# ---- Fatigue level (% change from baseline) -------------------------------
# Three levels, as the assignment requires: Low / Medium / High.
# A level fires when AT LEAST 2 OF THE 3 conditions hold (TEAM-DECISION: the sheet
# lists all three; requiring all three almost never happens with real data).
# The sheet's old "Low" (5 %) and "Very High" (25 %) bands are folded into Low and High.
LEVEL_THRESHOLDS = {  # level: (HR% above, RMSSD% below, SDNN% below)
    3: (15, -35, -30),  # High
    2: (10, -25, -20),  # Medium
}
LOW, MEDIUM, HIGH = 1, 2, 3  # anything below Medium is Low
CONDITIONS_NEEDED = 2
LEVEL_LABELS = {1: "Low Fatigue", 2: "Medium Fatigue", 3: "High Fatigue"}

# ---- Prediction rate --------------------------------------------------------
# The engine updates every 5 s for the dashboard, but a fatigue PREDICTION (the thing
# the 3-consecutive-High rule counts) is made once per HRV window. TEAM-DECISION.
PREDICTION_EVERY_S = 60

# ---- Fatigue score (framework "optional") ----------------------------------
# Each part is scaled 0..1 against the old sheet's Very High band (TEAM-DECISION).
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
# Only High is ever spoken. FEEDBACK is the first High feedback (after 3 consecutive High
# predictions). If High continues for another 3 predictions, ESCALATED is spoken instead
# (these are the sheet's old "Very High" lines).
FEEDBACK = {
    "cardiovascular": ("High fatigue detected. Slow your pace and recover.",
                       "Reduce workload intensity", "Restore physiological balance"),
    "prolonged_exertion": ("Take a short rest break if safe.",
                           "Initiate recovery break", "Prevent performance deterioration"),
    "heat": ("Move to a cooler area and hydrate if available.",
             "Reduce heat exposure", "Reduce heat strain"),
    "sleep": ("Increase caution and avoid rushing.",
              "Reduce task complexity", "Reduce error likelihood"),
    "long_work": ("Take a recovery break before continuing.",
                  "Pause task temporarily", "Improve recovery"),
    # TEAM-DECISION: the sheet has no line for "fatigue but no cause pattern matched"
    None: ("High fatigue detected. Take a short rest break if safe.",
           "Initiate recovery break", "Restore physiological balance"),
}
ESCALATED = {
    "cardiovascular": ("Fatigue is still high. Stop and recover now.",
                       "Suspend physical activity", "Prevent physiological overexertion"),
    "prolonged_exertion": ("Recovery is required before continuing work.",
                           "Mandatory break recommendation", "Prevent exhaustion"),
    "heat": ("Stop work and move to a recovery area.",
             "Remove worker from heat source", "Prevent heat illness"),
    "sleep": ("Fatigue is still high. Increase recovery before continuing.",
              "Delay safety-critical tasks", "Prevent fatigue-related incidents"),
    "long_work": ("Immediate recovery break recommended.",
                  "Trigger fatigue management protocol", "Restore physiological recovery"),
    None: ("Fatigue is still high. Stop and recover before continuing.",
           "Suspend physical activity", "Prevent exhaustion"),
}
