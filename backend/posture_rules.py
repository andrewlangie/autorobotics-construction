"""Posture decision table, transcribed from:
- "Polar H10 Angle-Based Posture Detection Framework" (angle / duration thresholds)
- "Feedback System - Posture" sheet (messages, closed-loop actions, durations, goals)

Values marked TEAM-DECISION are not in the sheets -- they are our defaults, change them here.
"""

LEVEL_LABELS = {1: "Good Posture", 2: "Poor Posture", 3: "Critical Posture"}

# ---- Angle thresholds (degrees from the worker's own upright calibration) ---
# cause: (measurement, poor above, critical above)
ANGLE_RULES = {
    "forward_bending": ("forward", 20, 45),  # pitch
    "side_bending": ("side", 10, 20),        # roll
}

# ---- Static posture ---------------------------------------------------------
# Framework: >5 min bad, >10 min critical. (The feedback sheet's 30-60 s column is for
# angles; for "holding still" the framework's minutes make more sense -- TEAM-DECISION.)
STATIC_MOVEMENT_G = 0.02   # TEAM-DECISION: average movement below this = holding still
STATIC_POOR_S = 5 * 60
STATIC_CRITICAL_S = 10 * 60

# ---- Duration gate for angle causes (sheet "Duration" column) ----------------
DURATION_S = {2: 30, 3: 60}  # poor after 30 s, critical after 60 s (lower end of the ranges)

# ---- Causes a chest strap cannot detect (kept so the messages exist) ---------
# twisting: needs a gyroscope; overhead_work: needs an arm/wrist sensor;
# asymmetric_lifting: needs load / both-side sensors. See posture_math.py.
NOT_DETECTED = ("twisting", "overhead_work", "asymmetric_lifting")

CAUSE_LABELS = {
    "forward_bending": "Forward Bending",
    "side_bending": "Side Bending",
    "static_posture": "Static Posture",
    "twisting": "Twisting / Rotation",
    "overhead_work": "Overhead Work",
    "asymmetric_lifting": "Asymmetric Lifting",
}

FEEDBACK = {
    (2, "forward_bending"): ("Straighten your back and lift your head slightly.",
                             "Encourage neutral spine alignment", "Reduce spinal loading"),
    (2, "twisting"): ("Reposition your feet before continuing.",
                      "Encourage whole-body movement instead of torso twisting", "Reduce torsional stress"),
    (2, "overhead_work"): ("Lower your arms briefly if safe.",
                           "Encourage posture reset", "Reduce shoulder fatigue"),
    (2, "asymmetric_lifting"): ("Distribute weight evenly and adjust your stance.",
                                "Improve lifting symmetry", "Reduce musculoskeletal strain"),
    (2, "static_posture"): ("Change position and stretch briefly.",
                            "Encourage posture variation", "Improve circulation and reduce stiffness"),
    (3, "forward_bending"): ("Unsafe posture detected. Stand upright and reposition.",
                             "Reduce trunk flexion immediately", "Prevent lower back injury"),
    (3, "twisting"): ("Stop and reposition before continuing.",
                      "Eliminate excessive trunk rotation", "Prevent spinal injury"),
    (3, "overhead_work"): ("Take a short arm recovery break if safe.",
                           "Reduce overhead exposure", "Reduce shoulder injury risk"),
    (3, "asymmetric_lifting"): ("Realign your body before lifting.",
                                "Correct lifting mechanics", "Reduce injury risk"),
    (3, "static_posture"): ("Move and stretch before continuing.",
                            "Force posture change or micro-break", "Reduce cumulative musculoskeletal fatigue"),
    # Side bending has no row in the sheet -> the sheet's general Poor / Critical lines
    (2, "side_bending"): ("Adjust your posture and maintain a neutral working position.",
                          "Encourage posture reset", "Reduce spinal loading"),
    (3, "side_bending"): ("Unsafe posture detected. Correct posture immediately if safe.",
                          "Reduce trunk side flexion immediately", "Prevent spinal injury"),
}
