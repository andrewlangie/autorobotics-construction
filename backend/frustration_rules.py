# backend/frustration_rules.py

def evaluate_cause_rules(pcts, context):
    """
    pcts: dict from frustration_math (eda_pct, scr_pct, hr_pct, rmssd_pct, sdnn_pct)
    context: dict containing contextual variables:
        - repeated_failures (int)
        - delay_pct (float)
        - interruptions (int)
        - waiting_min (float)
        - clarification_requests (int)
    """
    cause = "None"
    level = "Low"
    score = 0.20

    # 1. Rework / Repeated Task Failure
    rf = context.get("repeated_failures", 0)
    if pcts["eda_pct"] > 30 and pcts["scr_pct"] > 40 and pcts["hr_pct"] > 15 and pcts["rmssd_pct"] < -25 and pcts["sdnn_pct"] < -25 and rf > 5:
        return "Rework / Repeated Task Failure", "High", 0.85
    elif pcts["eda_pct"] > 20 and pcts["scr_pct"] > 25 and pcts["hr_pct"] > 10 and pcts["rmssd_pct"] < -15 and pcts["sdnn_pct"] < -15 and 4 <= rf <= 5:
        return "Rework / Repeated Task Failure", "Medium", 0.55
    elif pcts["eda_pct"] > 10 and pcts["scr_pct"] > 10 and pcts["hr_pct"] > 5 and pcts["rmssd_pct"] < -10 and pcts["sdnn_pct"] < -10 and 2 <= rf <= 3:
        return "Rework / Repeated Task Failure", "Low", 0.30

    # 2. Schedule Pressure
    dp = context.get("delay_pct", 0.0)
    if pcts["eda_pct"] > 25 and pcts["hr_pct"] > 20 and pcts["rmssd_pct"] < -30 and pcts["sdnn_pct"] < -30 and dp > 20:
        return "Schedule Pressure", "High", 0.85
    elif pcts["eda_pct"] > 15 and pcts["hr_pct"] > 15 and pcts["rmssd_pct"] < -20 and pcts["sdnn_pct"] < -20 and 10 < dp <= 20:
        return "Schedule Pressure", "Medium", 0.55
    elif pcts["eda_pct"] > 10 and pcts["hr_pct"] > 10 and pcts["rmssd_pct"] < -10 and pcts["sdnn_pct"] < -10 and 5 <= dp <= 10:
        return "Schedule Pressure", "Low", 0.30

    # 3. Equipment / Tool Problems
    intr = context.get("interruptions", 0)
    if pcts["eda_pct"] > 30 and pcts["scr_pct"] > 40 and pcts["hr_pct"] > 15 and pcts["rmssd_pct"] < -25 and pcts["sdnn_pct"] < -25 and intr > 3:
        return "Equipment / Tool Problems", "High", 0.85
    elif pcts["eda_pct"] > 20 and pcts["scr_pct"] > 25 and pcts["hr_pct"] > 10 and 2 <= intr <= 3:
        return "Equipment / Tool Problems", "Medium", 0.55
    elif pcts["eda_pct"] > 10 and pcts["scr_pct"] > 10 and pcts["hr_pct"] > 5 and intr == 1:
        return "Equipment / Tool Problems", "Low", 0.30

    # 4. Material Shortage / Waiting Time
    wm = context.get("waiting_min", 0.0)
    if pcts["eda_pct"] > 25 and pcts["hr_pct"] > 15 and pcts["rmssd_pct"] < -25 and pcts["sdnn_pct"] < -25 and wm > 30:
        return "Material Shortage / Waiting Time", "High", 0.85
    elif pcts["eda_pct"] > 15 and pcts["hr_pct"] > 10 and pcts["rmssd_pct"] < -15 and pcts["sdnn_pct"] < -15 and 10 < wm <= 30:
        return "Material Shortage / Waiting Time", "Medium", 0.55
    elif pcts["eda_pct"] > 5 and pcts["hr_pct"] > 5 and pcts["rmssd_pct"] < -10 and 5 <= wm <= 10:
        return "Material Shortage / Waiting Time", "Low", 0.30

    # 5. Poor Communication / Unclear Instructions
    cr = context.get("clarification_requests", 0)
    if pcts["eda_pct"] > 30 and pcts["scr_pct"] > 40 and pcts["hr_pct"] > 15 and pcts["rmssd_pct"] < -25 and pcts["sdnn_pct"] < -25 and cr > 3:
        return "Poor Communication", "High", 0.85
    elif pcts["eda_pct"] > 20 and pcts["hr_pct"] > 10 and pcts["rmssd_pct"] < -15 and pcts["sdnn_pct"] < -15 and 2 <= cr <= 3:
        return "Poor Communication", "Medium", 0.55
    elif pcts["eda_pct"] > 10 and pcts["hr_pct"] > 5 and cr == 1:
        return "Poor Communication", "Low", 0.30

    return "General", "Low", 0.20