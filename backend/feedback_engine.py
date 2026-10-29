"""Decision engine for the Polar H10: fatigue (HR + HRV) and posture (accelerometer).

Fatigue pipeline:
    heartbeats -> 60 s HRV window -> % change from baseline -> Low / Medium / High
    -> one prediction per minute -> 3 consecutive High predictions -> Decision
Low and Medium are never spoken; a Low or Medium prediction resets the count
(notify_policy.ConsecutiveHighRule). Posture still uses the duration gate + ThirdTimeRule.
A Decision carries the sheet's message, closed-loop action and feedback goal.

Usage (see polar_coach.py):
    fatigue, posture = FatigueEngine(), PostureEngine()
    fatigue.add_beat(t, hr, rr_ms)        # every heartbeat
    posture.add_acc(t, ax, ay, az)        # every accelerometer sample
    d = fatigue.evaluate(t)               # every few seconds -> Decision or None
    if d is spoken: fatigue.mark_spoken(d)
"""
from collections import deque
from dataclasses import asdict, dataclass

import fatigue_rules as FR
import posture_rules as PR
from hrv import RRWindow, pct_change
from notify_policy import ConsecutiveHighRule, ThirdTimeRule
from posture_math import PostureTracker

POSTURE_REPEAT_S = 120  # TEAM-DECISION: repeat a still-active posture message after 2 min
CALM_RESET_S = 60       # level must stay <= 1 this long before the same level can fire again

# Which message wins when both tracks want to speak: higher urgency first,
# then track order (posture = immediate injury risk). TEAM-DECISION.
URGENCY = {"fatigue": {"high": 2, "escalated": 3}, "posture": {2: 1, 3: 3}}
TRACK_ORDER = {"posture": 0, "fatigue": 1}


@dataclass(frozen=True)
class Decision:
    track: str          # "fatigue" or "posture"
    time: float         # seconds
    level: int
    level_label: str
    cause: str | None
    cause_label: str
    message: str        # what the worker hears
    action: str         # closed-loop action (logged)
    goal: str           # feedback goal (logged)
    urgency: int

    def to_dict(self):
        return asdict(self)


class Gate:
    """Duration gate from the sheets: a level must hold for DURATION_S[level] before it is spoken.

    Also stops the same level repeating every few seconds, lets an escalation through at
    once (after its own duration), and optionally lets a rapid jump to the top level skip
    the wait ("rapid deterioration").
    """

    def __init__(self, durations, repeat_s, rapid_jump=None, rapid_window_s=None):
        self.durations, self.repeat_s = durations, repeat_s
        self.rapid_jump, self.rapid_window_s = rapid_jump, rapid_window_s
        self.since = {}           # level -> time it was first reached (continuously)
        self.history = deque()    # (t, level) for the rapid check
        self.spoken_level, self.spoken_t, self.calm_since = 0, None, None

    def observe(self, t, level):
        for lvl in self.durations:
            if level >= lvl:
                self.since.setdefault(lvl, t)
            else:
                self.since.pop(lvl, None)
        self.history.append((t, level))
        while self.rapid_window_s and t - self.history[0][0] > self.rapid_window_s:
            self.history.popleft()
        if level <= 1:
            self.calm_since = self.calm_since if self.calm_since is not None else t
            if t - self.calm_since >= CALM_RESET_S:
                self.spoken_level = 0
        else:
            self.calm_since = None

    def ready(self, t, level):
        """Level that may be spoken now (0 = nothing)."""
        gated = 0
        for lvl in sorted(self.durations, reverse=True):
            if lvl in self.since and t - self.since[lvl] >= self.durations[lvl]:
                gated = lvl
                break
        top = max(self.durations)
        if self.rapid_jump and level == top:
            if min(lvl for _, lvl in self.history) <= top - self.rapid_jump:
                gated = top
        if gated < 2:
            return 0
        if gated > self.spoken_level:
            return gated
        if self.spoken_t is not None and t - self.spoken_t >= self.repeat_s:
            return gated
        return 0

    def mark_spoken(self, t, level):
        self.spoken_level, self.spoken_t = level, t


# --------------------------------------------------------------------------- fatigue

def _band(value, bands, higher):
    for name, limit in bands:
        if (value > limit) if higher else (value < limit):
            return name
    return None


def fatigue_level(hr_pct, rmssd_pct, sdnn_pct):
    """Low (1), Medium (2) or High (3): highest level where at least CONDITIONS_NEEDED
    of the 3 thresholds are met; below Medium is Low."""
    for level in (FR.HIGH, FR.MEDIUM):
        hr_t, rmssd_t, sdnn_t = FR.LEVEL_THRESHOLDS[level]
        met = (hr_pct > hr_t) + (rmssd_pct < rmssd_t) + (sdnn_pct < sdnn_t)
        if met >= FR.CONDITIONS_NEEDED:
            return level
    return FR.LOW


def fatigue_cause(hr_pct, rmssd_pct, sdnn_pct, minutes_on_shift=0.0, skin_temp=None):
    """Match the arrows of each cause pattern; return the best cause or None."""
    if skin_temp is not None and skin_temp >= FR.HEAT_SKIN_TEMP_C and hr_pct > 10:
        return "heat"
    observed = (
        _band(hr_pct, FR.HR_BANDS, higher=True),
        _band(rmssd_pct, FR.HRV_BANDS, higher=False),
        _band(sdnn_pct, FR.HRV_BANDS, higher=False),
    )
    best, best_matches = None, 0
    for cause, pattern in FR.CAUSE_PATTERNS.items():
        matches = sum(o == p for o, p in zip(observed, pattern))
        if matches > best_matches:  # strict ">" keeps the earlier cause on ties
            best, best_matches = cause, matches
    if best_matches < FR.MIN_MATCHES:
        return None
    if best == "cardiovascular" and minutes_on_shift >= FR.LONG_WORK_AFTER_MIN:
        return "long_work"  # identical arrows -> time on shift decides
    return best


def fatigue_score(hr_pct, rmssd_pct, sdnn_pct):
    parts = [max(0.0, min(1.0, v / s)) for v, s in zip((hr_pct, rmssd_pct, sdnn_pct), FR.SCORE_SCALE)]
    return sum(w * p for w, p in zip(FR.SCORE_WEIGHTS, parts))


class FatigueEngine:
    def __init__(self):
        self.window = RRWindow(60)
        self.baseline_samples = []
        self.baseline = None   # (hr, rmssd, sdnn)
        self.first_t = None
        self.first_metric_t = None
        self.smooth = None       # smoothed (HR%, RMSSD%, SDNN%)
        self.last_eval_t = None
        self.rule = ConsecutiveHighRule()  # spoken only after 3 consecutive High predictions
        self.next_prediction_t = None
        self.predictions = []    # (t, level) of every fatigue prediction (one per minute)
        self.pending = None      # triggered feedback waiting for the speaker
        self.state = {"status": "waiting for heartbeats"}

    def add_beat(self, t, hr, rr_ms):
        if self.first_t is None:
            self.first_t = t
        self.window.add(t, rr_ms)

    def evaluate(self, t, skin_temp=None):
        m = self.window.metrics()
        if m is None:
            self.state = {"status": "collecting beats"}
            return None
        if self.baseline is None:
            if self.first_metric_t is None:
                self.first_metric_t = t
            self.baseline_samples.append((m.hr, m.rmssd, m.sdnn))
            if t - self.first_metric_t >= FR.BASELINE_S:
                n = len(self.baseline_samples)
                self.baseline = tuple(sum(s[i] for s in self.baseline_samples) / n for i in range(3))
            self.state = {"status": "measuring baseline", "hr": m.hr, "rmssd": m.rmssd, "sdnn": m.sdnn}
            return None

        raw = (pct_change(m.hr, self.baseline[0]),
               pct_change(m.rmssd, self.baseline[1]),
               pct_change(m.sdnn, self.baseline[2]))
        # Smooth the % changes (HRV from 60 s of beats is noisy; one noisy window must
        # not trigger a "very high fatigue" alarm). Exponential moving average.
        if self.smooth is None:
            self.smooth = raw
        else:
            dt = max(t - self.last_eval_t, 1e-3)
            a = dt / (FR.SMOOTHING_S + dt)
            self.smooth = tuple(a * r + (1 - a) * s for r, s in zip(raw, self.smooth))
        self.last_eval_t = t
        hr_p, rmssd_p, sdnn_p = self.smooth
        level = fatigue_level(hr_p, rmssd_p, sdnn_p)
        minutes = (t - self.first_t) / 60
        cause = fatigue_cause(hr_p, rmssd_p, sdnn_p, minutes, skin_temp) if level >= 2 else None
        self.state = {
            "status": "ok", "level": level, "label": FR.LEVEL_LABELS[level], "cause": cause,
            "hr": m.hr, "rmssd": m.rmssd, "sdnn": m.sdnn,
            "hr_pct": hr_p, "rmssd_pct": rmssd_p, "sdnn_pct": sdnn_p,
            "score": fatigue_score(hr_p, rmssd_p, sdnn_p),
        }
        self.state["occurrence"] = self.rule.occurrence
        if self.next_prediction_t is None:
            self.next_prediction_t = t + FR.PREDICTION_EVERY_S
        if t < self.next_prediction_t:
            return self.pending  # not a new prediction yet (pending = deferred feedback)
        while self.next_prediction_t <= t:
            self.next_prediction_t += FR.PREDICTION_EVERY_S

        # ---- one fatigue prediction: Low / Medium / High ----
        self.predictions.append((t, level))
        fired = self.rule.update(level == FR.HIGH)   # Low or Medium resets the count
        self.state["occurrence"] = self.rule.occurrence
        if level != FR.HIGH:
            self.pending = None   # Low / Medium is never spoken
            return None
        if not fired:
            return self.pending   # High #1 / #2 in a row: tracked, not spoken
        escalated = self.rule.times_fired >= 2
        message, action, goal = (FR.ESCALATED if escalated else FR.FEEDBACK)[cause]
        label = FR.LEVEL_LABELS[FR.HIGH] + (" (continuing)" if escalated else "")
        self.pending = Decision("fatigue", t, FR.HIGH, label, cause, FR.CAUSE_LABELS[cause],
                                message, action, goal,
                                URGENCY["fatigue"]["escalated" if escalated else "high"])
        return self.pending

    def mark_spoken(self, decision):
        self.pending = None


# --------------------------------------------------------------------------- posture

def angle_level(forward_deg, side_deg):
    """(level, cause) from the angle thresholds; level 1 = good posture."""
    values = {"forward": forward_deg, "side": side_deg}
    best_level, best_cause = 1, None
    for cause, (measure, poor, critical) in PR.ANGLE_RULES.items():
        v = values[measure]
        level = 3 if v > critical else 2 if v > poor else 1
        if level > best_level:  # ties keep forward bending (listed first)
            best_level, best_cause = level, cause
    return best_level, best_cause


class PostureEngine:
    def __init__(self):
        self.tracker = PostureTracker()
        self.angle_gate = Gate(PR.DURATION_S, POSTURE_REPEAT_S)
        self.static_gate = Gate({2: 0, 3: 0}, POSTURE_REPEAT_S)  # static timing is in the level itself
        self.rule = ThirdTimeRule()  # Poor/Critical spoken only on the 3rd bad-posture occurrence
        self.still_since = None
        self.state = {"status": "waiting for accelerometer"}

    def add_acc(self, t, ax, ay, az):
        self.tracker.update(t, ax, ay, az)

    def evaluate(self, t):
        angles = self.tracker.angles()
        if angles is None:
            self.state = {"status": "calibrating -- stand upright"}
            return None
        forward, side = angles
        a_level, a_cause = angle_level(forward, side)

        movement = self.tracker.movement()
        if self.tracker.movement_window_full() and movement < PR.STATIC_MOVEMENT_G:
            self.still_since = self.still_since if self.still_since is not None else t
        else:
            self.still_since = None
        still_s = 0 if self.still_since is None else t - self.still_since
        s_level = 3 if still_s >= PR.STATIC_CRITICAL_S else 2 if still_s >= PR.STATIC_POOR_S else 1

        level, cause = max((a_level, a_cause), (s_level, "static_posture" if s_level > 1 else None),
                           key=lambda x: x[0])
        self.state = {
            "status": "ok", "level": level, "label": PR.LEVEL_LABELS[level], "cause": cause,
            "forward_deg": forward, "side_deg": side, "movement_g": movement, "still_s": still_s,
        }
        self.angle_gate.observe(t, a_level)
        self.static_gate.observe(t, s_level)
        self.rule.observe(t, level >= 2)
        self.state["occurrence"] = self.rule.occurrence
        a_ready = self.angle_gate.ready(t, a_level)
        s_ready = self.static_gate.ready(t, s_level)
        if not a_ready and not s_ready:
            return None
        if not self.rule.allow(t):  # 1st / 2nd bad-posture occurrence: tracked, not spoken
            if a_ready:
                self.angle_gate.mark_spoken(t, a_ready)
            if s_ready:
                self.static_gate.mark_spoken(t, s_ready)
            self.state["occurrence"] = self.rule.occurrence
            return None
        self.state["occurrence"] = self.rule.occurrence
        if a_ready >= s_ready:
            gated, cause = a_ready, a_cause
        else:
            gated, cause = s_ready, "static_posture"
        if cause is None:  # angle level dropped back to good inside the same tick
            return None
        message, action, goal = PR.FEEDBACK[(gated, cause)]
        return Decision("posture", t, gated, PR.LEVEL_LABELS[gated], cause, PR.CAUSE_LABELS[cause],
                        message, action, goal, URGENCY["posture"][gated])

    def mark_spoken(self, decision):
        gate = self.static_gate if decision.cause == "static_posture" else self.angle_gate
        gate.mark_spoken(decision.time, decision.level)


def choose_primary(decisions):
    """One message at a time: most urgent first, ties -> posture before fatigue."""
    decisions = [d for d in decisions if d is not None]
    if not decisions:
        return None
    return max(decisions, key=lambda d: (d.urgency, -TRACK_ORDER[d.track]))
