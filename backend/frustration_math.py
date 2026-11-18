"""Frustration calculation step (math side only -- the decision rules live in frustration_rules.py).

Converts raw Empatica readings into percent change from baseline:

    X% = (X_current - X_baseline) / X_baseline * 100

for EDA (uS), SCR (uS or peaks/min), HR (bpm), RMSSD (ms) and SDNN (ms).

The output dict uses exactly the keys frustration_rules.evaluate() expects:
    {"eda_pct", "scr_pct", "hr_pct", "rmssd_pct", "sdnn_pct"}

A signal that is missing, NaN, non-numeric, or has a missing/zero baseline comes back as None.
frustration_rules treats None as "condition not met", so bad data never triggers a match.

Typical use:
    from frustration_math import compute_signals, evaluate_raw
    signals = compute_signals(current, baseline)
    result = frustration_rules.evaluate(signals, context)
    # or in one step:
    result = evaluate_raw(current, baseline, context)
"""
import math

# Raw input key -> output key used by frustration_rules
SIGNAL_KEYS = {
    "eda": "eda_pct",        # Electrodermal Activity, uS
    "scr": "scr_pct",        # Skin Conductance Response, uS or peaks/minute
    "hr": "hr_pct",          # Heart Rate, bpm
    "rmssd": "rmssd_pct",    # RMSSD, ms
    "sdnn": "sdnn_pct",      # SDNN, ms
}


def _valid(value):
    """True for a real, finite number (bools excluded)."""
    if isinstance(value, bool) or value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def percent_change(current, baseline):
    """(current - baseline) / baseline * 100, or None if it cannot be computed.

    A zero baseline cannot be divided by, so it returns None rather than inf.
    A negative baseline is not physiologically meaningful for these signals, so it
    also returns None.
    """
    if not _valid(current) or not _valid(baseline):
        return None
    current, baseline = float(current), float(baseline)
    if baseline <= 0:
        return None
    return (current - baseline) / baseline * 100.0


def mean_of(samples):
    """Mean of the valid numbers in samples; None if there are none.
    Handy for turning a window of readings into one current/baseline value."""
    vals = [float(s) for s in (samples or []) if _valid(s)]
    return sum(vals) / len(vals) if vals else None


class BaselineTracker:
    """Running personal baseline: the mean of the first `minutes` minute-samples per signal.

    add(current) is called once per new minute-level sample (current = {"eda":..., "scr":...,
    "hr":...}); missing values are skipped. After `minutes` samples the baseline is frozen,
    so values() can be passed to compute_signals() as the baseline.
    """

    def __init__(self, minutes=3, keys=("eda", "scr", "hr")):
        self.minutes, self.keys = minutes, tuple(keys)
        self.samples = {k: [] for k in self.keys}
        self.count = 0

    @property
    def ready(self):
        return self.count >= self.minutes

    def add(self, current):
        if self.ready:
            return
        self.count += 1
        for k in self.keys:
            value = (current or {}).get(k)
            if _valid(value):
                self.samples[k].append(float(value))

    def values(self):
        return {k: mean_of(self.samples[k]) for k in self.keys}


def compute_signals(current, baseline):
    """Percent change from baseline for every signal.

    current  = {"eda": 3.2, "scr": 5.5, "hr": 84, "rmssd": 31, "sdnn": 40}
    baseline = {"eda": 2.5, "scr": 4.2, "hr": 75, "rmssd": 38, "sdnn": 48}
    -> {"eda_pct": 28.0, "scr_pct": 30.95..., "hr_pct": 12.0, "rmssd_pct": -18.42..., "sdnn_pct": -16.67...}
    """
    current = current or {}
    baseline = baseline or {}
    return {out_key: percent_change(current.get(raw_key), baseline.get(raw_key))
            for raw_key, out_key in SIGNAL_KEYS.items()}


def evaluate_raw(current, baseline, context=None):
    """Convenience: raw readings + baseline + context -> FrustrationResult."""
    import frustration_rules
    return frustration_rules.evaluate(compute_signals(current, baseline), context)


if __name__ == "__main__":
    cur = {"eda": 3.2, "scr": 5.5, "hr": 84, "rmssd": 31, "sdnn": 40}
    base = {"eda": 2.5, "scr": 4.2, "hr": 75, "rmssd": 38, "sdnn": 48}
    print(compute_signals(cur, base))
