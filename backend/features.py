"""Causal, scale-free per-minute features shared by training and the live runtime.

Every feature for minute t uses only minutes <= t, and is expressed relative to the
worker's own recent history (the previous HISTORY minutes). Because each feature is a
ratio of changes to that history's spread, it does not depend on the units of the
signal, so the same code works on the 0-1 min-max training files and on raw
uS / degC values from the live wristband.

Usage
-----
Offline (training):  frame = build_feature_frame(eda_array, temp_array)
Live (runtime):      buf = LiveFeatureBuffer(); feats = buf.push(eda, temp)  # None until ready
"""
from collections import deque

import numpy as np
import pandas as pd

SIGNALS = ("eda", "temp")
WINDOW = 5      # trailing minutes used for slope / variability / rising fraction
HISTORY = 20    # trailing minutes (before the current one) used as the self-baseline
_CLIP = 5.0

_KINDS = ("dev", "slope", "rising", "varratio", "rank")
FEATURE_NAMES = [f"{s}_{k}" for s in SIGNALS for k in _KINDS]


def _clip(x, lim=_CLIP):
    return float(np.clip(x, -lim, lim))


def _slope(y):
    """Least-squares slope per minute of a short window."""
    x = np.arange(len(y), dtype=float)
    x -= x.mean()
    denom = float((x ** 2).sum())
    return float((x * (y - y.mean())).sum() / denom) if denom > 0 else 0.0


def _signal_features(v, name):
    """v: 1-D float array, last element is the current minute."""
    cur = v[-1]
    hist = v[:-1][-HISTORY:]
    win = v[-WINDOW:]
    rng = float(hist.max() - hist.min())

    if rng <= 0:
        dev = slope = var_ratio = 0.0
    else:
        # Spread of the worker's own recent history; floor stops a flat history
        # from blowing the ratios up. Both terms scale with the signal's units.
        scale = max(float(hist.std()), 0.1 * rng)
        dev = (cur - float(np.median(hist))) / scale               # distance from own median
        slope = _slope(win) * (len(win) - 1) / scale               # change across window
        var_ratio = float(win.std()) / scale                       # recent vs. historical variability

    rising = float(np.mean(np.diff(win) > 0))                      # fraction of minutes going up
    rank = float(np.mean(hist < cur) + 0.5 * np.mean(hist == cur)) # where cur sits in own history

    return {
        f"{name}_dev": _clip(dev),
        f"{name}_slope": _clip(slope),
        f"{name}_rising": rising,
        f"{name}_varratio": float(min(var_ratio, _CLIP)),
        f"{name}_rank": rank,
    }


def features_at(eda, temp):
    """Features for the LAST element of eda/temp (values up to and including this minute).

    Returns None until WINDOW minutes are available or if any value in the span is NaN.
    """
    e = np.asarray(eda, dtype=float)[-(HISTORY + 1):]
    t = np.asarray(temp, dtype=float)[-(HISTORY + 1):]
    if len(e) < WINDOW or len(t) < WINDOW or np.isnan(e).any() or np.isnan(t).any():
        return None
    out = {}
    out.update(_signal_features(e, "eda"))
    out.update(_signal_features(t, "temp"))
    return out


def build_feature_frame(eda, temp):
    """One feature row per minute (offline). Columns: minute_index + FEATURE_NAMES.

    NaNs inside the series are forward-filled; leading NaNs are dropped, matching what
    LiveFeatureBuffer does. minute_index is the position in the original series.
    """
    e = pd.Series(np.asarray(eda, dtype=float)).ffill().to_numpy()
    t = pd.Series(np.asarray(temp, dtype=float)).ffill().to_numpy()
    valid = ~(np.isnan(e) | np.isnan(t))
    if not valid.any():
        return pd.DataFrame(columns=["minute_index"] + FEATURE_NAMES)
    start = int(np.argmax(valid))
    e, t = e[start:], t[start:]

    rows = []
    for i in range(WINDOW - 1, len(e)):
        feats = features_at(e[: i + 1], t[: i + 1])
        if feats is None:
            continue
        feats["minute_index"] = start + i
        rows.append(feats)
    return pd.DataFrame(rows, columns=["minute_index"] + FEATURE_NAMES)


class LiveFeatureBuffer:
    """Streaming version: call push() once per new aggregated minute."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._eda = deque(maxlen=HISTORY + 1)
        self._temp = deque(maxlen=HISTORY + 1)
        self._last_eda = np.nan
        self._last_temp = np.nan

    def push(self, eda, temp):
        """Returns a dict of FEATURE_NAMES, or None while warming up."""
        eda = np.nan if eda is None else float(eda)
        temp = np.nan if temp is None else float(temp)
        if np.isnan(eda):
            eda = self._last_eda
        if np.isnan(temp):
            temp = self._last_temp
        if np.isnan(eda) or np.isnan(temp):
            return None            # nothing valid seen yet
        self._last_eda, self._last_temp = eda, temp
        self._eda.append(eda)
        self._temp.append(temp)
        return features_at(list(self._eda), list(self._temp))
