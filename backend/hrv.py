"""Heart-rate variability from Polar H10 RR intervals.

RR interval = time between two heartbeats, in milliseconds.
RMSSD = root mean square of successive RR differences (short-term recovery).
SDNN  = standard deviation of RR intervals (overall variability).
Both drop when a person is fatigued or strained; heart rate goes up.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

MIN_RR_MS = 300    # 200 bpm -- anything shorter is a sensor glitch
MAX_RR_MS = 2000   # 30 bpm  -- anything longer is a missed beat
MAX_JUMP = 0.20    # a beat that differs >20% from the previous one is an artifact
MIN_BEATS = 20     # need at least this many clean beats for a reliable value


def clean_rr(rr_ms):
    """Drop impossible values and sudden jumps (artifacts from movement)."""
    rr = np.asarray(rr_ms, dtype=float)
    rr = rr[(rr > MIN_RR_MS) & (rr < MAX_RR_MS)]
    kept = []
    for value in rr:
        # compare with the last GOOD beat, so one glitch doesn't also remove the next beat
        if not kept or abs(value - kept[-1]) / kept[-1] < MAX_JUMP:
            kept.append(value)
    return np.array(kept)


def rmssd(rr_ms):
    rr = np.asarray(rr_ms, dtype=float)
    return float(np.sqrt(np.mean(np.diff(rr) ** 2)))


def sdnn(rr_ms):
    return float(np.std(np.asarray(rr_ms, dtype=float), ddof=1))


@dataclass
class HrvMetrics:
    hr: float      # bpm, from the mean RR interval
    rmssd: float   # ms
    sdnn: float    # ms
    beats: int     # clean beats used


class RRWindow:
    """Keeps the last `window_s` seconds of beats and computes HR/RMSSD/SDNN."""

    def __init__(self, window_s=60.0):
        self.window_s = window_s
        self.beats = deque()  # (time_s, rr_ms)

    def add(self, t, rr_ms):
        self.beats.append((t, rr_ms))
        while self.beats and t - self.beats[0][0] > self.window_s:
            self.beats.popleft()

    def metrics(self):
        """Return HrvMetrics, or None if there aren't enough clean beats yet."""
        rr = clean_rr([b[1] for b in self.beats])
        if len(rr) < MIN_BEATS:
            return None
        return HrvMetrics(
            hr=60000.0 / float(np.mean(rr)),
            rmssd=rmssd(rr),
            sdnn=sdnn(rr),
            beats=len(rr),
        )


def pct_change(current, baseline):
    """Percent change from baseline: the framework's HR% / RMSSD% / SDNN% formula."""
    return (current - baseline) / baseline * 100.0
