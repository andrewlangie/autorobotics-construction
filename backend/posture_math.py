"""Trunk posture from the Polar H10 accelerometer (chest strap).

For every accelerometer sample (x, y, z in g):
1. Low-pass filter -> the slowly changing part is gravity = how the chest is tilted.
2. Calibration (first 10 s, standing upright): the axis carrying most of gravity is "up".
3. Forward bend = rotation of gravity from "up" toward the FORWARD axis,
   side bend    = rotation of gravity from "up" toward the remaining (side) axis,
   both relative to the upright calibration, in degrees.
4. Movement = how much the raw signal differs from gravity (used for "static posture").

Why not the framework's pitch = atan2(Ax, sqrt(Ay^2 + Az^2)) directly? Those formulas
assume a sensor lying flat (gravity on Z). On a chest strap gravity is on X or Y while
standing, and then a side bend also changes "pitch". Measuring each bend against the
detected up-axis keeps forward and side bending separate. The thresholds are unchanged.

FORWARD_AXIS: the H10 axis pointing out of the chest. We assume z (perpendicular to the
sensor's face). If your recording shows bending forward changes the SIDE angle instead,
change FORWARD_AXIS to the other non-up axis (0 = x, 1 = y, 2 = z).

What a chest accelerometer CANNOT see: twisting (rotation around the vertical axis does
not change gravity) and arm position (overhead work). Those need another sensor.
"""
import math
from collections import deque

FORWARD_AXIS = 2               # z = out of the chest
GRAVITY_TIME_CONSTANT_S = 1.0  # longer = smoother angles, slower to react
CALIBRATION_S = 10.0           # stand upright this long when the session starts
MOVEMENT_WINDOW_S = 60.0       # movement is averaged over this window


class PostureTracker:
    def __init__(self):
        self.g = None              # current gravity estimate (gx, gy, gz)
        self.last_t = self.start_t = None
        self.calib = []            # gravity samples while calibrating
        self.axes = None           # (up, forward, side, sign of up)
        self.base = None           # upright (forward, side) angles
        self.move = deque()        # (t, movement in g)
        self.move_sum = 0.0

    @property
    def calibrated(self):
        return self.base is not None

    def update(self, t, ax, ay, az):
        """Feed one sample (time in seconds, acceleration in g)."""
        if self.g is None:
            self.g, self.last_t, self.start_t = (ax, ay, az), t, t
        dt = max(t - self.last_t, 1e-3)
        self.last_t = t
        alpha = dt / (GRAVITY_TIME_CONSTANT_S + dt)  # low-pass filter strength
        self.g = tuple(alpha * a + (1 - alpha) * g for a, g in zip((ax, ay, az), self.g))

        m = math.dist((ax, ay, az), self.g)  # movement = raw sample minus gravity
        self.move.append((t, m))
        self.move_sum += m
        while t - self.move[0][0] > MOVEMENT_WINDOW_S:
            self.move_sum -= self.move.popleft()[1]

        if self.base is None:
            self.calib.append(self.g)
            if t - self.start_t >= CALIBRATION_S:
                self._finish_calibration()

    def _finish_calibration(self):
        n = len(self.calib)
        g0 = [sum(s[i] for s in self.calib) / n for i in range(3)]
        up = max(range(3), key=lambda i: abs(g0[i]))
        forward = FORWARD_AXIS if FORWARD_AXIS != up else next(i for i in range(3) if i != up)
        side = next(i for i in range(3) if i not in (up, forward))
        self.axes = (up, forward, side, 1.0 if g0[up] >= 0 else -1.0)
        self.base = self._raw_angles(g0)

    def _raw_angles(self, g):
        up, forward, side, sign = self.axes
        vertical = sign * g[up]
        return (math.degrees(math.atan2(g[forward], vertical)),
                math.degrees(math.atan2(g[side], vertical)))

    def angles(self):
        """(forward_bend_deg, side_bend_deg) relative to upright, or None before calibration."""
        if self.base is None:
            return None
        f, s = self._raw_angles(self.g)
        return abs(f - self.base[0]), abs(s - self.base[1])

    def movement(self):
        """Average movement (g) over the last MOVEMENT_WINDOW_S; low = holding still."""
        return self.move_sum / len(self.move) if self.move else 0.0

    def movement_window_full(self):
        return bool(self.move) and self.move[-1][0] - self.move[0][0] >= MOVEMENT_WINDOW_S * 0.95
