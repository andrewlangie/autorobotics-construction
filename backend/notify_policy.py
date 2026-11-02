"""Notification rules.

ConsecutiveHighRule -- the VIP assignment rule (Fall 2026), used for FRUSTRATION and FATIGUE:

    * Predictions are Low, Medium or High.
    * Low and Medium are never spoken (Medium is only shown on the dashboard).
    * High-risk feedback is triggered only after THREE CONSECUTIVE High predictions.
    * Any Low or Medium prediction resets the count to zero.
    * After feedback is triggered the count starts again, so if High continues the next
      feedback comes after another three High predictions (TEAM-DECISION).
    * A missing / not-ready / low-confidence prediction is skipped: it neither counts
      nor resets (TEAM-DECISION).

ThirdTimeRule -- the older episode-based rule, still used by the POSTURE engine (Team 2).

Old rule text (Akshaya, Oct 2026):
    * Only HIGH and VERY HIGH levels are spoken (Medium is shown on the dashboard, not spoken).
      For posture the two alert levels are Poor and Critical.
    * The first two High/Very High occurrences are tracked silently; feedback is given
      when the THIRD occurrence is detected. Then the count starts again.

An "occurrence" is one episode: the level reaches High (after its duration gate), and the
episode only ends once the level has stayed below High for EPISODE_RESET_S. So one long
High stretch counts once, not every few seconds. TEAM-DECISION: EPISODE_RESET_S.
"""
NOTIFY_MIN_LEVEL_NAME = "High"
OCCURRENCES_NEEDED = 3
EPISODE_RESET_S = 60


class ThirdTimeRule:
    def __init__(self, needed=OCCURRENCES_NEEDED, reset_s=EPISODE_RESET_S):
        self.needed, self.reset_s = needed, reset_s
        self.count = 0              # High episodes since the last spoken one
        self.in_episode = False
        self.notify_episode = False  # True while in the episode that is being spoken
        self.low_since = None

    def observe(self, t, high):
        """Call every evaluation with t in seconds and whether the level is High or above now."""
        if high:
            self.low_since = None
            return
        if self.low_since is None:
            self.low_since = t
        if self.in_episode and t - self.low_since >= self.reset_s:
            self.in_episode = self.notify_episode = False

    def allow(self, t):
        """A High/Very High alert is ready now. Returns True if it should be spoken."""
        if not self.in_episode:
            self.in_episode = True
            self.count += 1
            if self.count >= self.needed:
                self.count = 0
                self.notify_episode = True
        return self.notify_episode

    @property
    def occurrence(self):
        """Shown to the user: which occurrence we are on (1, 2, ... ; 0 = none yet)."""
        return self.needed if self.notify_episode else self.count


class ConsecutiveHighRule:
    """Feedback only after `needed` consecutive High predictions; Low/Medium resets the count.

    Call update() exactly once per NEW prediction:
        high=True  -> High prediction
        high=False -> Low or Medium prediction (resets)
        high=None  -> no usable prediction (ignored)
    """

    def __init__(self, needed=OCCURRENCES_NEEDED):
        self.needed = needed
        self.count = 0          # consecutive High predictions since the last reset / feedback
        self.times_fired = 0    # feedbacks given during the current unbroken run of Highs
        self.last_reset_by = None

    def update(self, high):
        """Returns True when High-risk feedback should be triggered now."""
        if high is None:
            return False
        if not high:
            if self.count or self.times_fired:
                self.last_reset_by = "low/medium"
            self.count, self.times_fired = 0, 0
            return False
        self.count += 1
        if self.count >= self.needed:
            self.count = 0
            self.times_fired += 1
            return True
        return False

    @property
    def occurrence(self):
        """Consecutive High predictions so far (0 .. needed-1); shown on the dashboard."""
        return self.count
