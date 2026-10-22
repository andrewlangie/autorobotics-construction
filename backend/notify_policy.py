"""Team notification rule (Akshaya, Oct 2026), shared by the fatigue, stress and posture coaches:

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
