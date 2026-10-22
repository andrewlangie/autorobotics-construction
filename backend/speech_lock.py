"""Stops two voice processes (stress coach + Polar coach) from talking over each other.

Both processes call `with speaking():` around their speech. On macOS/Linux this is a file
lock in the temp folder; on Windows it does nothing (messages may overlap there).
"""
import os
import tempfile
from contextlib import contextmanager

LOCK_PATH = os.path.join(tempfile.gettempdir(), "vip_voice_coach.lock")

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None


@contextmanager
def speaking():
    if fcntl is None:
        yield
        return
    with open(LOCK_PATH, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # waits while the other process is speaking
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
