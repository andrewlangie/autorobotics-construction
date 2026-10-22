"""Voice selection: neural voice when online, offline fallback otherwise.
python -m unittest discover -s tests -v"""
import contextlib
import io
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import speak_biomarker_labels as sbl  # noqa: E402


class FakeCommunicate:
    calls = []

    def __init__(self, text, voice, rate="+0%", connect_timeout=None):
        FakeCommunicate.calls.append((text, voice, rate))

    async def save(self, path):
        Path(path).write_bytes(b"mp3")


class VoiceTests(unittest.TestCase):
    def setUp(self):
        FakeCommunicate.calls = []
        sbl._edge_ok = True

    def test_neural_voice_by_tone_and_language(self):
        fake = types.SimpleNamespace(Communicate=FakeCommunicate)
        with mock.patch.dict(sys.modules, {"edge_tts": fake}), \
             mock.patch.object(sbl, "_play_mp3") as play, \
             contextlib.redirect_stdout(io.StringIO()):
            sbl.coach_speak("Take a break.", "firm", "en", False, {})
            sbl.coach_speak("Tome un descanso.", "calm", "es", False, {})
        self.assertEqual(FakeCommunicate.calls[0], ("Take a break.", "en-US-AndrewNeural", "+0%"))
        self.assertEqual(FakeCommunicate.calls[1][1], "es-MX-DaliaNeural")
        self.assertEqual(play.call_count, 2)

    def test_offline_falls_back_once_and_stays_offline(self):
        def broken(*a, **k):
            raise OSError("no internet")
        fake = types.SimpleNamespace(Communicate=broken)
        with mock.patch.dict(sys.modules, {"edge_tts": fake}), \
             mock.patch.object(sbl.platform, "system", return_value="Linux"), \
             mock.patch.object(sbl, "speak_isolated") as offline, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            sbl.coach_speak("One.", "calm", "en", False, {})
            sbl.coach_speak("Two.", "calm", "en", False, {})
        self.assertFalse(sbl._edge_ok)       # no second network attempt
        self.assertEqual(offline.call_count, 2)

    def test_dry_run_makes_no_sound(self):
        with mock.patch.object(sbl, "speak_edge") as edge, contextlib.redirect_stdout(io.StringIO()) as out:
            sbl.coach_speak("Quiet.", "calm", "en", True, {})
        edge.assert_not_called()
        self.assertIn("[speaking:calm] Quiet.", out.getvalue())


if __name__ == "__main__":
    unittest.main()
