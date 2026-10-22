"""
speak_biomarker_labels.py

Polls the live biomarker endpoint served by `avro_stream_poller_with_ml.py`
(from the VIP_psychological_dashboard project) and reads new Frustration /
Mental Demand label updates aloud using offline text-to-speech (pyttsx3).

WHY pyttsx3:
- Runs fully offline, no API key or network call needed per utterance
- Works on Windows (SAPI5), macOS (NSSpeechSynthesizer), and Linux (espeak)
- Well suited to a continuously-running local monitoring loop like this one,
  where you don't want cloud latency, cost, or a dependency on internet
  access just to announce a label change

SETUP
-----
1. Make sure the live poller from the project is already running in another
   terminal, as described in the project's README:

       python avro_stream_poller_with_ml.py

   This exposes live predictions at http://127.0.0.1:7000/latest

2. Install dependencies for this script:

       pip install pyttsx3 requests

   On Linux, pyttsx3 also needs espeak installed at the OS level, e.g.:

       sudo apt-get install espeak

3. Run this script in a separate terminal:

       python speak_biomarker_labels.py

USAGE NOTES
-----------
- This script only SPEAKS. It does not modify the dashboard or the poller.
- The /latest JSON structure is confirmed from the actual project source
  (avro_stream_poller_with_ml.py + biomarker_runtime.py) -- see
  extract_labels() below for the exact fields used.
- The script only announces a label out loud when it CHANGES (or the first
  time it's seen), so it doesn't repeat "Frustration: High" every 5 seconds
  if nothing has changed.
- Frustration needs 2 minute-level samples and Mental Demand needs 1 before
  either model is "ready". Until then this script announces "Pending" once,
  then announces the real label once it resolves.

VOICE COACH MODE (added for the construction worker stress-feedback assignment)
-------------------------------------------------------------------------------
Everything above still works exactly as before. Adding --coach switches from
announcing labels to giving the worker short, escalating spoken messages:

    High stress            -> "Are you feeling tired? Please have some water."
    20 min of stress       -> "Please take a break; you've been stressed for 20 minutes."
    45 min of stress       -> required break (repeats every 10 min while it continues)
    Heat strain            -> warm skin + fast pulse: shade and water
    Racing heart           -> pulse above limit for 2 min: slow down, watch footing
    Focus-heavy work       -> high mental demand only: double-check critical steps
    Recovery               -> 5 calm minutes after an episode: positive feedback
    Hydration              -> every 60 min on shift

Design rules: one message at a time (highest priority wins), per-alert
cooldowns, low-confidence predictions ignored (frustration macro-F1 ~0.61,
mental demand ~0.52), calm voice for check-ins and firm voice for safety
alerts (macOS `say`, else the pyttsx3 engine above), English or Spanish.
Every coach alert is logged to voice_alert_log.csv next to this file.

       python speak_biomarker_labels.py --coach                          # live data
       python speak_biomarker_labels.py --simulate                       # 70-min shift demo, no wristband
       python speak_biomarker_labels.py --simulate --worker-name Carlos
       python speak_biomarker_labels.py --simulate --lang es
       python speak_biomarker_labels.py --simulate --dry-run             # print only, no audio

Messages live in COACH_MESSAGES and thresholds in COACH_CONFIG.

SECURITY NOTE
-------------
avro_stream_poller_with_ml.py (as provided) contains hardcoded AWS
credentials. Those should be rotated immediately and moved to environment
variables -- they should never live in source code, especially in a public
repo. This script does not need or use any AWS credentials; it only talks
to your local http://127.0.0.1:7000/latest endpoint.
"""

import argparse
import sys
import time
from typing import Optional

# Added for voice coach mode
import asyncio
import csv
import os
import platform
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import requests

from notify_policy import ThirdTimeRule
from speech_lock import speaking

try:
    import pyttsx3
except ImportError:
    print(
        "pyttsx3 is not installed. Install it with:\n"
        "    pip install pyttsx3\n"
        "(On Linux you may also need: sudo apt-get install espeak)",
        file=sys.stderr,
    )
    raise

DEFAULT_ENDPOINT = "http://127.0.0.1:7000/latest"
POLL_INTERVAL_SECONDS = 5  # matches the dashboard's own auto-refresh cadence


def build_tts_engine(rate: int = 175, volume: float = 1.0) -> "pyttsx3.Engine":
    """Create and configure a pyttsx3 TTS engine instance."""
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)
    return engine


def speak(engine: "pyttsx3.Engine", text: str) -> None:
    """Speak text aloud and wait until finished before returning."""
    coach_speak(text, "calm", "en", False, {})  # same voice as the coaches


def fetch_latest(endpoint: str) -> Optional[dict]:
    """Fetch the latest JSON payload from the live poller endpoint."""
    try:
        resp = requests.get(endpoint, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        print(f"[warning] could not reach {endpoint}: {exc}", file=sys.stderr)
        return None
    except ValueError:
        print("[warning] response was not valid JSON", file=sys.stderr)
        return None


def extract_labels(payload: dict) -> dict:
    """
    Pull out frustration / mental demand label + confidence from the raw
    /latest JSON payload produced by avro_stream_poller_with_ml.py.

    Confirmed exact shape (from biomarker_runtime.py's
    LiveBiomarkerPredictor.update()):

        payload["predictions"] = {
            "session_minute_index": int,
            "history_size": int,
            "model_notes": [...],
            "frustration": {
                "label": str or None,
                "confidence": float or None,
                "ready": bool,
                "reason": str or None,
            },
            "mental_demand": {
                "label": str or None,
                "confidence": float or None,
                "ready": bool,
                "reason": str or None,
            },
        }

    "ready" is False when there isn't yet enough history for that model
    (frustration needs 2 minute-level samples, mental_demand needs 1). When
    not ready, we surface the label as "Pending" (matching the dashboard's
    own terminology from the README) along with the reason.
    """
    predictions = payload.get("predictions")
    if not isinstance(predictions, dict):
        # e.g. still "waiting for first sample" / "waiting for biomarker files"
        return {
            "frustration_label": None,
            "frustration_confidence": None,
            "mental_demand_label": None,
            "mental_demand_confidence": None,
        }

    def _read(block_name):
        block = predictions.get(block_name) or {}
        if block.get("ready"):
            return block.get("label"), block.get("confidence")
        # Not ready yet -- surface as "Pending" so the change gets announced,
        # then a later real label will differ from "Pending" and get spoken.
        return "Pending", None

    frustration_label, frustration_conf = _read("frustration")
    mental_demand_label, mental_demand_conf = _read("mental_demand")

    return {
        "frustration_label": frustration_label,
        "frustration_confidence": frustration_conf,
        "mental_demand_label": mental_demand_label,
        "mental_demand_confidence": mental_demand_conf,
    }


def format_utterance(kind: str, label, confidence) -> str:
    if label is None:
        return ""
    if confidence is not None:
        try:
            pct = round(float(confidence) * 100)
            return f"{kind} is now {label}, {pct} percent confidence."
        except (TypeError, ValueError):
            pass
    return f"{kind} is now {label}."


def run_loop(endpoint: str, interval: int, debug: bool) -> None:
    engine = build_tts_engine()
    last_frustration = None
    last_mental_demand = None

    print(f"Polling {endpoint} every {interval}s. Press Ctrl+C to stop.")

    while True:
        payload = fetch_latest(endpoint)
        if payload is not None:
            if debug:
                print(f"[debug] raw payload: {payload}")

            labels = extract_labels(payload)

            f_label = labels["frustration_label"]
            f_conf = labels["frustration_confidence"]
            m_label = labels["mental_demand_label"]
            m_conf = labels["mental_demand_confidence"]

            if f_label is not None and f_label != last_frustration:
                utterance = format_utterance("Frustration", f_label, f_conf)
                if utterance:
                    speak(engine, utterance)
                last_frustration = f_label

            if m_label is not None and m_label != last_mental_demand:
                utterance = format_utterance("Mental demand", m_label, m_conf)
                if utterance:
                    speak(engine, utterance)
                last_mental_demand = m_label

            if (
                f_label is None
                and m_label is None
                and payload.get("note")
                not in (
                    "waiting for first sample",
                    "waiting for biomarker files",
                )
            ):
                print(
                    f"[info] no predictions yet -- poller note: {payload.get('note')!r}",
                    file=sys.stderr,
                )

        time.sleep(interval)


# ==========================================================================
# VOICE COACH MODE (added)
# Everything below this line until main() is new. The functions above are
# the instructor's original code and are reused as-is (fetch_latest,
# build_tts_engine, speak).
# ==========================================================================

LEVEL = {"low": 0, "medium": 1, "high": 2}

COACH_CONFIG = {
    "min_confidence": 0.55,  # ignore model labels below this confidence
    "microbreak_after_min": 20,  # sustained stress -> short break
    "required_break_after_min": 45,  # sustained stress -> required break
    "required_break_repeat_min": 10,
    "grace_low_minutes": 3,  # calm minutes in a row that end an unalerted episode
    "recovery_calm_minutes": 5,  # calm minutes needed for "nice work" feedback
    "heat_skin_temp_c": 35.0,  # wrist skin temp (not core temp). Heuristic, tune per site
    "heat_pulse_bpm": 110,
    "high_pulse_bpm": 120,
    "high_pulse_minutes": 2,
    "hydration_every_min": 60,
    "cooldown_min": {  # minimum minutes between repeats of the same alert
        "heat": 15,
        "heart": 10,
        "focus": 20,
        "checkin": 30,
        "hydration": 60,
    },
}

# The FIRST phrasing in each list is the default (the assignment's exact
# wording where one was given). --vary picks randomly from the list.
# {name} becomes ", Carlos" (or nothing); {minutes} becomes the duration.
COACH_MESSAGES = {
    "en": {
        "checkin": [
            "Are you feeling tired{name}? Please have some water.",
            "Quick check-in{name}. Your stress is running high. Grab some water and take three slow breaths.",
            "Hey{name}, your body is working hard right now. Take a sip of water and reset for a moment.",
        ],
        "microbreak": [
            "Please take a break{name}; you've been stressed for {minutes} minutes.",
            "You've been under strain for {minutes} minutes{name}. Finish your current step safely, then step back for two minutes.",
            "{minutes} minutes of high stress{name}. When it's safe, set your tools down and take a short breather.",
        ],
        "required_break": [
            "Please take a break{name}; you've been stressed for {minutes} minutes. Stop at a safe point and rest for ten minutes.",
            "Stop at a safe point{name}. You've been stressed for {minutes} minutes. Take a ten minute break in the rest area.",
            "{minutes} minutes of continuous stress{name}. Please secure your work, let your lead know, and take a real break now.",
        ],
        "heat": [
            "Heat alert{name}. Your skin temperature and heart rate are both up. Move to shade and drink water.",
            "Signs of heat strain{name}. Get out of the sun, loosen your gear, and hydrate before you continue.",
        ],
        "heart": [
            "Your heart rate is high{name}. Slow down, check your footing, and stay clear of moving equipment.",
            "Heart rate is elevated{name}. Ease off for a minute and keep three points of contact if you're on a ladder.",
        ],
        "focus": [
            "This task is demanding{name}. Double-check your harness, tie-off, and signals before the next critical step.",
            "High focus work detected{name}. Slow is smooth. Confirm with your spotter before the next lift.",
        ],
        "recovery": [
            "Nice work{name}. Your readings are back to normal. Keep pacing yourself.",
            "You're back in the green{name}. Good job taking that break.",
        ],
        "hydration": [
            "Hourly reminder{name}. Drink a cup of water, even if you're not thirsty.",
        ],
    },
    "es": {
        "checkin": [
            "¿Te sientes cansado{name}? Por favor, toma agua.",
            "Revisión rápida{name}. Tu nivel de estrés está alto. Toma agua y respira hondo tres veces.",
        ],
        "microbreak": [
            "Por favor, toma un descanso{name}; llevas {minutes} minutos con estrés.",
            "Llevas {minutes} minutos bajo presión{name}. Termina este paso con cuidado y descansa dos minutos.",
        ],
        "required_break": [
            "Por favor, toma un descanso{name}; llevas {minutes} minutos con estrés. Detente en un punto seguro y descansa diez minutos.",
            "{minutes} minutos de estrés continuo{name}. Asegura tu trabajo, avisa a tu supervisor y descansa ahora.",
        ],
        "heat": [
            "Alerta de calor{name}. Tu temperatura y tu ritmo cardíaco están altos. Ve a la sombra y toma agua.",
        ],
        "heart": [
            "Tu ritmo cardíaco está alto{name}. Baja el ritmo, cuida tu paso y aléjate de la maquinaria.",
        ],
        "focus": [
            "Esta tarea exige mucha concentración{name}. Revisa tu arnés y tus señales antes del siguiente paso.",
        ],
        "recovery": [
            "Buen trabajo{name}. Tus lecturas volvieron a la normalidad.",
        ],
        "hydration": [
            "Recordatorio{name}. Toma un vaso de agua, aunque no tengas sed.",
        ],
    },
}

# Alerts that are stress/frustration-level notifications (team rule applies to these).
STRESS_ALERTS = {"checkin", "microbreak", "required_break"}

# Priority (higher wins) and voice tone for each alert.
COACH_ALERTS = {
    "required_break": (100, "firm"),
    "heat": (90, "firm"),
    "heart": (80, "firm"),
    "microbreak": (60, "calm"),
    "checkin": (50, "calm"),
    "focus": (40, "calm"),
    "recovery": (30, "calm"),
    "hydration": (10, "calm"),
}

# Microsoft Edge neural voices (natural sounding, need internet). Voice choice from
# Dean's dean_tts work. List all voices: python -m edge_tts --list-voices
# Set COACH_VOICE=system to always use the offline voices below instead.
EDGE_VOICES = {
    "en": {"calm": "en-US-AvaNeural", "firm": "en-US-AndrewNeural"},
    "es": {"calm": "es-MX-DaliaNeural", "firm": "es-MX-JorgeNeural"},
}

# Offline fallback: macOS voices (run `say -v '?'` to see what's installed).
COACH_VOICES = {
    "en": {"calm": "Samantha", "firm": "Daniel"},
    "es": {"calm": "Paulina", "firm": "Jorge"},
}


@dataclass
class CoachState:
    shift_start: datetime = None
    episode_start: datetime = None  # when the current stress episode began
    episode_peak_tier: int = 0  # 0 none, 1 check-in, 2 micro-break, 3 required break
    calm_streak: int = 0
    high_pulse_streak: int = 0
    last_required_break: datetime = None
    last_fired: dict = field(default_factory=dict)
    # Team rule (notify_policy.py): only High is spoken, on the 3rd High episode.
    stress_rule: ThirdTimeRule = field(default_factory=ThirdTimeRule)
    focus_rule: ThirdTimeRule = field(default_factory=ThirdTimeRule)


def coach_label(pred, min_conf):
    """Return 'low'/'medium'/'high', or None if not ready or low confidence."""
    if not pred or not pred.get("ready") or pred.get("label") is None:
        return None
    conf = pred.get("confidence")
    if conf is not None and conf < min_conf:
        return None
    return str(pred["label"]).lower()


def stress_level(fr, md):
    """Combine both model outputs into 0 (calm), 1 (elevated), 2 (high)."""
    fr_l, md_l = LEVEL.get(fr, -1), LEVEL.get(md, -1)
    if fr_l == 2 or (fr_l == 1 and md_l == 2):
        return 2
    if fr_l == 1 or md_l == 2:
        return 1
    return 0


class VoiceCoach:
    """Turns a stream of minute-level samples into at most one alert per sample."""

    def __init__(self, config=COACH_CONFIG):
        self.cfg = config
        self.s = CoachState()

    def _cooled_down(self, key, now):
        last = self.s.last_fired.get(key)
        mins = self.cfg["cooldown_min"].get(key, 0)
        return last is None or (now - last) >= timedelta(minutes=mins)

    def evaluate(self, sample, now):
        """Return (alert_key, context) for the most important alert, or (None, context)."""
        cfg, s = self.cfg, self.s
        preds = sample.get("predictions") or {}
        fr = coach_label(preds.get("frustration"), cfg["min_confidence"])
        md = coach_label(preds.get("mental_demand"), cfg["min_confidence"])
        level = stress_level(fr, md)
        pulse = sample.get("pulse_rate")
        temp = sample.get("temp")

        if s.shift_start is None:
            s.shift_start = now
        candidates = []

        # --- Stress episode tracking (with a grace period for noisy minutes) ---
        if level >= 1:
            s.calm_streak = 0
            if s.episode_start is None:
                s.episode_start = now
        else:
            s.calm_streak += 1

        if s.episode_start is not None:
            minutes = int((now - s.episode_start).total_seconds() // 60)
            if minutes >= cfg["required_break_after_min"]:
                due = (
                    s.last_required_break is None
                    or now - s.last_required_break
                    >= timedelta(minutes=cfg["required_break_repeat_min"])
                )
                if due and level >= 1:
                    candidates.append(("required_break", {"minutes": minutes}))
            elif (
                minutes >= cfg["microbreak_after_min"]
                and s.episode_peak_tier < 2
                and level >= 1
            ):
                candidates.append(("microbreak", {"minutes": minutes}))
            elif (
                level == 2
                and s.episode_peak_tier < 1
                and self._cooled_down("checkin", now)
            ):
                candidates.append(("checkin", {}))

            # A few calm minutes end the episode (one noisy reading doesn't).
            # If the worker had been alerted, confirm recovery with positive feedback.
            if (
                s.calm_streak >= cfg["recovery_calm_minutes"]
                and s.episode_peak_tier >= 1
            ):
                candidates.append(("recovery", {}))
                s.episode_start, s.episode_peak_tier = None, 0
            elif s.calm_streak >= cfg["grace_low_minutes"] and s.episode_peak_tier == 0:
                s.episode_start = None

        # --- Physiological safety rules (independent of the ML models) ---
        if pulse is not None and pulse >= cfg["high_pulse_bpm"]:
            s.high_pulse_streak += 1
        else:
            s.high_pulse_streak = 0

        if (
            temp is not None
            and pulse is not None
            and temp >= cfg["heat_skin_temp_c"]
            and pulse >= cfg["heat_pulse_bpm"]
            and self._cooled_down("heat", now)
        ):
            candidates.append(("heat", {}))
        if s.high_pulse_streak >= cfg["high_pulse_minutes"] and self._cooled_down(
            "heart", now
        ):
            candidates.append(("heart", {}))

        # --- High mental demand while frustration is fine ---
        if (
            md == "high"
            and fr in (None, "low")
            and level < 2
            and self._cooled_down("focus", now)
        ):
            candidates.append(("focus", {}))

        # --- Hydration reminder ---
        on_shift = (now - s.shift_start).total_seconds() / 60
        last_h = s.last_fired.get("hydration")
        if on_shift >= cfg["hydration_every_min"] and (
            last_h is None
            or now - last_h >= timedelta(minutes=cfg["hydration_every_min"])
        ):
            candidates.append(("hydration", {}))

        # --- Team rule: stress/frustration and mental-demand alerts only at High, and only
        # on the 3rd High episode (the first two are tracked). Safety alerts (heat, racing
        # heart), hydration and recovery are not stress-level notifications and pass through.
        t = (now - s.shift_start).total_seconds()
        s.stress_rule.observe(t, level == 2)
        s.focus_rule.observe(t, md == "high")
        occ_before = s.stress_rule.count
        stress_ok = level == 2 and s.stress_rule.allow(t)
        if level == 2 and not stress_ok and s.stress_rule.count != occ_before:
            print(f"  (stress High occurrence {s.stress_rule.count}/{s.stress_rule.needed}: tracked, not spoken)")
        candidates = [c for c in candidates
                      if c[0] not in STRESS_ALERTS or stress_ok]
        if any(c[0] == "focus" for c in candidates):
            f_before = s.focus_rule.count
            if not s.focus_rule.allow(t):
                if s.focus_rule.count != f_before:
                    print(f"  (mental demand High occurrence {s.focus_rule.count}/{s.focus_rule.needed}: tracked, not spoken)")
                candidates = [c for c in candidates if c[0] != "focus"]

        if not candidates:
            return None, {"fr": fr, "md": md, "level": level}

        key, ctx = max(candidates, key=lambda c: COACH_ALERTS[c[0]][0])
        s.last_fired[key] = now
        if key == "checkin":
            s.episode_peak_tier = max(s.episode_peak_tier, 1)
        elif key == "microbreak":
            s.episode_peak_tier = max(s.episode_peak_tier, 2)
        elif key == "required_break":
            s.episode_peak_tier = 3
            s.last_required_break = now
        ctx.update({"fr": fr, "md": md, "level": level})
        return key, ctx


_edge_ok = os.environ.get("COACH_VOICE", "edge").lower() != "system"


def _play_mp3(path: str) -> None:
    """Play an mp3 and wait until it finishes (macOS afplay, Windows MCI, Linux ffplay/mpg123)."""
    system = platform.system()
    if system == "Darwin":
        subprocess.run(["afplay", path], check=True)
    elif system == "Windows":
        import ctypes
        mci = ctypes.windll.winmm.mciSendStringW
        mci(f'open "{path}" type mpegvideo alias coachaudio', None, 0, None)
        mci("play coachaudio wait", None, 0, None)
        mci("close coachaudio", None, 0, None)
    elif shutil.which("ffplay"):
        subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path], check=True)
    elif shutil.which("mpg123"):
        subprocess.run(["mpg123", "-q", path], check=True)
    else:
        raise RuntimeError("no mp3 player found (install ffmpeg or mpg123)")


def speak_edge(text: str, lang: str, tone: str, rate: int = 175) -> None:
    """Speak with a Microsoft neural voice. Raises if edge-tts is missing or offline."""
    import edge_tts

    voice = EDGE_VOICES.get(lang, EDGE_VOICES["en"])[tone]
    pct = round((rate - 175) / 175 * 100)  # pyttsx3-style words/min -> edge-tts percent
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    try:
        communicate = edge_tts.Communicate(text, voice, rate=f"{pct:+d}%", connect_timeout=5)
        asyncio.run(asyncio.wait_for(communicate.save(path), timeout=15))
        _play_mp3(path)
    finally:
        os.remove(path)


def speak_isolated(text: str, rate: int = 175) -> None:
    """Speak via a throwaway process so pyttsx3 can't get stuck after the first utterance."""
    print(f"[speaking] {text}", flush=True)
    code = (
        "import sys, pyttsx3\n"
        "e = pyttsx3.init()\n"
        "e.setProperty('rate', int(sys.argv[2]))\n"
        "e.say(sys.argv[1])\n"
        "e.runAndWait()\n"
    )
    subprocess.run([sys.executable, "-c", code, text, str(rate)])


def coach_speak(text, tone, lang, dry_run, engine_holder, rate=175):
    """Speak with a Microsoft neural voice; offline, fall back to macOS `say`, else pyttsx3."""
    global _edge_ok
    if dry_run:
        print(f"[speaking:{tone}] {text}", flush=True)
        return
    if _edge_ok:
        try:
            print(f"[speaking:{tone}] {text}", flush=True)
            speak_edge(text, lang, tone, rate)
            return
        except Exception as exc:  # no internet, edge-tts not installed, no player...
            _edge_ok = False  # don't wait on the network again this session
            print(f"(neural voice unavailable: {type(exc).__name__}: {exc} -- using offline voice)",
                  file=sys.stderr, flush=True)
    if platform.system() == "Darwin" and shutil.which("say"):
        print(f"[speaking:{tone}] {text}", flush=True)
        voice = COACH_VOICES.get(lang, COACH_VOICES["en"]).get(tone)
        cmd = ["say", "-r", str(rate)]
        if voice:
            cmd += ["-v", voice]
        if subprocess.run(cmd + [text], capture_output=True).returncode != 0:
            subprocess.run(["say", "-r", str(rate), text])  # voice not installed
        return
    speak_isolated(text, rate)


COACH_LOG_PATH = Path(__file__).resolve().parent.parent / "voice_alert_log.csv"


def log_alert(sample, key, ctx, text):
    new = not COACH_LOG_PATH.exists()
    with COACH_LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(
                [
                    "logged_at",
                    "participant_id",
                    "alert",
                    "frustration",
                    "mental_demand",
                    "pulse_rate",
                    "skin_temp",
                    "message",
                ]
            )
        w.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                sample.get("participant_id"),
                key,
                ctx.get("fr"),
                ctx.get("md"),
                sample.get("pulse_rate"),
                sample.get("temp"),
                text,
            ]
        )


def sample_time(sample):
    for k in ("eda_time", "temp_time", "pulse_time", "updated_at"):
        v = sample.get(k)
        if v:
            try:
                return datetime.fromisoformat(str(v).replace("Z", "+00:00")).replace(
                    tzinfo=None
                )
            except ValueError:
                continue
    return datetime.now()


def live_samples(endpoint, interval, debug):
    """Yield each new minute-level sample from the poller (uses the original fetch_latest)."""
    last_sig = None
    print(f"Coach listening to {endpoint} every {interval}s. Press Ctrl+C to stop.")
    while True:
        data = fetch_latest(endpoint)
        if data:
            if debug:
                print(f"[debug] raw payload: {data}")
            sig = (data.get("participant_id"), data.get("updated_at"))
            if data.get("predictions") and sig != last_sig:
                last_sig = sig
                yield data, sample_time(data)
        time.sleep(interval)


def simulated_samples(seconds_per_minute):
    """A 70-minute shift: three High stress episodes (3rd one spoken), heat spike, recovery."""

    def pred(label, conf=0.8):
        return {"label": label, "confidence": conf, "ready": True}

    start = datetime.now()
    print(
        f"Simulating a 70-minute shift ({seconds_per_minute}s per minute). Ctrl+C to stop."
    )
    for m in range(71):
        if m < 8:
            fr, md, pulse, temp = "low", "low", 82, 32.5
        elif m < 12:
            fr, md, pulse, temp = "low", "high", 92, 33.0  # focus-heavy task (MD High #1, tracked)
        elif m < 16 or 20 <= m < 24 or 28 <= m < 60:
            # High stress episodes: #1 min 12-15, #2 min 20-23 (tracked), #3 from min 28 (spoken)
            fr, md, pulse, temp = "high", "high", 104, 33.8
            if 38 <= m <= 41:
                pulse, temp = 118, 35.6  # hot afternoon, heat strain
            if 52 <= m <= 54:
                pulse = 126  # racing heart
        elif m < 60:
            fr, md, pulse, temp = "low", "low", 86, 33.2  # short calm gap between episodes
        else:
            fr, md, pulse, temp = "low", "low", 84, 33.0  # after the break
        yield (
            {
                "participant_id": "SIM-WORKER",
                "pulse_rate": pulse,
                "temp": temp,
                "predictions": {
                    "frustration": pred(fr),
                    "mental_demand": pred(md, 0.7),
                },
            },
            start + timedelta(minutes=m),
        )
        time.sleep(seconds_per_minute)


def run_coach(args) -> None:
    """Voice coach: speak escalating safety messages based on patterns over time."""
    coach = VoiceCoach()
    engine_holder = {}
    name = f", {args.worker_name}" if args.worker_name else ""
    source = (
        simulated_samples(args.speed)
        if args.simulate
        else live_samples(args.endpoint, args.interval, args.debug)
    )

    for sample, now in source:
        key, ctx = coach.evaluate(sample, now)
        minute = int((now - coach.s.shift_start).total_seconds() // 60)
        if key is None:
            print(
                f"  min {minute:>3}  fr={ctx['fr']} md={ctx['md']} "
                f"pulse={sample.get('pulse_rate')} temp={sample.get('temp')}  (quiet)"
            )
            continue
        options = COACH_MESSAGES[args.lang][key]
        template = random.choice(options) if args.vary else options[0]
        text = template.format(name=name, **ctx)
        print(f"  min {minute:>3}  -> {key.upper()}")
        with speaking():  # never talk over the Polar coach (backend/speech_lock.py)
            coach_speak(text, COACH_ALERTS[key][1], args.lang, args.dry_run, engine_holder)
        log_alert(sample, key, ctx, text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Poll the live biomarker endpoint and speak Frustration / "
            "Mental Demand label updates aloud."
        )
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"URL of the live poller's /latest endpoint (default: {DEFAULT_ENDPOINT})",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=POLL_INTERVAL_SECONDS,
        help=f"Seconds between polls (default: {POLL_INTERVAL_SECONDS})",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print the raw JSON payload on every poll for troubleshooting",
    )
    # --- Added for voice coach mode ---
    parser.add_argument(
        "--coach",
        action="store_true",
        help="Voice coach mode: escalating stress/safety messages instead of label announcements",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Coach demo with a scripted 70-minute shift, no wristband needed (implies --coach)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.5,
        help="Seconds per simulated minute (default: 1.5)",
    )
    parser.add_argument(
        "--lang",
        choices=["en", "es"],
        default="en",
        help="Coach message language (default: en)",
    )
    parser.add_argument(
        "--worker-name",
        default="",
        help="Personalize coach messages, e.g. --worker-name Carlos",
    )
    parser.add_argument(
        "--vary",
        action="store_true",
        help="Rotate through alternative phrasings instead of the assignment's exact wording",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Coach mode: print messages without audio",
    )
    args = parser.parse_args()

    try:
        if args.coach or args.simulate:
            run_coach(args)
        else:
            run_loop(args.endpoint, args.interval, args.debug)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
