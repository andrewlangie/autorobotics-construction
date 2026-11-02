"""Polar H10 voice coach: speaks fatigue and posture feedback.

    python backend/polar_coach.py --simulate --dry-run --speed 0   # instant test, text only
    python backend/polar_coach.py --simulate                       # hear a 60-min shift in ~60 s
    python backend/polar_coach.py --replay recordings/session1     # your own recording
    python backend/polar_coach.py                                  # live strap

Every spoken decision is logged to polar_alert_log.csv (with the closed-loop action and goal).
Run it next to `python speak_biomarker_labels.py --coach` -- the two never talk at once.
"""
import argparse
import csv
import heapq
import sys
import time
from datetime import datetime
from pathlib import Path

from fatigue_rules import HIGH
from feedback_engine import FatigueEngine, PostureEngine, choose_primary

LEVEL_NAMES = {1: "Low", 2: "Medium", 3: "High"}
from polar_status import StatusWriter
from speak_biomarker_labels import coach_speak
from speech_lock import speaking

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "polar_alert_log.csv"
EVALUATE_EVERY_S = 5   # how often the engines decide (in sensor time)
MIN_GAP_S = 10         # at least this long between two spoken messages
STATUS_EVERY_S = 60    # print a status line this often


def replay_events(folder):
    """Read a recording made with polar_reader.py --record and yield events in time order."""
    folder = Path(folder)

    def beats():
        with open(folder / "hr.csv") as f:
            for r in csv.DictReader(f):
                yield (float(r["t"]), 0, ("beat", float(r["t"]), float(r["hr"]), float(r["rr_ms"])))

    def accs():
        with open(folder / "acc.csv") as f:
            for r in csv.DictReader(f):
                yield (float(r["t"]), 1, ("acc", float(r["t"]), float(r["ax"]), float(r["ay"]), float(r["az"])))

    for _, _, ev in heapq.merge(beats(), accs()):
        yield ev


def log_decision(d, fatigue_state, posture_state):
    new = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["logged_at", "track", "level", "level_label", "cause", "message",
                        "closed_loop_action", "feedback_goal", "hr_pct", "rmssd_pct", "sdnn_pct",
                        "forward_deg", "side_deg"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), d.track, d.level, d.level_label,
                    d.cause_label, d.message, d.action, d.goal,
                    _fmt(fatigue_state.get("hr_pct")), _fmt(fatigue_state.get("rmssd_pct")),
                    _fmt(fatigue_state.get("sdnn_pct")), _fmt(posture_state.get("forward_deg")),
                    _fmt(posture_state.get("side_deg"))])


def _fmt(v):
    return "" if v is None else f"{v:.1f}"


def status_line(minute, fs, ps):
    if fs.get("status") == "ok":
        f = (f"fatigue L{fs['level']} HR{fs['hr_pct']:+.0f}% RMSSD{fs['rmssd_pct']:+.0f}% "
             f"SDNN{fs['sdnn_pct']:+.0f}%")
    else:
        f = f"fatigue: {fs.get('status')}"
    if ps.get("status") == "ok":
        p = f"posture L{ps['level']} fwd {ps['forward_deg']:.0f} side {ps['side_deg']:.0f} deg"
    else:
        p = f"posture: {ps.get('status')}"
    return f"  min {minute:>3}  {f} | {p}"


def run(events, speed, dry_run, status=None):
    fatigue, posture = FatigueEngine(), PostureEngine()
    start = next_eval = None
    last_spoken = next_status = -1e9
    last_occ = last_pocc = n_pred = 0
    last_level = None
    wall_start = time.time()
    for ev in events:
        t = ev[1]
        if start is None:
            start, next_eval = t, t + EVALUATE_EVERY_S
        if ev[0] == "beat":
            fatigue.add_beat(t, ev[2], ev[3])
        else:
            posture.add_acc(t, ev[2], ev[3], ev[4])
        if t < next_eval:
            continue
        next_eval = t + EVALUATE_EVERY_S

        if speed > 0:  # pace simulations/replays: `speed` sensor-seconds per real second
            wait = wall_start + (t - start) / speed - time.time()
            if wait > 0:
                time.sleep(wait)

        decisions = [fatigue.evaluate(t), posture.evaluate(t)]
        minute = int((t - start) // 60)
        if status:  # share the live state with the dashboard (polar_status.py)
            status.update((t - start) / 60, fatigue.state, posture.state)
        if t >= next_status:
            print(status_line(minute, fatigue.state, posture.state))
            next_status = t + STATUS_EVERY_S
        if len(fatigue.predictions) != n_pred:  # a new fatigue prediction was made
            n_pred = len(fatigue.predictions)
            level = fatigue.predictions[-1][1]
            if level == HIGH and fatigue.rule.occurrence:
                note = f"High {fatigue.rule.occurrence}/{fatigue.rule.needed} in a row: tracked, not spoken"
            elif level == HIGH:
                note = f"High {fatigue.rule.needed}/{fatigue.rule.needed} in a row: feedback"
            elif last_level == HIGH or last_occ:
                note = f"{LEVEL_NAMES[level]}: count reset, not spoken"
            else:
                note = f"{LEVEL_NAMES[level]}: not spoken"
            print(f"  min {minute:>3}  (fatigue prediction {note})")
            last_level, last_occ = level, fatigue.rule.occurrence
        pocc = posture.state.get("occurrence", 0)
        if pocc != last_pocc and 0 < pocc < posture.rule.needed:
            print(f"  min {minute:>3}  (posture Poor occurrence {pocc}/{posture.rule.needed}: tracked, not spoken)")
        last_pocc = pocc
        d = choose_primary(decisions)
        if d is None or t - last_spoken < MIN_GAP_S:
            continue  # a deferred decision is offered again on the next evaluation
        (fatigue if d.track == "fatigue" else posture).mark_spoken(d)
        last_spoken = t
        print(f"  min {minute:>3}  -> {d.track.upper()} L{d.level} {d.level_label} / {d.cause_label}")
        with speaking():
            coach_speak(d.message, "firm" if d.urgency >= 3 else "calm", "en", dry_run, {})
        log_decision(d, fatigue.state, posture.state)
        if status:
            status.spoken(d, (t - start) / 60)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--simulate", action="store_true", help="fake 60-minute shift, no strap needed")
    src.add_argument("--replay", metavar="FOLDER", help="replay a recording (hr.csv + acc.csv)")
    p.add_argument("--speed", type=float, default=60,
                   help="simulate/replay: sensor seconds per real second (0 = as fast as possible)")
    p.add_argument("--dry-run", action="store_true", help="print messages instead of speaking")
    p.add_argument("--no-dashboard", action="store_true",
                   help="don't write polar_status.json for the dashboard panel")
    args = p.parse_args()

    if args.simulate:
        from polar_simulator import simulated_events
        events, speed, source = simulated_events(), args.speed, "simulation"
    elif args.replay:
        events, speed, source = replay_events(args.replay), args.speed, "replay"
    else:
        from polar_reader import live_events
        events, speed, source = live_events(), 0, "live"  # live data arrives in real time already
    status = None if args.no_dashboard else StatusWriter(source=source)
    try:
        run(events, speed, args.dry_run, status)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    sys.exit(main())
