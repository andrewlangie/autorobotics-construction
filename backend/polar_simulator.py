"""Fake Polar H10 data for a 60-minute shift, so everything can be tested without the strap.

Produces the same events as polar_reader.live_events():
    ('beat', t, hr, rr_ms)   and   ('acc', t, ax, ay, az)   (acceleration in g)

Each phase sets how heart rate / HRV differ from baseline and how the trunk is tilted.
RR intervals come from a random process tuned so RMSSD and SDNN hit the targets.
"""
import math
import random

BASE_RR_MS = 800.0      # 75 bpm at rest
BASE_RMSSD = 45.0       # ms
BASE_SDNN = 55.0        # ms
ACC_RATE = 25           # Hz (lower than the real 50 Hz to keep the simulation fast)
WORK_MOVEMENT_G = 0.06  # normal working movement
STILL_MOVEMENT_G = 0.004

# (start_min, end_min, HR%, RMSSD%, SDNN%, forward_deg, side_deg, moving, what you should hear)
PHASES = [
    (0, 5, 0, 0, 0, 0, 0, True, "calibration + baseline (silent)"),
    (5, 8, 0, 0, 0, 30, 0, True, "forward bending 30 deg -> Poor posture"),
    (8, 10, 0, 0, 0, 55, 0, True, "forward bending 55 deg -> Critical posture"),
    (10, 12, 0, 0, 0, 0, 0, True, "upright again (silent)"),
    (12, 15, 0, 0, 0, 0, 15, True, "side bending 15 deg -> Poor posture"),
    (15, 19, 12, -28, -22, 0, 0, True, "Medium fatigue (shown on dashboard, never spoken)"),
    (19, 21, 20, -40, -35, 0, 0, True, "High fatigue x2 (tracked, silent)"),
    (21, 24, 12, -28, -22, 0, 0, True, "Medium -> High count resets (silent)"),
    (24, 25.75, 20, -40, -35, 0, 0, True, "High fatigue x2 again (tracked, silent)"),
    (25.75, 29, 0, 0, 0, 0, 0, True, "Low -> count resets (silent)"),
    (29, 40, 20, -40, -35, 0, 0, True, "High x3 in a row -> spoken; still High x3 -> escalation spoken"),
    (40, 45, 0, 0, 0, 0, 0, True, "recovery (silent)"),
    (45, 57, 0, 0, 0, 0, 0, False, "standing still -> Static posture after ~5 and ~10 min"),
    (57, 60, 0, 0, 0, 0, 0, True, "moving again (silent)"),
]


def phase_at(minute):
    for p in PHASES:
        if p[0] <= minute < p[1]:
            return p
    return PHASES[-1]


def simulated_events(seed=7, start_t=0.0):
    rnd = random.Random(seed)
    t_beat = start_t
    a = 0.0  # AR(1) state for RR variability
    end_s = PHASES[-1][1] * 60
    acc_dt = 1.0 / ACC_RATE
    t_acc = start_t
    sway = 0.0
    while t_beat - start_t < end_s:
        _, _, hr_p, rmssd_p, sdnn_p, fwd, side, moving, _ = phase_at((t_beat - start_t) / 60)
        mean_rr = BASE_RR_MS / (1 + hr_p / 100)
        target_rmssd = BASE_RMSSD * (1 + rmssd_p / 100)
        target_sdnn = BASE_SDNN * (1 + sdnn_p / 100)
        # AR(1): RMSSD = SDNN * sqrt(2 * (1 - phi))  ->  phi = 1 - RMSSD^2 / (2 * SDNN^2)
        phi = max(-0.9, min(0.95, 1 - target_rmssd ** 2 / (2 * target_sdnn ** 2)))
        a = phi * a + rnd.gauss(0, target_sdnn * math.sqrt(1 - phi ** 2))
        rr = mean_rr + a
        t_beat += rr / 1000

        # accelerometer samples up to this beat
        while t_acc < t_beat:
            _, _, _, _, _, fwd, side, moving, _ = phase_at((t_acc - start_t) / 60)
            f, s = math.radians(fwd), math.radians(side)
            # gravity when upright is along +X; forward bend tips it toward Z, side bend toward Y
            gx = math.cos(f) * math.cos(s)
            gy = math.sin(s)
            gz = math.sin(f) * math.cos(s)
            noise = WORK_MOVEMENT_G if moving else STILL_MOVEMENT_G
            sway = 0.9 * sway + rnd.gauss(0, noise * 0.3)  # slow body sway
            yield ("acc", t_acc, gx + rnd.gauss(0, noise) + sway, gy + rnd.gauss(0, noise),
                   gz + rnd.gauss(0, noise) - sway)
            t_acc += acc_dt
        yield ("beat", t_beat, 60000 / rr, rr)


# What really happens in each phase -- used as labels.csv by --save (for evaluate_rules.py)
SIM_LABELS = [
    ("0:00", "5:00", "posture", "good", ""),
    ("5:00", "8:00", "posture", "poor", "forward_bending"),
    ("8:00", "10:00", "posture", "critical", "forward_bending"),
    ("10:00", "12:00", "posture", "good", ""),
    ("12:00", "15:00", "posture", "poor", "side_bending"),
    ("15:00", "50:00", "posture", "good", ""),
    ("50:00", "55:00", "posture", "poor", "static_posture"),
    ("55:00", "57:00", "posture", "critical", "static_posture"),
    ("57:00", "60:00", "posture", "good", ""),
    ("0:00", "15:00", "fatigue", "low", ""),
    ("15:00", "19:00", "fatigue", "medium", ""),
    ("19:00", "21:00", "fatigue", "high", "prolonged_exertion"),
    ("21:00", "24:00", "fatigue", "medium", ""),
    ("24:00", "25:45", "fatigue", "high", "prolonged_exertion"),
    ("25:45", "29:00", "fatigue", "low", ""),
    ("29:00", "40:00", "fatigue", "high", "prolonged_exertion"),
    ("40:00", "60:00", "fatigue", "low", ""),
]


def save(folder, seed=7):
    """Write the simulated shift as a recording (hr.csv, acc.csv) plus labels.csv."""
    import csv
    from pathlib import Path

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "hr.csv", "w", newline="") as fh, open(folder / "acc.csv", "w", newline="") as fa:
        hw, aw = csv.writer(fh), csv.writer(fa)
        hw.writerow(["t", "hr", "rr_ms"])
        aw.writerow(["t", "ax", "ay", "az"])
        for ev in simulated_events(seed=seed):
            if ev[0] == "beat":
                hw.writerow([f"{ev[1]:.3f}", f"{ev[2]:.0f}", f"{ev[3]:.1f}"])
            else:
                aw.writerow([f"{ev[1]:.3f}", f"{ev[2]:.4f}", f"{ev[3]:.4f}", f"{ev[4]:.4f}"])
    with open(folder / "labels.csv", "w", newline="") as fl:
        w = csv.writer(fl)
        w.writerow(["start", "end", "track", "label", "cause"])
        w.writerows(SIM_LABELS)
    print(f"Saved simulated recording + labels.csv to {folder}/")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Simulated Polar H10 shift")
    ap.add_argument("--save", metavar="FOLDER", help="write hr.csv, acc.csv and labels.csv to FOLDER")
    args = ap.parse_args()
    if args.save:
        save(args.save)
    else:
        n = sum(1 for _ in simulated_events())
        print(f"{n} simulated events over {PHASES[-1][1]} minutes")
        for p in PHASES:
            print(f"  {p[0]:>2}-{p[1]:<2} min  {p[8]}")
