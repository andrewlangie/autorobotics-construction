"""Measure how accurate the fatigue / posture RULES are against your own labels.

1. Record a session:   python backend/polar_reader.py --record recordings/session1
2. Write labels.csv in that folder -- what really happened, as time ranges from the
   start of the recording (mm:ss or seconds):

       start,end,track,label,cause
       0:00,5:00,posture,good,
       5:00,7:00,posture,poor,forward_bending
       7:00,9:00,posture,critical,forward_bending
       0:00,15:00,fatigue,low,
       15:00,20:00,fatigue,medium,

   track = posture | fatigue
   label = posture: good / poor / critical      (or 1 / 2 / 3)
           fatigue: low / medium / high / very_high   (or 1 / 2 / 3 / 4)
   cause = optional (e.g. forward_bending, side_bending, static_posture,
           cardiovascular, prolonged_exertion, heat, sleep, long_work)
   Time not covered by a label is ignored. Overlapping labels: the last one wins.

3. Run:   python backend/evaluate_rules.py recordings/session1
          python backend/evaluate_rules.py recordings/session1 --out predictions.csv

Every 5 s of the recording the engines' CURRENT level (before the duration gate -- we
measure the classification, not when a message is spoken) is compared with your label.
Calibration and the fatigue baseline (first ~4 min) are skipped. Fatigue "no fatigue"
(below Low) counts as Low, because both give no feedback.

Several sessions: pass several folders; results are pooled.
"""
import argparse
import csv
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from feedback_engine import FatigueEngine, PostureEngine
from polar_coach import EVALUATE_EVERY_S, replay_events

POSTURE_NAMES = {"good": 1, "poor": 2, "bad": 2, "critical": 3}
FATIGUE_NAMES = {"low": 1, "none": 1, "medium": 2, "high": 3, "very_high": 4, "very high": 4}
LEVEL_LABELS = {
    "posture": {1: "good", 2: "poor", 3: "critical"},
    "fatigue": {1: "low", 2: "medium", 3: "high", 4: "very_high"},
}


def parse_time(text):
    """'5:30' -> 330.0, '330' -> 330.0"""
    text = text.strip()
    if ":" in text:
        parts = [float(p) for p in text.split(":")]
        seconds = 0.0
        for p in parts:
            seconds = seconds * 60 + p
        return seconds
    return float(text)


def parse_level(track, text):
    text = text.strip().lower()
    if text.isdigit():
        return int(text)
    names = POSTURE_NAMES if track == "posture" else FATIGUE_NAMES
    if text not in names:
        raise ValueError(f"unknown {track} label {text!r} (use {sorted(set(names))})")
    return names[text]


def load_labels(path):
    """List of (start_s, end_s, track, level, cause_or_None)."""
    labels = []
    with open(path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            track = row["track"].strip().lower()
            if track not in ("posture", "fatigue"):
                raise ValueError(f"{path} line {i}: track must be posture or fatigue")
            cause = (row.get("cause") or "").strip().lower() or None
            labels.append((parse_time(row["start"]), parse_time(row["end"]), track,
                           parse_level(track, row["label"]), cause))
    return labels


def label_at(labels, track, t):
    found = None
    for start, end, tr, level, cause in labels:
        if tr == track and start <= t < end:
            found = (level, cause)  # last matching label wins
    return found


def predict(events):
    """Run the engines over a recording; yield (t_from_start, fatigue_state, posture_state)."""
    fatigue, posture = FatigueEngine(), PostureEngine()
    start = next_eval = None
    for ev in events:
        t = ev[1]
        if start is None:
            start, next_eval = t, t + EVALUATE_EVERY_S
        if ev[0] == "beat":
            fatigue.add_beat(t, ev[2], ev[3])
        else:
            posture.add_acc(t, ev[2], ev[3], ev[4])
        if t >= next_eval:
            next_eval = t + EVALUATE_EVERY_S
            fatigue.evaluate(t)
            posture.evaluate(t)
            yield t - start, dict(fatigue.state), dict(posture.state)


def compare(sessions):
    """sessions: list of (name, events, labels). Returns rows of predicted vs true."""
    rows = []
    for name, events, labels in sessions:
        for t, fs, ps in predict(events):
            for track, state in (("fatigue", fs), ("posture", ps)):
                truth = label_at(labels, track, t)
                if truth is None or state.get("status") != "ok":
                    continue
                pred = max(state["level"], 1)  # fatigue level 0 (none) counts as Low
                rows.append({
                    "session": name, "t_s": round(t, 1), "track": track,
                    "true_level": truth[0], "pred_level": pred,
                    "true_cause": truth[1] or "", "pred_cause": state.get("cause") or "",
                })
    return rows


def report(rows, track):
    rows = [r for r in rows if r["track"] == track]
    print(f"\n=== {track.upper()} ===")
    if not rows:
        print("  no labelled time to compare (check labels.csv and that the session is long enough)")
        return None
    y_true = [r["true_level"] for r in rows]
    y_pred = [r["pred_level"] for r in rows]
    levels = sorted(LEVEL_LABELS[track])
    names = [LEVEL_LABELS[track][lv] for lv in levels]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, labels=levels, average="macro", zero_division=0)
    print(f"  compared: {len(rows)} decisions ({len(rows) * EVALUATE_EVERY_S / 60:.1f} min)")
    print(f"  accuracy: {acc:.3f}     macro-F1: {f1:.3f}")
    print("\n  per level:")
    rep = classification_report(y_true, y_pred, labels=levels, target_names=names, zero_division=0)
    print("  " + rep.replace("\n", "\n  "))
    cm = confusion_matrix(y_true, y_pred, labels=levels)
    width = max(len(n) for n in names) + 2
    print("  confusion matrix (rows = true label, columns = rules' prediction)")
    print("  " + " " * width + "".join(f"{n:>{width}}" for n in names))
    for name, line in zip(names, cm):
        print("  " + f"{name:<{width}}" + "".join(f"{v:>{width}}" for v in line))

    cause_rows = [r for r in rows if r["true_cause"] and r["true_level"] >= 2]
    if cause_rows:
        hits = sum(r["true_cause"] == r["pred_cause"] for r in cause_rows)
        print(f"\n  cause accuracy (labelled causes, level >= 2): {hits / len(cause_rows):.3f} "
              f"({hits}/{len(cause_rows)})")
    return {"accuracy": acc, "macro_f1": f1, "n": len(rows)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folders", nargs="+", help="recording folders (hr.csv, acc.csv, labels.csv)")
    p.add_argument("--labels", help="labels file name inside each folder (default labels.csv)",
                   default="labels.csv")
    p.add_argument("--out", help="also save every compared decision to this CSV")
    args = p.parse_args()

    sessions = []
    for folder in args.folders:
        folder = Path(folder)
        sessions.append((folder.name, replay_events(folder), load_labels(folder / args.labels)))
    rows = compare(sessions)
    report(rows, "posture")
    report(rows, "fatigue")
    if args.out and rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
