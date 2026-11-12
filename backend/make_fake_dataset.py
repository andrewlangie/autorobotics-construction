#!/usr/bin/env python3
"""Generate a fake labeled dataset in the layout train_loo.py reads.

    <out>/Frustration/P01_active_high.csv ...
    <out>/Mental Demand/P01_passive_low.csv ...

Columns: timestamp_local, eda, temp, bvp (eda/temp/bvp min-max scaled per file, like the real merge).
One label per file, 2 files (active, passive) per participant.

--effect controls how much the label changes the signal:
    1.0  labels leave a (synthetic) trace in the signal  -> tests that the pipeline can learn
    0.0  pure noise, labels carry NO information         -> scores should sit near the dummy baseline;
                                                            if they don't, something is leaking
Scores on this data say nothing about real workers; it only tests the plumbing.

    python make_fake_dataset.py --out fake_data --effect 0
    python backend/train_loo.py --dataset fake_data --id-regex "P(\\d+)"
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["low", "medium", "high"]


def minmax(x):
    return (x - x.min()) / (x.max() - x.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--participants", type=int, default=19)
    ap.add_argument("--minutes", type=int, default=40)
    ap.add_argument("--effect", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.minutes * 6  # one sample every 10 s
    ts = pd.date_range("2025-01-01 10:00", periods=n, freq="10s")

    for construct in ("Frustration", "Mental Demand"):
        folder = Path(args.out) / construct
        folder.mkdir(parents=True, exist_ok=True)
        for p in range(1, args.participants + 1):
            gain = rng.lognormal(0, 0.3)  # person-to-person differences in signal variability
            for kind in ("active", "passive"):
                label = LABELS[rng.integers(0, 3)]
                k = LABELS.index(label)
                eda = np.cumsum(rng.normal(0.002 * args.effect * (k - 1),
                                           gain * (0.02 + 0.02 * args.effect * k), n))
                temp = np.cumsum(rng.normal(0.0003 * args.effect * k, 0.01 * gain, n))
                df = pd.DataFrame({
                    "timestamp_local": ts, "eda": minmax(eda), "temp": minmax(temp),
                    "bvp": rng.normal(size=n),
                })
                df.to_csv(folder / f"P{p:02d}_{kind}_{label}.csv", index=False)
    print(f"Wrote {args.participants * 4} files under {args.out}")


if __name__ == "__main__":
    main()
