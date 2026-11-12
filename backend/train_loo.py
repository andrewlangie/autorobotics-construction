#!/usr/bin/env python3
"""Leave-one-participant-out evaluation with causal, scale-free features.

Run from the project root:
    python backend/train_loo.py --dataset <folder|zip> --id-regex "(\\d+)"

Compares the legacy raw features (eda, temp, minute_index) against the scale-free window
features from features.py, with logistic regression and gradient boosting, under the same
leave-one-participant-out split, and prints accuracy / precision / recall / F1.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURE_NAMES, build_feature_frame  # noqa: E402
from train_biomarker_models import load_and_resample, maybe_unzip_dataset  # noqa: E402

LABELS = ["low", "medium", "high"]

FEATURE_SETS = {
    "raw": ["eda", "temp", "minute_index"],                    # what the current models use
    "scalefree": FEATURE_NAMES,                                # causal, unit-free window features
    "scalefree+minute": FEATURE_NAMES + ["minute_index"],
}


def make_models():
    return {
        "logreg": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]),
        "gboost": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=150,
            class_weight="balanced", random_state=42,
        ),
    }


def participant_id(file_name, id_regex):
    m = re.search(id_regex, Path(file_name).stem, re.I)
    if not m:
        raise ValueError(f"--id-regex {id_regex!r} found no participant in {file_name!r}")
    return m.group(1)


def build_dataset(task_dir, id_regex):
    frames = []
    for csv_path in sorted(Path(task_dir).glob("*.csv")):
        d = load_and_resample(csv_path)
        if d is None:
            continue
        f = build_feature_frame(d["eda"].to_numpy(), d["temp"].to_numpy())
        if f.empty:
            continue
        idx = f["minute_index"].to_numpy()
        f["eda"] = d["eda"].to_numpy()[idx]
        f["temp"] = d["temp"].to_numpy()[idx]
        f["file"] = csv_path.name
        f["label"] = d["label"].iloc[0]
        f["participant"] = participant_id(csv_path.name, id_regex)
        f["type"] = ("active" if "active" in csv_path.stem.lower().replace("passive", "")
                     else "passive" if "passive" in csv_path.stem.lower() else "unknown")
        frames.append(f)
    if not frames:
        raise SystemExit(f"No usable CSVs in {task_dir}")
    return pd.concat(frames, ignore_index=True)


def score(y, pred):
    y = np.asarray(y)
    labels = [l for l in LABELS if l in set(y)]
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    return {
        "labels": labels, "accuracy": accuracy_score(y, pred),
        "p": p, "r": r, "f1": f, "support": s,
        "macro_p": p.mean(), "macro_r": r.mean(), "macro_f1": f.mean(),
    }


def print_block(title, sc):
    print(f"  {title}")
    print(f"    accuracy {sc['accuracy']:.3f} | macro precision {sc['macro_p']:.3f} "
          f"recall {sc['macro_r']:.3f} f1 {sc['macro_f1']:.3f}")
    print(f"    {'class':<8}{'precision':>10}{'recall':>9}{'f1':>8}{'support':>9}")
    for i, lab in enumerate(sc["labels"]):
        print(f"    {lab:<8}{sc['p'][i]:>10.3f}{sc['r'][i]:>9.3f}{sc['f1'][i]:>8.3f}{int(sc['support'][i]):>9}")


def loo_predict(df, cols, model, y=None):
    y = df["label"] if y is None else y
    return cross_val_predict(clone(model), df[cols], y, cv=LeaveOneGroupOut(), groups=df["participant"])


def file_level(df, pred):
    tmp = pd.DataFrame({"file": df["file"].values, "y": df["label"].values, "p": pred})
    agg = tmp.groupby("file").agg(y=("y", "first"), p=("p", lambda s: s.mode().iloc[0]))
    return agg["y"].to_numpy(), agg["p"].to_numpy()


def permutation_p(df, cols, model, observed_f1, n, seed=0):
    rng = np.random.default_rng(seed)
    files = df.groupby("file")["label"].first()
    hits = 0
    for _ in range(n):
        shuffled = pd.Series(rng.permutation(files.to_numpy()), index=files.index)
        y = df["file"].map(shuffled)
        pred = loo_predict(df, cols, model, y=y)
        hits += score(y, pred)["macro_f1"] >= observed_f1
    return (1 + hits) / (n + 1)


def describe(df, name):
    files = df.drop_duplicates("file")
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    print(f"participants: {df['participant'].nunique()} | files: {len(files)} | minute rows: {len(df)}")
    print("files per participant:", files.groupby("participant").size().value_counts().sort_index().to_dict())
    print("files per class:", files["label"].value_counts().reindex(LABELS).fillna(0).astype(int).to_dict())
    if files["type"].nunique() > 1 or files["type"].iloc[0] != "unknown":
        print("label x exoskeleton type (files):")
        print(pd.crosstab(files["label"], files["type"]).reindex(LABELS).fillna(0).astype(int).to_string())


def run_construct(name, df, args, summary):
    describe(df, name)
    cv_groups = df["participant"].nunique()
    if cv_groups < 2:
        raise SystemExit("Need at least 2 participants for leave-one-participant-out CV")
    models = make_models()

    dummy_pred = loo_predict(df, ["minute_index"], DummyClassifier(strategy="most_frequent"))
    dummy = score(df["label"], dummy_pred)
    print(f"\nDummy (always majority class): accuracy {dummy['accuracy']:.3f}, macro F1 {dummy['macro_f1']:.3f}")

    for fs_name in args.feature_sets:
        cols = FEATURE_SETS[fs_name]
        for m_name, model in models.items():
            print(f"\n--- {name} | features={fs_name} | model={m_name} | LOPO folds={cv_groups} ---")
            pred = loo_predict(df, cols, model)
            sc = score(df["label"], pred)
            print_block("Minute level (pooled over held-out participants)", sc)
            fy, fp = file_level(df, pred)
            fsc = score(fy, fp)
            print_block(f"File level (majority vote per file, n={len(fy)})", fsc)
            print("    confusion (rows=true, cols=pred, order low/medium/high):")
            for lab, row in zip(LABELS, confusion_matrix(df["label"], pred, labels=LABELS)):
                print(f"      {lab:<7}{row.tolist()}")
            p_val = None
            if args.permutations > 0:
                p_val = permutation_p(df, cols, model, sc["macro_f1"], args.permutations)
                print(f"    permutation p-value (macro F1, {args.permutations} shuffles): {p_val:.3f}")
            summary.append({
                "construct": name, "features": fs_name, "model": m_name,
                "acc": sc["accuracy"], "prec": sc["macro_p"], "rec": sc["macro_r"], "f1": sc["macro_f1"],
                "file_f1": fsc["macro_f1"], "dummy_f1": dummy["macro_f1"], "perm_p": p_val,
            })


def fit_and_save(name, df, args, outdir):
    cols = FEATURE_SETS[args.final_set]
    model = make_models()["logreg"].fit(df[cols], df["label"])
    slug = name.lower().replace(" ", "_")
    joblib.dump(model, outdir / f"{slug}_loo_model.joblib")
    return {"features": cols, "model": "logreg", "feature_set": args.final_set}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="folder or .zip with Frustration/ and Mental Demand/")
    ap.add_argument("--constructs", nargs="+", default=["Frustration", "Mental Demand"])
    ap.add_argument("--id-regex", default=r"(\d+)",
                    help="regex with ONE group that extracts the participant id from the filename stem")
    ap.add_argument("--feature-sets", nargs="+", default=list(FEATURE_SETS), choices=list(FEATURE_SETS))
    ap.add_argument("--permutations", type=int, default=0, help="label-permutation test shuffles (0 = off)")
    ap.add_argument("--outdir", default=None, help="if set, fit a final model per construct and save it here")
    ap.add_argument("--final-set", default="scalefree", choices=list(FEATURE_SETS))
    args = ap.parse_args()

    root = maybe_unzip_dataset(Path(args.dataset).resolve(), Path(args.outdir or ".").resolve())
    summary, meta = [], {}
    for name in args.constructs:
        df = build_dataset(root / name, args.id_regex)
        run_construct(name, df, args, summary)
        if args.outdir:
            outdir = Path(args.outdir).resolve()
            outdir.mkdir(parents=True, exist_ok=True)
            meta[name] = fit_and_save(name, df, args, outdir)

    print(f"\n{'=' * 78}\nSUMMARY (leave-one-participant-out; higher is better)\n{'=' * 78}")
    table = pd.DataFrame(summary)
    print(table.round(3).to_string(index=False))
    print("\nRead this as: does `scalefree` beat `raw`, and does either beat dummy_f1? "
          "With ~19 participants, differences under ~0.05 F1 are noise.")

    if args.outdir:
        (Path(args.outdir).resolve() / "loo_metadata.json").write_text(
            json.dumps({"id_regex": args.id_regex, "models": meta, "summary": summary}, indent=2, default=float))


if __name__ == "__main__":
    main()
