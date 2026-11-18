import os
import time
from datetime import datetime, timezone, timedelta

import boto3
import pandas as pd
from flask import Flask, jsonify

from biomarker_runtime import LiveBiomarkerPredictor

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = "us-east-1"

S3_BUCKET = "empatica-us-east-1-prod-data"
S3_BASE_PREFIX = "v2/1324/1/1/participant_data/"
POLL_SECONDS = 30

FORCE_DOWNLOAD_EVERY_POLL = True

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_BIOMARKER_DIR = os.path.join(PROJECT_ROOT, "biomarker_data")
os.makedirs(LOCAL_BIOMARKER_DIR, exist_ok=True)

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
predictor = LiveBiomarkerPredictor(MODEL_DIR)

app = Flask(__name__)
latest_sample = {
    "eda": None,
    "temp": None,
    "pulse_rate": None,
    "scr": None,
    "hr": None,
    "rmssd": None,
    "sdnn": None,
    "repeated_failures": 0,
    "delay_pct": 0.0,
    "interruptions": 0,
    "waiting_min": 0.0,
    "clarification_requests": 0,
    "eda_time": None,
    "temp_time": None,
    "pulse_time": None,
    "participant_id": None,
    "source_prefix": None,
    "updated_at": None,
    "note": "waiting for first sample",
    "predictions": None,
}


@app.route("/latest", methods=["GET"])
def latest():
    return jsonify(latest_sample)


def make_s3_client():
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def list_objects_all(s3, prefix):
    out = []
    token = None
    while True:
        args = {"Bucket": S3_BUCKET, "Prefix": prefix}
        if token:
            args["ContinuationToken"] = token
        resp = s3.list_objects_v2(**args)
        out.extend(resp.get("Contents", []))
        if resp.get("IsTruncated"):
            token = resp["NextContinuationToken"]
        else:
            break
    return out


def download_key(s3, key, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(S3_BUCKET, key, local_path)


def looks_like_epoch_ms(x):
    try:
        return float(x) > 1e11
    except Exception:
        return False


def epoch_ms_to_iso(v):
    try:
        dt = datetime.fromtimestamp(float(v) / 1000.0, tz=timezone.utc).astimezone()
        return dt.isoformat(timespec="seconds")
    except Exception:
        return None


def find_timestamp_col(df):
    for c in df.columns:
        cl = c.lower()
        if "timestamp_iso" == cl:
            return c
    for c in df.columns:
        cl = c.lower()
        if "timestamp_unix" == cl:
            return c
    for c in df.columns:
        cl = c.lower()
        if "time" in cl or "stamp" in cl or "unix" in cl or "epoch" in cl:
            return c
    return None


def choose_value_col(df, kind):
    preferred = {
        "eda": ["eda_scl_usiemens", "eda", "EDA", "value"],
        "temp": ["temperature_celsius", "temp_celsius", "temperature", "temp", "value"],
        "pulse": [
            "pulse_rate",
            "pulse-rate",
            "pulse",
            "hr",
            "heart_rate",
            "bpm",
            "value",
        ],
        "scr": ["scr", "skin_conductance_response", "peaks_per_minute", "value"],
        "hr": ["hr", "heart_rate", "bpm", "value"],
        "rmssd": ["rmssd", "RMSSD", "value"],
        "sdnn": ["sdnn", "SDNN", "value"],
    }[kind]

    for c in preferred:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            return c

    ts_col = find_timestamp_col(df)
    for c in df.columns:
        if c != ts_col and pd.api.types.is_numeric_dtype(df[c]):
            try:
                if not looks_like_epoch_ms(df.iloc[-1][c]):
                    return c
            except Exception:
                return c
    return None


def read_last_value(csv_path, kind):
    df = pd.read_csv(csv_path)
    if df.empty:
        return None, None

    ts_col = find_timestamp_col(df)
    val_col = choose_value_col(df, kind)
    if not val_col:
        print(f"[DEBUG] No value column for {kind}: {list(df.columns)}")
        return None, None

    for i in range(len(df) - 1, -1, -1):
        row = df.iloc[i]
        val = row.get(val_col, None)
        if pd.isna(val) or looks_like_epoch_ms(val):
            continue

        ts_val = None
        if ts_col and ts_col in df.columns:
            ts_raw = row.get(ts_col, None)
            if pd.notna(ts_raw):
                if looks_like_epoch_ms(ts_raw):
                    ts_val = epoch_ms_to_iso(ts_raw)
                else:
                    ts_val = str(ts_raw)

        return ts_val, float(val)

    return None, None


def read_all_values(csv_path, kind):
    """Every valid minute in the file as {timestamp: value}, oldest first.

    Empatica uploads the aggregated-per-minute CSVs in batches (~every 10 min), so one
    download can hold many new minutes. read_last_value() only returns the newest one;
    this returns them all so each minute can be fed to the predictor.
    """
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    if df.empty:
        return {}
    ts_col = find_timestamp_col(df)
    val_col = choose_value_col(df, kind)
    if not ts_col or not val_col:
        return {}
    out = {}
    for ts_raw, val in zip(df[ts_col], df[val_col]):
        if pd.isna(val) or pd.isna(ts_raw) or looks_like_epoch_ms(val):
            continue
        ts = epoch_ms_to_iso(ts_raw) if looks_like_epoch_ms(ts_raw) else str(ts_raw)
        if ts:
            out[ts] = float(val)
    return dict(sorted(out.items()))


def utc_dates_to_check(days_back=1):
    now_utc = datetime.now(timezone.utc)
    return [
        (now_utc - timedelta(days=d)).strftime("%Y-%m-%d")
        for d in range(0, days_back + 1)
    ]


def pick_most_recent_participant_and_keys(s3, date_prefixes, date_strings):
    candidates = []

    for prefix, day in zip(date_prefixes, date_strings):
        objs = list_objects_all(s3, prefix)
        for o in objs:
            k = o["Key"]
            if "/digital_biomarkers/aggregated_per_minute/" not in k:
                continue
            if not k.lower().endswith(".csv"):
                continue
            if f"_{day}_" not in k:
                continue
            candidates.append(o)

    if not candidates:
        return None, {}, {}, None

    groups = {}
    for o in candidates:
        parts = o["Key"].split("/")
        pid = None
        for day in date_strings:
            if day in parts:
                try:
                    pid = parts[parts.index(day) + 1]
                except Exception:
                    pid = None
                if pid:
                    break
        if not pid:
            continue
        groups.setdefault(pid, []).append(o)

    if not groups:
        return None, {}, {}, None

    best_pid = max(groups, key=lambda p: max(x["LastModified"] for x in groups[p]))
    keys = {}
    mods = {}
    used_prefix = None
    best_objs = groups[best_pid]

    def newest_ending(suffix):
        matches = [o for o in best_objs if o["Key"].endswith(suffix)]
        if not matches:
            return None
        return max(matches, key=lambda x: x["LastModified"])

    eda_o = newest_ending("_eda.csv")
    temp_o = newest_ending("_temperature.csv")
    pulse_o = newest_ending("_pulse-rate.csv")
    scr_o = newest_ending("_scr.csv")
    hr_o = newest_ending("_hr.csv")
    rmssd_o = newest_ending("_rmssd.csv")
    sdnn_o = newest_ending("_sdnn.csv")

    if eda_o:
        keys["eda"] = eda_o["Key"]
        mods["eda"] = eda_o["LastModified"]
    if temp_o:
        keys["temp"] = temp_o["Key"]
        mods["temp"] = temp_o["LastModified"]
    if pulse_o:
        keys["pulse"] = pulse_o["Key"]
        mods["pulse"] = pulse_o["LastModified"]
    if scr_o:
        keys["scr"] = scr_o["Key"]
        mods["scr"] = scr_o["LastModified"]
    if hr_o:
        keys["hr"] = hr_o["Key"]
        mods["hr"] = hr_o["LastModified"]
    if rmssd_o:
        keys["rmssd"] = rmssd_o["Key"]
        mods["rmssd"] = rmssd_o["LastModified"]
    if sdnn_o:
        keys["sdnn"] = sdnn_o["Key"]
        mods["sdnn"] = sdnn_o["LastModified"]


    newest_obj = max(best_objs, key=lambda x: x["LastModified"])
    newest_key_parts = newest_obj["Key"].split("/")
    for day in date_strings:
        if day in newest_key_parts:
            used_prefix = f"{S3_BASE_PREFIX}{day}/"
            break

    return best_pid, keys, mods, used_prefix


def poll_loop():
    s3 = make_s3_client()
    last_ts = None   # newest minute already fed to the predictor
    last_pid = None
    last_mods = {}

    print("Polling biomarker CSV updates...")

    while True:
        try:
            days = utc_dates_to_check(days_back=1)
            prefixes = [f"{S3_BASE_PREFIX}{d}/" for d in days]
            pid, keys, mods, used_prefix = pick_most_recent_participant_and_keys(
                s3, prefixes, days
            )

            if not pid or not all(k in keys for k in ("eda", "temp", "pulse")):
                latest_sample["note"] = "waiting for biomarker files"
                time.sleep(POLL_SECONDS)
                continue

            changed = FORCE_DOWNLOAD_EVERY_POLL or any(
                mods[k] != last_mods.get(k) for k in mods
            )

            print(f"[DEBUG] Participant: {pid}")
            print(
                f"[DEBUG] S3 LastModified eda={mods.get('eda')} temp={mods.get('temp')} pulse={mods.get('pulse')}"
            )
            print(f"[DEBUG] changed={changed} (force={FORCE_DOWNLOAD_EVERY_POLL})")

            if not changed:
                time.sleep(POLL_SECONDS)
                continue

            local_day = used_prefix.split("/")[-2] if used_prefix else days[0]
            local_dir = os.path.join(LOCAL_BIOMARKER_DIR, local_day, pid)
            paths = {
                "eda": os.path.join(local_dir, "eda.csv"),
                "temp": os.path.join(local_dir, "temperature.csv"),
                "pulse": os.path.join(local_dir, "pulse-rate.csv"),
                "scr": os.path.join(local_dir, "scr.csv"),
                "hr": os.path.join(local_dir, "hr.csv"),
                "rmssd": os.path.join(local_dir, "rmssd.csv"),
                "sdnn": os.path.join(local_dir, "sdnn.csv"),
            }

            for k in paths:
                if k in keys:
                    download_key(s3, keys[k], paths[k])
                    last_mods[k] = mods[k]

            # Feed EVERY new minute in the download to the predictor, oldest first.
            # (Empatica uploads ~10 minutes at a time; reading only the last row skipped
            # the rest and made the runtime see 10-minute gaps, which reset the session.)
            if pid != last_pid:
                last_ts, last_pid = None, pid
            series = {k: read_all_values(paths[k], k) for k in paths if k in keys}
            new_minutes = [t for t in series.get("eda", {}) if last_ts is None or t > last_ts]

            if not new_minutes:
                latest_sample["note"] = "S3 updated but no new minute rows yet"
                print("[DEBUG] no new minute rows since", last_ts)
            else:
                for ts in new_minutes:
                    val = {k: series[k].get(ts) for k in series}
                    pulse_val = val.get("pulse")
                    hr_val = val.get("hr")
                    latest_sample.update(
                        {
                            "eda": val.get("eda"),
                            "temp": val.get("temp"),
                            "pulse_rate": pulse_val,
                            "scr": val.get("scr"),
                            "hr": hr_val if hr_val is not None else pulse_val,
                            "rmssd": val.get("rmssd"),
                            "sdnn": val.get("sdnn"),
                            "eda_time": ts,
                            "temp_time": ts,
                            "pulse_time": ts,
                            "participant_id": pid,
                            "source_prefix": used_prefix if used_prefix else prefixes[0],
                            "updated_at": datetime.now().isoformat(timespec="seconds"),
                            "note": "new minute detected",
                        }
                    )
                    latest_sample["predictions"] = predictor.update(latest_sample)
                    last_ts = ts
                fr = (latest_sample["predictions"] or {}).get("frustration") or {}
                print(f"Processed {len(new_minutes)} new minute(s) up to {last_ts}; "
                      f"frustration={fr.get('label')} reason={fr.get('reason')}")
                print("Updated /latest:", latest_sample)

        except Exception as e:
            print("Poll error:", e)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    import threading

    threading.Thread(target=poll_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=7000)