# CODE TRAIN RANDOM FOREST

import boto3
import numpy as np
import json
import os
import io
import csv
from datetime import datetime, timezone

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

s3 = boto3.client("s3")

# =========================
# [UBAH] ENV VARIABLES
# =========================
CLEAN_BUCKET = os.environ["CLEAN_BUCKET_NAME"]
CLEAN_PREFIX = os.environ["CLEAN_PREFIX"]

MODEL_BUCKET = os.environ["MODEL_BUCKET_NAME"]

# model RF (joblib) tetap di folder model RF
RF_MODEL_PREFIX = os.environ.get("RF_MODEL_PREFIX", "models_randomforest/")

# [UBAH] metrics RF dipindah ke metrics/
RF_METRICS_PREFIX = os.environ.get("RF_METRICS_PREFIX", "metrics/random_forest/")

FEATURES = ["distance_m", "duration_s", "duration_in_traffic_s"]
TARGET = "speed_kmh"

# [BARU] Ambil N file cleaned terbaru (default 1)
LATEST_N_FILES = int(os.environ.get("LATEST_N_FILES", "1"))

# [BARU] Split evaluasi
TEST_SIZE = float(os.environ.get("TEST_SIZE", "0.2"))
RANDOM_STATE = int(os.environ.get("RANDOM_STATE", "42"))

PHASES = [1, 2, 3, 4]


# =========================
# HELPERS
# =========================
def utc_ts_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")


def list_latest_csv_keys(bucket: str, prefix: str, n: int) -> list:
    paginator = s3.get_paginator("list_objects_v2")
    objs = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj.get("Key", "")
            if k.endswith(".csv"):
                objs.append((k, obj["LastModified"]))

    if not objs:
        raise Exception(f"Tidak ada CSV di s3://{bucket}/{prefix}")

    objs.sort(key=lambda x: x[1], reverse=True)
    n = max(1, min(n, len(objs)))
    return [k for (k, _) in objs[:n]]


def read_rows_from_csv_s3(bucket: str, key: str):
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read().decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(data))
    for row in reader:
        yield row


def to_float(row, col):
    v = row.get(col, "")
    if v is None:
        return None
    v = str(v).strip()
    if v == "" or v.lower() == "nan":
        return None
    try:
        return float(v)
    except:
        return None


def train_test_split_np(X: np.ndarray, y: np.ndarray, test_size: float, seed: int):
    n = len(y)
    if n < 10 or test_size <= 0:
        return X, X[:0], y, y[:0]

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    split = int(n * (1.0 - test_size))
    split = max(1, min(split, n - 1))

    train_idx = idx[:split]
    test_idx = idx[split:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def save_json(bucket: str, key: str, data: dict):
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False),
        ContentType="application/json"
    )


def save_model_joblib(bucket: str, key: str, model):
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=buf.read(),
        ContentType="application/octet-stream"
    )


# =========================
# MAIN HANDLER
# =========================
def lambda_handler(event, context):
    print("=== START TRAINING (RANDOM FOREST) ===")

    latest_keys = list_latest_csv_keys(CLEAN_BUCKET, CLEAN_PREFIX, LATEST_N_FILES)
    print("Latest cleaned CSV keys:", latest_keys)

    # kumpulkan data per phase (dari N file terbaru)
    per_phase_X = {p: [] for p in PHASES}
    per_phase_y = {p: [] for p in PHASES}

    for key in latest_keys:
        print("Load:", key)
        for row in read_rows_from_csv_s3(CLEAN_BUCKET, key):
            phase_val = to_float(row, "phase")
            if phase_val is None:
                continue
            phase = int(phase_val)
            if phase not in PHASES:
                continue

            x = []
            ok = True
            for f in FEATURES:
                fv = to_float(row, f)
                if fv is None:
                    ok = False
                    break
                x.append(fv)

            yv = to_float(row, TARGET)
            if not ok or yv is None:
                continue

            per_phase_X[phase].append(x)
            per_phase_y[phase].append(yv)

    results = {}
    ts = utc_ts_compact()

    for phase in PHASES:
        X_list = per_phase_X[phase]
        y_list = per_phase_y[phase]
        rows = len(y_list)

        print(f"\n--- PHASE {phase} --- rows={rows}")
        if rows < 20:
            print("SKIP — data terlalu sedikit")
            continue

        X = np.array(X_list, dtype=float)
        y = np.array(y_list, dtype=float)

        # =========================
        # [BARU] SPLIT untuk evaluasi
        # =========================
        X_train, X_test, y_train, y_test = train_test_split_np(
            X, y, test_size=TEST_SIZE, seed=RANDOM_STATE + phase
        )

        # ====== 1) model evaluasi: fit di train ======
        model_eval = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42
        )
        model_eval.fit(X_train, y_train)

        # eval train
        y_pred_train = model_eval.predict(X_train)
        r2_train = float(r2_score(y_train, y_pred_train))
        mae_train = float(mean_absolute_error(y_train, y_pred_train))
        rmse_train = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))

        # eval test
        if len(y_test) > 0:
            y_pred_test = model_eval.predict(X_test)
            r2_test = float(r2_score(y_test, y_pred_test))
            mae_test = float(mean_absolute_error(y_test, y_pred_test))
            rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
        else:
            r2_test, mae_test, rmse_test = None, None, None

        # ====== 2) model DEPLOYMENT: fit ulang pakai SEMUA data ======
        model_full = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42
        )
        model_full.fit(X, y)

        model_key = f"{RF_MODEL_PREFIX}phase_{phase}/rf_model_{ts}.joblib"
        metrics_key = f"{RF_METRICS_PREFIX}phase_{phase}/rf_metrics_{ts}.json"

        save_model_joblib(MODEL_BUCKET, model_key, model_full)

        metrics = {
            "type": "random_forest",
            "phase": int(phase),
            "features": FEATURES,
            "target": TARGET,
            "timestamp_utc": ts,
            "source_cleaned_csv_keys": latest_keys,
            "split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE},
            "rows_total": int(rows),
            "rows_train": int(len(y_train)),
            "rows_test": int(len(y_test)),
            "eval_on": "holdout_test",
            "rf_params": {
                "n_estimators": 200,
                "max_depth": 12,
                "min_samples_leaf": 2,
                "random_state": 42
            },
            "train": {"r2": r2_train, "mae": mae_train, "rmse": rmse_train},
            "test": {"r2": r2_test, "mae": mae_test, "rmse": rmse_test},
            "model_fit": "full_after_eval"
        }
        save_json(MODEL_BUCKET, metrics_key, metrics)

        print("Saved model:", model_key)
        print("Saved metrics:", metrics_key)
        print(
            f"EVAL TRAIN r2={r2_train:.4f} mae={mae_train:.4f} rmse={rmse_train:.4f} | "
            + (f"EVAL TEST r2={r2_test:.4f} mae={mae_test:.4f} rmse={rmse_test:.4f}" if r2_test is not None else "EVAL TEST n/a")
        )

        results[f"phase_{phase}"] = {
            "model_s3_key": model_key,
            "metrics_s3_key": metrics_key,
            "rows_total": int(rows),
            "rows_train": int(len(y_train)),
            "rows_test": int(len(y_test)),
            "train": {"r2": r2_train, "mae": mae_train, "rmse": rmse_train},
            "test": {"r2": r2_test, "mae": mae_test, "rmse": rmse_test},
        }

    print("=== TRAINING COMPLETE (RF) ===")
    return {"status": "SUCCESS", "latest_cleaned_csv_keys": latest_keys, "models": results}
