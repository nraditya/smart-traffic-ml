KODE TRAIN LR

import boto3
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timezone

s3 = boto3.client("s3")

# =========================
# [UBAH] ENV VARIABLES
# =========================
CLEAN_BUCKET = os.environ["CLEAN_BUCKET_NAME"]
CLEAN_PREFIX = os.environ["CLEAN_PREFIX"]  # contoh: "cleaned/"

MODEL_BUCKET = os.environ["MODEL_BUCKET_NAME"]
MODEL_PREFIX = os.environ["MODEL_PREFIX"]  # contoh: "models/"  (khusus model LR untuk inference)

# [BARU] Metrics dipisah agar tidak ikut terbaca oleh Lambda Inference
LR_METRICS_PREFIX = os.environ.get("LR_METRICS_PREFIX", "metrics/linear_regression/")

# [BARU] Ambil N file cleaned terbaru (default 1)
LATEST_N_FILES = int(os.environ.get("LATEST_N_FILES", "1"))

# [BARU] Split untuk evaluasi (hindari overfitting / angka "terlalu sempurna")
TEST_SIZE = float(os.environ.get("TEST_SIZE", "0.2"))  # 0.2 = 80% train, 20% test
RANDOM_STATE = int(os.environ.get("RANDOM_STATE", "42"))

PHASES = [1, 2, 3, 4]
FEATURES = ["distance_m", "duration_s", "duration_in_traffic_s"]
TARGET = "speed_kmh"


# =========================
# HELPERS
# =========================
def utc_ts_compact() -> str:
    # aman untuk nama file S3 (tanpa ':')
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")


def list_latest_csv_keys(bucket: str, prefix: str, n: int) -> list:
    """[BARU] Ambil N file CSV cleaned TERBARU berdasarkan LastModified."""
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


def train_test_split_np(X: np.ndarray, y: np.ndarray, test_size: float, seed: int):
    n = len(y)
    if n < 10 or test_size <= 0:
        return X, X[:0], y, y[:0]

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    split = int(n * (1.0 - test_size))
    split = max(1, min(split, n - 1))  # pastikan ada train & test

    train_idx = idx[:split]
    test_idx = idx[split:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def metrics_regression(y_true: np.ndarray, y_pred: np.ndarray):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(y_true.mean())) ** 2))
    r2 = 1.0 if ss_tot == 0 else float(1.0 - (ss_res / ss_tot))

    return r2, mae, rmse


def fit_lr_closed_form(X: np.ndarray, y: np.ndarray):
    """Linear Regression: beta = (X'X)^-1 X'y (pakai pseudo-inverse agar aman)."""
    X_mat = np.hstack([np.ones((X.shape[0], 1)), X])  # + intercept
    beta = np.linalg.pinv(X_mat.T @ X_mat) @ (X_mat.T @ y)

    intercept = float(beta[0])
    coef = [float(c) for c in beta[1:]]
    return intercept, coef


def predict_lr(intercept: float, coef: list, X: np.ndarray):
    c = np.array(coef, dtype=float)
    return (X @ c) + intercept


def s3_put_json(bucket: str, key: str, payload: dict):
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, ensure_ascii=False),
        ContentType="application/json",
    )


# =========================
# MAIN HANDLER
# =========================
def lambda_handler(event, context):
    print("=== START TRAINING (LINEAR REGRESSION) ===")
    print("Clean bucket:", CLEAN_BUCKET, "prefix:", CLEAN_PREFIX)

    latest_keys = list_latest_csv_keys(CLEAN_BUCKET, CLEAN_PREFIX, LATEST_N_FILES)
    print("Latest cleaned CSV keys:", latest_keys)

    # [UBAH] gabungkan hanya N file terbaru
    frames = []
    for k in latest_keys:
        obj = s3.get_object(Bucket=CLEAN_BUCKET, Key=k)
        df_tmp = pd.read_csv(obj["Body"])
        df_tmp["__source_key"] = k
        frames.append(df_tmp)

    df = pd.concat(frames, ignore_index=True)

    # rapikan tipe data
    df = df.dropna()
    df["phase"] = pd.to_numeric(df["phase"], errors="coerce")
    for c in FEATURES + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()

    needed_cols = ["phase"] + FEATURES + [TARGET] + ["__source_key"]
    df = df[needed_cols].dropna()

    ts = utc_ts_compact()
    results = {}

    for phase in PHASES:
        df_phase = df[df["phase"] == phase]
        rows = len(df_phase)
        print(f"\n--- PHASE {phase} --- rows={rows}")

        if rows < 20:
            print("SKIP — data terlalu sedikit")
            continue

        X = df_phase[FEATURES].to_numpy(dtype=float)
        y = df_phase[TARGET].to_numpy(dtype=float)

        # =========================
        # [BARU] SPLIT untuk evaluasi
        # =========================
        X_train, X_test, y_train, y_test = train_test_split_np(
            X, y, test_size=TEST_SIZE, seed=RANDOM_STATE + phase
        )

        # ====== 1) model evaluasi: fit di train ======
        intercept_eval, coef_eval = fit_lr_closed_form(X_train, y_train)

        # eval train
        y_pred_train = predict_lr(intercept_eval, coef_eval, X_train)
        r2_train, mae_train, rmse_train = metrics_regression(y_train, y_pred_train)

        # eval test
        if len(y_test) > 0:
            y_pred_test = predict_lr(intercept_eval, coef_eval, X_test)
            r2_test, mae_test, rmse_test = metrics_regression(y_test, y_pred_test)
        else:
            r2_test, mae_test, rmse_test = None, None, None

        # ====== 2) model DEPLOYMENT: fit ulang pakai SEMUA data (train+test) ======
        intercept_full, coef_full = fit_lr_closed_form(X, y)

        # =========================
        # SAVE MODEL (untuk inference) -> pakai FULL fit
        # =========================
        model_key = f"{MODEL_PREFIX}phase_{phase}/lr_model_{ts}.json"
        model_payload = {
            "type": "linear_regression",
            "phase": int(phase),
            "features": FEATURES,
            "intercept": intercept_full,
            "coef": coef_full,
            "timestamp_utc": ts,
            "source_cleaned_csv_keys": latest_keys,
            "model_fit": "full_after_eval",
        }
        s3_put_json(MODEL_BUCKET, model_key, model_payload)

        # =========================
        # SAVE METRICS (dipisah folder metrics/)
        # =========================
        metrics_key = f"{LR_METRICS_PREFIX}phase_{phase}/lr_metrics_{ts}.json"
        metrics_payload = {
            "type": "linear_regression",
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
            "train": {"r2": r2_train, "mae": mae_train, "rmse": rmse_train},
            "test": {"r2": r2_test, "mae": mae_test, "rmse": rmse_test},
        }
        s3_put_json(MODEL_BUCKET, metrics_key, metrics_payload)

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

    print("=== TRAINING COMPLETE (LR) ===")
    return {"status": "SUCCESS", "latest_cleaned_csv_keys": latest_keys, "models": results}
