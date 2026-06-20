# SMART TRAFFIC PREPROCESS / CLEANING

import boto3
import pandas as pd
import io
from datetime import datetime, timezone
import os

s3 = boto3.client("s3")

RAW_BUCKET = os.environ["RAW_BUCKET"]
RAW_PREFIX = os.environ["RAW_PREFIX"]
CLEAN_BUCKET = os.environ["CLEAN_BUCKET"]
CLEAN_PREFIX = os.environ["CLEAN_PREFIX"]

def lambda_handler(event, context):
    print("=== START PREPROCESS ===")

    # 1. List semua raw file harian
    response = s3.list_objects_v2(Bucket=RAW_BUCKET, Prefix=RAW_PREFIX)
    files = [
        obj["Key"] for obj in response.get("Contents", [])
        if obj["Key"].endswith(".csv")
    ]

    if not files:
        print("TIDAK ADA FILE RAW")
        return {"status": "NO_RAW_FILES"}

    print("Ditemukan RAW files:", files)

    frames = []

    # 2. Download & load semua CSV raw (SKIP baris rusak)
    for key in files:
        print(f"[LOAD] {key}")
        obj = s3.get_object(Bucket=RAW_BUCKET, Key=key)
        body = obj["Body"].read().decode("utf-8")

        df = pd.read_csv(
            io.StringIO(body),
            on_bad_lines="skip"   
        )

        print(f"  Rows loaded: {len(df)}")
        frames.append(df)

    # 3. Gabungkan seluruh dataset
    full = pd.concat(frames, ignore_index=True)
    print("Total rows combined:", len(full))

    # 4. CLEANING
    before = len(full)
    full = full.dropna()
    print("Drop NA:", before - len(full))

    before = len(full)
    full = full[
        (full["duration_s"] > 0) &
        (full["distance_m"] > 0) &
        (full["speed_kmh"] > 0)
    ]
    print("Drop anomaly:", before - len(full))

    full["timestamp_utc"] = pd.to_datetime(full["timestamp_utc"], errors="coerce")
    full = full.dropna(subset=["timestamp_utc"])
    full = full.sort_values("timestamp_utc")
    full.reset_index(drop=True, inplace=True)

    print("Final cleaned rows:", len(full))

    # 5. Simpan ke cleaned/
    today = datetime.now(timezone.utc).date().isoformat()
    output_key = f"{CLEAN_PREFIX}cleaned_{today}.csv"

    csv_buf = io.StringIO()
    full.to_csv(csv_buf, index=False)

    s3.put_object(
        Bucket=CLEAN_BUCKET,
        Key=output_key,
        Body=csv_buf.getvalue().encode("utf-8"),
        ContentType="text/csv"
    )

    print(f"UPLOAD OK → s3://{CLEAN_BUCKET}/{output_key}")

    return {
        "status": "SUCCESS",
        "rows_cleaned": len(full),
        "saved_to": output_key
    }
