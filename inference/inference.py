SMART TRAFFIC INFERENCE

import boto3
import os
import json
import csv
import time
import logging
from datetime import datetime, timezone

# ======================= CLIENTS =======================
s3 = boto3.client("s3")
iot = boto3.client("iot-data")

# ======================= LOGGING =======================
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def utc_now_iso() -> str:
    # ISO 8601 with milliseconds and Z suffix
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def utc_now_epoch_ms() -> int:
    # Epoch milliseconds (UTC)
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def log_event(event: str, **fields):
    payload = {"ts_utc": utc_now_iso(), "event": event, **fields}
    logger.info(json.dumps(payload, ensure_ascii=False))


# =================== ENV VARIABLES =====================
MODEL_BUCKET = os.environ["MODEL_BUCKET"]
MODEL_PREFIX = os.environ["MODEL_PREFIX"]
RAW_REAL_BUCKET = os.environ["RAW_REAL_BUCKET"]
RAW_REAL_PREFIX = os.environ["RAW_REAL_PREFIX"]
IOT_TOPIC = os.environ["IOT_TOPIC"]

# Optional
G_MIN = float(os.environ.get("G_MIN", "10"))  
G_MAX = float(os.environ.get("G_MAX", "50"))  
HYBRID_W_REAL = float(os.environ.get("HYBRID_W_REAL", "0.6"))
HYBRID_W_PRED = float(os.environ.get("HYBRID_W_PRED", "0.4"))

# ESP apply policy (metadata)
APPLY_POLICY = os.environ.get("APPLY_POLICY", "NEXT_CYCLE")  # NEXT_CYCLE / IMMEDIATE
# =======================================================


# --------------------------------------------------------
# HELPERS: parse timestamp_utc from CSV to epoch ms
# --------------------------------------------------------
def parse_ts_to_epoch_ms(ts: str) -> int:
    """
    Supports:
    - 2026-01-19T00:12:51.298307+00:00
    - 2026-01-19T04:39:10.982Z
    """
    if not ts:
        return 0
    try:
        if ts.endswith("Z"):
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(ts)
        return int(dt.timestamp() * 1000)
    except Exception:
        return 0


# --------------------------------------------------------
# LOAD MODEL JSON TERBARU UNTUK SETIAP PHASE
# (dibuat lebih robust: pakai paginator agar tidak mentok 1000 objek)
# --------------------------------------------------------
def load_latest_model(phase: int):
    prefix = f"{MODEL_PREFIX}phase_{phase}"

    paginator = s3.get_paginator("list_objects_v2")

    newest_key = None
    newest_last_modified = None

    for page in paginator.paginate(Bucket=MODEL_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            if not key or key.endswith("/"):
                continue
            lm = obj["LastModified"]
            if newest_last_modified is None or lm > newest_last_modified:
                newest_last_modified = lm
                newest_key = key

    if newest_key is None:
        raise Exception(f"Tidak ada model untuk phase {phase} (prefix: {prefix})")

    last_modified = newest_last_modified.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    obj = s3.get_object(Bucket=MODEL_BUCKET, Key=newest_key)
    model = json.loads(obj["Body"].read().decode("utf-8"))

    return model, newest_key, last_modified


# --------------------------------------------------------
# LOAD REALTIME CSV TERBARU
# + parsing timestamp_utc agar bisa pilih row terbaru per phase jika dalam 1 CSV ada banyak timestamp
# --------------------------------------------------------
def load_latest_realtime_file():
    paginator = s3.get_paginator("list_objects_v2")

    newest_key = None
    newest_last_modified = None

    for page in paginator.paginate(Bucket=RAW_REAL_BUCKET, Prefix=RAW_REAL_PREFIX):
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            if not key or key.endswith("/"):
                continue
            lm = obj["LastModified"]
            if newest_last_modified is None or lm > newest_last_modified:
                newest_last_modified = lm
                newest_key = key

    if newest_key is None:
        raise Exception(f"Folder raw_real kosong! (prefix: {RAW_REAL_PREFIX})")

    last_modified = newest_last_modified.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    obj = s3.get_object(Bucket=RAW_REAL_BUCKET, Key=newest_key)
    body = obj["Body"].read().decode("utf-8")

    rows = []
    reader = csv.DictReader(body.splitlines())

    for r in reader:
        try:
            phase = int(float(r["phase"]))  # robust kalau kebaca "1.0"
            rows.append(
                {
                    "ts_ms": parse_ts_to_epoch_ms(r.get("timestamp_utc", "")),
                    "phase": phase,
                    "distance": float(r["distance_m"]),
                    "duration": float(r["duration_s"]),
                    "duration_traffic": float(r["duration_in_traffic_s"]),
                    "speed": float(r["speed_kmh"]),
                }
            )
        except Exception:
            continue

    return rows, newest_key, last_modified


# --------------------------------------------------------
# PREDIKSI SPEED DARI MODEL ML
# --------------------------------------------------------
def predict_speed(model: dict, distance: float, duration: float, duration_traffic: float) -> float:
    a, b, c = model["coef"]
    intercept = model["intercept"]
    return (a * distance) + (b * duration) + (c * duration_traffic) + intercept


# --------------------------------------------------------
# HITUNG GREEN TIME (MODEL HYBRID)
# --------------------------------------------------------
def calculate_green_time(speed_real: float, speed_pred: float) -> float:
    hybrid = (speed_real * HYBRID_W_REAL) + (speed_pred * HYBRID_W_PRED)

    # mapping yang dipakai
    green = 18 * (12 / max(hybrid, 1))

    # clamp
    return max(G_MIN, min(green, G_MAX))


# --------------------------------------------------------
# MAIN HANDLER
# --------------------------------------------------------
def lambda_handler(event, context):
    t0_perf = time.perf_counter()

    request_id = getattr(context, "aws_request_id", None)
    function_name = getattr(context, "function_name", None)

    log_event(
        "LAMBDA_START",
        request_id=request_id,
        function=function_name,
        apply_policy=APPLY_POLICY,
        topic=IOT_TOPIC,
        qos=1,
    )

    # ---------- Load latest realtime file ----------
    rt_t0 = time.perf_counter()
    realtime, rt_key, rt_last_modified = load_latest_realtime_file()
    rt_ms = (time.perf_counter() - rt_t0) * 1000.0

    if not realtime:
        raise Exception("Data realtime kosong setelah parsing CSV!")

    log_event(
        "REALTIME_LOADED",
        s3_bucket=RAW_REAL_BUCKET,
        s3_key=rt_key,
        last_modified_utc=rt_last_modified,
        rows_parsed=len(realtime),
        elapsed_ms=round(rt_ms, 3),
    )

    # ======================================================
    # TIMESTAMP UNTUK END-TO-END
    # - timestamp_utc: untuk trace manusia
    # - timestamp_ms : untuk end-to-end (ESP tinggal subtract)
    # ======================================================
    payload_timestamp_utc = utc_now_iso()
    payload_timestamp_ms = utc_now_epoch_ms()

    # Result object (format tetap, ditambah timestamp_ms)
    result = {
        "timestamp_utc": payload_timestamp_utc,
        "timestamp_ms": payload_timestamp_ms,
    }

    # ---------- Compute per phase ----------
    for phase in [1, 2, 3, 4]:
        phase_rows = [r for r in realtime if r["phase"] == phase]
        if not phase_rows:
            log_event("PHASE_SKIP_NO_DATA", phase=phase)
            continue

        # FIX: kalau dalam 1 CSV ada banyak timestamp, ambil yang PALING BARU untuk tiap phase
        data = max(phase_rows, key=lambda x: x.get("ts_ms", 0))

        speed_real = data["speed"]

        # load latest model per phase
        m_t0 = time.perf_counter()
        model, model_key, model_last_modified = load_latest_model(phase)
        m_ms = (time.perf_counter() - m_t0) * 1000.0

        log_event(
            "MODEL_LOADED",
            phase=phase,
            s3_bucket=MODEL_BUCKET,
            s3_key=model_key,
            last_modified_utc=model_last_modified,
            elapsed_ms=round(m_ms, 3),
        )

        speed_pred = predict_speed(
            model=model,
            distance=data["distance"],
            duration=data["duration"],
            duration_traffic=data["duration_traffic"],
        )

        green = calculate_green_time(speed_real=speed_real, speed_pred=speed_pred)

        result[str(phase)] = {
            "speed_real": speed_real,
            "speed_pred": speed_pred,
            "green_time": green,
        }

        log_event(
            "PHASE_RESULT",
            phase=phase,
            speed_real=round(speed_real, 6),
            speed_pred=round(speed_pred, 6),
            green_time=round(green, 6),
            # extra debug (aman): tunjukkan row timestamp dalam CSV yang kepilih
            row_ts_ms=data.get("ts_ms", 0),
        )

    # ---------- Publish to IoT ----------
    publish_payload = {
        "status": "SUCCESS",
        "apply_policy": APPLY_POLICY,
        "result": result,
        # optional trace (tidak wajib, tapi enak buat debugging)
        "request_id": request_id,
    }

    p_t0 = time.perf_counter()
    log_event(
        "PUBLISH_START",
        topic=IOT_TOPIC,
        qos=1,
        payload_timestamp_utc=payload_timestamp_utc,
        payload_timestamp_ms=payload_timestamp_ms,
    )

    iot.publish(
        topic=IOT_TOPIC,
        qos=1,
        payload=json.dumps(publish_payload, separators=(",", ":")),
    )

    p_ms = (time.perf_counter() - p_t0) * 1000.0
    log_event("PUBLISH_DONE", topic=IOT_TOPIC, qos=1, elapsed_ms=round(p_ms, 3))

    total_ms = (time.perf_counter() - t0_perf) * 1000.0
    log_event("LAMBDA_DONE", request_id=request_id, total_elapsed_ms=round(total_ms, 3))

    return publish_payload
