"""

5-minute historical snapshot collector for FR24 "historic flight positions (light)" endpoint.
- Uses FR24_API_TOKEN from environment variables.
- Writes append-only JSONL snapshots, one file per UTC day.
- Tightened altitude range (100–1000 ft) to reduce ground clutter + credit burn.
- Resume-safe: continues from the last written timestamp in the output folder.
- Handles 429 (rate limit) and transient network errors with exponential backoff.
- Fails fast on 402 (payment/credit limit) so you don't waste time retrying.


Optional overrides:
  $env:FR24_START_UTC="2025-01-01T00:00:00Z"
  $env:FR24_END_UTC="2025-07-01T00:00:00Z"
"""

import os
import sys
import time
import json
import re
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fr24sdk.client import Client
from fr24sdk.exceptions import RateLimitError, TransportError, PaymentRequiredError, AuthenticationError, ApiError

print("Python executable:", sys.executable)

API_TOKEN = os.getenv("FR24_API_TOKEN")
if not API_TOKEN:
    raise RuntimeError("Missing FR24_API_TOKEN environment variable.")

# ----------------------------
# Collection parameters
# ----------------------------
# north,south,west,east (PANC bounding box)
BOUNDS = "61.20,61.15,-150.05,-149.95"

# Collect only airborne-ish low altitude band to reduce ground ops
ALTITUDE_RANGES = ["50-1000"]  # list is required by the SDK

# 5-minute cadence
STEP = timedelta(minutes=5)

# Default 6-month window (override via env vars if desired)
DEFAULT_START = datetime(2025, 1, 1, tzinfo=timezone.utc)
DEFAULT_END   = datetime(2026, 1, 1, tzinfo=timezone.utc)

def parse_env_dt(var_name: str, default: datetime) -> datetime:
    s = os.getenv(var_name)
    if not s:
        return default
    # Accept "Z" suffix
    s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s).astimezone(timezone.utc)

START = parse_env_dt("FR24_START_UTC", DEFAULT_START)
END   = parse_env_dt("FR24_END_UTC", DEFAULT_END)

OUT_DIR = Path("data/fr24_raw_5min")
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("OUT_DIR resolves to:", OUT_DIR.resolve())

# ----------------------------
# Throttling / backoff controls
# ----------------------------
REQUEST_DELAY_S = 5.0     # sleep after every successful request
MAX_RETRIES = 8            # per timestamp
BACKOFF_BASE_S = 5.0       # 5,10,20,40,... seconds (plus jitter)
JITTER_S = 1.5             # random jitter to avoid synchronized retries

def ts_range(start: datetime, end: datetime, step: timedelta):
    t = start
    while t < end:
        yield t
        t += step

def resp_to_dict(resp):
    # robust conversion for SDK pydantic models
    if hasattr(resp, "model_dump_json"):
        return json.loads(resp.model_dump_json())
    if hasattr(resp, "model_dump"):
        return resp.model_dump(mode="json")
    return resp if isinstance(resp, dict) else {}

def _status_from_error(e: Exception):
    # fr24sdk ApiError often includes "500 Internal Server Error" in str(e)
    m = re.search(r"\b(\d{3})\b", str(e))
    return int(m.group(1)) if m else None

def fetch_snapshot(client: Client, ts: datetime) -> dict:
    attempt = 0
    while True:
        try:
            resp = client.historic.flight_positions.get_light(
                bounds=BOUNDS,
                altitude_ranges=ALTITUDE_RANGES,
                timestamp=ts
            )
            return resp_to_dict(resp)

        except PaymentRequiredError as e:
            # Stop immediately—retrying won't help.
            print(f"[402] Payment/Credit limit reached at {ts.isoformat()}. Stopping. Details: {e}")
            raise

        except RateLimitError:
            attempt += 1
            if attempt > MAX_RETRIES:
                raise
            sleep_s = BACKOFF_BASE_S * (2 ** (attempt - 1)) + random.random() * JITTER_S
            print(f"[429] {ts.isoformat()} attempt {attempt}/{MAX_RETRIES}. Sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)

        except TransportError as e:
            attempt += 1
            if attempt > MAX_RETRIES:
                print(f"[NET] {ts.isoformat()} failed after {MAX_RETRIES} retries: {e}. Skipping timestamp.")
                return {"data": []}
            sleep_s = BACKOFF_BASE_S * (2 ** (attempt - 1)) + random.random() * JITTER_S
            print(f"[NET] {ts.isoformat()} transport error: {e}. Sleep {sleep_s:.1f}s (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(sleep_s)

        except AuthenticationError as e:
            print(f"[401] Authentication failed. Check FR24_API_TOKEN. Details: {e}")
            raise

        except ApiError as e:
            # Retry on server errors (5xx). Skip timestamp after retries.
            status = _status_from_error(e)
            if status and 500 <= status <= 599:
                attempt += 1
                if attempt > MAX_RETRIES:
                    print(f"[{status}] {ts.isoformat()} server error persists after {MAX_RETRIES} retries. Skipping timestamp.")
                    return {"data": []}
                sleep_s = BACKOFF_BASE_S * (2 ** (attempt - 1)) + random.random() * 2.0
                print(f"[{status}] {ts.isoformat()} server error. Sleep {sleep_s:.1f}s then retry (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(sleep_s)
            else:
                # Non-5xx ApiError: treat as fatal so you notice and fix parameters/auth
                print(f"[API] Non-5xx ApiError at {ts.isoformat()}: {e}")
                raise

def get_last_written_timestamp(out_dir: Path):
    files = sorted(out_dir.glob("*.jsonl"))
    if not files:
        return None

    last_file = files[-1]
    with last_file.open("rb") as f:
        f.seek(0, 2)
        end = f.tell()
        chunk = 8192
        f.seek(max(0, end - chunk))
        lines = f.read().splitlines()

    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            rec = json.loads(line.decode("utf-8"))
            return datetime.fromisoformat(rec["timestamp"]).astimezone(timezone.utc)
        except Exception:
            continue

    return None

# ----------------------------
# Resume logic
# ----------------------------
last_ts = get_last_written_timestamp(OUT_DIR)
if last_ts:
    START = max(START, last_ts + STEP)
    print("Resuming from:", START.isoformat())
else:
    print("Starting fresh from:", START.isoformat())

print("Collection window:", START.isoformat(), "->", END.isoformat())
print("Bounds:", BOUNDS, "| Altitude ranges:", ALTITUDE_RANGES, "| Step:", STEP)

with Client(api_token=API_TOKEN) as client:
    for ts in ts_range(START, END, STEP):
        resp_dict = fetch_snapshot(client, ts)
        data_list = resp_dict.get("data", [])

        out_file = OUT_DIR / f"{ts.strftime('%Y-%m-%d')}.jsonl"
        snapshot_record = {
            "timestamp": ts.isoformat(),
            "bounds": BOUNDS,
            "altitude_ranges": ALTITUDE_RANGES,
            "data": data_list,
        }

        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot_record) + "\n")

        # progress print every hour
        if ts.minute == 0:
            print(ts.isoformat(), "records:", len(data_list))

        time.sleep(REQUEST_DELAY_S)
