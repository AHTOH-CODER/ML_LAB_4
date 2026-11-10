import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
expected = json.loads((ROOT / "expected_predictions.json").read_text(encoding="utf-8"))
sample = json.loads((ROOT / "sample_request.json").read_text(encoding="utf-8"))

health = requests.get(f"{BASE}/health", timeout=10)
health.raise_for_status()
print("HEALTH", health.json())

meta = requests.get(f"{BASE}/metadata", timeout=10)
meta.raise_for_status()
print("METADATA", json.dumps(meta.json(), ensure_ascii=False)[:400])

response = requests.post(f"{BASE}/predict", json=sample, timeout=30)
response.raise_for_status()
payload = response.json()
got = payload["predictions"]
want = expected["predictions"]
print("PREDICTIONS", got)
print("EXPECTED", want)
print("VERSION", payload["model_version"])
print("LATENCY_MS", payload["latency_ms"])
if got != want:
    raise SystemExit("PREDICTIONS_MISMATCH")
print("CHECK_API_PASS")
