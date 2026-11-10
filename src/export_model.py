import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from data import CLASSES, FEATURES, SEED, file_sha256, load_xy, split

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "digits-classifier"
EXPORT_DIR = ROOT / "exported_model"

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
client = MlflowClient()
version = client.get_model_version_by_alias(MODEL_NAME, "champion")
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")

staging = ROOT / "_export_staging"
if staging.exists():
    shutil.rmtree(staging)
staging.mkdir(parents=True)
local_path = Path(mlflow.artifacts.download_artifacts(
    artifact_uri=version.source, dst_path=str(staging)
))
if EXPORT_DIR.exists():
    shutil.rmtree(EXPORT_DIR)
shutil.copytree(local_path, EXPORT_DIR)
shutil.rmtree(staging)
print("exported", EXPORT_DIR)

_, x, y = load_xy()
_, _, x_test, _, _, y_test = split(x, y)
sample = x_test.head(5)
predictions = model.predict(sample)
probabilities = model.predict_proba(sample)

request = {
    "instances": [
        {"pixels": [int(value) for value in row]}
        for row in sample[FEATURES].to_numpy().tolist()
    ]
}
expected = {
    "predictions": [int(value) for value in predictions],
    "probabilities": [
        [round(float(item), 6) for item in row] for row in probabilities
    ],
}
(ROOT / "sample_request.json").write_text(
    json.dumps(request, indent=2), encoding="utf-8"
)
(ROOT / "expected_predictions.json").write_text(
    json.dumps(expected, indent=2), encoding="utf-8"
)

run = client.get_run(version.run_id)
meta = {
    "model_name": MODEL_NAME,
    "model_version": version.version,
    "run_id": version.run_id,
    "run_name": run.info.run_name,
    "alias": "champion",
    "features": FEATURES,
    "classes": CLASSES,
    "pixel_order": "row-major 8x8",
    "pixel_range": [0, 16],
    "seed": SEED,
    "dataset_sha256": file_sha256(),
    "params": run.data.params,
}
(EXPORT_DIR / "metadata.json").write_text(
    json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps({"version": version.version, "run_id": version.run_id, "run_name": run.info.run_name}, indent=2))
