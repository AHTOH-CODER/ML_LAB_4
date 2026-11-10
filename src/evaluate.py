import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from data import load_xy, split

ROOT = Path(__file__).resolve().parent.parent
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
model = mlflow.sklearn.load_model(str(ROOT / "exported_model"))
_, x, y = load_xy()
_, _, x_test, _, _, y_test = split(x, y)
pred = model.predict(x_test)
metrics = {
    "n_test": int(len(y_test)),
    "test_f1_macro": float(f1_score(y_test, pred, average="macro")),
    "test_precision_macro": float(precision_score(y_test, pred, average="macro", zero_division=0)),
    "test_recall_macro": float(recall_score(y_test, pred, average="macro", zero_division=0)),
    "test_accuracy": float(accuracy_score(y_test, pred)),
}
(ROOT / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2))
