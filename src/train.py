import argparse
import inspect
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import (
    CLASSES,
    FEATURES,
    RAW_PATH,
    SEED,
    SOURCE_URL,
    file_sha256,
    load_xy,
    split,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "digits-classifier"


def tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def log_sklearn_model(model, x_example):
    kwargs = {
        "input_example": x_example,
        "signature": infer_signature(x_example, model.predict(x_example)),
        "pip_requirements": (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines(),
    }
    params = inspect.signature(mlflow.sklearn.log_model).parameters
    if "name" in params:
        return mlflow.sklearn.log_model(model, name="model", **kwargs)
    return mlflow.sklearn.log_model(model, "model", **kwargs)


p = argparse.ArgumentParser()
p.add_argument("--algorithm", choices=["lr", "rf"], default="lr")
p.add_argument("--c", type=float, default=1.0)
p.add_argument("--trees", type=int, default=200)
p.add_argument("--depth", type=int, default=12)
p.add_argument("--min-leaf", type=int, default=1)
p.add_argument("--register", action="store_true")
a = p.parse_args()

mlflow.set_tracking_uri(tracking_uri())
mlflow.set_experiment("lab4-digits")

df, x, y = load_xy()
x_train, x_val, x_test, y_train, y_val, y_test = split(x, y)

if a.algorithm == "lr":
    clf = LogisticRegression(C=a.c, max_iter=3000, random_state=SEED)
    run_name = f"lr-C{a.c:g}"
    model_params = {"C": a.c}
else:
    clf = RandomForestClassifier(
        n_estimators=a.trees,
        max_depth=a.depth if a.depth > 0 else None,
        min_samples_leaf=a.min_leaf,
        random_state=SEED,
        n_jobs=-1,
    )
    run_name = f"rf-{a.trees}x{a.depth}-leaf{a.min_leaf}"
    model_params = {"trees": a.trees, "depth": a.depth, "min_leaf": a.min_leaf}

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", clf),
])

with mlflow.start_run(run_name=run_name) as run:
    mlflow.log_params({
        "algorithm": a.algorithm,
        "seed": SEED,
        "dataset": "sklearn-load-digits",
        "pixel_order": "row-major-8x8",
        "pixel_range": "0-16",
        "n_train": len(x_train),
        "n_validation": len(x_val),
        "n_test": len(x_test),
        **model_params,
    })
    mlflow.set_tags({
        "dataset_sha256": file_sha256(),
        "purpose": "lab4",
        "variant": "4",
        "selection_metric": "val_f1_macro",
        "source_url": SOURCE_URL,
        "raw_path": str(RAW_PATH),
    })

    model.fit(x_train, y_train)
    pred = model.predict(x_val)
    metrics = {
        "val_f1_macro": float(f1_score(y_val, pred, average="macro")),
        "val_precision_macro": float(precision_score(y_val, pred, average="macro", zero_division=0)),
        "val_recall_macro": float(recall_score(y_val, pred, average="macro", zero_division=0)),
        "val_accuracy": float(accuracy_score(y_val, pred)),
        "train_f1_macro": float(f1_score(y_train, model.predict(x_train), average="macro")),
    }
    mlflow.log_metrics(metrics)

    mlflow.log_dict({
        "train_indices": x_train.index.tolist(),
        "validation_indices": x_val.index.tolist(),
        "test_indices": x_test.index.tolist(),
        "features": FEATURES,
        "classes": CLASSES,
        "pixel_order": "row-major 8x8, pixel_00 is the top-left cell",
        "seed": SEED,
    }, "split.json")
    mlflow.log_artifact(__file__, artifact_path="source")
    mlflow.log_artifact(str(Path(__file__).parent / "data.py"), artifact_path="source")
    mlflow.log_artifact(str(ROOT / "requirements.txt"), artifact_path="source")

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_val, pred, display_labels=CLASSES, ax=ax)
    ax.set_title(run_name)
    mlflow.log_figure(fig, "validation_confusion_matrix.png")
    plt.close(fig)

    info = log_sklearn_model(model, x_train.head(2))
    payload = {
        "run_id": run.info.run_id,
        "run_name": run_name,
        "model_uri": info.model_uri,
        **metrics,
    }
    print(json.dumps(payload, indent=2))
    if a.register:
        mv = mlflow.register_model(info.model_uri, MODEL_NAME)
        print("REGISTERED_VERSION=", mv.version)
