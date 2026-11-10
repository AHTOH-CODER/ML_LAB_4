import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from data import CLASSES, FEATURES

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path("/app/exported_model")
if not MODEL_DIR.exists():
    MODEL_DIR = ROOT / "exported_model"

model = mlflow.sklearn.load_model(str(MODEL_DIR))
metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))

app = FastAPI(title="Сервис распознавания рукописных цифр", version=str(metadata["model_version"]))


class Instance(BaseModel):
    pixels: list[int] = Field(min_length=64, max_length=64)

    @field_validator("pixels")
    @classmethod
    def check_range(cls, value):
        for index, item in enumerate(value):
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError(f"pixels[{index}] должен быть целым числом")
            if item < 0 or item > 16:
                raise ValueError(f"pixels[{index}]={item} вне диапазона 0–16")
        return value


class PredictRequest(BaseModel):
    instances: list[Instance] = Field(min_length=1, max_length=100)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": metadata["model_version"]}


@app.get("/metadata")
def get_metadata():
    return metadata


@app.post("/predict")
def predict(body: PredictRequest):
    started = time.perf_counter()
    frame = pd.DataFrame(
        [item.pixels for item in body.instances],
        columns=FEATURES,
    )
    labels = model.predict(frame)
    probabilities = model.predict_proba(frame)
    elapsed = (time.perf_counter() - started) * 1000
    return {
        "predictions": [int(value) for value in labels],
        "class_names": [CLASSES[int(value)] for value in labels],
        "probabilities": [
            [round(float(item), 6) for item in row] for row in probabilities
        ],
        "model_version": str(metadata["model_version"]),
        "latency_ms": round(elapsed, 3),
    }
