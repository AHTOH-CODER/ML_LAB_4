import argparse
import os

import mlflow
from mlflow import MlflowClient

MODEL_NAME = "digits-classifier"

p = argparse.ArgumentParser()
p.add_argument("--version", required=True)
p.add_argument("--alias", default="champion")
a = p.parse_args()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
client = MlflowClient()
version = client.get_model_version(MODEL_NAME, a.version)
client.set_registered_model_alias(MODEL_NAME, a.alias, a.version)
print(f"{a.alias} -> version {version.version}, run {version.run_id}")
