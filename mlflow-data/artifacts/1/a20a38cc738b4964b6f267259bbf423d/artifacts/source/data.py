"""Загрузка digits (sklearn) и фиксированное разбиение 60/20/20.

Общий модуль для train.py, export_model.py и evaluate.py: все скрипты
используют один и тот же файл, один и тот же порядок пикселей и одно
и то же разбиение (random_state=42, stratify=y).
"""
import hashlib
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "digits.csv"
SOURCE_URL = "https://scikit-learn.org/stable/datasets/toy_dataset.html"
SEED = 42

TARGET = "digit"
# Порядок развёртки 8x8: слева направо, сверху вниз, как в sklearn.load_digits.
PIXELS = [f"pixel_{i:02d}" for i in range(64)]
FEATURES = PIXELS
CLASSES = [str(i) for i in range(10)]


def file_sha256(path: Path = RAW_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_raw() -> Path:
    """Сохраняет load_digits в CSV один раз. Повторный вызов файл не переписывает."""
    if RAW_PATH.exists() and RAW_PATH.stat().st_size > 0:
        return RAW_PATH
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    bunch = load_digits()
    frame = pd.DataFrame(bunch.data, columns=PIXELS)
    frame[TARGET] = bunch.target.astype(int)
    frame.to_csv(RAW_PATH, index=False)
    return RAW_PATH


def load_frame() -> pd.DataFrame:
    ensure_raw()
    return pd.read_csv(RAW_PATH)


def load_xy():
    df = load_frame()
    x = df[FEATURES].copy()
    y = df[TARGET].astype(int)
    return df, x, y


def split(x, y):
    """60/20/20: сначала 20 % в test, затем 25 % остатка в validation."""
    x_dev, x_test, y_dev, y_test = train_test_split(
        x, y, test_size=0.2, random_state=SEED, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_dev, y_dev, test_size=0.25, random_state=SEED, stratify=y_dev
    )
    return x_train, x_val, x_test, y_train, y_val, y_test
