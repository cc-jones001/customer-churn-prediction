"""Loading and cleaning the IBM Telco Customer Churn dataset."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"

# Official source: IBM's public repository (same file Kaggle hosts as blastchar/telco-customer-churn)
SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"

TARGET = "Churn"
ID_COL = "customerID"

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_COLS = ["SeniorCitizen"]
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
RAW_FEATURES = NUMERIC_COLS + BINARY_COLS + CATEGORICAL_COLS


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(path: Path = DATA_PATH) -> Path:
    """Download the dataset from IBM's repository if it is not already present."""
    if not path.exists():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SOURCE_URL, path)
    return path


def verify(path: Path = DATA_PATH) -> None:
    digest = sha256(path)
    if digest != EXPECTED_SHA256:
        raise ValueError(
            f"Checksum mismatch for {path}: got {digest}, expected {EXPECTED_SHA256}"
        )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix types. Works on raw rows with or without the target column."""
    df = df.copy()
    # TotalCharges is stored as text; 11 brand-new customers (tenure 0) have a blank value.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0
    if TARGET in df.columns and not pd.api.types.is_numeric_dtype(df[TARGET]):
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)
    return df


def load(path: Path = DATA_PATH, check: bool = True) -> pd.DataFrame:
    download(path)
    if check:
        verify(path)
    return clean(pd.read_csv(path))


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[RAW_FEATURES], df[TARGET]
