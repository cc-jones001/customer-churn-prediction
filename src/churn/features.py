"""Feature engineering and preprocessing.

Everything here lives inside the saved sklearn Pipeline, so the model accepts
raw customer rows (same columns as the CSV) at prediction time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import BINARY_COLS, CATEGORICAL_COLS, NUMERIC_COLS

ADDON_SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
ENGINEERED_NUMERIC = [
    "num_addon_services", "avg_monthly_spend", "charge_delta",
    "is_new_customer",
]
ENGINEERED_BINARY = ["auto_payment", "no_protection_bundle"]


class ChurnFeatures(BaseEstimator, TransformerMixin):
    """Adds domain features derived from the raw columns."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
        X["num_addon_services"] = (X[ADDON_SERVICES] == "Yes").sum(axis=1)
        tenure = X["tenure"].astype(float)
        # Average historical monthly spend; for new customers fall back to current charge
        X["avg_monthly_spend"] = np.where(
            tenure > 0, X["TotalCharges"] / tenure.replace(0, np.nan), X["MonthlyCharges"]
        )
        # Positive = paying more now than their historical average (recent price increase / upgrade)
        X["charge_delta"] = X["MonthlyCharges"] - X["avg_monthly_spend"]
        X["is_new_customer"] = (tenure <= 6).astype(int)
        X["auto_payment"] = X["PaymentMethod"].str.contains("automatic").astype(int)
        X["no_protection_bundle"] = (
            (X["InternetService"] != "No")
            & (X["OnlineSecurity"] == "No")
            & (X["TechSupport"] == "No")
        ).astype(int)
        return X


def build_preprocessor(scale: bool = True) -> Pipeline:
    numeric = NUMERIC_COLS + ENGINEERED_NUMERIC
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    columns = ColumnTransformer(
        [
            ("num", Pipeline(num_steps), numeric),
            ("bin", "passthrough", BINARY_COLS + ENGINEERED_BINARY),
            (
                "cat",
                OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("features", ChurnFeatures()), ("columns", columns)])
