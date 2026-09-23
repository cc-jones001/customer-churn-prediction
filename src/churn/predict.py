"""Score customers with the trained churn model.

Usage:
    python -m churn.predict data/Telco-Customer-Churn.csv -o predictions.csv
    python -m churn.predict new_customers.csv --top 50

Input: a CSV with the same columns as the training data (customerID and Churn optional).
Output: churn_probability, churn_prediction (0/1 at the tuned threshold) and a risk_band.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .data import ID_COL, PROJECT_ROOT, RAW_FEATURES, clean

MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"


def load_model(path: Path = MODEL_PATH) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"No model at {path}. Run `python -m churn.train` first.")
    return joblib.load(path)


def predict(df: pd.DataFrame, artifact: dict | None = None) -> pd.DataFrame:
    artifact = artifact or load_model()
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    X = clean(df)[RAW_FEATURES]
    proba = artifact["pipeline"].predict_proba(X)[:, 1]
    out = pd.DataFrame(index=df.index)
    if ID_COL in df.columns:
        out[ID_COL] = df[ID_COL]
    out["churn_probability"] = proba.round(4)
    out["churn_prediction"] = (proba >= artifact["threshold"]).astype(int)
    out["risk_band"] = pd.cut(
        proba, bins=[-0.01, 0.2, 0.5, 0.7, 1.0], labels=["Low", "Medium", "High", "Very high"]
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="CSV of customers to score")
    ap.add_argument("-o", "--output", type=Path, help="where to write predictions (CSV)")
    ap.add_argument("--top", type=int, default=10, help="print the N highest-risk customers")
    args = ap.parse_args()

    artifact = load_model()
    preds = predict(pd.read_csv(args.input), artifact)
    print(f"Model: {artifact['model_name']}  |  decision threshold: {artifact['threshold']:.3f}")
    print(f"Scored {len(preds):,} customers; {preds['churn_prediction'].sum():,} flagged as likely to churn.\n")
    print(preds.sort_values("churn_probability", ascending=False).head(args.top).to_string(index=False))
    if args.output:
        preds.to_csv(args.output, index=False)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
