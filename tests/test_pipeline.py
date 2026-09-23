import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn import data as D
from churn.features import ChurnFeatures, build_preprocessor
from churn.predict import MODEL_PATH, load_model, predict


@pytest.fixture(scope="module")
def df():
    return D.load()


def test_dataset_integrity(df):
    assert len(df) == 7043
    assert df[D.ID_COL].is_unique
    assert set(df[D.TARGET].unique()) == {0, 1}
    assert abs(df[D.TARGET].mean() - 0.2654) < 1e-3
    assert df[D.RAW_FEATURES].isna().sum().sum() == 0


def test_total_charges_blank_handled():
    raw = pd.read_csv(D.DATA_PATH)
    blanks = raw["TotalCharges"].str.strip() == ""
    assert blanks.sum() == 11
    cleaned = D.clean(raw)
    assert (cleaned.loc[blanks, "TotalCharges"] == 0).all()
    assert (cleaned.loc[blanks, "tenure"] == 0).all()


def test_engineered_features(df):
    X = ChurnFeatures().transform(df[D.RAW_FEATURES].head(200))
    assert X["num_addon_services"].between(0, 6).all()
    assert np.isfinite(X["avg_monthly_spend"]).all()
    assert set(X["auto_payment"].unique()) <= {0, 1}


def test_preprocessor_shapes_and_unseen_category(df):
    X, y = D.split_xy(df)
    pipe = Pipeline([("prep", build_preprocessor()), ("model", LogisticRegression(max_iter=1000))])
    pipe.fit(X, y)
    row = X.head(1).copy()
    row["PaymentMethod"] = "Crypto wallet"  # never seen in training
    p = pipe.predict_proba(row)[:, 1]
    assert 0 <= p[0] <= 1


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="train the model first")
def test_saved_model_predicts_raw_rows(df):
    artifact = load_model()
    raw = pd.read_csv(D.DATA_PATH).head(50)  # raw strings, including target column
    out = predict(raw, artifact)
    assert len(out) == 50
    assert out["churn_probability"].between(0, 1).all()
    assert set(out["churn_prediction"].unique()) <= {0, 1}
    # A risky profile should score higher than a loyal one
    base = raw.iloc[[0]].copy()
    risky = base.assign(Contract="Month-to-month", tenure=1, InternetService="Fiber optic",
                        PaymentMethod="Electronic check", OnlineSecurity="No", TechSupport="No",
                        MonthlyCharges=95.0, TotalCharges="95.0")
    loyal = base.assign(Contract="Two year", tenure=70, InternetService="DSL",
                        PaymentMethod="Credit card (automatic)", OnlineSecurity="Yes",
                        TechSupport="Yes", MonthlyCharges=60.0, TotalCharges="4200.0")
    p_risky = predict(risky, artifact)["churn_probability"].iloc[0]
    p_loyal = predict(loyal, artifact)["churn_probability"].iloc[0]
    assert p_risky > 0.5 > p_loyal


def test_predict_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        predict(pd.DataFrame({"tenure": [1]}), artifact={"pipeline": None, "threshold": 0.5})
