"""Train, compare, tune and evaluate churn models.

Usage:
    python -m churn.train            # full run (~5-10 min on a laptop)
    python -m churn.train --quick    # fewer tuning iterations, for smoke tests
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from scipy.stats import loguniform, randint, uniform
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import data as D
from .features import build_preprocessor
from .plots import AQUA, BLUE, INK_2, MUTED, ORANGE, apply_style, plt

SEED = 42
ROOT = D.PROJECT_ROOT
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"


# --------------------------------------------------------------------------- models
def candidate_models() -> dict[str, Pipeline]:
    """No class re-weighting: keeps predicted probabilities calibrated. The 27/73 class
    imbalance is handled instead by tuning the decision threshold (step 3)."""
    def pipe(est, scale=True):
        return Pipeline([("prep", build_preprocessor(scale=scale)), ("model", est)])

    return {
        "Baseline (majority class)": pipe(DummyClassifier(strategy="prior")),
        "Logistic Regression": pipe(
            LogisticRegression(max_iter=5000, random_state=SEED)
        ),
        "Random Forest": pipe(
            RandomForestClassifier(
                n_estimators=400, min_samples_leaf=5,
                n_jobs=-1, random_state=SEED,
            ),
            scale=False,
        ),
        "Hist Gradient Boosting": pipe(
            HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=300, random_state=SEED
            ),
            scale=False,
        ),
        "XGBoost": pipe(
            XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                colsample_bytree=0.8, eval_metric="logloss",
                n_jobs=-1, random_state=SEED,
            ),
            scale=False,
        ),
    }


SEARCH_SPACES = {
    "Logistic Regression": {
        "model__C": loguniform(1e-3, 1e2),
        "model__l1_ratio": uniform(0, 1),  # elastic net: 0 = ridge (L2), 1 = lasso (L1)
        "model__solver": ["saga"],
    },
    "Random Forest": {
        "model__n_estimators": randint(200, 800),
        "model__max_depth": [None, 6, 10, 16],
        "model__min_samples_leaf": randint(1, 30),
        "model__max_features": ["sqrt", 0.3, 0.5],
    },
    "Hist Gradient Boosting": {
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__max_iter": randint(100, 600),
        "model__max_leaf_nodes": randint(8, 48),
        "model__min_samples_leaf": randint(10, 100),
        "model__l2_regularization": loguniform(1e-3, 10),
    },
    "XGBoost": {
        "model__n_estimators": randint(150, 700),
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__max_depth": randint(2, 7),
        "model__min_child_weight": randint(1, 20),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.5, 0.5),
        "model__reg_lambda": loguniform(1e-2, 20),
        "model__gamma": uniform(0, 2),
    },
}


# --------------------------------------------------------------------------- metrics
def lift_at(y_true, proba, frac: float) -> tuple[float, float]:
    """Churn rate and share of all churners captured in the top `frac` riskiest customers."""
    y_true = np.asarray(y_true)
    n = max(1, int(round(len(proba) * frac)))
    top = np.argsort(-proba)[:n]
    rate = y_true[top].mean()
    return rate / y_true.mean(), y_true[top].sum() / y_true.sum()


def evaluate(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    lift10, cap10 = lift_at(y_true, proba, 0.10)
    lift20, cap20 = lift_at(y_true, proba, 0.20)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "brier": brier_score_loss(y_true, proba),
        "lift_top10pct": lift10,
        "capture_top10pct": cap10,
        "lift_top20pct": lift20,
        "capture_top20pct": cap20,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": threshold,
    }


def best_f1_threshold(y_true, proba) -> float:
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[np.nanargmax(f1[:-1])])


# --------------------------------------------------------------------------- figures
def fig_churn_drivers(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    overall = df[D.TARGET].mean()
    for ax, col in zip(axes, ["Contract", "InternetService", "PaymentMethod"]):
        rates = df.groupby(col)[D.TARGET].mean().sort_values()
        bars = ax.barh(rates.index, rates.values, color=BLUE, height=0.6)
        ax.axvline(overall, color=MUTED, lw=1, ls="--")
        for b, v in zip(bars, rates.values):
            ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.0%}", va="center",
                    color=INK_2, fontsize=9)
        ax.set_xlim(0, max(0.6, rates.max() + 0.1))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.set_title(f"Churn rate by {col}")
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"Dashed line = overall churn rate ({overall:.0%})", x=0.01, y=0.02,
                 ha="left", va="bottom", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG_DIR / "churn_rate_by_segment.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    bins = np.arange(0, 78, 6)
    for label, color, name in [(0, BLUE, "Stayed"), (1, ORANGE, "Churned")]:
        ax.hist(df.loc[df[D.TARGET] == label, "tenure"], bins=bins, color=color,
                alpha=0.75, label=name, edgecolor="white", linewidth=1.5, density=True)
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Density")
    ax.set_title("Churners are concentrated in the first year")
    ax.legend()
    fig.savefig(FIG_DIR / "tenure_distribution.png")
    plt.close(fig)


def fig_model_comparison(cv_table: pd.DataFrame) -> None:
    t = cv_table.sort_values("roc_auc_mean")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(t.index, t["roc_auc_mean"], xerr=t["roc_auc_std"], color=BLUE, height=0.55,
            error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 3})
    for i, v in enumerate(t["roc_auc_mean"]):
        ax.text(v + 0.012, i, f"{v:.3f}", va="center", color=INK_2, fontsize=9)
    ax.set_xlim(0.45, 0.92)
    ax.set_xlabel("ROC-AUC (5-fold CV on training set, ±1 sd)")
    ax.set_title("Model comparison (default hyperparameters)")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG_DIR / "model_comparison.png")
    plt.close(fig)


def fig_evaluation(y, proba, threshold, name) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    # ROC
    fpr, tpr, _ = roc_curve(y, proba)
    ax = axes[0]
    ax.plot(fpr, tpr, color=BLUE, label=f"{name} (AUC {roc_auc_score(y, proba):.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="Random")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test set)")
    ax.legend(loc="lower right")
    # PR
    prec, rec, _ = precision_recall_curve(y, proba)
    ax = axes[1]
    ax.plot(rec, prec, color=BLUE, label=f"AP {average_precision_score(y, proba):.3f}")
    ax.axhline(np.mean(y), color=MUTED, lw=1, ls="--", label=f"Base rate {np.mean(y):.2f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision–recall curve (test set)",
           ylim=(0, 1.02))
    ax.legend(loc="upper right")
    # Confusion matrix
    ax = axes[2]
    cm = confusion_matrix(y, (proba >= threshold).astype(int))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    labels = [["True negative", "False positive"], ["False negative", "True positive"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n{labels[i][j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "#0b0b0b", fontsize=10)
    ax.set_xticks([0, 1], ["Predicted stay", "Predicted churn"])
    ax.set_yticks([0, 1], ["Actual stay", "Actual churn"])
    ax.set_title(f"Confusion matrix (threshold {threshold:.2f})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "evaluation.png")
    plt.close(fig)

    # Cumulative gains + calibration
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    order = np.argsort(-proba)
    gains = np.cumsum(np.asarray(y)[order]) / np.sum(y)
    frac = np.arange(1, len(y) + 1) / len(y)
    ax = axes[0]
    ax.plot(frac, gains, color=BLUE, label="Model")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="Random targeting")
    k = int(0.2 * len(y))
    ax.scatter([frac[k]], [gains[k]], color=ORANGE, s=40, zorder=3,
               edgecolor="white", linewidth=1.5)
    ax.annotate(f"Top 20% of customers\ncontains {gains[k]:.0%} of churners",
                (frac[k], gains[k]), xytext=(0.35, 0.45), color=INK_2, fontsize=9,
                arrowprops={"arrowstyle": "-", "color": MUTED})
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set(xlabel="Share of customers contacted (highest risk first)",
           ylabel="Share of churners reached", title="Cumulative gains (test set)")
    ax.legend(loc="lower right")
    ax = axes[1]
    pt, pp = calibration_curve(y, proba, n_bins=10, strategy="quantile")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="Perfect calibration")
    ax.plot(pp, pt, color=BLUE, marker="o", markersize=6, markeredgecolor="white", label=name)
    ax.set(xlabel="Predicted churn probability", ylabel="Observed churn rate",
           title="Calibration (test set)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gains_and_calibration.png")
    plt.close(fig)


def fig_importance(imp: pd.DataFrame) -> None:
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.index, top["mean"], xerr=top["std"], color=AQUA, height=0.6,
            error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2})
    ax.set_xlabel("Drop in ROC-AUC when feature is shuffled")
    ax.set_title("Permutation importance (test set, top 15)")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG_DIR / "feature_importance.png")
    plt.close(fig)


def fig_shap(pipe: Pipeline, X_test: pd.DataFrame) -> bool:
    try:
        import shap
    except ImportError:
        return False
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    Xt = pd.DataFrame(prep.transform(X_test), columns=prep.named_steps["columns"].get_feature_names_out())
    try:
        if isinstance(model, LogisticRegression):
            explainer = shap.LinearExplainer(model, Xt)
        else:
            explainer = shap.TreeExplainer(model)
        values = explainer(Xt)
        if values.values.ndim == 3:  # (n, features, classes) -> churn class
            values = values[:, :, 1]
        plt.figure()
        shap.plots.beeswarm(values, max_display=15, show=False)
        plt.title("SHAP: how each feature pushes churn risk up or down", loc="left")
        plt.savefig(FIG_DIR / "shap_summary.png")
        plt.close("all")
        return True
    except Exception as exc:  # SHAP is a nice-to-have; never fail training on it
        print(f"  (skipping SHAP plot: {exc})")
        return False


# --------------------------------------------------------------------------- main
def main(quick: bool = False) -> dict:
    t0 = time.time()
    apply_style()
    for d in (MODEL_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    df = D.load()
    X, y = D.split_xy(df)
    print(f"Loaded {len(df):,} customers, churn rate {y.mean():.1%}")
    fig_churn_drivers(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1", "recall": "recall"}

    # 1. Compare model families with default settings
    print("\n[1/4] Cross-validating candidate models")
    rows = {}
    for name, pipe in candidate_models().items():
        res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        rows[name] = {f"{m}_{s}": getattr(res[f"test_{m}"], s)() for m in scoring for s in ("mean", "std")}
        print(f"  {name:28s} ROC-AUC {rows[name]['roc_auc_mean']:.4f} ± {rows[name]['roc_auc_std']:.4f}")
    cv_table = pd.DataFrame(rows).T.sort_values("roc_auc_mean", ascending=False)
    cv_table.round(4).to_csv(REPORT_DIR / "cv_model_comparison.csv")
    fig_model_comparison(cv_table)

    # 2. Tune every non-baseline family, keep the best by CV ROC-AUC
    print("\n[2/4] Hyperparameter search (RandomizedSearchCV, 5-fold, ROC-AUC)")
    n_iter = 5 if quick else 40
    tuned = {}
    base = candidate_models()
    for name, space in SEARCH_SPACES.items():
        search = RandomizedSearchCV(
            base[name], space, n_iter=n_iter, scoring="roc_auc", cv=cv,
            random_state=SEED, n_jobs=-1, refit=False,
        )
        search.fit(X_train, y_train)
        tuned[name] = (search.best_score_, search.best_params_)
        print(f"  {name:28s} best CV ROC-AUC {search.best_score_:.4f}")
    best_name = max(tuned, key=lambda k: tuned[k][0])
    best_cv_auc, best_params = tuned[best_name]
    final = clone(base[best_name]).set_params(**best_params)
    print(f"  -> selected: {best_name}")

    # 3. Pick decision threshold on out-of-fold training predictions (never on the test set)
    print("\n[3/4] Choosing decision threshold from out-of-fold predictions")
    oof = cross_val_predict(final, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    threshold = best_f1_threshold(y_train, oof)
    print(f"  F1-optimal threshold: {threshold:.3f}")

    # 4. Fit on the full training set, evaluate once on the untouched test set
    print("\n[4/4] Final fit and held-out evaluation")
    final.fit(X_train, y_train)
    proba = final.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, proba, threshold)
    test_metrics_default = evaluate(y_test, proba, 0.5)
    for k in ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"):
        print(f"  {k:10s} {test_metrics[k]:.4f}")

    fig_evaluation(y_test.values, proba, threshold, best_name)
    perm = permutation_importance(final, X_test, y_test, scoring="roc_auc",
                                  n_repeats=10 if not quick else 3, random_state=SEED, n_jobs=-1)
    imp = pd.DataFrame({"mean": perm.importances_mean, "std": perm.importances_std},
                       index=X_test.columns).sort_values("mean", ascending=False)
    imp.round(5).to_csv(REPORT_DIR / "feature_importance.csv")
    fig_importance(imp)
    has_shap = fig_shap(final, X_test)

    # Refit on ALL data for the shipped model (evaluation above stays on the held-out split)
    shipped = clone(final).fit(X, y)
    artifact = {
        "pipeline": shipped,
        "threshold": threshold,
        "model_name": best_name,
        "features": D.RAW_FEATURES,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(artifact, MODEL_DIR / "churn_model.joblib", compress=3)

    def clean_params(p):
        return {k.replace("model__", ""): (v.item() if hasattr(v, "item") else v) for k, v in p.items()}

    report = {
        "dataset": {
            "name": "IBM Telco Customer Churn",
            "source": D.SOURCE_URL,
            "sha256": D.EXPECTED_SHA256,
            "rows": int(len(df)),
            "churn_rate": float(y.mean()),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "selected_model": best_name,
        "best_params": clean_params(best_params),
        "cv_roc_auc_tuned": {k: float(v[0]) for k, v in tuned.items()},
        "test_metrics": test_metrics,
        "test_metrics_threshold_0.5": test_metrics_default,
        "shap_plot": has_shap,
        "runtime_seconds": round(time.time() - t0, 1),
        "quick_mode": quick,
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nSaved model -> {MODEL_DIR / 'churn_model.joblib'}")
    print(f"Saved report -> {REPORT_DIR / 'metrics.json'}  ({report['runtime_seconds']}s)")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="fewer search iterations (smoke test)")
    main(quick=ap.parse_args().quick)
