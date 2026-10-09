"""Train one Random Forest per target (overall CAD, LAD, LCX, RCA).

Run:  python -m src.train
Metrics come from 5-fold stratified cross-validation, so they are honest
estimates on a small dataset (303 patients), not training accuracy.
"""
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, METRICS_PATH, TARGETS, FEATURE_COLUMNS
from src.data import load_dataset

RELIABLE_AUC = 0.80  # below this the UI marks the artery as "low confidence"


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=1)),
    ])


def train() -> dict:
    df = load_dataset()
    X = df[FEATURE_COLUMNS]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    MODELS_DIR.mkdir(exist_ok=True)

    metrics = {"n_patients": int(len(df)), "cv_folds": 5,
               "features": FEATURE_COLUMNS, "targets": {}}

    for key, (column, positive) in TARGETS.items():
        y = (df[column] == positive).astype(int)
        scores = cross_validate(
            build_pipeline(), X, y, cv=cv,
            scoring=["accuracy", "roc_auc", "f1", "recall", "precision"])
        rate = float(y.mean())
        auc = float(scores["test_roc_auc"].mean())
        metrics["targets"][key] = {
            "positive_rate": round(rate, 3),
            "baseline_accuracy": round(max(rate, 1 - rate), 3),
            "accuracy": round(float(scores["test_accuracy"].mean()), 3),
            "roc_auc": round(auc, 3),
            "f1": round(float(scores["test_f1"].mean()), 3),
            "recall": round(float(scores["test_recall"].mean()), 3),
            "precision": round(float(scores["test_precision"].mean()), 3),
            "confidence": "reliable" if auc >= RELIABLE_AUC else "low",
        }
        model = build_pipeline().fit(X, y)
        joblib.dump(model, MODELS_DIR / f"{key.lower()}_model.pkl")
        m = metrics["targets"][key]
        print(f"{key}: accuracy {m['accuracy']:.3f} (baseline {m['baseline_accuracy']:.3f}) "
              f"| AUC {m['roc_auc']:.3f} | {m['confidence']}")

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Saved models and metrics to {MODELS_DIR}")
    return metrics


if __name__ == "__main__":
    train()
