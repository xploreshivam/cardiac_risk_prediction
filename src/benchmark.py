"""Benchmark Random Forest vs. GradientBoosting, XGBoost, and CalibratedClassifierCV.

Compares model discrimination (ROC-AUC, F1, Accuracy, Sensitivity, Precision)
and probability calibration (Brier Score) across 5-fold Stratified CV on the
Z-Alizadeh Sani dataset, focusing especially on LCX and RCA.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import BASE_DIR, FEATURE_COLUMNS, TARGETS
from src.data import load_dataset


def evaluate_cv(build_estimator_fn, X, y, cv):
    """Run cross-validation collecting out-of-fold predictions for exact evaluation."""
    y_true_list = []
    y_pred_list = []
    y_prob_list = []

    fold_aucs = []
    fold_briers = []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        estimator = build_estimator_fn(y_train)
        estimator.fit(X_train_scaled, y_train)

        probs = estimator.predict_proba(X_val_scaled)[:, 1]
        preds = (probs >= 0.5).astype(int)

        y_true_list.extend(y_val)
        y_pred_list.extend(preds)
        y_prob_list.extend(probs)

        fold_aucs.append(roc_auc_score(y_val, probs))
        fold_briers.append(brier_score_loss(y_val, probs))

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    y_prob = np.array(y_prob_list)

    return {
        "roc_auc_mean": round(float(np.mean(fold_aucs)), 3),
        "roc_auc_std": round(float(np.std(fold_aucs)), 3),
        "brier_score": round(float(np.mean(fold_briers)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 3),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 3),
    }


def run_benchmark():
    df = load_dataset()
    X = df[FEATURE_COLUMNS]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Models to evaluate
    model_factories = {
        "Random Forest (Baseline)": lambda y_tr: RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=1
        ),
        "RF + Calibrated (Sigmoid)": lambda y_tr: CalibratedClassifierCV(
            estimator=RandomForestClassifier(
                n_estimators=300, max_depth=6, min_samples_leaf=2,
                class_weight="balanced", random_state=42, n_jobs=1
            ),
            method="sigmoid",
            cv=3
        ),
        "RF + Calibrated (Isotonic)": lambda y_tr: CalibratedClassifierCV(
            estimator=RandomForestClassifier(
                n_estimators=300, max_depth=6, min_samples_leaf=2,
                class_weight="balanced", random_state=42, n_jobs=1
            ),
            method="isotonic",
            cv=3
        ),
        "Gradient Boosting (Sklearn)": lambda y_tr: GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=3,
            subsample=0.8, random_state=42
        ),
        "Gradient Boosting + Calibrated": lambda y_tr: CalibratedClassifierCV(
            estimator=GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.05, max_depth=3,
                subsample=0.8, random_state=42
            ),
            method="sigmoid",
            cv=3
        ),
        "XGBoost": lambda y_tr: XGBClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(len(y_tr) - sum(y_tr)) / max(sum(y_tr), 1),
            random_state=42, eval_metric="logloss"
        ),
        "XGBoost + Calibrated": lambda y_tr: CalibratedClassifierCV(
            estimator=XGBClassifier(
                n_estimators=100, learning_rate=0.05, max_depth=3,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=(len(y_tr) - sum(y_tr)) / max(sum(y_tr), 1),
                random_state=42, eval_metric="logloss"
            ),
            method="sigmoid",
            cv=3
        ),
    }

    results = {}
    print("=" * 80)
    print("BENCHMARK: Random Forest vs. GradientBoosting vs. XGBoost vs. Calibrated")
    print("Z-Alizadeh Sani Dataset (N=303), 5-Fold Stratified Cross-Validation")
    print("=" * 80)

    for target_name, (column, positive) in TARGETS.items():
        y = (df[column] == positive).astype(int)
        pos_rate = float(y.mean())
        print(f"\nTarget: {target_name} ({pos_rate * 100:.1f}% positive)")
        print("-" * 80)
        print(f"{'Model':<32} | {'ROC-AUC':<9} | {'Brier (Cal)':<11} | {'Accuracy':<8} | {'Recall':<7} | {'Precision':<9} | {'F1':<6}")
        print("-" * 80)

        results[target_name] = {
            "positive_rate": round(pos_rate, 3),
            "models": {}
        }

        for model_name, factory in model_factories.items():
            res = evaluate_cv(factory, X, y, cv)
            results[target_name]["models"][model_name] = res
            auc_str = f"{res['roc_auc_mean']:.3f}±{res['roc_auc_std']:.2f}"
            print(f"{model_name:<32} | {auc_str:<9} | {res['brier_score']:<11.4f} | {res['accuracy']:<8.3f} | {res['recall']:<7.3f} | {res['precision']:<9.3f} | {res['f1']:<6.3f}")

    # Output JSON
    output_path = BASE_DIR / "docs" / "benchmark_results.json"
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved benchmark metrics to {output_path}")

    return results


if __name__ == "__main__":
    run_benchmark()
