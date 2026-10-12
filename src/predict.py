# imports
import json
import joblib
import pandas as pd

from src.config import MODELS_DIR, METRICS_PATH, FEATURES, FEATURE_COLUMNS, TARGETS, ARTERIES


import math

# levels
def rl(p: float) -> str:
    if p < 0.25:
        return "Low"
    if p < 0.50:
        return "Moderate"
    if p < 0.75:
        return "High"
    return "Very high"


# coercion
def cr(f: dict, r):
    if r is None or r == "":
        return float(f["default"])
    try:
        v = float(r)
    except (TypeError, ValueError):
        raise ValueError(f"{f['label']} must be a number.")
    if not math.isfinite(v):
        raise ValueError(f"{f['label']} must be a valid finite number.")
    if f["type"] == "number":
        if not f["min"] <= v <= f["max"]:
            raise ValueError(f"{f['label']} must be between {f['min']} and {f['max']}.")
    elif v not in {o[1] for o in f["options"]}:
        raise ValueError(f"{f['label']} has an invalid value.")
    return v


# patient display formatting
def format_patient_value(f: dict, raw_val: float) -> str:
    if f.get("type") == "select":
        for label, val in f.get("options", []):
            if abs(float(val) - float(raw_val)) < 1e-5:
                return label
        return "Yes" if raw_val == 1 else "No"
    unit = f.get("unit", "")
    num_str = str(int(raw_val)) if float(raw_val).is_integer() else f"{raw_val:.1f}"
    return f"{num_str} {unit}".strip()


# robust binary classification SHAP extractor across library versions
def extract_binary_shap(raw_sv, raw_base):
    vals = raw_sv.values if hasattr(raw_sv, "values") else raw_sv
    if isinstance(vals, list):
        pos = vals[1] if len(vals) > 1 else vals[0]
        pos = pos[0] if getattr(pos, "ndim", 0) > 1 else pos
    elif getattr(vals, "ndim", 0) == 3:
        pos = vals[0, :, 1]
    elif getattr(vals, "ndim", 0) == 2:
        pos = vals[0]
    else:
        pos = vals

    b = raw_base.values if hasattr(raw_base, "values") else raw_base
    if isinstance(b, (list, tuple)):
        base_val = float(b[1] if len(b) > 1 else b[0])
    elif getattr(b, "ndim", 0) == 1 and getattr(b, "size", 0) > 1:
        base_val = float(b[1])
    else:
        base_val = float(b)

    return pos, base_val


# predictor
class CardiacRiskPredictor:

    # loading
    def __init__(self):
        if not METRICS_PATH.exists():
            raise FileNotFoundError("Models not found. Run: python -m src.train")
        self.metrics = json.loads(METRICS_PATH.read_text())
        self.models = {k: joblib.load(MODELS_DIR / f"{k.lower()}_model.pkl") for k in TARGETS}

        # Pre-cache TreeExplainer for all 4 targets for instant response times
        self.explainers = {}
        try:
            import shap
            for k, pipe in self.models.items():
                self.explainers[k] = shap.TreeExplainer(pipe.named_steps["rf"])
        except Exception:
            pass

    # formatting
    def _row(self, p: dict) -> pd.DataFrame:
        v = [cr(f, p.get(f["id"])) for f in FEATURES]
        return pd.DataFrame([v], columns=FEATURE_COLUMNS)

    # scoring
    def _result(self, k: str, w: pd.DataFrame) -> dict:
        pr = float(self.models[k].predict_proba(w)[0][1])
        m = self.metrics["targets"][k]
        return {
            "risk": round(pr, 3),
            "level": rl(pr),
            "confidence": m["confidence"],
            "accuracy": m["accuracy"],
            "auc": m["roc_auc"],
            "recall": m.get("recall", 0),
            "precision": m.get("precision", 0),
            "f1": m.get("f1", 0)
        }

    # SHAP explanation for a target: input is transformed via scaler first
    def _explain_target(self, k: str, rw: pd.DataFrame) -> dict:
        pipe = self.models[k]
        scaler = pipe.named_steps["scaler"]
        rf = pipe.named_steps["rf"]

        # Input wahi lo jo scaler ke baad model ko jaata hai
        x_scaled = scaler.transform(rw)

        if k not in self.explainers:
            import shap
            self.explainers[k] = shap.TreeExplainer(rf)

        explainer = self.explainers[k]
        raw_sv = explainer.shap_values(x_scaled)
        pos_sv, base_val = extract_binary_shap(raw_sv, explainer.expected_value)

        features_list = []
        for i, f in enumerate(FEATURES):
            val_raw = float(rw.iloc[0, i])
            sv_num = round(float(pos_sv[i]), 4)
            features_list.append({
                "id": f["id"],
                "column": f["column"],
                "label": f["label"],
                "patient_value": format_patient_value(f, val_raw),
                "shap_value": sv_num,
                "effect": "increase" if sv_num > 0 else ("decrease" if sv_num < 0 else "neutral")
            })

        # Sort features by absolute contribution magnitude (descending)
        features_list.sort(key=lambda item: abs(item["shap_value"]), reverse=True)

        return {
            "target": k,
            "base_value": round(base_val, 4),
            "features": features_list
        }

    # inference with all 4 targets explained
    def predict(self, q: dict) -> dict:
        rw = self._row(q)
        overall = self._result("CAD", rw)
        overall["shap"] = self._explain_target("CAD", rw)

        arteries = {}
        for k in ARTERIES:
            res_k = self._result(k, rw)
            res_k["shap"] = self._explain_target(k, rw)
            arteries[k] = res_k

        return {
            "overall": overall,
            "arteries": arteries,
            "explanations": {
                "CAD": overall["shap"],
                "LAD": arteries["LAD"]["shap"],
                "LCX": arteries["LCX"]["shap"],
                "RCA": arteries["RCA"]["shap"]
            }
        }
