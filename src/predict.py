# imports
import json
import joblib
import pandas as pd

from src.config import MODELS_DIR, METRICS_PATH, FEATURES, FEATURE_COLUMNS, TARGETS, ARTERIES


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
    if f["type"] == "number":
        if not f["min"] <= v <= f["max"]:
            raise ValueError(f"{f['label']} must be between {f['min']} and {f['max']}.")
    elif v not in {o[1] for o in f["options"]}:
        raise ValueError(f"{f['label']} has an invalid value.")
    return v


# predictor
class CardiacRiskPredictor:

    # loading
    def __init__(self):
        if not METRICS_PATH.exists():
            raise FileNotFoundError("Models not found. Run: python -m src.train")
        self.metrics = json.loads(METRICS_PATH.read_text())
        self.models = {k: joblib.load(MODELS_DIR / f"{k.lower()}_model.pkl") for k in TARGETS}

    # formatting
    def _row(self, p: dict) -> pd.DataFrame:
        v = [cr(f, p.get(f["id"])) for f in FEATURES]
        return pd.DataFrame([v], columns=FEATURE_COLUMNS)

    # scoring
    def _result(self, k: str, w: pd.DataFrame) -> dict:
        pr = float(self.models[k].predict_proba(w)[0][1])
        m = self.metrics["targets"][k]
        return {"risk": round(pr, 3), "level": rl(pr), "confidence": m["confidence"],
                "accuracy": m["accuracy"], "auc": m["roc_auc"]}

    # inference
    def predict(self, q: dict) -> dict:
        rw = self._row(q)
        return {
            "overall": self._result("CAD", rw),
            "arteries": {k: self._result(k, rw) for k in ARTERIES}
        }
