"""Single source of truth: paths, model targets and the patient form fields."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "extention_of_Z-Alizadeh_sani_dataset.xlsx"
SHEET = "Sheet 1 - Table 1"
MODELS_DIR = BASE_DIR / "models"
METRICS_PATH = MODELS_DIR / "model_metrics.json"

# key -> (dataset column, value that counts as "positive")
# LAD / LCX / RCA / Cath all come from the same angiography, so they are never
# used as input features for each other (that would leak the answer).
TARGETS = {
    "CAD": ("Cath", "CAD"),
    "LAD": ("LAD", "Stenotic"),
    "LCX": ("LCX", "Stenotic"),
    "RCA": ("RCA", "Stenotic"),
}
ARTERIES = ["LAD", "LCX", "RCA"]

YES_NO = [("No", 0), ("Yes", 1)]

# Order here = column order the models are trained on.
FEATURES = [
    {"id": "age", "column": "Age", "label": "Age", "group": "Patient",
     "type": "number", "min": 20, "max": 90, "step": 1, "default": 55, "unit": "years"},
    {"id": "sex", "column": "Sex", "label": "Sex", "group": "Patient",
     "type": "select", "options": [("Male", 1), ("Female", 0)], "default": 1},
    {"id": "bmi", "column": "BMI", "label": "Body mass index", "group": "Patient",
     "type": "number", "min": 14, "max": 50, "step": 0.1, "default": 26, "unit": "kg/m²"},
    {"id": "dm", "column": "DM", "label": "Diabetes", "group": "Patient",
     "type": "select", "options": YES_NO, "default": 0},
    {"id": "htn", "column": "HTN", "label": "Hypertension", "group": "Patient",
     "type": "select", "options": YES_NO, "default": 0},
    {"id": "smoker", "column": "Current Smoker", "label": "Current smoker", "group": "Patient",
     "type": "select", "options": YES_NO, "default": 0},
    {"id": "bp", "column": "BP", "label": "Systolic blood pressure", "group": "Patient",
     "type": "number", "min": 80, "max": 220, "step": 1, "default": 120, "unit": "mmHg"},
    {"id": "typical_cp", "column": "Typical Chest Pain", "label": "Typical chest pain",
     "group": "Symptoms and ECG", "type": "select", "options": YES_NO, "default": 0},
    {"id": "atypical_cp", "column": "Atypical", "label": "Atypical chest pain",
     "group": "Symptoms and ECG", "type": "select", "options": YES_NO, "default": 0},
    {"id": "t_inversion", "column": "Tinversion", "label": "T-wave inversion on ECG",
     "group": "Symptoms and ECG", "type": "select", "options": YES_NO, "default": 0},
    {"id": "ef", "column": "EF-TTE", "label": "Ejection fraction", "group": "Echocardiogram",
     "type": "number", "min": 15, "max": 70, "step": 1, "default": 55, "unit": "%"},
    {"id": "rwma", "column": "Region RWMA", "label": "Regions with wall motion abnormality",
     "group": "Echocardiogram", "type": "number", "min": 0, "max": 4, "step": 1, "default": 0},
]
FEATURE_COLUMNS = [f["column"] for f in FEATURES]
