import pandas as pd
from src.config import DATA_PATH, SHEET, FEATURES, FEATURE_COLUMNS


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Download 'Extension of Z-Alizadeh Sani "
            "dataset' from the UCI repository and place the .xlsx file in data/raw/."
        )
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET)

    # The dataset spells female as "Fmale"
    df["Sex"] = (df["Sex"].astype(str).str.strip().str.lower() == "male").astype(int)

    # Clean and cast each feature according to its defined type
    for f in FEATURES:
        col = f["column"]
        if f["type"] == "number":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(f["default"]).astype(float)
        elif col != "Sex":
            df[col] = df[col].astype(str).str.strip().str.upper().isin(["Y", "YES", "1", "TRUE"]).astype(int)

    for col in ("Cath", "LAD", "LCX", "RCA"):
        df[col] = df[col].astype(str).str.strip()
    return df
