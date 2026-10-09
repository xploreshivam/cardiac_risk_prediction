"""Dataset loading and cleaning."""
import pandas as pd
from src.config import DATA_PATH, SHEET, FEATURE_COLUMNS


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Download 'Extension of Z-Alizadeh Sani "
            "dataset' from the UCI repository and place the .xlsx file in data/raw/."
        )
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET)

    # The dataset spells female as "Fmale"
    df["Sex"] = (df["Sex"].astype(str).str.strip().str.lower() == "male").astype(int)

    # Y/N text columns -> 1/0 (numeric columns are left alone)
    for col in FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip().str.upper().isin(["Y", "YES"]).astype(int)

    for col in ("Cath", "LAD", "LCX", "RCA"):
        df[col] = df[col].astype(str).str.strip()
    return df
