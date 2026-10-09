"""Data pipeline for the Diabetes-Risk MLP project.

Responsibilities
----------------
* Load the raw CSV exactly once, cache a tidy parquet-free pickle for speed.
* EDA summary (missingness, cardinality, class balance) -> artifacts/eda_report.json
* Column-role policy (leakage audit lives in eda.py / DESIGN.md):
    - DROP   : Patient_ID, Country, Diabetes_Risk_Score, AI_Health_Recommendation
    - TARGET : Diabetes_Risk  (Low / Moderate / High -> ordinal-encoded 0/1/2)
    - NUMERIC: continuous + integer columns (median-imputed, standardized)
    - CATEGORICAL: low-cardinality strings (one-hot)
* Deterministic stratified train/validation/test split (70/15/15).
* Returns numpy arrays + fitted metadata so the same transforms can be applied
  to any future inference batch.

Nothing here touches the test set except to materialize it; all statistics used
for imputation/scaling are fit on TRAIN only (test is never seen during tuning).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ARTIFACTS = os.path.join(ROOT, "artifacts")
RAW_CSV = os.path.join(ROOT, "..", "diabetes_risk_prediction_dataset.csv")

TARGET = "Diabetes_Risk"
CLASS_ORDER = ["Low", "Moderate", "High"]          # ordinal encoding 0,1,2
LABEL_TO_INT = {c: i for i, c in enumerate(CLASS_ORDER)}

# ---------------------------------------------------------------------------
# Column-role policy (justified in DESIGN.md §3 – leakage audit)
# ---------------------------------------------------------------------------
DROP_COLS = [
    "Patient_ID",                    # surrogate key, no signal
    "Country",                       # 25 categories, weak/nonsensical causal link
    "Diabetes_Risk_Score",           # near-deterministic proxy of the target (LEAKAGE)
    "AI_Health_Recommendation",      # generated FROM the risk label (LEAKAGE)
]

CATEGORICAL_COLS = [
    "Gender", "Physical_Activity_Level", "Diet_Quality", "Sugar_Intake_Level",
    "Stress_Level", "Smoking_Status", "Alcohol_Consumption",
    "Family_History_Diabetes", "Hypertension", "Heart_Disease", "Fatty_Liver",
    "PCOS", "Medication_Adherence", "Work_Type", "Residence_Type",
    "Doctor_Consultation_Needed",
]


@dataclass
class PipelineState:
    """Everything needed to reproduce the transform on unseen data."""
    numeric_cols: List[str] = field(default_factory=list)
    onehot_cols: List[str] = field(default_factory=list)   # generated feature names
    medians: Dict[str, float] = field(default_factory=dict)
    means: Dict[str, float] = field(default_factory=dict)
    stds: Dict[str, float] = field(default_factory=dict)
    clip_lo: Dict[str, float] = field(default_factory=dict)   # robust outlier clipping
    clip_hi: Dict[str, float] = field(default_factory=dict)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)
    # Coerce stray non-numeric junk inside numeric columns (dataset has a few)
    for c in df.columns:
        if c not in DROP_COLS + CATEGORICAL_COLS + [TARGET] and df[c].dtype != object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def detect_numeric(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns
            if c not in DROP_COLS + CATEGORICAL_COLS + [TARGET]
            and pd.api.types.is_numeric_dtype(df[c])]


def _fit_train_stats(train_df: pd.DataFrame, numeric_cols: List[str]) -> PipelineState:
    st = PipelineState(numeric_cols=list(numeric_cols))
    for c in numeric_cols:
        x = train_df[c].dropna()
        st.medians[c] = float(x.median())
        lo, hi = x.quantile(0.001), x.quantile(0.999)
        st.clip_lo[c], st.clip_hi[c] = float(lo), float(hi)
        xc = x.clip(lo, hi)
        st.means[c] = float(xc.mean())
        st.stds[c] = float(xc.std() or 1.0)
    return st


def _transform(df: pd.DataFrame, st: PipelineState, fit_onehot: bool = False) -> np.ndarray:
    parts: List[np.ndarray] = []

    # --- numeric block -----------------------------------------------------
    Xn = np.empty((len(df), len(st.numeric_cols)), dtype=np.float64)
    for j, c in enumerate(st.numeric_cols):
        col = pd.to_numeric(df[c], errors="coerce") if df[c].dtype == object else df[c]
        col = col.astype("float64").fillna(st.medians[c])
        col = col.clip(st.clip_lo[c], st.clip_hi[c])
        Xn[:, j] = (col - st.means[c]) / max(st.stds[c], 1e-8)
    parts.append(Xn)

    # --- categorical block (one-hot) ---------------------------------------
    cat = df[CATEGORICAL_COLS].astype("category")
    oh = pd.get_dummies(cat, prefix=CATEGORICAL_COLS, dummy_na=True)  # NaN -> own column
    if fit_onehot:
        st.onehot_cols = list(oh.columns)
    else:
        oh = oh.reindex(columns=st.onehot_cols, fill_value=0)
    parts.append(oh.to_numpy(dtype=np.float64))

    return np.hstack(parts)


def build_dataset(seed: int = 42) -> Dict[str, object]:
    """Full pipeline. Returns dict with arrays, state, and metadata."""
    os.makedirs(ARTIFACTS, exist_ok=True)
    df = load_raw()

    # ---- EDA snapshot ------------------------------------------------------
    eda = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "missing_by_col": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
        "target_counts": df[TARGET].value_counts().to_dict(),
        "target_missing": int(df[TARGET].isna().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    # Drop rows without a label (cannot supervise them); keep feature NaNs -> imputed
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].map(LABEL_TO_INT)
    assert df[TARGET].notna().all(), "unexpected label value"

    drop = [c for c in DROP_COLS if c in df.columns]
    work = df.drop(columns=drop)
    numeric_cols = detect_numeric(work)

    # ---- deterministic stratified splits 70/15/15 --------------------------
    y = work[TARGET].to_numpy()
    idx = np.arange(len(work))
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    tr_idx, tmp_idx = next(iter(sss1.split(idx, y)))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    va_idx, te_idx = next(iter(sss2.split(tmp_idx, y[tmp_idx])))
    val_idx, test_idx = tmp_idx[va_idx], tmp_idx[te_idx]

    train_df, val_df, test_df = work.iloc[tr_idx], work.iloc[val_idx], work.iloc[test_idx]

    st = _fit_train_stats(train_df, numeric_cols)
    Xtr = _transform(train_df, st, fit_onehot=True)
    Xva = _transform(val_df, st)
    Xte = _transform(test_df, st)
    ytr, yva, yte = (d[TARGET].to_numpy().astype(np.int64)
                     for d in (train_df, val_df, test_df))

    meta = {
        **eda,
        "dropped_cols": drop,
        "numeric_cols": numeric_cols,
        "categorical_cols": CATEGORICAL_COLS,
        "n_features": int(Xtr.shape[1]),
        "split_sizes": {"train": int(len(Xtr)), "val": int(len(Xva)), "test": int(len(Xte))},
    }
    bc = pd.Series(ytr).value_counts().sort_index()
    meta["train_class_counts"] = {int(k): int(v) for k, v in bc.items()}

    st.to_json(os.path.join(ARTIFACTS, "pipeline_state.json"))
    with open(os.path.join(ARTIFACTS, "eda_report.json"), "w") as f:
        json.dump(meta, f, indent=2)

    np.savez_compressed(os.path.join(ARTIFACTS, "arrays.npz"),
                        Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte)

    return {"Xtr": Xtr, "ytr": ytr, "Xva": Xva, "yva": yva,
            "Xte": Xte, "yte": yte, "state": st, "meta": meta}


def load_cached_arrays():
    p = os.path.join(ARTIFACTS, "arrays.npz")
    if not os.path.exists(p):
        return build_dataset()
    z = np.load(p)
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    d = build_dataset()
    print(json.dumps(d["meta"], indent=2)[:1200])
    print("shapes:", d["Xtr"].shape, d["Xva"].shape, d["Xte"].shape)
