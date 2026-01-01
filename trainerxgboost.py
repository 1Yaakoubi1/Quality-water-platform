"""Train an XGBoost multi-class model for Ichkeul water quality.

Creates a deployable pickle package 'water-model2.pkl' compatible with predict.py.

Usage
-----
python trainerxgboost.py \
  --csv ichkeul_water_quality_ml_dataset.csv \
  --out water-model2.pkl

The output pickle contains:
{
  'model': XGBClassifier,
  'imputer': SimpleImputer,
  'label_encoder': LabelEncoder,
  'station_id_encoder': LabelEncoder,
  'features': [...],
  'feature_order': [...],
  'best_params': {...},
  'metrics': {...}
}
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from xgboost import XGBClassifier


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Temporal features
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["day"] = out["date"].dt.day
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--out", type=str, default="water-model2.pkl", help="Output pickle path")
    parser.add_argument("--target", type=str, default="water_quality_label")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=650, help=">=500")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if args.target not in df.columns:
        raise ValueError(f"Target column missing: {args.target}")

    df = build_features(df)

    # Encode station_id
    sid_enc = LabelEncoder()
    df["station_id_encoded"] = sid_enc.fit_transform(df["station_id"].astype(str))

    drop_cols = ["date", "station_id", args.target]
    base_features = [c for c in df.columns if c not in drop_cols]

    # Coerce numerics
    for c in base_features:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Target encoding
    y_enc = LabelEncoder()
    y = y_enc.fit_transform(df[args.target].astype(str))
    X = df[base_features].copy()

    # Class imbalance: sample_weight (multi-class safe)
    counts = np.bincount(y)
    class_w = (len(y) / (len(counts) * counts))
    sw = class_w[y]

    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
        X, y, sw, test_size=0.2, stratify=y, random_state=args.random_state
    )

    # Imputer
    imp = SimpleImputer(strategy="median")
    X_train_imp = imp.fit_transform(X_train)
    X_test_imp = imp.transform(X_test)

    # --- Lightweight random search with 5-fold CV (macro F1) ---
    rng = np.random.default_rng(args.random_state)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)

    def sample_params() -> Dict[str, Any]:
        return {
            "learning_rate": float(rng.uniform(0.03, 0.12)),
            "max_depth": int(rng.integers(3, 8)),
            "subsample": float(rng.choice([0.8, 0.9, 1.0])),
            "colsample_bytree": float(rng.choice([0.7, 0.85, 1.0])),
            "min_child_weight": int(rng.choice([1, 3, 5, 7])),
            "gamma": float(rng.choice([0.0, 0.1, 0.2])),
            "reg_lambda": float(rng.choice([1.0, 1.2, 1.5])),
        }

    candidates = [sample_params() for _ in range(8)]

    def cv_macro_f1(params: Dict[str, Any]) -> float:
        scores = []
        for tr_idx, va_idx in cv.split(X_train_imp, y_train):
            clf = XGBClassifier(
                objective="multi:softprob",
                num_class=len(y_enc.classes_),
                n_estimators=200,
                tree_method="hist",
                n_jobs=-1,
                random_state=args.random_state,
                eval_metric="mlogloss",
                **params,
            )
            clf.fit(X_train_imp[tr_idx], y_train[tr_idx], sample_weight=sw_train[tr_idx])
            pred = clf.predict(X_train_imp[va_idx])
            scores.append(f1_score(y_train[va_idx], pred, average="macro"))
        return float(np.mean(scores))

    scored = [(cv_macro_f1(p), p) for p in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    best_cv_f1, best_params = scored[0]

    # Final training with required estimators
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(y_enc.classes_),
        n_estimators=max(args.n_estimators, 500),
        tree_method="hist",
        n_jobs=-1,
        random_state=args.random_state,
        eval_metric="mlogloss",
        **best_params,
    )
    model.fit(X_train_imp, y_train, sample_weight=sw_train)

    pred = model.predict(X_test_imp)
    metrics = {
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_f1_macro": float(f1_score(y_test, pred, average="macro")),
        "cv_f1_macro": float(best_cv_f1),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classes": list(y_enc.classes_),
        "class_counts": counts.tolist(),
    }

    out_obj = {
        "model": model,
        "imputer": imp,
        "label_encoder": y_enc,
        "station_id_encoder": sid_enc,
        "features": base_features,
        "feature_order": base_features,
        "best_params": best_params,
        "metrics": metrics,
        "source_csv": str(csv_path.name),
    }

    out_path = Path(args.out)
    with open(out_path, "wb") as f:
        pickle.dump(out_obj, f)

    print("✅ Saved:", out_path)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
