from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from feature_engineering import build_features


DBO_MODEL_PATH = os.path.join("models", "dbo_model.pkl")
DCO_MODEL_PATH = os.path.join("models", "dco_model.pkl")


def _load_pickle(path: str) -> Any:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _extract_expected_features(obj: Any) -> list[str] | None:
    if obj is None:
        return None

    if isinstance(obj, dict):
        for key in ["feature_columns", "features", "feature_order"]:
            if key in obj and obj[key]:
                return list(obj[key])

    if hasattr(obj, "feature_names_in_"):
        try:
            return list(obj.feature_names_in_)
        except Exception:
            pass

    return None


def _extract_model_and_preprocessor(obj: Any) -> Tuple[Any, Any, list[str] | None]:
    if obj is None:
        return None, None, None

    if isinstance(obj, dict):
        model = obj.get("model")
        imputer = obj.get("imputer")
        features = _extract_expected_features(obj)
        return model, imputer, features

    return obj, None, _extract_expected_features(obj)


def _align_features(df: pd.DataFrame, feature_list: list[str] | None) -> pd.DataFrame:
    out = df.copy()
    if not feature_list:
        return out

    for col in feature_list:
        if col not in out.columns:
            out[col] = np.nan

    return out[feature_list]


def _predict_with_model(model_obj: Any, df_features: pd.DataFrame) -> Tuple[np.ndarray | None, pd.DataFrame]:
    model, imputer, expected_features = _extract_model_and_preprocessor(model_obj)

    if model is None:
        return None, df_features

    X = _align_features(df_features, expected_features)
    X = X.apply(pd.to_numeric, errors="coerce")

    if imputer is not None:
        X_infer = imputer.transform(X)
    else:
        X_infer = X

    preds = model.predict(X_infer)
    return preds, X


def estimate_bod_cod(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estime DBO et DCO à partir des mesures capteurs/features.
    Retourne le DataFrame enrichi.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df doit être un DataFrame pandas")

    if df.empty:
        return df.copy()

    features_df = build_features(df)

    dbo_obj = _load_pickle(DBO_MODEL_PATH)
    dco_obj = _load_pickle(DCO_MODEL_PATH)

    out = df.copy()

    dbo_preds, _ = _predict_with_model(dbo_obj, features_df)
    dco_preds, _ = _predict_with_model(dco_obj, features_df)

    if dbo_preds is not None:
        out["estimated_dbo"] = pd.to_numeric(pd.Series(dbo_preds), errors="coerce")
    else:
        out["estimated_dbo"] = np.nan

    if dco_preds is not None:
        out["estimated_dco"] = pd.to_numeric(pd.Series(dco_preds), errors="coerce")
    else:
        out["estimated_dco"] = np.nan

    out["soft_sensing_available"] = int(dbo_obj is not None or dco_obj is not None)

    # Heuristique simple de confiance
    confidence = []
    for _, row in features_df.iterrows():
        score = 1.0

        critical_fields = [
            row.get("temperature"),
            row.get("ph"),
            row.get("turbidity"),
            row.get("dissolved_oxygen"),
        ]
        missing_ratio = sum(pd.isna(v) for v in critical_fields) / max(len(critical_fields), 1)
        score -= 0.5 * missing_ratio

        outlier_flags = [
            row.get("flag_temperature_outlier", 0),
            row.get("flag_ph_outlier", 0),
            row.get("flag_do_outlier", 0),
            row.get("flag_turbidity_outlier", 0),
            row.get("flag_conductivity_outlier", 0),
        ]
        score -= 0.1 * sum(int(v) for v in outlier_flags if not pd.isna(v))

        score = max(0.0, min(1.0, score))
        confidence.append(score)

    out["soft_sensing_confidence"] = confidence

    return out


def estimate_single_record(record: Dict[str, Any]) -> Dict[str, Any]:
    df = pd.DataFrame([record])
    pred_df = estimate_bod_cod(df)

    if pred_df.empty:
        return {
            "estimated_dbo": None,
            "estimated_dco": None,
            "soft_sensing_confidence": 0.0,
            "soft_sensing_available": 0,
        }

    row = pred_df.iloc[0]
    return {
        "estimated_dbo": _safe_float(row.get("estimated_dbo")),
        "estimated_dco": _safe_float(row.get("estimated_dco")),
        "soft_sensing_confidence": _safe_float(row.get("soft_sensing_confidence")),
        "soft_sensing_available": int(row.get("soft_sensing_available", 0)),
    }