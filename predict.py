"""Prediction utilities for CleanWater (Ichkeul).

This module is used by the Streamlit app (main.py).

It supports two model packaging formats:

1) Random Forest package (water-model1.pkl)
   - dict with keys: model (sklearn Pipeline), label_encoder, feature_columns

2) XGBoost package (water-model2.pkl)
   - dict with keys: model (xgboost.XGBClassifier), imputer (SimpleImputer),
     label_encoder, features, station_id_encoder (LabelEncoder),
     and optionally: feature_order

The predictor is defensive: it aligns columns, engineers temporal features,
encodes station_id when required, imputes missing values, then returns
predicted labels and probabilities.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd


def _ensure_dataframe(x: Any, name: str) -> pd.DataFrame:
    if not isinstance(x, pd.DataFrame):
        raise TypeError(f"{name} doit être un DataFrame pandas")
    return x


def _load_pickle(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _engineer_time_station_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add year/month/day from date if present, keep original columns."""
    out = df.copy()
    if "date" in out.columns:
        dt = pd.to_datetime(out["date"], errors="coerce")
        out["year"] = dt.dt.year
        out["month"] = dt.dt.month
        out["day"] = dt.dt.day
    return out


def _align_features(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    """Ensure all features exist; create missing as NaN; drop extras; keep order."""
    out = df.copy()
    for c in feature_list:
        if c not in out.columns:
            out[c] = np.nan
    return out[feature_list]


def predict_quality(df2: pd.DataFrame, data: pd.DataFrame, model_path: str = "water-model1.pkl") -> pd.DataFrame:
    """Predict water quality labels.

    Parameters
    ----------
    df2 : pd.DataFrame
        Original dataframe (kept for returning results).
    data : pd.DataFrame
        Dataframe used for prediction (can be same as df2).
    model_path : str
        Path to pickle model.

    Returns
    -------
    pd.DataFrame
        df2 with added prediction columns.
    """

    df2 = _ensure_dataframe(df2, "df2")
    data = _ensure_dataframe(data, "data")

    obj = _load_pickle(model_path)

    # ----------------------------
    # Case A: Packaged dict (recommended)
    # ----------------------------
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]

        # --- Random Forest package (sklearn Pipeline expects DataFrame) ---
        if "feature_columns" in obj or "features" in obj and hasattr(model, "predict") and "imputer" not in obj:
            feature_cols = obj.get("feature_columns") or obj.get("features")
            X = data.copy()
            # Align columns if feature list is provided
            if feature_cols is not None:
                X = _align_features(X, list(feature_cols))
            y_pred = model.predict(X)
            proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None

            le = obj.get("label_encoder")
            if le is not None and hasattr(le, "inverse_transform"):
                labels = le.inverse_transform(y_pred)
                class_names = list(le.classes_)
            else:
                labels = y_pred
                class_names = None

        # --- XGBoost package ---
        else:
            # Expect keys: imputer, label_encoder, features; optional: station_id_encoder
            features = obj.get("feature_order") or obj.get("features")
            if not features:
                raise ValueError("Le package XGBoost doit contenir la liste 'features' (ou 'feature_order').")

            Xdf = _engineer_time_station_features(data)

            # station_id encoding if required
            if "station_id_encoded" in features:
                if "station_id" not in Xdf.columns:
                    Xdf["station_id"] = "unknown"
                sid_enc = obj.get("station_id_encoder")
                if sid_enc is not None and hasattr(sid_enc, "transform"):
                    # map unseen to -1
                    known = set(getattr(sid_enc, "classes_", []))
                    sid = Xdf["station_id"].astype(str)
                    sid_safe = sid.where(sid.isin(list(known)), other=str(list(known)[0]) if known else "unknown")
                    Xdf["station_id_encoded"] = sid_enc.transform(sid_safe)
                else:
                    # fallback simple factorize
                    Xdf["station_id_encoded"] = pd.factorize(Xdf["station_id"].astype(str))[0]

            # Align, numeric coercion
            Xdf = _align_features(Xdf, list(features))
            Xdf = Xdf.apply(pd.to_numeric, errors="coerce")

            imp = obj.get("imputer")
            if imp is None:
                raise ValueError("Le package XGBoost doit contenir 'imputer' (SimpleImputer).")
            X = imp.transform(Xdf)

            y_pred = model.predict(X)
            proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None

            le = obj.get("label_encoder")
            if le is not None and hasattr(le, "inverse_transform"):
                labels = le.inverse_transform(y_pred)
                class_names = list(le.classes_)
            else:
                labels = y_pred
                class_names = None

        # Build output
        out = df2.copy()
        out["predicted_water_quality"] = labels
        if proba is not None and class_names is not None:
            for i, cname in enumerate(class_names):
                out[f"proba_{cname}"] = proba[:, i]
        return out

    # ----------------------------
    # Case B: Raw XGBClassifier pickle (legacy)
    # ----------------------------
    try:
        from xgboost import XGBClassifier  # noqa

        if obj.__class__.__name__ == "XGBClassifier":
            model = obj

            # Best-effort: infer expected features from booster
            booster = model.get_booster()
            fn = booster.feature_names
            if not fn:
                raise ValueError("Le modèle XGBoost ne contient pas de noms de features; fournissez un package dict.")

            Xdf = _engineer_time_station_features(data)
            if "station_id_encoded" in fn:
                if "station_id" not in Xdf.columns:
                    Xdf["station_id"] = "unknown"
                Xdf["station_id_encoded"] = pd.factorize(Xdf["station_id"].astype(str))[0]

            Xdf = _align_features(Xdf, list(fn)).apply(pd.to_numeric, errors="coerce")
            # Fit a temporary imputer on provided data (not ideal, but avoids crash)
            from sklearn.impute import SimpleImputer

            imp = SimpleImputer(strategy="median")
            X = imp.fit_transform(Xdf)

            y_pred = model.predict(X)
            proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None

            out = df2.copy()
            out["predicted_water_quality"] = y_pred
            if proba is not None:
                for i in range(proba.shape[1]):
                    out[f"proba_class_{i}"] = proba[:, i]
            return out
    except Exception:
        pass

    raise TypeError(
        "Format de modèle non supporté. Utilisez un pickle dict avec la clé 'model' (recommandé)."
    )
