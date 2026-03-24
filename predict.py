from __future__ import annotations

import os
import pickle
from typing import Any, List

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
    out = df.copy()
    if "date" in out.columns:
        dt = pd.to_datetime(out["date"], errors="coerce")
        out["year"] = dt.dt.year
        out["month"] = dt.dt.month
        out["day"] = dt.dt.day
    return out


def _align_features(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in feature_list:
        if c not in out.columns:
            out[c] = np.nan
    return out[feature_list]


def predict_quality(df2: pd.DataFrame, data: pd.DataFrame, model_path: str = "water-model1.pkl") -> pd.DataFrame:
    df2 = _ensure_dataframe(df2, "df2")
    data = _ensure_dataframe(data, "data")

    obj = _load_pickle(model_path)

    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]

        if ("feature_columns" in obj or "features" in obj) and "imputer" not in obj:
            feature_cols = obj.get("feature_columns") or obj.get("features")
            X = data.copy()
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

        else:
            features = obj.get("feature_order") or obj.get("features")
            if not features:
                raise ValueError("Le package XGBoost doit contenir 'features' ou 'feature_order'.")

            Xdf = _engineer_time_station_features(data)

            if "station_id_encoded" in features:
                if "station_id" not in Xdf.columns:
                    Xdf["station_id"] = "unknown"
                sid_enc = obj.get("station_id_encoder")
                if sid_enc is not None and hasattr(sid_enc, "transform"):
                    known = set(getattr(sid_enc, "classes_", []))
                    sid = Xdf["station_id"].astype(str)
                    sid_safe = sid.where(sid.isin(list(known)), other=str(list(known)[0]) if known else "unknown")
                    Xdf["station_id_encoded"] = sid_enc.transform(sid_safe)
                else:
                    Xdf["station_id_encoded"] = pd.factorize(Xdf["station_id"].astype(str))[0]

            Xdf = _align_features(Xdf, list(features))
            Xdf = Xdf.apply(pd.to_numeric, errors="coerce")

            imp = obj.get("imputer")
            if imp is None:
                raise ValueError("Le package XGBoost doit contenir 'imputer'.")

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

        out = df2.copy()
        out["predicted_water_quality"] = labels
        if proba is not None and class_names is not None:
            for i, cname in enumerate(class_names):
                out[f"proba_{cname}"] = proba[:, i]
        return out

    raise TypeError("Format de modèle non supporté.")