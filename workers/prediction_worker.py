from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from firebase_service import (
    add_alert,
    add_prediction,
    get_latest_raw_measurements,
    upsert_latest_station_state,
)
from predict import predict_quality
from soft_sensing import estimate_bod_cod
from scoring import compute_all_scores
from alert_engine import evaluate_all_alerts, deduplicate_alerts


CLASSIFICATION_MODELS = {
    "Random Forest": "models/water-model1.pkl",
    "XGBoost": "models/water-model2.pkl",
}


def _safe_float(value: Any):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _prepare_classification_input(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    rename_map = {
        "temperature": "Temperature",
        "ph": "pH",
        "turbidity": "Turbidity",
        "dissolved_oxygen": "Dissolved Oxygen",
        "salinity": "Salinity",
        "chlorophyll": "Chlorophyll",
        "dissolved_organic_matter": "Dissolved Organic Matter",
        "suspended_matter": "Suspended Matter",
    }

    out = out.rename(columns=rename_map)
    return out


def run_quality_prediction_for_dataframe(
    df: pd.DataFrame,
    model_name: str = "Random Forest",
) -> pd.DataFrame:
    if model_name not in CLASSIFICATION_MODELS:
        raise ValueError(f"Modèle non supporté : {model_name}")

    model_path = CLASSIFICATION_MODELS[model_name]
    X = _prepare_classification_input(df)
    pred_df = predict_quality(X, X.copy(), model_path)
    return pred_df


def build_prediction_payload(row: pd.Series) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}

    if "predicted_water_quality" in row.index:
        payload["predicted_water_quality"] = row.get("predicted_water_quality")

    proba_cols = [c for c in row.index if str(c).startswith("proba_")]
    if proba_cols:
        payload["probabilities"] = {
            c: _safe_float(row.get(c))
            for c in proba_cols
            if _safe_float(row.get(c)) is not None
        }

    if "estimated_dbo" in row.index:
        payload["estimated_dbo"] = _safe_float(row.get("estimated_dbo"))

    if "estimated_dco" in row.index:
        payload["estimated_dco"] = _safe_float(row.get("estimated_dco"))

    if "soft_sensing_confidence" in row.index:
        payload["soft_sensing_confidence"] = _safe_float(row.get("soft_sensing_confidence"))

    if "soft_sensing_available" in row.index:
        payload["soft_sensing_available"] = int(row.get("soft_sensing_available", 0))

    if "water_quality_score" in row.index:
        payload["water_quality_score"] = _safe_float(row.get("water_quality_score"))

    if "water_quality_label" in row.index:
        payload["water_quality_label"] = row.get("water_quality_label")

    if "risk_score" in row.index:
        payload["risk_score"] = _safe_float(row.get("risk_score"))

    if "risk_level" in row.index:
        payload["risk_level"] = row.get("risk_level")

    if "data_reliability_score" in row.index:
        payload["data_reliability_score"] = _safe_float(row.get("data_reliability_score"))

    if "data_reliability_level" in row.index:
        payload["data_reliability_level"] = row.get("data_reliability_level")

    return payload


def _save_alerts_from_row(row_dict: Dict[str, Any]) -> int:
    alerts = deduplicate_alerts(evaluate_all_alerts(row_dict))
    saved_count = 0

    for alert in alerts:
        add_alert(
            station_id=alert["station_id"],
            alert_type=alert["alert_type"],
            severity=alert["severity"],
            message=alert["message"],
            recommendation=alert.get("recommendation", ""),
            metadata=alert.get("metadata", {}),
        )
        saved_count += 1

    return saved_count


def run_predictions_for_latest_measurements(
    station_id: str | None = None,
    limit: int = 50,
    classification_model_name: str = "Random Forest",
) -> Dict[str, Any]:
    df = get_latest_raw_measurements(limit=limit, station_id=station_id)

    if df.empty:
        return {"status": "empty", "rows_processed": 0}

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp")

    # 1) Classification qualité
    quality_df = run_quality_prediction_for_dataframe(
        df.copy(),
        model_name=classification_model_name,
    )

    # 2) Soft-sensing
    soft_df = estimate_bod_cod(df.copy())

    # 3) Merge
    merged = df.copy()

    if "predicted_water_quality" in quality_df.columns:
        merged["predicted_water_quality"] = quality_df["predicted_water_quality"]

    for col in quality_df.columns:
        if str(col).startswith("proba_"):
            merged[col] = quality_df[col]

    for col in ["estimated_dbo", "estimated_dco", "soft_sensing_confidence", "soft_sensing_available"]:
        if col in soft_df.columns:
            merged[col] = soft_df[col]

    # 4) Scores
    score_rows: List[Dict[str, Any]] = []
    for _, row in merged.iterrows():
        scores = compute_all_scores(row.to_dict())
        score_rows.append(scores)

    scores_df = pd.DataFrame(score_rows)
    for col in scores_df.columns:
        merged[col] = scores_df[col]

    saved_predictions: List[Dict[str, Any]] = []
    total_alerts_saved = 0

    # 5) Save predictions + latest state + alerts
    for _, row in merged.iterrows():
        station = str(row.get("station_id", "unknown"))
        payload = build_prediction_payload(row)

        saved_predictions.append(
            add_prediction(
                station_id=station,
                prediction_type="water_quality_and_soft_sensing",
                model_name=classification_model_name,
                result=payload,
            )
        )

        latest_patch = {
            "station_id": station,
            "last_timestamp": row.get("timestamp").isoformat() if pd.notna(row.get("timestamp")) else row.get("timestamp"),
            "latitude": row.get("latitude"),
            "longitude": row.get("longitude"),
            "temperature": _safe_float(row.get("temperature")),
            "ph": _safe_float(row.get("ph")),
            "turbidity": _safe_float(row.get("turbidity")),
            "dissolved_oxygen": _safe_float(row.get("dissolved_oxygen")),
            "salinity": _safe_float(row.get("salinity")),
            "conductivity": _safe_float(row.get("conductivity")),
            "ndwi": _safe_float(row.get("ndwi")),
            "ndti": _safe_float(row.get("ndti")),
            "ndci": _safe_float(row.get("ndci")),
            "chlorophyll": _safe_float(row.get("chlorophyll")),
            "dissolved_organic_matter": _safe_float(row.get("dissolved_organic_matter")),
            "suspended_matter": _safe_float(row.get("suspended_matter")),
            "predicted_water_quality": payload.get("predicted_water_quality"),
            "prediction_probabilities": payload.get("probabilities", {}),
            "estimated_dbo": payload.get("estimated_dbo"),
            "estimated_dco": payload.get("estimated_dco"),
            "soft_sensing_confidence": payload.get("soft_sensing_confidence"),
            "soft_sensing_available": payload.get("soft_sensing_available"),
            "water_quality_score": payload.get("water_quality_score"),
            "water_quality_label": payload.get("water_quality_label"),
            "risk_score": payload.get("risk_score"),
            "risk_level": payload.get("risk_level"),
            "data_reliability_score": payload.get("data_reliability_score"),
            "data_reliability_level": payload.get("data_reliability_level"),
            "source": row.get("source", "mqtt"),
            "quality_flag": row.get("quality_flag", "valid"),
            "validation_errors": row.get("validation_errors", []),
        }

        upsert_latest_station_state(station, latest_patch)

        total_alerts_saved += _save_alerts_from_row(latest_patch)

    return {
        "status": "ok",
        "rows_processed": len(merged),
        "predictions_saved": len(saved_predictions),
        "alerts_saved": total_alerts_saved,
        "model_used": classification_model_name,
    }