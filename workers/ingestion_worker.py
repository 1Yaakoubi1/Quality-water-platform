from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from alert_engine import deduplicate_alerts, evaluate_all_alerts
from firebase_service import (
    add_alert,
    add_raw_measurement,
    upsert_latest_station_state,
)
from alert_engine import evaluate_all_alerts
from firebase_service import add_alert

alerts = evaluate_all_alerts(payload)
for a in alerts:
    add_alert(
        station_id=a["station_id"],
        alert_type=a["alert_type"],
        severity=a["severity"],
        message=a["message"],
        recommendation=a["recommendation"],
        metadata=a.get("metadata", {}),
    )

REQUIRED_MIN_FIELDS = ["station_id"]
NUMERIC_FIELDS = [
    "temperature",
    "ph",
    "turbidity",
    "dissolved_oxygen",
    "salinity",
    "conductivity",
    "ndwi",
    "ndti",
    "ndci",
    "chlorophyll",
    "dissolved_organic_matter",
    "suspended_matter",
]


def _safe_float(value: Any):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload)

    if "station_id" not in out or out["station_id"] in [None, ""]:
        out["station_id"] = "unknown"

    out["station_id"] = str(out["station_id"])

    if "timestamp" not in out or not out["timestamp"]:
        out["timestamp"] = datetime.utcnow().isoformat()

    for field in NUMERIC_FIELDS:
        if field in out:
            out[field] = _safe_float(out[field])

    if "source" not in out:
        out["source"] = "mqtt"

    if "quality_flag" not in out:
        out["quality_flag"] = "raw"

    if "metadata" not in out or not isinstance(out.get("metadata"), dict):
        out["metadata"] = {}

    return out


def validate_payload(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    for field in REQUIRED_MIN_FIELDS:
        if field not in payload or payload[field] in [None, ""]:
            errors.append(f"Champ requis manquant : {field}")

    ph = payload.get("ph")
    if ph is not None and not (0 <= ph <= 14):
        errors.append("pH hors plage [0, 14]")

    temperature = payload.get("temperature")
    if temperature is not None and not (-5 <= temperature <= 60):
        errors.append("Température hors plage [-5, 60]")

    dissolved_oxygen = payload.get("dissolved_oxygen")
    if dissolved_oxygen is not None and not (0 <= dissolved_oxygen <= 25):
        errors.append("Oxygène dissous hors plage [0, 25]")

    turbidity = payload.get("turbidity")
    if turbidity is not None and turbidity < 0:
        errors.append("Turbidité négative")

    conductivity = payload.get("conductivity")
    if conductivity is not None and conductivity < 0:
        errors.append("Conductivité négative")

    return errors


def process_measurement(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Traite une mesure unitaire :
    - normalisation
    - validation
    - stockage brut
    - mise à jour latest state
    - génération alertes
    """
    normalized = normalize_payload(payload)
    errors = validate_payload(normalized)

    quality_flag = "valid" if not errors else "invalid"
    normalized["quality_flag"] = quality_flag

    if errors:
        normalized["metadata"] = {
            **normalized.get("metadata", {}),
            "validation_errors": errors,
        }

    saved_measurement = add_raw_measurement(normalized)

    latest_state = {
        "last_timestamp": normalized.get("timestamp"),
        "temperature": normalized.get("temperature"),
        "ph": normalized.get("ph"),
        "turbidity": normalized.get("turbidity"),
        "dissolved_oxygen": normalized.get("dissolved_oxygen"),
        "salinity": normalized.get("salinity"),
        "conductivity": normalized.get("conductivity"),
        "ndwi": normalized.get("ndwi"),
        "ndti": normalized.get("ndti"),
        "ndci": normalized.get("ndci"),
        "chlorophyll": normalized.get("chlorophyll"),
        "dissolved_organic_matter": normalized.get("dissolved_organic_matter"),
        "suspended_matter": normalized.get("suspended_matter"),
        "source": normalized.get("source", "mqtt"),
        "quality_flag": quality_flag,
        "last_measurement_doc_id": saved_measurement.get("doc_id"),
        "validation_errors": errors,
    }

    upsert_latest_station_state(normalized["station_id"], latest_state)

    alerts = deduplicate_alerts(evaluate_all_alerts(normalized))
    saved_alerts = []

    for alert in alerts:
        saved_alerts.append(
            add_alert(
                station_id=alert["station_id"],
                alert_type=alert["alert_type"],
                severity=alert["severity"],
                message=alert["message"],
                recommendation=alert.get("recommendation", ""),
                metadata=alert.get("metadata", {}),
            )
        )

    return {
        "status": "ok",
        "measurement": saved_measurement,
        "alerts_count": len(saved_alerts),
        "alerts": saved_alerts,
        "validation_errors": errors,
    }