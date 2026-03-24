from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from firebase_admin import firestore

from firebase_config import init_firebase


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat()


def _clean_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {}
    for k, v in data.items():
        if isinstance(v, float) and pd.isna(v):
            cleaned[k] = None
        else:
            cleaned[k] = v
    return cleaned


# =========================================================
# STATIONS
# =========================================================
def upsert_station(
    station_id: str,
    name: str,
    latitude: float | None = None,
    longitude: float | None = None,
    description: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    db = init_firebase()

    payload = {
        "station_id": str(station_id),
        "name": name,
        "latitude": latitude,
        "longitude": longitude,
        "description": description,
        "metadata": metadata or {},
        "updated_at": _utc_now_iso(),
    }

    db.collection("stations").document(str(station_id)).set(_clean_payload(payload), merge=True)
    return payload


def get_station(station_id: str) -> Optional[Dict[str, Any]]:
    db = init_firebase()
    doc = db.collection("stations").document(str(station_id)).get()
    if not doc.exists:
        return None
    data = doc.to_dict()
    data["doc_id"] = doc.id
    return data


def get_all_stations() -> pd.DataFrame:
    db = init_firebase()
    docs = db.collection("stations").stream()

    rows: List[Dict[str, Any]] = []
    for doc in docs:
        item = doc.to_dict()
        item["doc_id"] = doc.id
        rows.append(item)

    return pd.DataFrame(rows)


# =========================================================
# RAW MEASUREMENTS
# =========================================================
def add_raw_measurement(data: Dict[str, Any]) -> Dict[str, Any]:
    db = init_firebase()

    payload = {
        "station_id": str(data.get("station_id", "unknown")),
        "temperature": data.get("temperature"),
        "ph": data.get("ph"),
        "turbidity": data.get("turbidity"),
        "dissolved_oxygen": data.get("dissolved_oxygen"),
        "salinity": data.get("salinity"),
        "conductivity": data.get("conductivity"),
        "timestamp": data.get("timestamp", _utc_now_iso()),
        "source": data.get("source", "mqtt"),
        "quality_flag": data.get("quality_flag", "raw"),
        "created_at": _utc_now_iso(),
        "metadata": data.get("metadata", {}),
    }

    ref = db.collection("raw_measurements").add(_clean_payload(payload))
    payload["doc_id"] = ref[1].id
    return payload


def add_raw_measurements_batch(rows: List[Dict[str, Any]]) -> int:
    db = init_firebase()
    batch = db.batch()

    count = 0
    for row in rows:
        payload = {
            "station_id": str(row.get("station_id", "unknown")),
            "temperature": row.get("temperature"),
            "ph": row.get("ph"),
            "turbidity": row.get("turbidity"),
            "dissolved_oxygen": row.get("dissolved_oxygen"),
            "salinity": row.get("salinity"),
            "conductivity": row.get("conductivity"),
            "timestamp": row.get("timestamp", _utc_now_iso()),
            "source": row.get("source", "mqtt"),
            "quality_flag": row.get("quality_flag", "raw"),
            "created_at": _utc_now_iso(),
            "metadata": row.get("metadata", {}),
        }
        ref = db.collection("raw_measurements").document()
        batch.set(ref, _clean_payload(payload))
        count += 1

    if count > 0:
        batch.commit()

    return count


def get_latest_raw_measurements(limit: int = 50, station_id: str | None = None) -> pd.DataFrame:
    db = init_firebase()

    query = db.collection("raw_measurements")
    if station_id:
        query = query.where("station_id", "==", str(station_id))

    docs = query.order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit).stream()

    rows: List[Dict[str, Any]] = []
    for doc in docs:
        item = doc.to_dict()
        item["doc_id"] = doc.id
        rows.append(item)

    return pd.DataFrame(rows)


# =========================================================
# LATEST STATION STATE
# =========================================================
def upsert_latest_station_state(
    station_id: str,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    db = init_firebase()

    payload = {
        "station_id": str(station_id),
        **state,
        "updated_at": _utc_now_iso(),
    }

    db.collection("latest_station_state").document(str(station_id)).set(_clean_payload(payload), merge=True)
    return payload


def get_latest_station_state(station_id: str) -> Optional[Dict[str, Any]]:
    db = init_firebase()
    doc = db.collection("latest_station_state").document(str(station_id)).get()
    if not doc.exists:
        return None
    data = doc.to_dict()
    data["doc_id"] = doc.id
    return data


def get_all_latest_station_states() -> pd.DataFrame:
    db = init_firebase()
    docs = db.collection("latest_station_state").stream()

    rows: List[Dict[str, Any]] = []
    for doc in docs:
        item = doc.to_dict()
        item["doc_id"] = doc.id
        rows.append(item)

    return pd.DataFrame(rows)


# =========================================================
# PREDICTIONS
# =========================================================
def add_prediction(
    station_id: str,
    prediction_type: str,
    model_name: str,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    db = init_firebase()

    payload = {
        "station_id": str(station_id),
        "prediction_type": prediction_type,
        "model_name": model_name,
        "result": result,
        "timestamp": _utc_now_iso(),
    }

    ref = db.collection("predictions").add(_clean_payload(payload))
    payload["doc_id"] = ref[1].id
    return payload


def get_latest_predictions(limit: int = 50, station_id: str | None = None) -> pd.DataFrame:
    db = init_firebase()

    query = db.collection("predictions")
    if station_id:
        query = query.where("station_id", "==", str(station_id))

    docs = query.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit).stream()

    rows: List[Dict[str, Any]] = []
    for doc in docs:
        item = doc.to_dict()
        item["doc_id"] = doc.id
        rows.append(item)

    return pd.DataFrame(rows)


# =========================================================
# ALERTS
# =========================================================
def add_alert(
    station_id: str,
    alert_type: str,
    severity: str,
    message: str,
    recommendation: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    db = init_firebase()

    payload = {
        "station_id": str(station_id),
        "alert_type": alert_type,
        "severity": severity,
        "message": message,
        "recommendation": recommendation,
        "metadata": metadata or {},
        "status": "open",
        "timestamp": _utc_now_iso(),
    }

    ref = db.collection("alerts").add(_clean_payload(payload))
    payload["doc_id"] = ref[1].id
    return payload


def get_latest_alerts(limit: int = 50, station_id: str | None = None, only_open: bool = False) -> pd.DataFrame:
    db = init_firebase()

    query = db.collection("alerts")
    if station_id:
        query = query.where("station_id", "==", str(station_id))
    if only_open:
        query = query.where("status", "==", "open")

    docs = query.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit).stream()

    rows: List[Dict[str, Any]] = []
    for doc in docs:
        item = doc.to_dict()
        item["doc_id"] = doc.id
        rows.append(item)

    return pd.DataFrame(rows)


def close_alert(alert_doc_id: str) -> None:
    db = init_firebase()
    db.collection("alerts").document(alert_doc_id).set(
        {
            "status": "closed",
            "closed_at": _utc_now_iso(),
        },
        merge=True,
    )


# =========================================================
# DAILY AGGREGATES
# =========================================================
def upsert_daily_aggregate(
    station_id: str,
    date_str: str,
    aggregate: Dict[str, Any],
) -> Dict[str, Any]:
    db = init_firebase()

    doc_id = f"{station_id}_{date_str}"
    payload = {
        "station_id": str(station_id),
        "date": date_str,
        **aggregate,
        "updated_at": _utc_now_iso(),
    }

    db.collection("daily_aggregates").document(doc_id).set(_clean_payload(payload), merge=True)
    return payload


def get_daily_aggregates(limit: int = 30, station_id: str | None = None) -> pd.DataFrame:
    db = init_firebase()

    query = db.collection("daily_aggregates")
    if station_id:
        query = query.where("station_id", "==", str(station_id))

    docs = query.order_by("date", direction=firestore.Query.DESCENDING).limit(limit).stream()

    rows: List[Dict[str, Any]] = []
    for doc in docs:
        item = doc.to_dict()
        item["doc_id"] = doc.id
        rows.append(item)

    return pd.DataFrame(rows)