from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from scoring import compute_all_scores


def _safe_float(value: Any):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def generate_diagnostic_messages(row: Dict[str, Any]) -> List[str]:
    messages: List[str] = []

    temperature = _safe_float(row.get("temperature"))
    ph = _safe_float(row.get("ph"))
    turbidity = _safe_float(row.get("turbidity"))
    dissolved_oxygen = _safe_float(row.get("dissolved_oxygen"))
    estimated_dbo = _safe_float(row.get("estimated_dbo"))
    estimated_dco = _safe_float(row.get("estimated_dco"))
    ndci = _safe_float(row.get("ndci"))
    ndti = _safe_float(row.get("ndti"))
    ndwi = _safe_float(row.get("ndwi"))
    chlorophyll = _safe_float(row.get("chlorophyll"))

    if dissolved_oxygen is not None and dissolved_oxygen < 5:
        messages.append("Oxygène dissous faible : suspicion de charge organique élevée ou dégradation biologique.")

    if turbidity is not None and turbidity > 20:
        messages.append("Turbidité élevée : possible ruissellement, remise en suspension ou pollution particulaire.")

    if ph is not None and (ph < 6.5 or ph > 8.5):
        messages.append("pH anormal : déséquilibre chimique possible, vérification terrain recommandée.")

    if estimated_dbo is not None and estimated_dbo > 6:
        messages.append("DBO estimée élevée : charge biodégradable potentiellement importante.")

    if estimated_dco is not None and estimated_dco > 40:
        messages.append("DCO estimée élevée : présence possible de matière organique ou polluants oxydables.")

    if ndci is not None and ndci > 0.2:
        messages.append("NDCI élevé : risque d’eutrophisation ou de prolifération algale.")

    if chlorophyll is not None and chlorophyll > 25:
        messages.append("Chlorophylle élevée : activité algale importante probable.")

    if ndti is not None and ndti > 0.15:
        messages.append("NDTI élevé : proxy satellite de turbidité élevé, cohérent avec possible charge sédimentaire.")

    if ndwi is not None and ndwi < 0.05:
        messages.append("NDWI faible : possible réduction locale de la présence d’eau ou assèchement relatif.")

    if temperature is not None and dissolved_oxygen is not None:
        if temperature > 30 and dissolved_oxygen < 6:
            messages.append("Température élevée combinée à un DO faible : stress écologique probable.")

    if not messages:
        messages.append("Aucune anomalie majeure détectée à partir des paramètres disponibles.")

    return messages


def generate_recommendations(row: Dict[str, Any]) -> List[str]:
    recs: List[str] = []

    turbidity = _safe_float(row.get("turbidity"))
    dissolved_oxygen = _safe_float(row.get("dissolved_oxygen"))
    estimated_dbo = _safe_float(row.get("estimated_dbo"))
    estimated_dco = _safe_float(row.get("estimated_dco"))
    ndci = _safe_float(row.get("ndci"))
    validation_errors = row.get("validation_errors", []) or []

    if dissolved_oxygen is not None and dissolved_oxygen < 5:
        recs.append("Déclencher un contrôle terrain rapide et vérifier la présence de rejets organiques.")

    if turbidity is not None and turbidity > 20:
        recs.append("Inspecter les apports sédimentaires et comparer avec la météo récente.")

    if estimated_dbo is not None and estimated_dbo > 6:
        recs.append("Programmer un prélèvement laboratoire pour confirmer la DBO.")

    if estimated_dco is not None and estimated_dco > 40:
        recs.append("Programmer une analyse DCO en laboratoire pour validation.")

    if ndci is not None and ndci > 0.2:
        recs.append("Renforcer le suivi chlorophylle/oxygène et surveiller le risque d’eutrophisation.")

    if validation_errors:
        recs.append("Vérifier les capteurs ou la chaîne de transmission car des anomalies de qualité de données existent.")

    if not recs:
        recs.append("Poursuivre la surveillance normale et suivre l’évolution temporelle.")

    return recs


def build_station_diagnostic(row: Dict[str, Any]) -> Dict[str, Any]:
    scores = compute_all_scores(row)
    messages = generate_diagnostic_messages(row)
    recommendations = generate_recommendations(row)

    station_id = str(row.get("station_id", "unknown"))

    summary = (
        f"Station {station_id} — qualité {scores['water_quality_label']} "
        f"(score={scores['water_quality_score']}/100), "
        f"risque {scores['risk_level']} "
        f"(score={scores['risk_score']}/100), "
        f"fiabilité données {scores['data_reliability_level']} "
        f"({scores['data_reliability_score']}/100)."
    )

    return {
        "station_id": station_id,
        "summary": summary,
        "scores": scores,
        "diagnostic_messages": messages,
        "recommendations": recommendations,
    }


def build_diagnostics_for_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        diag = build_station_diagnostic(row.to_dict())
        rows.append(
            {
                "station_id": diag["station_id"],
                "summary": diag["summary"],
                "water_quality_score": diag["scores"]["water_quality_score"],
                "water_quality_label": diag["scores"]["water_quality_label"],
                "risk_score": diag["scores"]["risk_score"],
                "risk_level": diag["scores"]["risk_level"],
                "data_reliability_score": diag["scores"]["data_reliability_score"],
                "data_reliability_level": diag["scores"]["data_reliability_level"],
                "diagnostic_messages": " | ".join(diag["diagnostic_messages"]),
                "recommendations": " | ".join(diag["recommendations"]),
            }
        )

    return pd.DataFrame(rows)