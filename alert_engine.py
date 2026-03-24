from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def _safe_float(value: Any):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _make_alert(
    station_id: str,
    alert_type: str,
    severity: str,
    message: str,
    recommendation: str,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "station_id": str(station_id),
        "alert_type": alert_type,
        "severity": severity,
        "message": message,
        "recommendation": recommendation,
        "metadata": metadata or {},
    }


# =========================================================
# 1) ALERTES MESURES CAPTEURS
# =========================================================
def evaluate_measurement_alerts(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []

    station_id = str(row.get("station_id", "unknown"))

    temperature = _safe_float(row.get("temperature"))
    ph = _safe_float(row.get("ph"))
    turbidity = _safe_float(row.get("turbidity"))
    dissolved_oxygen = _safe_float(row.get("dissolved_oxygen"))
    conductivity = _safe_float(row.get("conductivity"))
    salinity = _safe_float(row.get("salinity"))

    if ph is not None:
        if ph < 6.5 or ph > 8.5:
            alerts.append(
                _make_alert(
                    station_id,
                    "abnormal_ph",
                    "high",
                    f"pH anormal détecté ({ph:.2f}).",
                    "Vérifier le capteur pH et effectuer un prélèvement de confirmation.",
                    {"ph": ph},
                )
            )

    if dissolved_oxygen is not None:
        if dissolved_oxygen < 5:
            alerts.append(
                _make_alert(
                    station_id,
                    "low_dissolved_oxygen",
                    "critical",
                    f"Oxygène dissous faible ({dissolved_oxygen:.2f} mg/L).",
                    "Inspecter rapidement la zone et rechercher une charge organique excessive.",
                    {"dissolved_oxygen": dissolved_oxygen},
                )
            )
        elif dissolved_oxygen < 6:
            alerts.append(
                _make_alert(
                    station_id,
                    "moderate_dissolved_oxygen",
                    "medium",
                    f"Oxygène dissous modérément faible ({dissolved_oxygen:.2f} mg/L).",
                    "Surveiller l’évolution sur les prochaines heures.",
                    {"dissolved_oxygen": dissolved_oxygen},
                )
            )

    if turbidity is not None:
        if turbidity > 20:
            alerts.append(
                _make_alert(
                    station_id,
                    "high_turbidity",
                    "high",
                    f"Turbidité élevée ({turbidity:.2f} NTU).",
                    "Contrôler un apport sédimentaire, ruissellement ou remise en suspension.",
                    {"turbidity": turbidity},
                )
            )
        elif turbidity > 10:
            alerts.append(
                _make_alert(
                    station_id,
                    "moderate_turbidity",
                    "medium",
                    f"Turbidité modérée à élevée ({turbidity:.2f} NTU).",
                    "Comparer avec l’historique local et vérifier les conditions météo.",
                    {"turbidity": turbidity},
                )
            )

    if temperature is not None and dissolved_oxygen is not None:
        if temperature > 30 and dissolved_oxygen < 6:
            alerts.append(
                _make_alert(
                    station_id,
                    "thermal_oxygen_stress",
                    "high",
                    f"Stress thermique probable : température {temperature:.2f}°C et DO {dissolved_oxygen:.2f} mg/L.",
                    "Renforcer la surveillance, risque de dégradation rapide de la qualité.",
                    {"temperature": temperature, "dissolved_oxygen": dissolved_oxygen},
                )
            )

    if conductivity is not None and conductivity > 2000:
        alerts.append(
            _make_alert(
                station_id,
                "high_conductivity",
                "medium",
                f"Conductivité élevée ({conductivity:.2f}).",
                "Vérifier une augmentation de minéralisation ou une intrusion saline.",
                {"conductivity": conductivity},
            )
        )

    if salinity is not None and salinity > 5:
        alerts.append(
            _make_alert(
                station_id,
                "high_salinity",
                "medium",
                f"Salinité élevée ({salinity:.2f}).",
                "Comparer avec l’historique saisonnier et les observations satellite.",
                {"salinity": salinity},
            )
        )

    return alerts


# =========================================================
# 2) ALERTES SATELLITE (EXPERT 2)
# =========================================================
def evaluate_satellite_alerts(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []

    station_id = str(row.get("station_id", "unknown"))

    ndti = _safe_float(row.get("ndti"))
    ndci = _safe_float(row.get("ndci"))
    ndwi = _safe_float(row.get("ndwi"))
    chlorophyll = _safe_float(row.get("chlorophyll"))
    dissolved_organic_matter = _safe_float(row.get("dissolved_organic_matter"))

    if ndti is not None and ndti > 0.15:
        alerts.append(
            _make_alert(
                station_id,
                "satellite_high_turbidity_proxy",
                "medium",
                f"Proxy satellite de turbidité élevé (NDTI={ndti:.3f}).",
                "Comparer avec les capteurs in situ et vérifier les apports sédimentaires.",
                {"ndti": ndti},
            )
        )

    if ndci is not None and ndci > 0.20:
        alerts.append(
            _make_alert(
                station_id,
                "satellite_eutrophication_risk",
                "high",
                f"Risque d’eutrophisation détecté (NDCI={ndci:.3f}).",
                "Renforcer le suivi chlorophylle/oxygène dissous et envisager une inspection terrain.",
                {"ndci": ndci},
            )
        )

    if ndwi is not None and ndwi < 0.05:
        alerts.append(
            _make_alert(
                station_id,
                "low_water_presence",
                "medium",
                f"Faible présence d’eau détectée (NDWI={ndwi:.3f}).",
                "Vérifier un assèchement local ou une variabilité saisonnière importante.",
                {"ndwi": ndwi},
            )
        )

    if chlorophyll is not None and chlorophyll > 25:
        alerts.append(
            _make_alert(
                station_id,
                "high_chlorophyll",
                "high",
                f"Chlorophylle élevée ({chlorophyll:.2f}).",
                "Risque de prolifération algale, prévoir un suivi rapproché.",
                {"chlorophyll": chlorophyll},
            )
        )

    if dissolved_organic_matter is not None and dissolved_organic_matter > 10:
        alerts.append(
            _make_alert(
                station_id,
                "high_organic_matter_proxy",
                "medium",
                f"Matière organique dissoute élevée ({dissolved_organic_matter:.2f}).",
                "Croiser avec les estimations DBO/DCO et renforcer les analyses labo si besoin.",
                {"dissolved_organic_matter": dissolved_organic_matter},
            )
        )

    return alerts


# =========================================================
# 3) ALERTES IA / SOFT-SENSING / SCORING (EXPERT 3)
# =========================================================
def evaluate_ai_alerts(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []

    station_id = str(row.get("station_id", "unknown"))

    predicted_water_quality = row.get("predicted_water_quality")
    estimated_dbo = _safe_float(row.get("estimated_dbo"))
    estimated_dco = _safe_float(row.get("estimated_dco"))
    soft_sensing_confidence = _safe_float(row.get("soft_sensing_confidence"))

    # Ces champs viennent de scoring.py si on a enrichi latest_station_state
    water_quality_score = _safe_float(row.get("water_quality_score"))
    risk_score = _safe_float(row.get("risk_score"))
    risk_level = row.get("risk_level")

    if predicted_water_quality is not None and str(predicted_water_quality).lower() in ["poor", "critique", "dégradé", "bad"]:
        alerts.append(
            _make_alert(
                station_id,
                "predicted_poor_water_quality",
                "high",
                f"Le modèle IA prédit une qualité d’eau défavorable : {predicted_water_quality}.",
                "Vérifier la station, comparer à l’historique et confirmer par prélèvement terrain.",
                {"predicted_water_quality": predicted_water_quality},
            )
        )

    if estimated_dbo is not None:
        if estimated_dbo > 10:
            alerts.append(
                _make_alert(
                    station_id,
                    "high_estimated_dbo",
                    "high",
                    f"DBO estimée élevée ({estimated_dbo:.2f}).",
                    "Possible pollution organique importante, prévoir un contrôle laboratoire.",
                    {"estimated_dbo": estimated_dbo},
                )
            )
        elif estimated_dbo > 6:
            alerts.append(
                _make_alert(
                    station_id,
                    "moderate_estimated_dbo",
                    "medium",
                    f"DBO estimée modérément élevée ({estimated_dbo:.2f}).",
                    "Surveiller l’évolution et confirmer si la tendance persiste.",
                    {"estimated_dbo": estimated_dbo},
                )
            )

    if estimated_dco is not None:
        if estimated_dco > 80:
            alerts.append(
                _make_alert(
                    station_id,
                    "high_estimated_dco",
                    "high",
                    f"DCO estimée élevée ({estimated_dco:.2f}).",
                    "Possible charge chimique/organique importante, contrôle recommandé.",
                    {"estimated_dco": estimated_dco},
                )
            )
        elif estimated_dco > 40:
            alerts.append(
                _make_alert(
                    station_id,
                    "moderate_estimated_dco",
                    "medium",
                    f"DCO estimée modérément élevée ({estimated_dco:.2f}).",
                    "Vérifier les tendances et prévoir confirmation si nécessaire.",
                    {"estimated_dco": estimated_dco},
                )
            )

    if soft_sensing_confidence is not None and soft_sensing_confidence < 0.5:
        alerts.append(
            _make_alert(
                station_id,
                "low_soft_sensing_confidence",
                "low",
                f"Confiance faible dans l’estimation IA ({soft_sensing_confidence:.2f}).",
                "Interpréter les prédictions avec prudence et privilégier une validation terrain.",
                {"soft_sensing_confidence": soft_sensing_confidence},
            )
        )

    if risk_score is not None:
        if risk_score >= 75:
            alerts.append(
                _make_alert(
                    station_id,
                    "critical_risk_score",
                    "critical",
                    f"Score de risque critique ({risk_score:.2f}/100).",
                    "Déclencher une intervention prioritaire et consulter immédiatement le rapport station.",
                    {"risk_score": risk_score, "risk_level": risk_level},
                )
            )
        elif risk_score >= 50:
            alerts.append(
                _make_alert(
                    station_id,
                    "high_risk_score",
                    "high",
                    f"Score de risque élevé ({risk_score:.2f}/100).",
                    "Renforcer la surveillance et envisager une inspection terrain.",
                    {"risk_score": risk_score, "risk_level": risk_level},
                )
            )

    if water_quality_score is not None and water_quality_score < 35:
        alerts.append(
            _make_alert(
                station_id,
                "critical_water_quality_score",
                "critical",
                f"Score global de qualité très faible ({water_quality_score:.2f}/100).",
                "Situation critique, vérifier les mesures, consulter le diagnostic et agir rapidement.",
                {"water_quality_score": water_quality_score},
            )
        )

    return alerts


# =========================================================
# 4) ALERTES QUALITÉ DES DONNÉES
# =========================================================
def evaluate_data_quality_alerts(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    station_id = str(row.get("station_id", "unknown"))

    missing_fields = []
    for field in ["temperature", "ph", "turbidity", "dissolved_oxygen"]:
        val = row.get(field)
        if val is None or pd.isna(val):
            missing_fields.append(field)

    if missing_fields:
        alerts.append(
            _make_alert(
                station_id,
                "missing_sensor_fields",
                "low",
                f"Champs capteurs manquants : {', '.join(missing_fields)}.",
                "Vérifier la transmission MQTT et l’état de la station.",
                {"missing_fields": missing_fields},
            )
        )

    validation_errors = row.get("validation_errors", [])
    if validation_errors:
        alerts.append(
            _make_alert(
                station_id,
                "validation_errors_detected",
                "medium",
                f"Erreurs de validation détectées : {len(validation_errors)}.",
                "Contrôler la qualité des données entrantes et les capteurs concernés.",
                {"validation_errors": validation_errors},
            )
        )

    return alerts


# =========================================================
# 5) COHÉRENCE CAPTEUR / SATELLITE
# =========================================================
def evaluate_cross_source_alerts(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    station_id = str(row.get("station_id", "unknown"))

    turbidity = _safe_float(row.get("turbidity"))
    ndti = _safe_float(row.get("ndti"))

    chlorophyll = _safe_float(row.get("chlorophyll"))
    ndci = _safe_float(row.get("ndci"))

    if turbidity is not None and ndti is not None:
        if turbidity > 15 and ndti < 0.05:
            alerts.append(
                _make_alert(
                    station_id,
                    "sensor_satellite_turbidity_inconsistency",
                    "low",
                    "Incohérence entre turbidité capteur élevée et proxy satellite faible.",
                    "Vérifier le capteur, la date satellite utilisée et le contexte local.",
                    {"turbidity": turbidity, "ndti": ndti},
                )
            )

    if chlorophyll is not None and ndci is not None:
        if chlorophyll > 25 and ndci < 0.05:
            alerts.append(
                _make_alert(
                    station_id,
                    "sensor_satellite_chlorophyll_inconsistency",
                    "low",
                    "Incohérence entre chlorophylle élevée et signal satellite faible.",
                    "Contrôler la cohérence temporelle entre mesure terrain et image satellite.",
                    {"chlorophyll": chlorophyll, "ndci": ndci},
                )
            )

    return alerts


# =========================================================
# 6) ORCHESTRATION
# =========================================================
def evaluate_all_alerts(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    alerts.extend(evaluate_measurement_alerts(row))
    alerts.extend(evaluate_satellite_alerts(row))
    alerts.extend(evaluate_ai_alerts(row))
    alerts.extend(evaluate_data_quality_alerts(row))
    alerts.extend(evaluate_cross_source_alerts(row))
    return deduplicate_alerts(alerts)


def deduplicate_alerts(alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique_alerts = []

    for alert in alerts:
        key = (
            alert.get("station_id"),
            alert.get("alert_type"),
            alert.get("severity"),
            alert.get("message"),
        )
        if key not in seen:
            seen.add(key)
            unique_alerts.append(alert)

    return unique_alerts