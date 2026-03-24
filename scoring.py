from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def _safe_float(value: Any):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def score_label(score: float) -> str:
    if score >= 80:
        return "Excellent"
    if score >= 65:
        return "Good"
    if score >= 50:
        return "Moderate"
    if score >= 35:
        return "Poor"
    return "Very Poor"


def score_water_quality(row: Dict[str, Any]) -> Dict[str, Any]:
    ph = _safe_float(row.get("ph"))
    turbidity = _safe_float(row.get("turbidity"))
    dissolved_oxygen = _safe_float(row.get("dissolved_oxygen"))
    salinity = _safe_float(row.get("salinity"))
    temperature = _safe_float(row.get("temperature"))
    estimated_dbo = _safe_float(row.get("estimated_dbo"))
    estimated_dco = _safe_float(row.get("estimated_dco"))
    ndci = _safe_float(row.get("ndci"))
    ndti = _safe_float(row.get("ndti"))

    if ph is None:
        ph_score = 0.5
    elif 6.5 <= ph <= 8.5:
        ph_score = 1.0
    elif 6.0 <= ph < 6.5 or 8.5 < ph <= 9.0:
        ph_score = 0.6
    else:
        ph_score = 0.2

    if dissolved_oxygen is None:
        do_score = 0.5
    elif dissolved_oxygen >= 7:
        do_score = 1.0
    elif dissolved_oxygen >= 6:
        do_score = 0.8
    elif dissolved_oxygen >= 5:
        do_score = 0.5
    else:
        do_score = 0.2

    if turbidity is None:
        turbidity_score = 0.5
    elif turbidity <= 5:
        turbidity_score = 1.0
    elif turbidity <= 10:
        turbidity_score = 0.8
    elif turbidity <= 20:
        turbidity_score = 0.5
    else:
        turbidity_score = 0.2

    if temperature is None:
        temp_score = 0.5
    elif 10 <= temperature <= 28:
        temp_score = 1.0
    elif 28 < temperature <= 32:
        temp_score = 0.7
    else:
        temp_score = 0.4

    if salinity is None:
        salinity_score = 0.5
    elif salinity <= 1:
        salinity_score = 1.0
    elif salinity <= 3:
        salinity_score = 0.7
    elif salinity <= 5:
        salinity_score = 0.5
    else:
        salinity_score = 0.2

    if estimated_dbo is None:
        dbo_score = 0.5
    elif estimated_dbo <= 3:
        dbo_score = 1.0
    elif estimated_dbo <= 6:
        dbo_score = 0.7
    elif estimated_dbo <= 10:
        dbo_score = 0.4
    else:
        dbo_score = 0.2

    if estimated_dco is None:
        dco_score = 0.5
    elif estimated_dco <= 20:
        dco_score = 1.0
    elif estimated_dco <= 40:
        dco_score = 0.7
    elif estimated_dco <= 80:
        dco_score = 0.4
    else:
        dco_score = 0.2

    if ndci is None:
        ndci_score = 0.5
    elif ndci <= 0.1:
        ndci_score = 1.0
    elif ndci <= 0.2:
        ndci_score = 0.7
    else:
        ndci_score = 0.3

    if ndti is None:
        ndti_score = 0.5
    elif ndti <= 0.05:
        ndti_score = 1.0
    elif ndti <= 0.15:
        ndti_score = 0.7
    else:
        ndti_score = 0.3

    weights = {
        "ph": 0.10,
        "do": 0.20,
        "turbidity": 0.15,
        "temperature": 0.10,
        "salinity": 0.05,
        "dbo": 0.15,
        "dco": 0.15,
        "ndci": 0.05,
        "ndti": 0.05,
    }

    score = 100 * (
        weights["ph"] * ph_score
        + weights["do"] * do_score
        + weights["turbidity"] * turbidity_score
        + weights["temperature"] * temp_score
        + weights["salinity"] * salinity_score
        + weights["dbo"] * dbo_score
        + weights["dco"] * dco_score
        + weights["ndci"] * ndci_score
        + weights["ndti"] * ndti_score
    )

    score = round(score, 2)

    return {
        "water_quality_score": score,
        "water_quality_label": score_label(score),
        "details": {
            "ph_score": ph_score,
            "do_score": do_score,
            "turbidity_score": turbidity_score,
            "temperature_score": temp_score,
            "salinity_score": salinity_score,
            "dbo_score": dbo_score,
            "dco_score": dco_score,
            "ndci_score": ndci_score,
            "ndti_score": ndti_score,
        },
    }


def score_data_reliability(row: Dict[str, Any]) -> Dict[str, Any]:
    fields = [
        "temperature",
        "ph",
        "turbidity",
        "dissolved_oxygen",
        "salinity",
        "conductivity",
    ]

    present = 0
    for f in fields:
        v = row.get(f)
        if v is not None and not pd.isna(v):
            present += 1

    completeness = present / len(fields)

    penalty = 0.0
    for flag_col in [
        "flag_temperature_outlier",
        "flag_ph_outlier",
        "flag_do_outlier",
        "flag_turbidity_outlier",
        "flag_conductivity_outlier",
    ]:
        val = row.get(flag_col, 0)
        try:
            penalty += 0.10 * int(val)
        except Exception:
            pass

    score = 100 * clamp01(completeness - penalty)

    if score >= 85:
        label = "High"
    elif score >= 65:
        label = "Medium"
    else:
        label = "Low"

    return {
        "data_reliability_score": round(score, 2),
        "data_reliability_level": label,
        "details_reliability": {
            "completeness_ratio": round(completeness, 3),
            "penalty": round(penalty, 3),
        },
    }


def compute_risk_score(row: Dict[str, Any]) -> Dict[str, Any]:
    turbidity = _safe_float(row.get("turbidity"))
    dissolved_oxygen = _safe_float(row.get("dissolved_oxygen"))
    ndci = _safe_float(row.get("ndci"))
    ndti = _safe_float(row.get("ndti"))
    estimated_dbo = _safe_float(row.get("estimated_dbo"))
    estimated_dco = _safe_float(row.get("estimated_dco"))
    chlorophyll = _safe_float(row.get("chlorophyll"))

    risk = 0.0

    if turbidity is not None:
        if turbidity > 20:
            risk += 25
        elif turbidity > 10:
            risk += 15

    if dissolved_oxygen is not None:
        if dissolved_oxygen < 5:
            risk += 30
        elif dissolved_oxygen < 6:
            risk += 15

    if ndci is not None:
        if ndci > 0.20:
            risk += 20
        elif ndci > 0.10:
            risk += 10

    if ndti is not None:
        if ndti > 0.15:
            risk += 15
        elif ndti > 0.08:
            risk += 8

    if chlorophyll is not None:
        if chlorophyll > 25:
            risk += 15
        elif chlorophyll > 10:
            risk += 8

    if estimated_dbo is not None:
        if estimated_dbo > 10:
            risk += 20
        elif estimated_dbo > 6:
            risk += 10

    if estimated_dco is not None:
        if estimated_dco > 80:
            risk += 20
        elif estimated_dco > 40:
            risk += 10

    risk = min(100.0, risk)

    if risk >= 75:
        level = "Critical"
    elif risk >= 50:
        level = "High"
    elif risk >= 25:
        level = "Medium"
    else:
        level = "Low"

    return {
        "risk_score": round(risk, 2),
        "risk_level": level,
        "eutrophication_risk_score": round(risk, 2),
        "eutrophication_risk_label": level,
    }


def compute_all_scores(row: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    out.update(score_water_quality(row))
    out.update(score_data_reliability(row))
    out.update(compute_risk_score(row))
    return out