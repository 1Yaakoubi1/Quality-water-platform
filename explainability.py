from __future__ import annotations

from typing import Dict, Any, List


# =========================================================
# IMPORTANCE SIMPLE (HEURISTIQUE)
# =========================================================
def compute_feature_importance(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    features = []

    def add(name, value, weight):
        if value is not None:
            try:
                v = float(value)
                features.append({
                    "feature": name,
                    "value": v,
                    "impact": abs(v * weight),
                })
            except:
                pass

    add("turbidity", row.get("turbidity"), 2.0)
    add("ph", row.get("ph"), 1.5)
    add("temperature", row.get("temperature"), 1.0)
    add("dissolved_oxygen", row.get("dissolved_oxygen"), -2.0)
    add("estimated_dbo", row.get("estimated_dbo"), 2.5)
    add("estimated_dco", row.get("estimated_dco"), 2.0)

    features = sorted(features, key=lambda x: x["impact"], reverse=True)
    return features[:5]


# =========================================================
# EXPLICATION TEXTE
# =========================================================
def generate_explanation_text(row: Dict[str, Any]) -> str:
    important = compute_feature_importance(row)

    lines = []
    lines.append("Explication de la qualité de l’eau :")

    for f in important:
        name = f["feature"]
        value = f["value"]

        if name == "turbidity":
            lines.append(f"- Turbidité élevée ({value}) → dégradation de la qualité")
        elif name == "ph":
            lines.append(f"- pH ({value}) influence l’équilibre chimique")
        elif name == "dissolved_oxygen":
            lines.append(f"- Oxygène dissous ({value}) → impact écologique")
        elif name == "estimated_dbo":
            lines.append(f"- DBO estimée élevée ({value}) → pollution organique")
        elif name == "estimated_dco":
            lines.append(f"- DCO estimée élevée ({value}) → pollution chimique")
        else:
            lines.append(f"- {name} = {value}")

    return "\n".join(lines)