from __future__ import annotations

from io import BytesIO
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def _prepare_timeseries(df: pd.DataFrame, station_id: str | None = None) -> pd.DataFrame:
    out = df.copy()

    if station_id is not None and "station_id" in out.columns:
        out = out[out["station_id"].astype(str) == str(station_id)].copy()

    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out = out.sort_values("timestamp")

    return out


def _fig_to_png_bytes(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_score_bar_chart(
    water_quality_score: float | None,
    risk_score: float | None,
    data_reliability_score: float | None,
) -> bytes:
    labels = ["Qualité", "Risque", "Fiabilité"]
    values = [
        0 if water_quality_score is None else float(water_quality_score),
        0 if risk_score is None else float(risk_score),
        0 if data_reliability_score is None else float(data_reliability_score),
    ]

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    bars = ax.bar(labels, values)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score / 100")
    ax.set_title("Scores principaux")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1,
            f"{val:.1f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    return _fig_to_png_bytes(fig)


def generate_station_timeseries_chart(
    df: pd.DataFrame,
    station_id: str,
    columns: List[str] | None = None,
) -> bytes | None:
    columns = columns or ["temperature", "ph", "turbidity", "dissolved_oxygen"]
    out = _prepare_timeseries(df, station_id=station_id)

    if out.empty or "timestamp" not in out.columns:
        return None

    valid_cols = [c for c in columns if c in out.columns]
    if not valid_cols:
        return None

    fig, ax = plt.subplots(figsize=(8, 4.5))

    plotted = False
    for col in valid_cols:
        series = pd.to_numeric(out[col], errors="coerce")
        if series.notna().sum() == 0:
            continue
        ax.plot(out["timestamp"], series, label=col)
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.set_title(f"Évolution temporelle — {station_id}")
    ax.set_xlabel("Temps")
    ax.set_ylabel("Valeur")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    return _fig_to_png_bytes(fig)


def generate_softsensing_chart(
    df: pd.DataFrame,
    station_id: str,
) -> bytes | None:
    out = _prepare_timeseries(df, station_id=station_id)

    if out.empty or "timestamp" not in out.columns:
        return None

    cols = [c for c in ["estimated_dbo", "estimated_dco"] if c in out.columns]
    if not cols:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))

    plotted = False
    for col in cols:
        series = pd.to_numeric(out[col], errors="coerce")
        if series.notna().sum() == 0:
            continue
        ax.plot(out["timestamp"], series, label=col)
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.set_title(f"Soft-Sensing DBO / DCO — {station_id}")
    ax.set_xlabel("Temps")
    ax.set_ylabel("Valeur estimée")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    return _fig_to_png_bytes(fig)