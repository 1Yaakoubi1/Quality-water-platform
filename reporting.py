from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Dict, List, Any
import os
import tempfile

import pandas as pd

from diagnostics import build_station_diagnostic
from charts import generate_score_bar_chart, generate_station_timeseries_chart, generate_softsensing_chart
from gee_reporting import export_station_satellite_png, compute_satellite_summary, get_station_coords_and_dates


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        pass
    return str(value)


def _safe_station_id(row: Dict[str, Any]) -> str:
    return str(row.get("station_id", "unknown"))


# =========================================================
# TEXT REPORTS
# =========================================================
def generate_station_report_text(row: Dict[str, Any]) -> str:
    diag = build_station_diagnostic(row)

    scores = diag["scores"]
    messages = diag["diagnostic_messages"]
    recommendations = diag["recommendations"]

    lines: List[str] = []
    lines.append("CLEANWATER — RAPPORT STATION")
    lines.append("=" * 60)
    lines.append(f"Date génération : {datetime.utcnow().isoformat()} UTC")
    lines.append(f"Station : {_safe_station_id(row)}")
    lines.append("")

    lines.append("MESURES DISPONIBLES")
    lines.append("-" * 60)
    for key in [
        "last_timestamp",
        "latitude",
        "longitude",
        "temperature",
        "ph",
        "turbidity",
        "dissolved_oxygen",
        "salinity",
        "conductivity",
        "predicted_water_quality",
        "estimated_dbo",
        "estimated_dco",
        "soft_sensing_confidence",
        "ndwi",
        "ndti",
        "ndci",
        "chlorophyll",
        "dissolved_organic_matter",
        "suspended_matter",
    ]:
        lines.append(f"{key}: {_fmt(row.get(key))}")

    lines.append("")
    lines.append("SCORES")
    lines.append("-" * 60)
    lines.append(f"Water quality score : {scores['water_quality_score']}/100")
    lines.append(f"Water quality label : {scores['water_quality_label']}")
    lines.append(f"Risk score          : {scores['risk_score']}/100")
    lines.append(f"Risk level          : {scores['risk_level']}")
    lines.append(f"Data reliability    : {scores['data_reliability_score']}/100")
    lines.append(f"Reliability level   : {scores['data_reliability_level']}")

    lines.append("")
    lines.append("SYNTHÈSE")
    lines.append("-" * 60)
    lines.append(diag["summary"])

    lines.append("")
    lines.append("DIAGNOSTIC")
    lines.append("-" * 60)
    for msg in messages:
        lines.append(f"- {msg}")

    lines.append("")
    lines.append("RECOMMANDATIONS")
    lines.append("-" * 60)
    for rec in recommendations:
        lines.append(f"- {rec}")

    return "\n".join(lines)


def generate_multi_station_summary(df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("CLEANWATER — RAPPORT GLOBAL")
    lines.append("=" * 70)
    lines.append(f"Date génération : {datetime.utcnow().isoformat()} UTC")
    lines.append(f"Nombre de stations : {len(df)}")
    lines.append("")

    if df.empty:
        lines.append("Aucune donnée disponible.")
        return "\n".join(lines)

    for _, row in df.iterrows():
        diag = build_station_diagnostic(row.to_dict())
        lines.append(diag["summary"])
        lines.append("Messages clés :")
        for msg in diag["diagnostic_messages"][:4]:
            lines.append(f"  - {msg}")
        lines.append("Recommandations :")
        for rec in diag["recommendations"][:3]:
            lines.append(f"  - {rec}")
        lines.append("")

    return "\n".join(lines)


# =========================================================
# PDF HELPERS
# =========================================================
def _build_pdf_styles():
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "cw_title",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        spaceAfter=12,
    )

    subtitle_style = ParagraphStyle(
        "cw_subtitle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        spaceBefore=8,
        spaceAfter=6,
    )

    normal_style = ParagraphStyle(
        "cw_normal",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        spaceAfter=4,
    )

    small_style = ParagraphStyle(
        "cw_small",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        spaceAfter=3,
    )

    return title_style, subtitle_style, normal_style, small_style


def _pdf_header_footer(canvas, doc):
    from reportlab.lib.colors import HexColor

    canvas.saveState()
    width, height = doc.pagesize

    canvas.setStrokeColor(HexColor("#1f4e79"))
    canvas.setLineWidth(1)
    canvas.line(doc.leftMargin, height - 20, width - doc.rightMargin, height - 20)

    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(doc.leftMargin, height - 15, "CleanWater")

    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(width - doc.rightMargin, height - 15, datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))

    canvas.setFont("Helvetica", 8)
    canvas.drawString(doc.leftMargin, 15, "Rapport généré automatiquement par CleanWater")
    canvas.drawRightString(width - doc.rightMargin, 15, f"Page {doc.page}")

    canvas.restoreState()


def _station_measures_table(row: Dict[str, Any]):
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    data = [["Paramètre", "Valeur"]]
    for key in [
        "last_timestamp",
        "latitude",
        "longitude",
        "temperature",
        "ph",
        "turbidity",
        "dissolved_oxygen",
        "salinity",
        "conductivity",
        "predicted_water_quality",
        "estimated_dbo",
        "estimated_dco",
        "soft_sensing_confidence",
        "ndwi",
        "ndti",
        "ndci",
        "chlorophyll",
        "dissolved_organic_matter",
        "suspended_matter",
    ]:
        data.append([key, _fmt(row.get(key))])

    table = Table(data, colWidths=[190, 220])
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#f3f6f9")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    return table


def _station_scores_table(scores: Dict[str, Any]):
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    data = [
        ["Indicateur", "Valeur"],
        ["Water quality score", f"{scores['water_quality_score']}/100"],
        ["Water quality label", f"{scores['water_quality_label']}"],
        ["Risk score", f"{scores['risk_score']}/100"],
        ["Risk level", f"{scores['risk_level']}"],
        ["Data reliability score", f"{scores['data_reliability_score']}/100"],
        ["Data reliability level", f"{scores['data_reliability_level']}"],
    ]

    table = Table(data, colWidths=[190, 220])
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2e7d32")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#eef7ee")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    return table


def _write_temp_image(image_bytes: bytes, suffix: str = ".png") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(image_bytes)
    tmp.flush()
    tmp.close()
    return tmp.name


# =========================================================
# PDF REPORTS
# =========================================================
def generate_station_report_pdf(row: Dict[str, Any], history_df: pd.DataFrame | None = None) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image

    diag = build_station_diagnostic(row)
    scores = diag["scores"]

    title_style, subtitle_style, normal_style, small_style = _build_pdf_styles()

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=2.0 * cm,
        bottomMargin=1.8 * cm,
    )

    story = []
    temp_files: List[str] = []

    # Logo optionnel
    logo_path = os.path.join("assets", "logo.png")
    if os.path.exists(logo_path):
        try:
            story.append(Image(logo_path, width=2.8 * cm, height=2.8 * cm))
            story.append(Spacer(1, 4))
        except Exception:
            pass

    story.append(Paragraph("CleanWater — Rapport Station", title_style))
    story.append(Paragraph(f"<b>Station :</b> {_safe_station_id(row)}", small_style))
    story.append(Paragraph(f"<b>Date génération :</b> {datetime.utcnow().isoformat()} UTC", small_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("1. Synthèse", subtitle_style))
    story.append(Paragraph(diag["summary"], normal_style))
    story.append(Spacer(1, 8))

    # Graphique des scores
    try:
        score_chart = generate_score_bar_chart(
            scores.get("water_quality_score"),
            scores.get("risk_score"),
            scores.get("data_reliability_score"),
        )
        score_chart_path = _write_temp_image(score_chart)
        temp_files.append(score_chart_path)

        story.append(Paragraph("2. Visualisation des scores", subtitle_style))
        story.append(Image(score_chart_path, width=14 * cm, height=7 * cm))
        story.append(Spacer(1, 8))
    except Exception:
        pass

    # Graphique temporel capteurs
    if history_df is not None and not history_df.empty:
        try:
            ts_chart = generate_station_timeseries_chart(history_df, _safe_station_id(row))
            if ts_chart:
                ts_chart_path = _write_temp_image(ts_chart)
                temp_files.append(ts_chart_path)

                story.append(Paragraph("3. Évolution temporelle des capteurs", subtitle_style))
                story.append(Image(ts_chart_path, width=15 * cm, height=8 * cm))
                story.append(Spacer(1, 8))
        except Exception:
            pass

        try:
            ss_chart = generate_softsensing_chart(history_df, _safe_station_id(row))
            if ss_chart:
                ss_chart_path = _write_temp_image(ss_chart)
                temp_files.append(ss_chart_path)

                story.append(Paragraph("4. Évolution DBO / DCO estimées", subtitle_style))
                story.append(Image(ss_chart_path, width=15 * cm, height=8 * cm))
                story.append(Spacer(1, 8))
        except Exception:
            pass

    # Sentinel dans le PDF
    lat, lon, start_date, end_date = get_station_coords_and_dates(row)
    if lat is not None and lon is not None:
        try:
            rgb_path = export_station_satellite_png(
                latitude=lat,
                longitude=lon,
                start_date=start_date,
                end_date=end_date,
                layer="RGB",
            )
            if rgb_path:
                temp_files.append(rgb_path)
                story.append(Paragraph("5. Vue Sentinel-2 (RGB)", subtitle_style))
                story.append(Image(rgb_path, width=15 * cm, height=15 * cm))
                story.append(Spacer(1, 8))
        except Exception:
            pass

        try:
            ndwi_path = export_station_satellite_png(
                latitude=lat,
                longitude=lon,
                start_date=start_date,
                end_date=end_date,
                layer="NDWI",
            )
            if ndwi_path:
                temp_files.append(ndwi_path)
                story.append(Paragraph("6. Vue satellite — NDWI", subtitle_style))
                story.append(Image(ndwi_path, width=15 * cm, height=15 * cm))
                story.append(Spacer(1, 8))
        except Exception:
            pass

        try:
            sat_summary = compute_satellite_summary(
                latitude=lat,
                longitude=lon,
                start_date=start_date,
                end_date=end_date,
            )
            story.append(Paragraph("7. Analyse satellite automatique", subtitle_style))

            if sat_summary.get("image_found"):
                if sat_summary.get("ndwi_mean") is not None:
                    story.append(Paragraph(f"NDWI moyen : {sat_summary['ndwi_mean']:.3f}", normal_style))
                if sat_summary.get("ndti_mean") is not None:
                    story.append(Paragraph(f"NDTI moyen : {sat_summary['ndti_mean']:.3f}", normal_style))
                if sat_summary.get("ndci_mean") is not None:
                    story.append(Paragraph(f"NDCI moyen : {sat_summary['ndci_mean']:.3f}", normal_style))

                for msg in sat_summary.get("summary_messages", []):
                    story.append(Paragraph(f"• {msg}", normal_style))
            else:
                story.append(Paragraph(sat_summary.get("message", "Résumé satellite indisponible."), normal_style))

            story.append(Spacer(1, 8))
        except Exception:
            pass

    story.append(Paragraph("8. Mesures disponibles", subtitle_style))
    story.append(_station_measures_table(row))
    story.append(Spacer(1, 10))

    story.append(Paragraph("9. Scores détaillés", subtitle_style))
    story.append(_station_scores_table(scores))
    story.append(Spacer(1, 10))

    story.append(Paragraph("10. Diagnostic", subtitle_style))
    for msg in diag["diagnostic_messages"]:
        story.append(Paragraph(f"• {msg}", normal_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("11. Recommandations", subtitle_style))
    for rec in diag["recommendations"]:
        story.append(Paragraph(f"• {rec}", normal_style))

    doc.build(story, onFirstPage=_pdf_header_footer, onLaterPages=_pdf_header_footer)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    for path in temp_files:
        try:
            os.remove(path)
        except Exception:
            pass

    return pdf_bytes


def generate_global_report_pdf(df: pd.DataFrame) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

    title_style, subtitle_style, normal_style, small_style = _build_pdf_styles()

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=2.0 * cm,
        bottomMargin=1.8 * cm,
    )

    story = []

    logo_path = os.path.join("assets", "logo.png")
    if os.path.exists(logo_path):
        try:
            story.append(Image(logo_path, width=2.8 * cm, height=2.8 * cm))
            story.append(Spacer(1, 4))
        except Exception:
            pass

    story.append(Paragraph("CleanWater — Rapport Global", title_style))
    story.append(Paragraph(f"<b>Date génération :</b> {datetime.utcnow().isoformat()} UTC", small_style))
    story.append(Paragraph(f"<b>Nombre de stations :</b> {len(df)}", small_style))
    story.append(Spacer(1, 10))

    if df.empty:
        story.append(Paragraph("Aucune donnée disponible.", normal_style))
        doc.build(story, onFirstPage=_pdf_header_footer, onLaterPages=_pdf_header_footer)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes

    story.append(Paragraph("1. Tableau de synthèse", subtitle_style))

    table_data = [[
        "Station",
        "Qualité",
        "Score qualité",
        "Risque",
        "Score risque",
        "Fiabilité",
    ]]

    diagnostics = []
    for _, row in df.iterrows():
        diag = build_station_diagnostic(row.to_dict())
        diagnostics.append(diag)
        s = diag["scores"]
        table_data.append([
            diag["station_id"],
            s["water_quality_label"],
            str(s["water_quality_score"]),
            s["risk_level"],
            str(s["risk_score"]),
            s["data_reliability_level"],
        ])

    summary_table = Table(
        table_data,
        colWidths=[70, 90, 80, 70, 70, 80],
        repeatRows=1,
    )
    summary_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#f3f6f9")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    story.append(summary_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("2. Diagnostics détaillés", subtitle_style))

    for idx, diag in enumerate(diagnostics, start=1):
        story.append(Paragraph(f"<b>{idx}. Station {diag['station_id']}</b>", normal_style))
        story.append(Paragraph(diag["summary"], small_style))
        story.append(Paragraph("<b>Messages clés :</b>", small_style))
        for msg in diag["diagnostic_messages"][:4]:
            story.append(Paragraph(f"• {msg}", small_style))
        story.append(Paragraph("<b>Recommandations :</b>", small_style))
        for rec in diag["recommendations"][:3]:
            story.append(Paragraph(f"• {rec}", small_style))
        story.append(Spacer(1, 8))

    doc.build(story, onFirstPage=_pdf_header_footer, onLaterPages=_pdf_header_footer)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes