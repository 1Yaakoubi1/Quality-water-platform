from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import ee

from firebase_service import (
    get_all_latest_station_states,
    get_latest_alerts,
    get_latest_predictions,
    get_latest_raw_measurements,
)
from workers.prediction_worker import run_predictions_for_latest_measurements
from diagnostics import build_station_diagnostic, build_diagnostics_for_dataframe
from scoring import compute_all_scores
from reporting import (
    generate_station_report_text,
    generate_multi_station_summary,
    generate_station_report_pdf,
    generate_global_report_pdf,
)

EMAIL_OK = True
try:
    from notification_service import (
        send_station_report_email,
        send_global_report_email,
    )
except Exception:
    EMAIL_OK = False


# =========================================================
# CONFIG PAGE
# =========================================================
st.set_page_config(
    page_title="CleanWater Decision Dashboard",
    page_icon="💧",
    layout="wide",
)

st.title("💧 CleanWater — Dashboard d'aide à la décision")
st.caption(
    "Surveillance temps réel, IA, soft-sensing, scoring, diagnostic, reporting, Sentinel-2 et notification email"
)


# =========================================================
# INIT GEE
# =========================================================
GEE_OK = True
try:
    ee.Initialize(project="cleanwater-460912")
except Exception:
    try:
        ee.Authenticate()
        ee.Initialize(project="cleanwater-460912")
    except Exception:
        GEE_OK = False


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Aller vers :",
    [
        "🏠 Vue globale",
        "🛰️ Sentinel-2 & GEE",
        "📡 Données temps réel",
        "🧠 Prédictions & Soft-Sensing",
        "🚨 Alertes",
        "🩺 Diagnostic",
        "📄 Reporting",
    ],
)

station_filter = st.sidebar.text_input("Filtrer par station_id", value="").strip()
limit = st.sidebar.slider("Nombre max de lignes", min_value=10, max_value=200, value=50, step=10)


# =========================================================
# HELPERS
# =========================================================
def load_raw_data() -> pd.DataFrame:
    if station_filter:
        return get_latest_raw_measurements(limit=limit, station_id=station_filter)
    return get_latest_raw_measurements(limit=limit)


def load_latest_states() -> pd.DataFrame:
    df = get_all_latest_station_states()
    if df.empty:
        return df
    if station_filter and "station_id" in df.columns:
        df = df[df["station_id"].astype(str) == station_filter]
    return df.reset_index(drop=True)


def load_alerts() -> pd.DataFrame:
    if station_filter:
        return get_latest_alerts(limit=limit, station_id=station_filter)
    return get_latest_alerts(limit=limit)


def load_predictions() -> pd.DataFrame:
    if station_filter:
        return get_latest_predictions(limit=limit, station_id=station_filter)
    return get_latest_predictions(limit=limit)


def dataframe_to_csv_download(df: pd.DataFrame, filename: str, label: str):
    if df.empty:
        st.info("Aucune donnée à exporter.")
        return
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def compute_kpis(df_states: pd.DataFrame, df_alerts: pd.DataFrame):
    n_stations = 0 if df_states.empty else len(df_states)

    open_alerts = 0
    critical_alerts = 0
    mean_quality = None
    mean_risk = None

    if not df_alerts.empty:
        open_alerts = len(df_alerts)
        if "severity" in df_alerts.columns:
            critical_alerts = int((df_alerts["severity"].astype(str).str.lower() == "critical").sum())

    if not df_states.empty:
        scores = []
        risks = []

        for _, row in df_states.iterrows():
            try:
                s = compute_all_scores(row.to_dict())
                quality_score = s.get("water_quality_score")
                risk_score = s.get("risk_score")

                if quality_score is not None:
                    scores.append(float(quality_score))

                if risk_score is not None:
                    risks.append(float(risk_score))
            except Exception as e:
                print(f"Erreur compute_all_scores pour station {row.get('station_id', 'unknown')} : {e}")

        if scores:
            mean_quality = round(sum(scores) / len(scores), 2)

        if risks:
            mean_risk = round(sum(risks) / len(risks), 2)

    return {
        "stations": n_stations,
        "open_alerts": open_alerts,
        "critical_alerts": critical_alerts,
        "mean_quality": mean_quality,
        "mean_risk": mean_risk,
    }


def safe_aoi_from_drawing(drawing, fallback):
    try:
        if not drawing or "geometry" not in drawing:
            return fallback
        geom = ee.Geometry(drawing["geometry"])
        area = geom.area(maxError=1).getInfo()
        if area is None or area <= 0:
            return fallback
        return geom
    except Exception:
        return fallback


def add_legend(m, title, vmin, vmax, colors, pos="left"):
    gradient = ", ".join(colors)
    left_css = "left: 30px;" if pos == "left" else "right: 30px;"
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 30px;
        {left_css}
        z-index: 9999;
        background: rgba(255,255,255,0.92);
        padding: 10px 12px;
        border: 1px solid #ccc;
        border-radius: 8px;
        font-size: 13px;
        width: 260px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    ">
        <div style="font-weight: 700; margin-bottom: 6px;">{title}</div>
        <div style="height: 12px; border-radius: 6px;
            background: linear-gradient(to right, {gradient});
            border: 1px solid #bbb;">
        </div>
        <div style="display:flex; justify-content: space-between; margin-top: 6px;">
            <span>{vmin}</span>
            <span>{vmax}</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


# =========================================================
# LOAD FIREBASE
# =========================================================
try:
    df_raw = load_raw_data()
    df_states = load_latest_states()
    df_alerts = load_alerts()
    df_predictions = load_predictions()
except Exception as e:
    st.error(f"Erreur de lecture Firebase : {e}")
    st.stop()


# =========================================================
# PAGE 1 — VUE GLOBALE
# =========================================================
if page == "🏠 Vue globale":
    st.subheader("Vue synthétique")

    kpis = compute_kpis(df_states, df_alerts)
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Stations actives", kpis["stations"])
    c2.metric("Alertes ouvertes", kpis["open_alerts"])
    c3.metric("Alertes critiques", kpis["critical_alerts"])
    c4.metric("Qualité moyenne", kpis["mean_quality"] if kpis["mean_quality"] is not None else "N/A")
    c5.metric("Risque moyen", kpis["mean_risk"] if kpis["mean_risk"] is not None else "N/A")

    st.markdown("---")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Dernier état des stations")
        if df_states.empty:
            st.info("Aucun état station disponible.")
        else:
            preview_rows = []
            for _, row in df_states.iterrows():
                scores = compute_all_scores(row.to_dict())
                preview_rows.append({
                    "station_id": row.get("station_id"),
                    "last_timestamp": row.get("last_timestamp"),
                    "latitude": row.get("latitude"),
                    "longitude": row.get("longitude"),
                    "temperature": row.get("temperature"),
                    "ph": row.get("ph"),
                    "turbidity": row.get("turbidity"),
                    "dissolved_oxygen": row.get("dissolved_oxygen"),
                    "predicted_water_quality": row.get("predicted_water_quality"),
                    "estimated_dbo": row.get("estimated_dbo"),
                    "estimated_dco": row.get("estimated_dco"),
                    "water_quality_score": scores.get("water_quality_score"),
                    "water_quality_label": scores.get("water_quality_label"),
                    "risk_score": scores.get("risk_score"),
                    "risk_level": scores.get("risk_level"),
                })
            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True)

    with col2:
        st.subheader("Dernières alertes")
        if df_alerts.empty:
            st.info("Aucune alerte disponible.")
        else:
            cols = [c for c in ["timestamp", "station_id", "severity", "alert_type", "message", "status"] if c in df_alerts.columns]
            st.dataframe(df_alerts[cols].head(15), use_container_width=True)

    st.markdown("---")
    st.subheader("Résumé décisionnel")
    if df_states.empty:
        st.info("Pas assez de données pour générer un résumé.")
    else:
        global_summary = generate_multi_station_summary(df_states)
        st.text_area("Rapport global", global_summary, height=350)
        st.download_button(
            "Télécharger le résumé global",
            data=global_summary.encode("utf-8"),
            file_name=f"cleanwater_global_summary_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )


# =========================================================
# PAGE 2 — SENTINEL
# =========================================================
elif page == "🛰️ Sentinel-2 & GEE":
    st.subheader("Sentinel-2 & indices GEE")

    if not GEE_OK:
        st.error("Google Earth Engine non initialisé.")
        st.stop()

    start_date = st.date_input("Date début", datetime(2024, 1, 1).date(), key="gee_start")
    end_date = st.date_input("Date fin", datetime(2024, 12, 31).date(), key="gee_end")
    cloud_max = st.slider("Nuages max (%)", 0, 80, 20, key="gee_cloud")

    st.markdown("### AOI — dessine un polygone")
    draw_map = folium.Map(location=[37.2, 9.67], zoom_start=11, control_scale=True)
    Draw(export=False).add_to(draw_map)
    st_map = st_folium(draw_map, height=420, width=900)

    default_aoi = ee.Geometry.Rectangle([9.55, 37.14, 9.80, 37.27])
    drawing = st_map.get("last_active_drawing")
    aoi = safe_aoi_from_drawing(drawing, default_aoi)
    region_safe = aoi.buffer(1)

    def mask_s2(img):
        qa = img.select("QA60")
        cloud = qa.bitwiseAnd(1 << 10).eq(0)
        cirrus = qa.bitwiseAnd(1 << 11).eq(0)
        return img.updateMask(cloud.And(cirrus)).divide(10000)

    def add_indices(img):
        ndci_img = img.normalizedDifference(["B5", "B4"]).rename("NDCI")
        ndti_img = img.normalizedDifference(["B4", "B3"]).rename("NDTI")
        return img.addBands([ndci_img, ndti_img])

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region_safe)
        .filterDate(str(start_date), str(end_date))
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_max))
        .map(mask_s2)
        .map(add_indices)
    )

    if s2.size().getInfo() == 0:
        st.warning("Aucune image Sentinel-2 trouvée sur cette période.")
        st.stop()

    img = s2.median().clip(region_safe)

    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = img.normalizedDifference(["B3", "B8"]).rename("NDWI")
    mndwi = img.normalizedDifference(["B3", "B11"]).rename("MNDWI")
    ndmi = img.normalizedDifference(["B8", "B11"]).rename("NDMI")
    ndci = img.select("NDCI")
    ndti = img.select("NDTI")

    try:
        B3 = img.select("B3")
        B8 = img.select("B8")
        B11 = img.select("B11")
        B12 = img.select("B12")
        awei = (
            B3.subtract(B11).multiply(4)
            .subtract(B8.multiply(0.25))
            .subtract(B12.multiply(2.75))
        ).rename("AWEI")
    except Exception:
        awei = ndwi.rename("AWEI")

    palette_br = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]
    vis_ndci = {"min": -0.2, "max": 0.5, "palette": palette_br}
    vis_ndti = {"min": -0.2, "max": 0.4, "palette": palette_br}

    layers = {
        "RGB": (
            img.select(["B4", "B3", "B2"]).rename(["R", "G", "B"]),
            {"bands": ["R", "G", "B"], "min": 0.02, "max": 0.30},
        ),
        "NDVI": (ndvi, {"min": -0.2, "max": 0.9}),
        "NDWI": (ndwi, {"min": -0.5, "max": 0.8}),
        "MNDWI": (mndwi, {"min": -0.5, "max": 0.8}),
        "NDMI": (ndmi, {"min": -0.5, "max": 0.7}),
        "AWEI": (awei, {"min": -2.0, "max": 2.0}),
        "NDCI (Chlorophylle)": (ndci, vis_ndci),
        "NDTI (Turbidité)": (ndti, vis_ndti),
    }

    m2 = folium.Map(location=[37.2, 9.67], zoom_start=11, control_scale=True)

    for name, (im_raw, vis) in layers.items():
        map_id = ee.Image(im_raw).getMapId(vis)
        folium.TileLayer(
            tiles=map_id["tile_fetcher"].url_format,
            attr="Google Earth Engine",
            name=name,
            overlay=True,
            control=True,
        ).add_to(m2)

    folium.LayerControl(collapsed=False).add_to(m2)
    add_legend(m2, "Palette (bleu→rouge)", "-0.2", "0.5", palette_br, pos="left")

    st.markdown("### Carte interactive")
    st_folium(m2, height=650, width=900)

    st.markdown("### Résumé indices")
    c1, c2, c3 = st.columns(3)
    c1.info("NDWI / MNDWI : détection eau")
    c2.info("NDTI : proxy turbidité")
    c3.info("NDCI : proxy chlorophylle / eutrophisation")


# =========================================================
# PAGE 3 — DONNÉES TEMPS RÉEL
# =========================================================
elif page == "📡 Données temps réel":
    st.subheader("Mesures brutes reçues")

    if df_raw.empty:
        st.info("Aucune mesure brute disponible.")
    else:
        st.dataframe(df_raw, use_container_width=True)
        dataframe_to_csv_download(
            df_raw,
            filename=f"raw_measurements_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
            label="Télécharger les mesures brutes",
        )

    st.markdown("---")
    st.subheader("État consolidé des stations")

    if df_states.empty:
        st.info("Aucun état consolidé disponible.")
    else:
        st.dataframe(df_states, use_container_width=True)
        dataframe_to_csv_download(
            df_states,
            filename=f"latest_station_state_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
            label="Télécharger les états station",
        )


# =========================================================
# PAGE 4 — PRÉDICTIONS
# =========================================================
elif page == "🧠 Prédictions & Soft-Sensing":
    st.subheader("Lancer les prédictions")

    model_choice = st.selectbox(
        "Choisir le modèle de classification",
        ["Random Forest", "XGBoost"],
        index=0,
    )

    pred_col1, pred_col2 = st.columns([1, 2])

    with pred_col1:
        run_button = st.button("Lancer prédictions sur les dernières mesures", type="primary")

    with pred_col2:
        st.caption("Cette action exécute la classification qualité + le soft-sensing DBO/DCO sur les dernières mesures.")

    if run_button:
        try:
            result = run_predictions_for_latest_measurements(
                station_id=station_filter if station_filter else None,
                limit=limit,
                classification_model_name=model_choice,
            )
            st.success(f"Prédictions terminées : {result}")
            df_predictions = load_predictions()
            df_states = load_latest_states()
        except Exception as e:
            st.error(f"Erreur pendant la prédiction : {e}")

    st.markdown("---")
    st.subheader("Historique des prédictions")

    if df_predictions.empty:
        st.info("Aucune prédiction disponible.")
    else:
        st.dataframe(df_predictions, use_container_width=True)
        dataframe_to_csv_download(
            df_predictions,
            filename=f"predictions_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
            label="Télécharger l'historique des prédictions",
        )

    st.markdown("---")
    st.subheader("Dernier état enrichi")

    if df_states.empty:
        st.info("Aucun état enrichi disponible.")
    else:
        preview_cols = [
            c for c in [
                "station_id",
                "last_timestamp",
                "predicted_water_quality",
                "estimated_dbo",
                "estimated_dco",
                "soft_sensing_confidence",
                "soft_sensing_available",
                "water_quality_score",
                "water_quality_label",
                "risk_score",
                "risk_level",
                "data_reliability_score",
            ] if c in df_states.columns
        ]
        st.dataframe(df_states[preview_cols], use_container_width=True)


# =========================================================
# PAGE 5 — ALERTES
# =========================================================
elif page == "🚨 Alertes":
    st.subheader("Mur d'alertes")

    if df_alerts.empty:
        st.info("Aucune alerte.")
    else:
        display_cols = [
            c for c in [
                "timestamp",
                "station_id",
                "severity",
                "alert_type",
                "message",
                "recommendation",
                "status",
            ] if c in df_alerts.columns
        ]
        st.dataframe(df_alerts[display_cols], use_container_width=True)

        st.markdown("### Répartition par sévérité")
        if "severity" in df_alerts.columns:
            sev_counts = df_alerts["severity"].astype(str).value_counts().reset_index()
            sev_counts.columns = ["severity", "count"]
            st.dataframe(sev_counts, use_container_width=True)

        dataframe_to_csv_download(
            df_alerts,
            filename=f"alerts_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
            label="Télécharger les alertes",
        )


# =========================================================
# PAGE 6 — DIAGNOSTIC
# =========================================================
elif page == "🩺 Diagnostic":
    st.subheader("Diagnostic station par station")

    if df_states.empty:
        st.info("Aucun état station disponible pour diagnostic.")
    else:
        diagnostics_df = build_diagnostics_for_dataframe(df_states)
        st.dataframe(diagnostics_df, use_container_width=True)

        st.markdown("---")
        st.subheader("Détail d'une station")

        station_ids = df_states["station_id"].astype(str).unique().tolist() if "station_id" in df_states.columns else []
        if station_ids:
            selected_station = st.selectbox("Choisir une station", station_ids)
            station_row = df_states[df_states["station_id"].astype(str) == str(selected_station)].iloc[0].to_dict()
            diag = build_station_diagnostic(station_row)

            st.markdown("### Synthèse")
            st.write(diag["summary"])

            col1, col2, col3 = st.columns(3)
            col1.metric("Qualité", diag["scores"]["water_quality_score"])
            col2.metric("Risque", diag["scores"]["risk_score"])
            col3.metric("Fiabilité", diag["scores"]["data_reliability_score"])

            st.markdown("### Messages de diagnostic")
            for msg in diag["diagnostic_messages"]:
                st.write(f"- {msg}")

            st.markdown("### Recommandations")
            for rec in diag["recommendations"]:
                st.write(f"- {rec}")


# =========================================================
# PAGE 7 — REPORTING
# =========================================================
elif page == "📄 Reporting":
    st.subheader("Génération de rapport professionnel")

    if df_states.empty:
        st.info("Aucune donnée disponible pour générer un rapport.")
    else:
        mode = st.radio("Type de rapport", ["Rapport station", "Rapport global"], horizontal=True)

        if mode == "Rapport station":
            station_ids = df_states["station_id"].astype(str).unique().tolist() if "station_id" in df_states.columns else []
            selected_station = st.selectbox("Station", station_ids)

            station_row = df_states[df_states["station_id"].astype(str) == str(selected_station)].iloc[0].to_dict()

            report_text = generate_station_report_text(station_row)
            pdf_bytes = generate_station_report_pdf(station_row, history_df=df_raw)

            st.text_area("Aperçu texte du rapport", report_text, height=450)

            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    "Télécharger rapport TXT",
                    data=report_text.encode("utf-8"),
                    file_name=f"station_report_{selected_station}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                )

            with col2:
                st.download_button(
                    "Télécharger rapport PDF",
                    data=pdf_bytes,
                    file_name=f"station_report_{selected_station}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                )

            st.markdown("---")
            st.subheader("Envoi du rapport station par email")

            if not EMAIL_OK:
                st.info("Module email indisponible. Vérifie notification_service.py")
            else:
                recipient_email = st.text_input("Email destinataire", value="", key="recipient_station_pdf")
                custom_message = st.text_area(
                    "Message personnalisé",
                    value=(
                        f"Bonjour,\n\n"
                        f"Veuillez trouver en pièce jointe le rapport CleanWater de la station {selected_station}.\n\n"
                        f"Cordialement."
                    ),
                    key="body_station_pdf",
                )

                if st.button("📩 Envoyer le rapport station par email"):
                    if not recipient_email.strip():
                        st.warning("Veuillez saisir une adresse email.")
                    else:
                        try:
                            send_station_report_email(
                                station_id=selected_station,
                                to_email=recipient_email.strip(),
                                pdf_bytes=pdf_bytes,
                                custom_message=custom_message,
                            )
                            st.success("Email envoyé avec succès.")
                        except Exception as e:
                            st.error(f"Erreur d'envoi email : {e}")

        else:
            report_text = generate_multi_station_summary(df_states)
            pdf_bytes = generate_global_report_pdf(df_states)

            st.text_area("Aperçu texte du rapport global", report_text, height=450)

            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    "Télécharger rapport global TXT",
                    data=report_text.encode("utf-8"),
                    file_name=f"global_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                )

            with col2:
                st.download_button(
                    "Télécharger rapport global PDF",
                    data=pdf_bytes,
                    file_name=f"global_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                )

            st.markdown("---")
            st.subheader("Envoi du rapport global par email")

            if not EMAIL_OK:
                st.info("Module email indisponible. Vérifie notification_service.py")
            else:
                recipient_email = st.text_input("Email destinataire", value="", key="recipient_global_pdf")
                custom_message = st.text_area(
                    "Message personnalisé",
                    value=(
                        "Bonjour,\n\n"
                        "Veuillez trouver en pièce jointe le rapport global CleanWater.\n\n"
                        "Cordialement."
                    ),
                    key="body_global_pdf",
                )

                if st.button("📩 Envoyer le rapport global par email"):
                    if not recipient_email.strip():
                        st.warning("Veuillez saisir une adresse email.")
                    else:
                        try:
                            send_global_report_email(
                                to_email=recipient_email.strip(),
                                pdf_bytes=pdf_bytes,
                                custom_message=custom_message,
                            )
                            st.success("Email envoyé avec succès.")
                        except Exception as e:
                            st.error(f"Erreur d'envoi email : {e}")