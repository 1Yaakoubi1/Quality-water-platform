# main.py — CleanWater Ichkeul (Green IA) — FINAL
# ✅ GEE Sentinel-2 indices + export GeoTIFF ZIP (stable)
# ✅ Analyse GeoTIFF ZIP (rasterio) : NDWI/NDTI/NDCI + Otsu (surface eau)
# ✅ Green IA : CodeCarbon (empreinte CO2)
# ✅ Diagnostic rasterio : centre vs bordures sud (auto)
# ✅ Interprétation rasterio + Score global 0–100 + jauge

import os
import io
import glob
import zipfile
import tempfile
import re
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import requests

# --- GEE ---
import ee
import geemap.foliumap as geemap

# --- Rasterio ---
import rasterio
from rasterio.merge import merge

# --- CodeCarbon (Green IA) ---
try:
    from codecarbon import EmissionsTracker
    CODECARBON_OK = True
except Exception:
    CODECARBON_OK = False

# Optional: predict_quality (si tu l’utilises ailleurs)
try:
    from predict import predict_quality
    PREDICT_OK = True
except Exception:
    PREDICT_OK = False


# =========================================================
# GEE init
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
# Sidebar
# =========================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Aller vers :",
    [
        "🏠 Carte & Indices Sentinel-2",
        "🧪 Analyse GeoTIFF ",
        "🧰 Analyse IA (CSV)",
        "📡 Acquisition capteurs",
    ],
)


# =========================================================
# Utils (general)
# =========================================================
def http_download_bytes(url: str, timeout_s: int = 600) -> bytes:
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    return r.content


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


def aoi_bounds_lonlat(geom: ee.Geometry):
    coords = geom.bounds(maxError=1).coordinates().getInfo()[0]
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    return min(xs), min(ys), max(xs), max(ys)


def tile_rects(xmin, ymin, xmax, ymax, n):
    dx = (xmax - xmin) / n
    dy = (ymax - ymin) / n
    rects = []
    for i in range(n):
        for j in range(n):
            x0 = xmin + i * dx
            x1 = xmin + (i + 1) * dx
            y0 = ymin + j * dy
            y1 = ymin + (j + 1) * dy
            rects.append(((i, j), ee.Geometry.Rectangle([x0, y0, x1, y1])))
    return rects


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
# Rasterio/Green IA — Otsu & plots
# =========================================================
def otsu_threshold(values, nbins=256):
    v = values[np.isfinite(values)]
    if v.size == 0:
        return None
    vmin, vmax = np.percentile(v, 1), np.percentile(v, 99)
    if vmin == vmax:
        return float(vmin)

    hist, edges = np.histogram(v, bins=nbins, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    p = hist / (hist.sum() + 1e-12)

    centers = (edges[:-1] + edges[1:]) / 2.0
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]

    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    idx = np.nanargmax(sigma_b2)
    return float(centers[idx])


def robust_limits(a, p1=2, p2=98):
    v = a[np.isfinite(a)]
    if v.size == 0:
        return None, None
    return float(np.percentile(v, p1)), float(np.percentile(v, p2))


def st_plot_heatmap(arr, title, cmap="RdYlBu_r"):
    vmin, vmax = robust_limits(arr)
    fig = plt.figure(figsize=(9, 6))
    plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    st.pyplot(fig)


def mosaic_paths(paths):
    if not paths:
        return None, None, None
    srcs = [rasterio.open(p) for p in paths]
    mosaic, transform = merge(srcs)
    meta0 = srcs[0].meta.copy()
    nodata = meta0.get("nodata", None)
    for s in srcs:
        s.close()
    arr = mosaic[0].astype("float32")
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr, transform, nodata


def water_surface_from_ndwi(ndwi, transform):
    v = ndwi[np.isfinite(ndwi)]
    if v.size == 0:
        return None, None, None, None

    # Green IA: subsample si grand raster
    if v.size > 2_000_000:
        v = v[::10]

    thr = otsu_threshold(v, nbins=256)
    if thr is None:
        return None, None, None, None

    water_mask = np.isfinite(ndwi) & (ndwi > thr)

    px_w = transform.a
    px_h = -transform.e
    pixel_area = abs(px_w * px_h)
    water_m2 = np.count_nonzero(water_mask) * pixel_area
    return float(thr), float(water_m2), float(water_m2 / 1e6), water_mask


def tile_stats(tile_path):
    with rasterio.open(tile_path) as src:
        a = src.read(1).astype("float32")
        nd = src.nodata
        if nd is not None:
            a[a == nd] = np.nan
        valid = np.isfinite(a)
        return {
            "file": os.path.basename(tile_path),
            "min": float(np.nanmin(a)),
            "p05": float(np.nanpercentile(a, 5)),
            "mean": float(np.nanmean(a)),
            "p95": float(np.nanpercentile(a, 95)),
            "max": float(np.nanmax(a)),
            "std": float(np.nanstd(a)),
            "valid_%": float(100.0 * np.count_nonzero(valid) / max(a.size, 1)),
        }


# =========================================================
# Auto-détection centre & bordures sud (tile_NxN_i_j)
# =========================================================
def parse_tile_ij(filename: str):
    base = os.path.basename(filename).lower()
    m = re.search(r"tile_(\d+)x\1_(\d+)_(\d+)", base)
    if not m:
        return None
    n = int(m.group(1))
    i = int(m.group(2))
    j = int(m.group(3))
    return n, i, j


def list_tiles(paths):
    tiles = []
    for p in paths:
        info = parse_tile_ij(p)
        if info:
            n, i, j = info
            tiles.append((p, n, i, j))
    return tiles


def pick_center_tile(tiles):
    if not tiles:
        return None
    N = tiles[0][1]
    ci = (N - 1) / 2.0
    cj = (N - 1) / 2.0
    return min(tiles, key=lambda t: (t[2] - ci) ** 2 + (t[3] - cj) ** 2)


def pick_south_tiles(tiles, south_is_j_max=False, k=2):
    if not tiles:
        return []
    south_j = max(t[3] for t in tiles) if south_is_j_max else min(t[3] for t in tiles)
    candidates = [t for t in tiles if t[3] == south_j]
    N = tiles[0][1]
    ci = (N - 1) / 2.0
    candidates = sorted(candidates, key=lambda t: abs(t[2] - ci))
    return candidates[:k]


def match_same_tile(reference_path, idx_paths):
    if reference_path is None or not idx_paths:
        return None
    info = parse_tile_ij(reference_path)
    if not info:
        return None
    n, i, j = info
    token = f"tile_{n}x{n}_{i}_{j}"
    return next((p for p in idx_paths if token in os.path.basename(p).lower()), None)


# =========================================================
# Diagnostic rasterio + interprétation + score
# =========================================================
def compute_zone_metrics(p_ref, ndwi_paths, ndti_paths, ndci_paths):
    if p_ref is None:
        return {"NDWI": None, "NDTI": None, "NDCI": None, "tile_file": None}
    out = {"tile_file": os.path.basename(p_ref)}
    out["NDWI"] = tile_stats(p_ref)

    p_ndti = match_same_tile(p_ref, ndti_paths)
    out["NDTI"] = tile_stats(p_ndti) if p_ndti else None

    p_ndci = match_same_tile(p_ref, ndci_paths)
    out["NDCI"] = tile_stats(p_ndci) if p_ndci else None
    return out


def diagnose_from_metrics(center, south1, south2, thr_ndwi):
    alerts = []

    def mean(met, idx):
        s = met.get(idx)
        return None if s is None else s.get("mean", None)

    c_ndwi = mean(center, "NDWI")
    s1_ndwi = mean(south1, "NDWI")
    s2_ndwi = mean(south2, "NDWI") if mean(south2, "NDWI") is not None else s1_ndwi

    c_ndti = mean(center, "NDTI")
    s1_ndti = mean(south1, "NDTI")
    s2_ndti = mean(south2, "NDTI") if mean(south2, "NDTI") is not None else s1_ndti

    c_ndci = mean(center, "NDCI")
    s1_ndci = mean(south1, "NDCI")
    s2_ndci = mean(south2, "NDCI") if mean(south2, "NDCI") is not None else s1_ndci

    # Assèchement (NDWI)
    if c_ndwi is not None and s1_ndwi is not None:
        ndwi_edges = float(np.nanmean([s1_ndwi, s2_ndwi]))
        if (ndwi_edges < 0.7 * c_ndwi) or ((c_ndwi - ndwi_edges) > 0.10):
            alerts.append(("warning", "Assèchement bordures sud probable (NDWI bordures << NDWI centre)."))
        else:
            alerts.append(("success", "NDWI bordures vs centre : pas d’assèchement marqué (selon critères)."))

    # Turbidité (NDTI)
    if s1_ndti is not None:
        ndti_edges = float(np.nanmean([s1_ndti, s2_ndti]))
        if (c_ndti is None and ndti_edges > 0.15) or (c_ndti is not None and (ndti_edges - c_ndti) > 0.08):
            alerts.append(("warning", "Turbidité bordures sud élevée (NDTI bordures élevé)."))
        else:
            alerts.append(("success", "NDTI bordures vs centre : pas d’anomalie forte (selon critères)."))

    # Chlorophylle (NDCI)
    if s1_ndci is not None:
        ndci_edges = float(np.nanmean([s1_ndci, s2_ndci]))
        if ndci_edges > 0.20 and (c_ndci is None or (ndci_edges - c_ndci) > 0.07):
            alerts.append(("warning", "Chlorophylle (NDCI) élevée sur bordures sud (risque eutrophisation)."))
        else:
            alerts.append(("success", "NDCI : pas de signal fort (selon seuils par défaut)."))

    alerts.append(("info", f"Seuil Otsu NDWI (eau libre) = {thr_ndwi:.3f}"))
    return alerts


def describe_water_state_from_rasterio(center_metrics, south1_metrics, south2_metrics, water_km2, thr_ndwi):
    def mean(met, idx):
        s = met.get(idx)
        return None if s is None else s.get("mean", None)

    c_ndwi = mean(center_metrics, "NDWI")
    s1_ndwi = mean(south1_metrics, "NDWI")
    s2_ndwi = mean(south2_metrics, "NDWI") if mean(south2_metrics, "NDWI") is not None else s1_ndwi

    c_ndti = mean(center_metrics, "NDTI")
    s1_ndti = mean(south1_metrics, "NDTI")
    s2_ndti = mean(south2_metrics, "NDTI") if mean(south2_metrics, "NDTI") is not None else s1_ndti

    c_ndci = mean(center_metrics, "NDCI")
    s1_ndci = mean(south1_metrics, "NDCI")
    s2_ndci = mean(south2_metrics, "NDCI") if mean(south2_metrics, "NDCI") is not None else s1_ndci

    ndwi_edges = np.nanmean([s1_ndwi, s2_ndwi]) if (s1_ndwi is not None) else None
    ndti_edges = np.nanmean([s1_ndti, s2_ndti]) if (s1_ndti is not None) else None
    ndci_edges = np.nanmean([s1_ndci, s2_ndci]) if (s1_ndci is not None) else None

    lines = []
    lines.append(f"Surface en eau libre (NDWI+Otsu) : {water_km2:.3f} km².")
    lines.append(f"Seuil Otsu NDWI : {thr_ndwi:.3f} (eau = NDWI > seuil).")

    if c_ndwi is not None and ndwi_edges is not None:
        diff = float(c_ndwi - ndwi_edges)
        if diff > 0.10 or ndwi_edges < 0.7 * c_ndwi:
            lines.append("Quantité : eau libre davantage au centre ; bordures sud avec NDWI plus faible (peu profond/vasières/assèchement relatif).")
        else:
            lines.append("Quantité : contraste centre/bordures modéré ; pas d’assèchement marqué selon NDWI.")
    else:
        lines.append("Quantité : comparaison NDWI centre/bordures non disponible (tuiles/indices manquants).")

    if ndti_edges is not None:
        if ndti_edges > 0.15:
            lines.append("Turbidité : NDTI élevé → forte charge sédimentaire/remise en suspension probable en bordures.")
        elif ndti_edges > 0.08:
            lines.append("Turbidité : NDTI modéré → turbidité présente mais non extrême.")
        else:
            lines.append("Turbidité : NDTI faible → eau relativement claire (signal sédiments faible).")
    else:
        lines.append("Turbidité : NDTI non disponible.")

    if ndci_edges is not None:
        if ndci_edges > 0.20:
            lines.append("Chlorophylle : NDCI élevé → chlorophylle (proxy) élevée, compatible avec eutrophisation/bloom.")
        elif ndci_edges > 0.10:
            lines.append("Chlorophylle : NDCI modéré → eutrophisation possible, à surveiller.")
        else:
            lines.append("Chlorophylle : NDCI faible → signal chlorophyllien faible.")
    else:
        lines.append("Chlorophylle : NDCI non disponible.")

    return "\n".join([f"- {x}" for x in lines])


def clamp01(x):
    return max(0.0, min(1.0, float(x)))


def compute_global_score(center_metrics, south1_metrics, south2_metrics, water_km2):
    """
    Score 0–100 basé uniquement sur rasterio :
      - Quantité d’eau : water_km2 (normalisée par une référence locale)
      - Qualité turbidité : NDTI bordures (plus haut = pire)
      - Qualité eutrophisation : NDCI bordures (plus haut = pire)

    IMPORTANT : les références ci-dessous sont "par défaut" et doivent être calibrées
    avec ton historique Ichkeul (multi-dates).
    """
    # Référence quantité (km²) : à calibrer sur historique
    # Exemple: 8 km² (faible) -> 0 ; 30 km² (bon) -> 1
    q_min, q_max = 8.0, 30.0
    q_norm = clamp01((water_km2 - q_min) / (q_max - q_min))

    def mean(met, idx):
        s = met.get(idx)
        return None if s is None else s.get("mean", None)

    s1_ndti = mean(south1_metrics, "NDTI")
    s2_ndti = mean(south2_metrics, "NDTI") if mean(south2_metrics, "NDTI") is not None else s1_ndti
    ndti_edges = np.nanmean([s1_ndti, s2_ndti]) if s1_ndti is not None else None

    s1_ndci = mean(south1_metrics, "NDCI")
    s2_ndci = mean(south2_metrics, "NDCI") if mean(south2_metrics, "NDCI") is not None else s1_ndci
    ndci_edges = np.nanmean([s1_ndci, s2_ndci]) if s1_ndci is not None else None

    # NDTI: 0.05 bon -> 1 ; 0.25 très turbide -> 0
    if ndti_edges is None:
        turb_norm = 0.5
    else:
        t_good, t_bad = 0.05, 0.25
        turb_norm = clamp01(1.0 - (ndti_edges - t_good) / (t_bad - t_good))

    # NDCI: 0.05 bon -> 1 ; 0.30 bloom -> 0
    if ndci_edges is None:
        chl_norm = 0.5
    else:
        c_good, c_bad = 0.05, 0.30
        chl_norm = clamp01(1.0 - (ndci_edges - c_good) / (c_bad - c_good))

    # Pondérations (quantité prioritaire en sécheresse)
    wQ, wT, wC = 0.45, 0.30, 0.25
    score = 100.0 * (wQ * q_norm + wT * turb_norm + wC * chl_norm)

    details = {
        "quantity_norm": q_norm,
        "turbidity_norm": turb_norm,
        "chlorophyll_norm": chl_norm,
        "ndti_edges": None if ndti_edges is None else float(ndti_edges),
        "ndci_edges": None if ndci_edges is None else float(ndci_edges),
    }
    return float(score), details


def score_label(score):
    if score >= 80:
        return "Excellent"
    if score >= 65:
        return "Good"
    if score >= 50:
        return "Moderate"
    if score >= 35:
        return "Poor"
    return "Very Poor"


# =========================================================
# PAGE 1 — GEE Carte + Export ZIP
# =========================================================
if page == "🏠 Carte & Indices Sentinel-2":
    st.title("🛰️ Sentinel-2 & Indices — Lac Ichkeul (GEE)")

    if not GEE_OK:
        st.error("Google Earth Engine non initialisé.")
        st.stop()

    start_date = st.date_input("Date début", datetime(2024, 1, 1).date())
    end_date = st.date_input("Date fin", datetime(2024, 12, 31).date())
    cloud_max = st.slider("Nuages max (%)", 0, 80, 20)

    st.subheader("📌 AOI (dessine un polygone)")
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
        ndci = img.normalizedDifference(["B5", "B4"]).rename("NDCI")
        ndti = img.normalizedDifference(["B4", "B3"]).rename("NDTI")
        return img.addBands([ndci, ndti])

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region_safe)
        .filterDate(str(start_date), str(end_date))
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_max))
        .map(mask_s2)
        .map(add_indices)
    )

    if s2.size().getInfo() == 0:
        st.error("Aucune image Sentinel-2 sur cette période/AOI.")
        st.stop()

    img = s2.median().clip(region_safe)

    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = img.normalizedDifference(["B3", "B8"]).rename("NDWI")
    mndwi = img.normalizedDifference(["B3", "B11"]).rename("MNDWI")
    ndmi = img.normalizedDifference(["B8", "B11"]).rename("NDMI")
    ndci = img.select("NDCI")
    ndti = img.select("NDTI")

    # AWEI
    try:
        B3 = img.select("B3")
        B8 = img.select("B8")
        B11 = img.select("B11")
        B12 = img.select("B12")
        awei = (B3.subtract(B11).multiply(4)
                .subtract(B8.multiply(0.25))
                .subtract(B12.multiply(2.75))).rename("AWEI")
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

    export_images = {
        "NDVI": ndvi,
        "NDWI": ndwi,
        "MNDWI": mndwi,
        "NDMI": ndmi,
        "AWEI": awei,
        "NDCI": ndci,
        "NDTI": ndti,
    }

    st.subheader("🗺️ Carte interactive (GEE tiles)")
    m2 = folium.Map(location=[37.2, 9.67], zoom_start=11, control_scale=True)
    for name, (im_raw, vis) in layers.items():
        tl = geemap.ee_tile_layer(im_raw, vis, name)
        folium.TileLayer(tiles=tl.tiles, attr="Google Earth Engine", name=name, overlay=True, control=True).add_to(m2)
    folium.LayerControl(collapsed=False).add_to(m2)
    add_legend(m2, "Palette (bleu→rouge)", "-0.2", "0.5", palette_br, pos="left")
    st_folium(m2, height=650, width=900)

    st.markdown("---")
    st.header("📦 Export GeoTIFF (tuilé) — ZIP téléchargeable")

    tile_grid = st.selectbox("Grille (NxN)", [2, 3, 4, 5, 6, 8], index=3)
    tile_scale = st.selectbox("Résolution (m)", [10, 20, 30, 60], index=1)
    tile_crs = st.text_input("CRS EPSG (optionnel)", value="")
    tile_max_pixels = st.number_input("maxPixels", value=1_000_000_000, step=100_000_000)

    export_mode = st.selectbox("Mode", ["BATCH INDICES (ZIP)", "INDICE UNIQUE"], index=0)
    indices_to_export = st.multiselect(
        "Indices à exporter (Batch)",
        list(export_images.keys()),
        default=["NDWI", "NDTI", "NDCI", "NDVI"],
    )
    index_one = st.selectbox("Indice unique", list(export_images.keys()), index=0)

    st.info("Si l’export échoue : mets NxN=8 et scale=30/60 m.")

    if "zip_ready_bytes" not in st.session_state:
        st.session_state.zip_ready_bytes = None
    if "zip_ready_name" not in st.session_state:
        st.session_state.zip_ready_name = None

    colA, colB = st.columns([1, 2])
    with colA:
        build_zip = st.button("📦 Générer ZIP GeoTIFF", type="primary")
    with colB:
        if st.session_state.zip_ready_bytes is not None:
            st.download_button(
                "⬇️ Télécharger ZIP (GeoTIFF)",
                data=st.session_state.zip_ready_bytes,
                file_name=st.session_state.zip_ready_name,
                mime="application/zip",
                use_container_width=True,
            )

    if build_zip:
        try:
            xmin, ymin, xmax, ymax = aoi_bounds_lonlat(region_safe)
            rects = tile_rects(xmin, ymin, xmax, ymax, int(tile_grid))

            products = []
            if export_mode == "BATCH INDICES (ZIP)":
                for idx in indices_to_export:
                    products.append((f"INDEX_{idx}", export_images[idx].clip(region_safe), int(tile_scale)))
            else:
                products.append((f"INDEX_{index_one}", export_images[index_one].clip(region_safe), int(tile_scale)))

            zip_buffer = io.BytesIO()
            with st.spinner("Génération ZIP…"):
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    total = max(len(products) * len(rects), 1)
                    done = 0
                    prog = st.progress(0)

                    for prod_name, prod_img, scale in products:
                        for (i, j), rect in rects:
                            tile_geom = rect.intersection(region_safe, maxError=1)
                            area = tile_geom.area(maxError=1).getInfo()
                            if area is None or area <= 0:
                                done += 1
                                prog.progress(min(done / total, 1.0))
                                continue

                            params = {
                                "region": tile_geom,
                                "scale": int(scale),
                                "format": "GEO_TIFF",
                                "filePerBand": False,
                                "maxPixels": int(tile_max_pixels),
                            }
                            if tile_crs.strip():
                                params["crs"] = tile_crs.strip()

                            url = prod_img.getDownloadURL(params)
                            content = http_download_bytes(url, timeout_s=600)

                            tif_name = (
                                f"Ichkeul_{prod_name}_{start_date}_{end_date}_"
                                f"tile_{tile_grid}x{tile_grid}_{i}_{j}_{scale}m.tif"
                            )
                            zf.writestr(tif_name, content)

                            done += 1
                            prog.progress(min(done / total, 1.0))

            zip_buffer.seek(0)
            zip_name = f"Ichkeul_EXPORT_{start_date}_{end_date}_TILED_{tile_grid}x{tile_grid}_{tile_scale}m.zip"
            st.session_state.zip_ready_bytes = zip_buffer.getvalue()
            st.session_state.zip_ready_name = zip_name

            st.success("✅ ZIP prêt. Clique sur Télécharger.")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Export échoué : {e}")
            st.session_state.zip_ready_bytes = None
            st.session_state.zip_ready_name = None


# =========================================================
# PAGE 2 — Analyse GeoTIFF (Green IA + CO2 + diagnostic + score)
# =========================================================
elif page == "🧪 Analyse GeoTIFF ":
    st.title("🧪 Analyse GeoTIFF — NDWI/NDTI/NDCI + Otsu + CO₂ + Diagnostic + Score")

    zip_file = st.file_uploader("📦 ZIP GeoTIFF", type=["zip"])
    south_is_j_max = st.checkbox("Axe Y inversé (prendre sud = j max)", value=False)

    if not CODECARBON_OK:
        st.warning("CodeCarbon non installé. Installe-le via:  pip install codecarbon")

    if zip_file:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "tiles.zip")
            with open(zip_path, "wb") as f:
                f.write(zip_file.getbuffer())

            tracker = None
            if CODECARBON_OK:
                tracker = EmissionsTracker(
                    project_name="CleanWater-Ichkeul-GeoTIFF",
                    output_dir=tmpdir,
                    log_level="error",
                )
                tracker.start()

            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(tmpdir)

                all_tifs = glob.glob(os.path.join(tmpdir, "**", "*.tif"), recursive=True)
                if not all_tifs:
                    st.error("Aucun .tif trouvé dans le ZIP.")
                    st.stop()

                ndwi_paths = sorted([p for p in all_tifs if "ndwi" in os.path.basename(p).lower()])
                ndti_paths = sorted([p for p in all_tifs if "ndti" in os.path.basename(p).lower()])
                ndci_paths = sorted([p for p in all_tifs if "ndci" in os.path.basename(p).lower()])
                ndvi_paths = sorted([p for p in all_tifs if "ndvi" in os.path.basename(p).lower()])

                if len(ndwi_paths) == 0:
                    st.error("NDWI manquant : impossible de calculer la surface en eau.")
                    st.stop()

                # Mosaïques indices
                ndwi, tr, _ = mosaic_paths(ndwi_paths)
                ndti, _, _ = mosaic_paths(ndti_paths) if ndti_paths else (None, None, None)
                ndci, _, _ = mosaic_paths(ndci_paths) if ndci_paths else (None, None, None)
                ndvi, _, _ = mosaic_paths(ndvi_paths) if ndvi_paths else (None, None, None)

                thr, water_m2, water_km2, water_mask = water_surface_from_ndwi(ndwi, tr)
                if thr is None:
                    st.error("Otsu impossible (valeurs NDWI invalides).")
                    st.stop()

                emissions = None
                if tracker is not None:
                    emissions = tracker.stop()

                # KPIs
                c1, c2, c3 = st.columns(3)
                c1.metric("💧 Surface eau (km²)", f"{water_km2:.3f}")
                c2.metric("📉 Seuil Otsu NDWI", f"{thr:.3f}")
                c3.metric("🌍 Empreinte CO₂ (kg)", f"{emissions:.6f}" if emissions is not None else "N/A")

                # Heatmaps
                st.subheader("🗺️ Cartes (heatmaps)")
                st_plot_heatmap(ndwi, f"NDWI (mosaïque) — Otsu={thr:.3f}")
                if ndti is not None:
                    st_plot_heatmap(ndti, "NDTI (mosaïque)")
                if ndci is not None:
                    st_plot_heatmap(ndci, "NDCI (mosaïque)")
                if ndvi is not None:
                    st_plot_heatmap(ndvi, "NDVI (mosaïque)")

                # Masque eau
                figm = plt.figure(figsize=(9, 6))
                plt.imshow(water_mask, cmap="gray", interpolation="nearest")
                plt.title("Masque eau libre (NDWI > Otsu)")
                plt.tight_layout()
                st.pyplot(figm)

                # Centre vs bordures sud (AUTO)
                st.subheader("📍 Centre vs bordures sud (détection auto)")
                tiles = list_tiles(ndwi_paths)
                center_tile = pick_center_tile(tiles)
                south_tiles = pick_south_tiles(tiles, south_is_j_max=south_is_j_max, k=2)

                if center_tile is None or len(south_tiles) == 0:
                    st.warning("Impossible de détecter tile_NxN_i_j dans les noms. Diagnostic tuiles non disponible.")
                    st.stop()

                p_center = center_tile[0]
                p_s1 = south_tiles[0][0]
                p_s2 = south_tiles[1][0] if len(south_tiles) > 1 else p_s1

                st.write("Tuiles détectées (NDWI) :")
                st.code(
                    f"CENTRE : {os.path.basename(p_center)}\n"
                    f"SUD_1  : {os.path.basename(p_s1)}\n"
                    f"SUD_2  : {os.path.basename(p_s2)}"
                )

                center_metrics = compute_zone_metrics(p_center, ndwi_paths, ndti_paths, ndci_paths)
                south1_metrics = compute_zone_metrics(p_s1, ndwi_paths, ndti_paths, ndci_paths)
                south2_metrics = compute_zone_metrics(p_s2, ndwi_paths, ndti_paths, ndci_paths)

                # Table stats
                rows = []
                for zone_name, met in [("CENTRE", center_metrics), ("SUD_1", south1_metrics), ("SUD_2", south2_metrics)]:
                    for idx_name in ["NDWI", "NDTI", "NDCI"]:
                        s = met.get(idx_name)
                        if s is None:
                            continue
                        rows.append({
                            "zone": zone_name,
                            "index": idx_name,
                            "mean": s["mean"],
                            "p05": s["p05"],
                            "p95": s["p95"],
                            "std": s["std"],
                            "valid_%": s["valid_%"],
                            "file": met.get("tile_file", ""),
                        })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

                # Diagnostic (rasterio only)
                st.subheader("🚨 Diagnostic automatique (basé rasterio)")
                alerts = diagnose_from_metrics(center_metrics, south1_metrics, south2_metrics, thr_ndwi=thr)
                for level, msg in alerts:
                    if level == "warning":
                        st.warning(msg)
                    elif level == "success":
                        st.success(msg)
                    else:
                        st.info(msg)

                # Interprétation descriptive (rasterio only)
                st.subheader("🧾 Interprétation (basée rasterio)")
                desc = describe_water_state_from_rasterio(
                    center_metrics=center_metrics,
                    south1_metrics=south1_metrics,
                    south2_metrics=south2_metrics,
                    water_km2=water_km2,
                    thr_ndwi=thr
                )
                st.markdown(desc)

                # Score global (0–100) + jauge
                st.subheader("📊 Score global Qualité/Quantité (0–100)")
                score, details = compute_global_score(center_metrics, south1_metrics, south2_metrics, water_km2)
                label = score_label(score)

                s1, s2, s3 = st.columns(3)
                s1.metric("Score", f"{score:.1f} / 100")
                s2.metric("Classe", label)
                s3.metric("NDWI eau (km²)", f"{water_km2:.3f}")

                st.progress(int(round(score)))

                with st.expander("Détails du score (normalisations rasterio)"):
                    st.write({
                        "quantity_norm(0-1)": round(details["quantity_norm"], 3),
                        "turbidity_norm(0-1)": round(details["turbidity_norm"], 3),
                        "chlorophyll_norm(0-1)": round(details["chlorophyll_norm"], 3),
                        "ndti_edges_mean": details["ndti_edges"],
                        "ndci_edges_mean": details["ndci_edges"],
                        "note": "Seuils et références à calibrer avec historique multi-dates Ichkeul."
                    })

            finally:
                if tracker is not None:
                    try:
                        tracker.stop()
                    except Exception:
                        pass


# =========================================================
# PAGE 3 — Analyse IA (CSV -> modèle)
# =========================================================
elif page == "🧰 Analyse IA (CSV)":
    st.title("🧰 Analyse IA — Qualité de l'eau (CSV)")

    if not PREDICT_OK:
        st.warning("predict.py non trouvé / predict_quality indisponible. La page IA est désactivée.")
    uploaded_file = st.file_uploader("Importer CSV", type=["csv"])
    model_choice = st.selectbox("Modèle", ["Random Forest", "XGBoost"])
    model_path = "water-model1.pkl" if model_choice == "Random Forest" else "water-model2.pkl"

    if uploaded_file and PREDICT_OK:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(10), use_container_width=True)
        try:
            df_pred = predict_quality(df, df.copy(), model_path)
            st.success("✅ Prédiction OK")
            st.dataframe(df_pred.head(20), use_container_width=True)
            st.download_button(
                "📥 Télécharger résultats",
                df_pred.to_csv(index=False).encode("utf-8"),
                "predictions.csv",
                "text/csv",
            )
        except Exception as e:
            st.error(f"Erreur prédiction : {e}")


# =========================================================
# PAGE 4 — Capteurs
# =========================================================
elif page == "📡 Acquisition capteurs":
    st.title("📡 Données capteurs")
    st.info("Module prêt pour intégration terrain (UART/LoRa/HTTP).")
