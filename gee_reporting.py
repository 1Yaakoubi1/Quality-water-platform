from __future__ import annotations

import os
import tempfile
from typing import Dict, Any, Tuple

import ee
import requests


GEE_OK = True
try:
    ee.Initialize(project="cleanwater-460912")
except Exception:
    try:
        ee.Authenticate()
        ee.Initialize(project="cleanwater-460912")
    except Exception:
        GEE_OK = False


def _download_file(url: str, output_path: str, timeout: int = 180) -> str:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(r.content)
    return output_path


def _point_buffer_region(longitude: float, latitude: float, buffer_m: int = 1500):
    point = ee.Geometry.Point([longitude, latitude])
    return point.buffer(buffer_m).bounds()


def _build_s2_image(region, start_date: str, end_date: str, cloud_max: int = 20):
    def mask_s2(img):
        qa = img.select("QA60")
        cloud = qa.bitwiseAnd(1 << 10).eq(0)
        cirrus = qa.bitwiseAnd(1 << 11).eq(0)
        return img.updateMask(cloud.And(cirrus)).divide(10000)

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_max))
        .map(mask_s2)
    )

    if s2.size().getInfo() == 0:
        return None

    return s2.median().clip(region)


def _image_download_url(image, region, vis_params: Dict[str, Any], dimensions: int = 1024) -> str:
    rgb_vis = image.visualize(**vis_params)
    return rgb_vis.getThumbURL({
        "region": region,
        "dimensions": dimensions,
        "format": "png",
    })


def export_station_satellite_png(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    output_path: str | None = None,
    cloud_max: int = 20,
    layer: str = "RGB",
) -> str | None:
    """
    Exporte une image PNG Sentinel-2 pour une station.
    layer: RGB | NDWI | NDTI | NDCI
    """
    if not GEE_OK:
        return None

    region = _point_buffer_region(longitude, latitude, buffer_m=1500)
    img = _build_s2_image(region, start_date, end_date, cloud_max=cloud_max)
    if img is None:
        return None

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        output_path = tmp.name
        tmp.close()

    layer = str(layer).upper()

    if layer == "NDWI":
        derived = img.normalizedDifference(["B3", "B8"])
        vis = {"min": -0.5, "max": 0.8, "palette": ["#8c510a", "#f6e8c3", "#c7eae5", "#01665e"]}
    elif layer == "NDTI":
        derived = img.normalizedDifference(["B4", "B3"])
        vis = {"min": -0.2, "max": 0.4, "palette": ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]}
    elif layer == "NDCI":
        derived = img.normalizedDifference(["B5", "B4"])
        vis = {"min": -0.2, "max": 0.5, "palette": ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]}
    else:
        derived = img.select(["B4", "B3", "B2"])
        vis = {"bands": ["B4", "B3", "B2"], "min": 0.02, "max": 0.30}

    url = _image_download_url(derived, region, vis_params=vis, dimensions=1024)
    return _download_file(url, output_path)


def compute_satellite_summary(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    cloud_max: int = 20,
) -> Dict[str, Any]:
    """
    Calcule un petit résumé satellite autour d'une station.
    """
    if not GEE_OK:
        return {
            "gee_available": False,
            "message": "Google Earth Engine non initialisé.",
        }

    region = _point_buffer_region(longitude, latitude, buffer_m=1500)
    img = _build_s2_image(region, start_date, end_date, cloud_max=cloud_max)
    if img is None:
        return {
            "gee_available": True,
            "image_found": False,
            "message": "Aucune image Sentinel-2 trouvée sur cette période.",
        }

    ndwi = img.normalizedDifference(["B3", "B8"]).rename("NDWI")
    ndti = img.normalizedDifference(["B4", "B3"]).rename("NDTI")
    ndci = img.normalizedDifference(["B5", "B4"]).rename("NDCI")

    reducer = ee.Reducer.mean()

    ndwi_mean = ndwi.reduceRegion(reducer=reducer, geometry=region, scale=20, maxPixels=1_000_000).get("NDWI")
    ndti_mean = ndti.reduceRegion(reducer=reducer, geometry=region, scale=20, maxPixels=1_000_000).get("NDTI")
    ndci_mean = ndci.reduceRegion(reducer=reducer, geometry=region, scale=20, maxPixels=1_000_000).get("NDCI")

    out = {
        "gee_available": True,
        "image_found": True,
        "ndwi_mean": float(ndwi_mean.getInfo()) if ndwi_mean is not None else None,
        "ndti_mean": float(ndti_mean.getInfo()) if ndti_mean is not None else None,
        "ndci_mean": float(ndci_mean.getInfo()) if ndci_mean is not None else None,
    }

    messages = []

    if out["ndwi_mean"] is not None:
        if out["ndwi_mean"] < 0.05:
            messages.append("Présence d’eau faible ou assèchement relatif probable autour de la station.")
        else:
            messages.append("Présence d’eau détectée autour de la station.")

    if out["ndti_mean"] is not None:
        if out["ndti_mean"] > 0.15:
            messages.append("Proxy satellite de turbidité élevé.")
        elif out["ndti_mean"] > 0.08:
            messages.append("Proxy satellite de turbidité modéré.")
        else:
            messages.append("Proxy satellite de turbidité faible.")

    if out["ndci_mean"] is not None:
        if out["ndci_mean"] > 0.20:
            messages.append("Risque d’eutrophisation ou chlorophylle élevée détecté par satellite.")
        elif out["ndci_mean"] > 0.10:
            messages.append("Signal chlorophyllien modéré à surveiller.")
        else:
            messages.append("Signal chlorophyllien faible.")

    out["summary_messages"] = messages
    return out


def get_station_coords_and_dates(row: Dict[str, Any]) -> Tuple[float | None, float | None, str, str]:
    """
    Extrait latitude/longitude depuis row, plus une plage de dates par défaut.
    """
    lat = row.get("latitude")
    lon = row.get("longitude")

    try:
        lat = None if lat is None else float(lat)
    except Exception:
        lat = None

    try:
        lon = None if lon is None else float(lon)
    except Exception:
        lon = None

    # Fenêtre simple par défaut : dernière année glissante fixe
    start_date = "2024-01-01"
    end_date = "2024-12-31"

    return lat, lon, start_date, end_date