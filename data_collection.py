import ipywidgets as widgets
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import geemap
import ee

try:
    ee.Initialize(project="cleanwater-460912")
except Exception:
    try:
        ee.Authenticate()
        ee.Initialize(project="cleanwater-460912")
    except Exception as e:
        raise RuntimeError(f"Google Earth Engine non initialisé : {e}")


def get_data(long, lat, start_date, end_date):
    Map = geemap.Map()
    geometry = ee.Geometry.Point([long, lat])

    image = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterBounds(geometry)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 20))
        .first()
    )

    ndwi = image.normalizedDifference(["B3", "B8"])
    ndwiMasked = ndwi.updateMask(ndwi.gte(0.4))
    ndwiMasked1 = ndwiMasked.toInt()

    vectors = ndwiMasked1.reduceToVectors(
        scale=30.0,
        geometryType="polygon",
        eightConnected=False,
        maxPixels=10000000,
        bestEffort=True,
    )

    Map.addLayer(geometry)

    sentinel = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterBounds(vectors)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .median()
    )

    mndwi = sentinel.normalizedDifference(["B3", "B11"]).rename("mndwi")
    mndwitr = mndwi.gt(0)
    ndsi = sentinel.normalizedDifference(["B11", "B12"]).rename("ndsi")
    ndti = sentinel.normalizedDifference(["B4", "B3"]).rename("ndti")
    ndci = sentinel.normalizedDifference(["B5", "B4"]).rename("ndci")

    ph = ee.Image(8.339).subtract(
        ee.Image(0.827).multiply(sentinel.select("B1").divide(sentinel.select("B8")))
    ).rename("ph")

    dissolvedoxygen = (
        ee.Image(-0.0167).multiply(sentinel.select("B8"))
        .add(ee.Image(0.0067).multiply(sentinel.select("B9")))
        .add(ee.Image(0.0083).multiply(sentinel.select("B11")))
        .add(ee.Image(9.577))
        .rename("dissolvedoxygen")
    )

    col = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterDate(start_date, end_date)
        .filterBounds(vectors)
        .median()
    )

    temp = col.select("ST_B.*").multiply(0.00341802).add(149.0).subtract(273.15).rename("temp")

    data = ee.ImageCollection("COPERNICUS/S3/OLCI").filterDate(start_date, end_date).filterBounds(vectors)

    rgb = (
        data.select(["Oa08_radiance", "Oa06_radiance", "Oa04_radiance"])
        .median()
        .multiply(ee.Image([0.00876539, 0.0123538, 0.0115198]))
        .clip(vectors)
    )

    dom = rgb.select("Oa08_radiance").divide(rgb.select("Oa04_radiance")).rename("dom")
    sm = rgb.select("Oa08_radiance").divide(rgb.select("Oa06_radiance")).rename("suspended_matter")

    def extract_array(image_band, band_name, scale=100, tile_scale=None):
        kwargs = {
            "reducer": ee.Reducer.toList(),
            "geometry": vectors,
            "scale": scale,
        }
        if tile_scale is not None:
            kwargs["tileScale"] = tile_scale

        latlon = ee.Image.pixelLonLat().addBands(image_band).reduceRegion(**kwargs)
        return np.array(ee.Array(latlon.get(band_name)).getInfo())

    data_dom = extract_array(dom, "dom", scale=100, tile_scale=16)
    data_sm = extract_array(sm, "suspended_matter", scale=100, tile_scale=16)
    data_lst = extract_array(temp, "temp", scale=100)
    data_ndti = extract_array(ndti, "ndti", scale=100)
    data_ndsi = extract_array(ndsi, "ndsi", scale=100)
    data_ndci = extract_array(ndci, "ndci", scale=100)
    data_do = extract_array(dissolvedoxygen, "dissolvedoxygen", scale=100, tile_scale=16)
    data_ph = extract_array(ph, "ph", scale=100)

    df = pd.concat(
        [
            pd.DataFrame(data_do, columns=["Dissolved Oxygen"]),
            pd.DataFrame(data_ndsi, columns=["Salinity"]),
            pd.DataFrame(data_lst, columns=["Temperature"]),
            pd.DataFrame(data_ph, columns=["pH"]),
            pd.DataFrame(data_ndti, columns=["Turbidity"]),
            pd.DataFrame(data_dom, columns=["Dissolved Organic Matter"]),
            pd.DataFrame(data_sm, columns=["Suspended Matter"]),
            pd.DataFrame(data_ndci, columns=["Chlorophyll"]),
        ],
        axis=1,
        sort=False,
    )

    geemap.ee_export_image(
        ndwi,
        filename="TUNISIE.tif",
        scale=10,
        region=geometry.buffer(1000).bounds().getInfo(),
        file_per_band=False,
    )

    return df


def send_df(df2):
    df2 = df2.dropna()
    df2["Dissolved Organic Matter"] = df2["Dissolved Organic Matter"] * 1000
    df2["Suspended Matter"] = df2["Suspended Matter"] * 1000
    test = pd.DataFrame(
        MinMaxScaler().fit_transform(df2.drop(["Salinity"], axis=1)),
        columns=df2.drop(["Salinity"], axis=1).columns,
    )
    return df2, test