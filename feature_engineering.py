from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd


SENSOR_COLUMNS = [
    "temperature",
    "ph",
    "turbidity",
    "dissolved_oxygen",
    "salinity",
    "conductivity",
]

SATELLITE_COLUMNS = [
    "ndwi",
    "ndti",
    "ndci",
    "chlorophyll",
    "dissolved_organic_matter",
    "suspended_matter",
]


def ensure_datetime(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    if col in out.columns:
        out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def sort_by_time(df: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
    out = ensure_datetime(df, time_col)
    if time_col in out.columns:
        out = out.sort_values(time_col)
    return out


def add_time_features(df: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
    out = ensure_datetime(df, time_col)

    if time_col not in out.columns:
        return out

    dt = out[time_col]
    out["year"] = dt.dt.year
    out["month"] = dt.dt.month
    out["day"] = dt.dt.day
    out["hour"] = dt.dt.hour
    out["dayofweek"] = dt.dt.dayofweek
    out["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)

    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)

    return out


def add_station_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "station_id" not in out.columns:
        out["station_id"] = "unknown"
    out["station_id"] = out["station_id"].astype(str)
    return out


def coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_basic_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if {"temperature", "dissolved_oxygen"}.issubset(out.columns):
        out["temp_do_ratio"] = out["temperature"] / (out["dissolved_oxygen"] + 1e-6)

    if {"turbidity", "chlorophyll"}.issubset(out.columns):
        out["turbidity_chlorophyll_ratio"] = out["turbidity"] / (out["chlorophyll"] + 1e-6)

    if {"salinity", "temperature"}.issubset(out.columns):
        out["salinity_temperature_product"] = out["salinity"] * out["temperature"]

    if {"conductivity", "salinity"}.issubset(out.columns):
        out["conductivity_salinity_ratio"] = out["conductivity"] / (out["salinity"] + 1e-6)

    if {"ndti", "ndci"}.issubset(out.columns):
        out["ndti_ndci_product"] = out["ndti"] * out["ndci"]

    return out


def add_lag_features(
    df: pd.DataFrame,
    columns: List[str] | None = None,
    lags: List[int] | None = None,
    group_col: str = "station_id",
) -> pd.DataFrame:
    out = sort_by_time(df)
    columns = columns or SENSOR_COLUMNS
    lags = lags or [1, 2, 3]

    if group_col not in out.columns:
        out[group_col] = "global"

    for col in columns:
        if col not in out.columns:
            continue
        for lag in lags:
            out[f"{col}_lag_{lag}"] = out.groupby(group_col)[col].shift(lag)

    return out


def add_rolling_features(
    df: pd.DataFrame,
    columns: List[str] | None = None,
    windows: List[int] | None = None,
    group_col: str = "station_id",
) -> pd.DataFrame:
    out = sort_by_time(df)
    columns = columns or SENSOR_COLUMNS
    windows = windows or [3, 6, 12]

    if group_col not in out.columns:
        out[group_col] = "global"

    for col in columns:
        if col not in out.columns:
            continue

        for w in windows:
            out[f"{col}_roll_mean_{w}"] = (
                out.groupby(group_col)[col]
                .transform(lambda s: s.rolling(window=w, min_periods=1).mean())
            )
            out[f"{col}_roll_std_{w}"] = (
                out.groupby(group_col)[col]
                .transform(lambda s: s.rolling(window=w, min_periods=1).std())
            )
            out[f"{col}_roll_min_{w}"] = (
                out.groupby(group_col)[col]
                .transform(lambda s: s.rolling(window=w, min_periods=1).min())
            )
            out[f"{col}_roll_max_{w}"] = (
                out.groupby(group_col)[col]
                .transform(lambda s: s.rolling(window=w, min_periods=1).max())
            )

    return out


def add_trend_features(
    df: pd.DataFrame,
    columns: List[str] | None = None,
    group_col: str = "station_id",
) -> pd.DataFrame:
    out = sort_by_time(df)
    columns = columns or SENSOR_COLUMNS

    if group_col not in out.columns:
        out[group_col] = "global"

    for col in columns:
        if col not in out.columns:
            continue
        out[f"{col}_diff_1"] = out.groupby(group_col)[col].diff(1)
        out[f"{col}_pct_change_1"] = out.groupby(group_col)[col].pct_change()

    return out


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "temperature" in out.columns:
        out["flag_temperature_outlier"] = ((out["temperature"] < 0) | (out["temperature"] > 45)).astype(int)

    if "ph" in out.columns:
        out["flag_ph_outlier"] = ((out["ph"] < 0) | (out["ph"] > 14)).astype(int)

    if "dissolved_oxygen" in out.columns:
        out["flag_do_outlier"] = ((out["dissolved_oxygen"] < 0) | (out["dissolved_oxygen"] > 20)).astype(int)

    if "turbidity" in out.columns:
        out["flag_turbidity_outlier"] = (out["turbidity"] < 0).astype(int)

    if "conductivity" in out.columns:
        out["flag_conductivity_outlier"] = (out["conductivity"] < 0).astype(int)

    return out


def build_features(
    df: pd.DataFrame,
    include_lags: bool = True,
    include_rolling: bool = True,
    include_trends: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    out = add_station_features(out)
    out = ensure_datetime(out, "timestamp")
    out = coerce_numeric_columns(out, SENSOR_COLUMNS + SATELLITE_COLUMNS)
    out = add_time_features(out, "timestamp")
    out = add_basic_interactions(out)

    if include_lags:
        out = add_lag_features(out, columns=SENSOR_COLUMNS)

    if include_rolling:
        out = add_rolling_features(out, columns=SENSOR_COLUMNS)

    if include_trends:
        out = add_trend_features(out, columns=SENSOR_COLUMNS)

    out = add_quality_flags(out)
    return out


def build_latest_station_features(df: pd.DataFrame) -> pd.DataFrame:
    out = build_features(df, include_lags=True, include_rolling=True, include_trends=True)
    out = sort_by_time(out)

    if "station_id" not in out.columns:
        return out.tail(1)

    latest = out.groupby("station_id", as_index=False).tail(1)
    return latest.reset_index(drop=True)