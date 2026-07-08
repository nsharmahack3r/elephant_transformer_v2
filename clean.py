#!/usr/bin/env python3
"""
clean_dataset.py — one-time cleaning of the raw elephant GPS export.

Reads the raw CSV at ACTUAL_PATH, applies quality filters, drops unusable /
constant / redundant columns, imputes covariate gaps, and writes:

  * a cleaned CSV      -> CLEANED_PATH        (from .env)
  * a small sample CSV -> SAMPLE_PATH         (from .env, used for smoke tests)
  * a summary JSON     -> <data_dir>/dataset.json

This script is intentionally self-contained (only pandas / numpy / dotenv) so it
can run BEFORE the rest of the package exists. Run it once:

    uv run scripts/clean_dataset.py

Thresholds can be overridden on the command line; see --help.

Paths come from .env (assumed to already exist):
    ACTUAL_PATH=data/actual_data.csv
    SAMPLE_PATH=data/sample.csv
    CLEANED_PATH=data/cleaned.csv
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

# Columns we keep. Everything else is dropped. The three identity/geometry
# columns plus the eight retained covariates. (NDWI and water_occ_1km are
# dropped: ~100% missing. sensor-type / taxon / study-name / tag id are
# constant or redundant. timestamp_ms* are numeric duplicates of `timestamp`.)
ID_COL = "individual-local-identifier"
TIME_COL = "timestamp"
LON_COL = "location-long"
LAT_COL = "location-lat"

CONTINUOUS_COVARS = [
    "aspect_deg",
    "human_settle",
    "elevation_m",
    "slope_deg",
    "EVI",
    "LST_celsius",
    "NDVI",
]
CATEGORICAL_COVARS = ["LULC_class"]  # coded land-cover class; do NOT interpolate

KEEP_COLS = (
    ["event-id", TIME_COL, LON_COL, LAT_COL, ID_COL]
    + CATEGORICAL_COVARS
    + CONTINUOUS_COVARS
)

# Plausible Etosha / northern-Namibia bounding box. The raw data contains at
# least one corrupt fix near lon=0.99 which this removes. Widen if your study
# region genuinely extends further.
DEFAULT_BBOX = {"lon_min": 13.0, "lon_max": 18.0, "lat_min": -21.0, "lat_max": -17.0}

# An elephant sustaining >7 m/s (~25 km/h) between fixes is a GPS error, not
# biology. Applied iteratively because removing one bad fix changes its
# neighbours' implied speeds.
DEFAULT_MAX_SPEED_MPS = 7.0
SPEED_FILTER_MAX_PASSES = 5

# Sample file: a few individuals, capped, contiguous in time — enough to
# exercise every code path in training/eval quickly.
SAMPLE_N_INDIVIDUALS = 3
SAMPLE_ROWS_PER_INDIVIDUAL = 6000

EARTH_RADIUS_M = 6_371_000.0


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres between arrays of points."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def log(msg: str) -> None:
    print(f"[clean] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Cleaning steps                                                               #
# --------------------------------------------------------------------------- #


def load_raw(path: Path) -> pd.DataFrame:
    log(f"reading raw data from {path} ...")
    df = pd.read_csv(path, low_memory=False)
    log(f"  raw shape: {df.shape}")
    return df


def apply_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Use Movebank flags before we drop the columns that carry them."""
    if "visible" in df.columns:
        before = len(df)
        vis = df["visible"]
        # accept true / "true" / 1
        keep = vis.astype(str).str.lower().isin(["true", "1", "1.0"])
        df = df[keep]
        log(f"  visible filter: dropped {before - len(df)} hidden rows")

    if "manually-marked-outlier" in df.columns:
        before = len(df)
        flagged = pd.to_numeric(df["manually-marked-outlier"], errors="coerce") == 1
        df = df[~flagged]
        log(f"  manual-outlier filter: dropped {before - len(df)} flagged rows")

    return df


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in KEEP_COLS if c in df.columns]
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        log(f"  WARNING: expected columns absent from raw file: {missing}")
    return df[present].copy()


def parse_and_basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True, errors="coerce")
    df[LON_COL] = pd.to_numeric(df[LON_COL], errors="coerce")
    df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors="coerce")

    before = len(df)
    df = df.dropna(subset=[TIME_COL, LON_COL, LAT_COL, ID_COL])
    log(f"  dropped {before - len(df)} rows missing time/lon/lat/id")

    df[ID_COL] = df[ID_COL].astype(str)
    return df


def bbox_filter(df: pd.DataFrame, bbox: dict) -> pd.DataFrame:
    before = len(df)
    m = df[LON_COL].between(bbox["lon_min"], bbox["lon_max"]) & df[LAT_COL].between(
        bbox["lat_min"], bbox["lat_max"]
    )
    df = df[m]
    log(f"  bbox filter: dropped {before - len(df)} out-of-region fixes")
    return df


def dedup(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.sort_values([ID_COL, TIME_COL], kind="mergesort")
    df = df.drop_duplicates(subset=[ID_COL, TIME_COL], keep="first")
    log(f"  dedup: dropped {before - len(df)} duplicate (id,timestamp) rows")
    return df


def _speed_mask(df: pd.DataFrame, max_speed: float) -> pd.Series:
    g = df.groupby(ID_COL, sort=False)
    dt = g[TIME_COL].diff().dt.total_seconds()
    plat = g[LAT_COL].shift()
    plon = g[LON_COL].shift()
    dist = haversine_m(plat.values, plon.values, df[LAT_COL].values, df[LON_COL].values)
    dist = pd.Series(dist, index=df.index)
    speed = dist / dt.replace(0.0, np.nan)
    # first fix of each track has NaN dt -> keep it; NaN speed -> keep
    return (speed.isna()) | (speed <= max_speed)


def speed_filter(df: pd.DataFrame, max_speed: float) -> pd.DataFrame:
    for i in range(SPEED_FILTER_MAX_PASSES):
        mask = _speed_mask(df, max_speed)
        n_bad = int((~mask).sum())
        if n_bad == 0:
            break
        df = df[mask]
        log(f"  speed filter pass {i + 1}: dropped {n_bad} impossible-speed fixes")
    return df


def impute_covariates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values([ID_COL, TIME_COL], kind="mergesort")

    # Continuous: time-agnostic linear interpolation within each individual,
    # then a global median fallback for any remaining gaps.
    for col in CONTINUOUS_COVARS:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df.groupby(ID_COL, sort=False)[col].transform(
            lambda s: s.interpolate(limit_direction="both")
        )
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Categorical land-cover: forward/backward fill within individual, then mode.
    for col in CATEGORICAL_COVARS:
        if col not in df.columns:
            continue
        df[col] = df.groupby(ID_COL, sort=False)[col].transform(
            lambda s: s.ffill().bfill()
        )
        if df[col].isna().any():
            mode = df[col].mode(dropna=True)
            if len(mode):
                df[col] = df[col].fillna(mode.iloc[0])

    return df


# --------------------------------------------------------------------------- #
# Summary + sample                                                            #
# --------------------------------------------------------------------------- #


def build_summary(df: pd.DataFrame) -> dict:
    g = df.groupby(ID_COL, sort=False)

    # sampling intervals across all tracks
    dt = g[TIME_COL].diff().dt.total_seconds().dropna()
    dt = dt[dt > 0]

    rows_per = df[ID_COL].value_counts().to_dict()
    periods = {
        eid: [str(sub[TIME_COL].min()), str(sub[TIME_COL].max())] for eid, sub in g
    }
    miss_counts = df.isna().sum().to_dict()
    miss_pct = (df.isna().mean() * 100).round(4).to_dict()

    return {
        "dataset_overview": {
            "total_rows": int(len(df)),
            "total_elephants": int(df[ID_COL].nunique()),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "time_range_utc": [str(df[TIME_COL].min()), str(df[TIME_COL].max())],
            "spatial_bounding_box": {
                "lat_min": float(df[LAT_COL].min()),
                "lat_max": float(df[LAT_COL].max()),
                "lon_min": float(df[LON_COL].min()),
                "lon_max": float(df[LON_COL].max()),
            },
            "row_meaning": (
                "Each row is one GPS fix for one elephant at one timestamp, "
                "with environmental covariates sampled at that location/time. "
                "Cleaned: quality-filtered, region-clipped, speed-filtered, "
                "covariate-imputed; unusable/constant/redundant columns removed."
            ),
        },
        "rows_per_elephant": {k: int(v) for k, v in rows_per.items()},
        "tracking_period_per_elephant": periods,
        "missing_data_counts": {k: int(v) for k, v in miss_counts.items()},
        "missing_data_pct": {k: float(v) for k, v in miss_pct.items()},
        "sampling_interval_seconds": {
            "median_seconds": float(dt.median()) if len(dt) else None,
            "mean_seconds": float(dt.mean()) if len(dt) else None,
            "p10_seconds": float(dt.quantile(0.10)) if len(dt) else None,
            "p90_seconds": float(dt.quantile(0.90)) if len(dt) else None,
            "min_seconds": float(dt.min()) if len(dt) else None,
            "max_seconds": float(dt.max()) if len(dt) else None,
            "n_intervals_sampled": int(len(dt)),
        },
    }


def build_sample(df: pd.DataFrame) -> pd.DataFrame:
    top = df[ID_COL].value_counts().head(SAMPLE_N_INDIVIDUALS).index.tolist()
    parts = []
    for eid in top:
        sub = df[df[ID_COL] == eid].sort_values(TIME_COL, kind="mergesort")
        parts.append(sub.head(SAMPLE_ROWS_PER_INDIVIDUAL))
    sample = pd.concat(parts, ignore_index=True)
    log(f"  sample: {len(sample)} rows across {len(top)} individuals {top}")
    return sample


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def main() -> None:
    load_dotenv()

    ap = argparse.ArgumentParser(description="One-time elephant GPS cleaning.")
    ap.add_argument("--actual-path", default=os.getenv("ACTUAL_PATH"))
    ap.add_argument("--cleaned-path", default=os.getenv("CLEANED_PATH"))
    ap.add_argument("--sample-path", default=os.getenv("SAMPLE_PATH"))
    ap.add_argument("--max-speed-mps", type=float, default=DEFAULT_MAX_SPEED_MPS)
    ap.add_argument("--lon-min", type=float, default=DEFAULT_BBOX["lon_min"])
    ap.add_argument("--lon-max", type=float, default=DEFAULT_BBOX["lon_max"])
    ap.add_argument("--lat-min", type=float, default=DEFAULT_BBOX["lat_min"])
    ap.add_argument("--lat-max", type=float, default=DEFAULT_BBOX["lat_max"])
    args = ap.parse_args()

    if not args.actual_path or not args.cleaned_path or not args.sample_path:
        raise SystemExit(
            "ACTUAL_PATH, CLEANED_PATH and SAMPLE_PATH must be set (in .env or via flags)."
        )

    actual = Path(args.actual_path)
    cleaned = Path(args.cleaned_path)
    sample = Path(args.sample_path)
    summary_path = cleaned.parent / "dataset.json"
    for p in (cleaned, sample, summary_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    bbox = {
        "lon_min": args.lon_min,
        "lon_max": args.lon_max,
        "lat_min": args.lat_min,
        "lat_max": args.lat_max,
    }

    df = load_raw(actual)
    df = apply_quality_flags(df)
    df = select_columns(df)
    df = parse_and_basic_clean(df)
    df = bbox_filter(df, bbox)
    df = dedup(df)
    df = speed_filter(df, args.max_speed_mps)
    df = impute_covariates(df)
    df = df.sort_values([ID_COL, TIME_COL], kind="mergesort").reset_index(drop=True)

    log(f"final cleaned shape: {df.shape}")
    df.to_csv(cleaned, index=False)
    log(f"wrote cleaned data -> {cleaned}")

    summary = build_summary(df)
    summary_path.write_text(json.dumps(summary, indent=2))
    log(f"wrote summary -> {summary_path}")

    sample_df = build_sample(df)
    sample_df.to_csv(sample, index=False)
    log(f"wrote sample -> {sample}")

    log("done.")


if __name__ == "__main__":
    main()
