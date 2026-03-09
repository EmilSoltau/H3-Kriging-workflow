"""Parquet -> Variogram -> Kriging workflow.

Same pipeline as kriging_osm.py but reads from a local Parquet file
(produced by fetch_data.py) instead of calling the CH API.

1. Load geochemistry data from data/till_al.parquet
2. Prepare and aggregate values into H3 cells
3. Build distance-based empirical variogram
4. Fit theoretical variogram (best of spherical/gaussian/exponential/matern)
5. Run ordinary kriging interpolation with local neighborhoods
6. Export PNG maps over OpenStreetMap
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import contextily as cx
import geopandas as gpd
import gstools as gs
import h3
import h3ronpy as h3r
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

# Conversion factors to normalize all concentration units to ppm
UNIT_TO_PPM_FACTOR = {
    "ppm": 1.0, "ppb": 1e-3, "pct": 1e4, "percent": 1e4,
    "%": 1e4, "mg/kg": 1.0, "g/t": 1.0,
}

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_PARQUET = DATA_DIR / "till_al.parquet"

DEFAULT_CONFIG: dict[str, Any] = {
    "parquet_path": str(DEFAULT_PARQUET),
    "bbox": {
        "min_lon": 25.9786957,
        "min_lat": 62.8128557,
        "max_lon": 27.4820141,
        "max_lat": 63.6108582,
    },
    "resolution": 7,
    "min_resolution": 7,
    "sampletype_id": 24,
    "element_code": "Al",
    "apply_log10": True,
    "out_dir": "output",
    "max_neighbors": 64,
}


# ---------------------------------------------------------------------------
# Data loading (Parquet)
# ---------------------------------------------------------------------------

def load_parquet(path: str | Path) -> pd.DataFrame:
    """Read raw geochemistry data from a Parquet file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Parquet file not found: {path}\n"
            "Run fetch_data.py first to download the data."
        )
    df = pd.read_parquet(path)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def normalize_h3_u64(series: pd.Series) -> pd.Series:
    """Parse H3 cell strings into uint64 values, setting invalid ones to NA."""
    parsed = h3r.cells_parse(series.astype(str).to_list(), set_failing_to_invalid=True).to_pylist()
    return pd.Series(parsed, dtype="UInt64")


def parse_h3_column(df: pd.DataFrame) -> pd.DataFrame:
    """Find the H3 column, parse to uint64, drop invalid rows."""
    if "h3_id" in df.columns:
        df = df.rename(columns={"h3_id": "h3_raw"})
    elif "h3_index" in df.columns:
        df = df.rename(columns={"h3_index": "h3_raw"})
    if "h3_raw" not in df.columns:
        raise ValueError(f"No H3 column found in {list(df.columns)}")

    df = df.copy()
    df["h3_u64"] = normalize_h3_u64(df["h3_raw"])
    n_before = len(df)
    df = df.dropna(subset=["h3_u64"]).copy()
    if n_before != len(df):
        logger.info("Dropped %d rows with invalid H3 cells", n_before - len(df))
    return df


# ---------------------------------------------------------------------------
# Value preparation
# ---------------------------------------------------------------------------

def prepare_values(df: pd.DataFrame, sampletype_id: int, element_code: str, apply_log10: bool) -> pd.DataFrame:
    logger.info(
        "Preparing values: sampletype_id=%d, element=%s, log10=%s, input_rows=%d",
        sampletype_id, element_code, apply_log10, len(df),
    )
    req = {"h3_u64", "sampletype_id", "element_code"}
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["sampletype_id"] = pd.to_numeric(out["sampletype_id"], errors="coerce")
    out["element_code"] = out["element_code"].astype(str).str.upper()
    out = out[
        (out["sampletype_id"] == sampletype_id)
        & (out["element_code"] == element_code.upper())
    ].copy()
    logger.debug("After sampletype/element filter: %d rows", len(out))
    if out.empty:
        raise ValueError("No rows after sampletype/element filtering.")

    # Prefer pre-computed value_ppm; fall back to raw value * unit conversion
    ppm = pd.Series(np.nan, index=out.index, dtype=float)
    if "value_ppm" in out.columns:
        ppm = pd.to_numeric(out["value_ppm"], errors="coerce")
    if "value" in out.columns:
        raw = pd.to_numeric(out["value"], errors="coerce")
        if "unit_name" in out.columns:
            units = out["unit_name"].astype(str).str.strip().str.lower()
            ppm = ppm.fillna(raw * units.map(UNIT_TO_PPM_FACTOR))
        else:
            ppm = ppm.fillna(raw)

    out["value_ppm_norm"] = ppm
    out = out.dropna(subset=["h3_u64", "value_ppm_norm"]).copy()
    n_non_positive = int((out["value_ppm_norm"] <= 0).sum())
    out = out[out["value_ppm_norm"] > 0].copy()
    if n_non_positive:
        logger.debug("Dropped %d non-positive values", n_non_positive)
    if out.empty:
        raise ValueError("No positive values remain for variogram calculation.")

    # Log-transform reduces skewness in geochemical data (common for trace elements)
    out["value_work"] = (
        np.log10(out["value_ppm_norm"]) if apply_log10 else out["value_ppm_norm"]
    )
    out["h3_u64"] = out["h3_u64"].astype("uint64")
    logger.info(
        "Prepared %d values, range=[%.4f, %.4f]",
        len(out), float(out["value_work"].min()), float(out["value_work"].max()),
    )
    return out[["h3_u64", "value_work"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# H3 spatial utilities
# ---------------------------------------------------------------------------

def aggregate_to_resolution(cell_values: pd.DataFrame, target_res: int) -> pd.DataFrame:
    """Re-parent cells to target resolution and average values per cell."""
    logger.debug("Aggregating %d cells to resolution %d", len(cell_values), target_res)
    cells = [int(v) for v in cell_values["h3_u64"].tolist() if pd.notna(v)]
    if not cells:
        return pd.DataFrame(columns=["h3_u64", "value"])

    changed = h3r.change_resolution_list(cells, target_res).to_pylist()
    parent_uint = [int(ch[0]) if ch else None for ch in changed]
    tbl = (
        pd.DataFrame({
            "h3_u64": parent_uint,
            "value": pd.to_numeric(cell_values["value_work"], errors="coerce"),
        })
        .dropna(subset=["h3_u64", "value"])
        .copy()
    )
    if tbl.empty:
        return pd.DataFrame(columns=["h3_u64", "value"])

    out = tbl.groupby("h3_u64", as_index=False)["value"].mean().sort_values("h3_u64")
    out["h3_u64"] = pd.to_numeric(out["h3_u64"], errors="coerce").astype("uint64")
    logger.debug("Aggregated to %d unique cells at resolution %d", len(out), target_res)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Distance-based empirical variogram
# ---------------------------------------------------------------------------

# Cap for O(n²) pairwise distance computation; kriging still uses all points
_VARIOGRAM_MAX_POINTS = 1500


def build_distance_variogram(
    points_df: pd.DataFrame,
    n_bins: int = 15,
    max_lag_fraction: float = 0.5,
) -> pd.DataFrame:
    """Compute empirical variogram from pairwise Haversine distances."""
    n_total = len(points_df)
    if n_total > _VARIOGRAM_MAX_POINTS:
        sample = points_df.sample(n=_VARIOGRAM_MAX_POINTS, random_state=42)
        logger.info(
            "Subsampled %d -> %d points for variogram (kriging uses all %d)",
            n_total, _VARIOGRAM_MAX_POINTS, n_total,
        )
    else:
        sample = points_df

    lat = sample["lat"].to_numpy(dtype=float)
    lon = sample["lon"].to_numpy(dtype=float)
    val = sample["value"].to_numpy(dtype=float)
    n = len(val)
    logger.info("Building distance-based variogram from %d points", n)

    # Upper-triangle pairwise Haversine distances (metres)
    d = _haversine_matrix(lat, lon, lat, lon)
    i_upper, j_upper = np.triu_indices(n, k=1)
    dists = d[i_upper, j_upper]
    diffs = val[i_upper] - val[j_upper]

    # Limit lags to max_lag_fraction of maximum distance
    max_lag = float(np.max(dists)) * max_lag_fraction
    mask = dists <= max_lag
    dists = dists[mask]
    diffs = diffs[mask]

    if len(dists) < 3:
        raise ValueError(
            f"Only {len(dists)} pairs within max lag {max_lag:.0f} m — "
            "not enough for variogram estimation."
        )

    # Bin pairs by equal-width distance intervals
    edges = np.linspace(0.0, max_lag, n_bins + 1)
    bin_idx = np.clip(np.digitize(dists, edges) - 1, 0, n_bins - 1)

    rows: list[dict[str, float]] = []
    for b in range(n_bins):
        in_bin = bin_idx == b
        n_pairs = int(in_bin.sum())
        if n_pairs < 1:
            continue
        bin_dists = dists[in_bin]
        bin_diffs = diffs[in_bin]
        lag_m = float(np.mean(bin_dists))
        abs_diff = np.abs(bin_diffs)
        mean_root_abs = float(np.mean(np.sqrt(abs_diff)))
        mean_sqdiff = float(np.mean(bin_diffs ** 2))
        # Cressie-Hawkins: robust to outliers unlike classical (Matheron) estimator
        denom = 0.457 + 0.494 / n_pairs + 0.045 / (n_pairs * n_pairs)
        gamma_robust = 0.5 * (mean_root_abs ** 4) / denom
        gamma_classic = 0.5 * mean_sqdiff
        rows.append({"lag_m": lag_m, "cressie_hawkins": gamma_robust,
                      "classical": gamma_classic, "n_pairs": float(n_pairs)})

    if not rows:
        raise ValueError("No empirical variogram bins produced.")

    result = pd.DataFrame(rows).sort_values("lag_m").reset_index(drop=True)
    logger.info(
        "Distance variogram: %d bins, lag range [%.0f, %.0f] m, %d total pairs",
        len(result), float(result["lag_m"].min()), float(result["lag_m"].max()),
        int(result["n_pairs"].sum()),
    )
    return result


# Average H3 cell area per resolution (km²) — for fast grid size estimation
_H3_AVG_AREA_KM2 = {
    0: 4_357_449.416, 1: 609_788.441, 2: 86_801.780, 3: 12_393.434,
    4: 1_770.348, 5: 252.903, 6: 36.129, 7: 5.161,
    8: 0.737, 9: 0.105, 10: 0.015, 11: 0.002, 12: 0.0003,
}


def choose_prediction_resolution(
    bbox: dict[str, float], n_obs: int,
    max_cells: int = 50_000, min_res: int = 4, max_res: int = 12,
) -> int:
    """Pick the finest H3 resolution that keeps the grid under max_cells."""
    lat_mid = np.deg2rad((bbox["min_lat"] + bbox["max_lat"]) / 2.0)
    lat_km = (bbox["max_lat"] - bbox["min_lat"]) * 111.32
    lon_km = (bbox["max_lon"] - bbox["min_lon"]) * 111.32 * np.cos(lat_mid)
    bbox_area_km2 = lat_km * lon_km

    best_res = min_res
    for res in range(max_res, min_res - 1, -1):
        est_cells = bbox_area_km2 / _H3_AVG_AREA_KM2.get(res, 0.0003)
        if est_cells <= max_cells * 1.2:
            best_res = res
            break

    cells = h3_cells_covering_bbox(bbox, best_res)
    if len(cells) > max_cells and best_res > min_res:
        best_res -= 1
        cells = h3_cells_covering_bbox(bbox, best_res)

    logger.info("Prediction resolution: %d (%d cells, cap=%d)", best_res, len(cells), max_cells)
    return best_res


# ---------------------------------------------------------------------------
# Variogram fitting (GSTools)
# ---------------------------------------------------------------------------

_EARTH_RADIUS_M = 6_371_000.0

# Candidate models — best fit (lowest weighted RSS) is selected automatically
_GSTOOLS_MODELS: dict[str, type[gs.CovModel]] = {
    "spherical": gs.Spherical,
    "gaussian": gs.Gaussian,
    "exponential": gs.Exponential,
    "matern": gs.Matern,
}


def _estimate_practical_range(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> tuple[float, float]:
    """Estimate sill and range from empirical variogram for optimizer seeding."""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    ws = w[order] if w is not None else np.ones_like(ys)
    n_plateau = max(1, len(ys) // 3)
    est_sill = float(np.average(ys[-n_plateau:], weights=ws[-n_plateau:]))
    threshold = 0.90 * est_sill
    cumavg = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    above = np.where(cumavg >= threshold)[0]
    est_range = float(xs[above[0]]) if len(above) > 0 else float(xs[-1])
    return est_sill, est_range


def fit_theoretical(variogram_df: pd.DataFrame) -> dict[str, Any]:
    """Fit theoretical variogram model, return best by WRSS."""
    x = variogram_df["lag_m"].to_numpy(dtype=float)
    y = variogram_df["cressie_hawkins"].to_numpy(dtype=float)
    w = variogram_df["n_pairs"].to_numpy(dtype=float)

    est_sill, est_range = _estimate_practical_range(x, y, w)
    est_nugget = max(0.0, float(np.min(y)) * 0.5)
    est_var = max(est_sill - est_nugget, 1e-10)
    logger.info(
        "Fitting variogram to %d points (est_sill=%.6f, est_range=%.0f, est_nugget=%.6f)",
        len(x), est_sill, est_range, est_nugget,
    )
    best: dict[str, Any] | None = None

    for name, ModelClass in _GSTOOLS_MODELS.items():
        model = ModelClass(
            dim=2, latlon=True, geo_scale=_EARTH_RADIUS_M,
            var=est_var, len_scale=est_range, nugget=est_nugget,
        )
        max_lag = float(np.max(x))
        model.set_arg_bounds(len_scale=[est_range * 0.3, max(est_range * 3.0, max_lag)])
        try:
            model.fit_variogram(x, y, init_guess="current")
        except Exception:
            logger.warning("Model '%s' fitting failed", name, exc_info=True)
            continue

        pred = model.variogram(x)
        wrss = float(np.sum(w * (y - pred) ** 2))
        if best is None or wrss < best["wrss"]:
            best = {"name": name, "model": model, "pred": pred, "wrss": wrss}

    if best is None:
        raise ValueError("Could not fit theoretical model.")

    model = best["model"]
    pred = best["pred"]

    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    wrmse = float(np.sqrt(np.sum(w * (y - pred) ** 2) / np.sum(w)))

    # Practical range: distance where γ reaches 95% of sill
    sill_val = float(model.sill)
    target_gamma = 0.95 * sill_val
    search_d = np.linspace(0.0, float(model.len_scale) * 10.0, 10000)
    search_gamma = model.variogram(search_d)
    above_95 = np.where(search_gamma >= target_gamma)[0]
    practical_range = float(search_d[above_95[0]]) if len(above_95) > 0 else float(model.len_scale)

    logger.info(
        "Best model: %s (nugget=%.6f, sill=%.6f, practical_range=%.0f m, r2=%s)",
        best["name"], float(model.nugget), float(model.sill), practical_range, r2,
    )

    x_dense = np.linspace(0.0, max(float(np.max(x)), practical_range * 1.2), 240)
    y_dense = model.variogram(x_dense)

    return {
        "modelType": best["name"],
        "gsmodel": model,
        "params": {
            "nugget": float(model.nugget), "partialSill": float(model.var),
            "sill": float(model.sill), "range": practical_range,
            "len_scale": float(model.len_scale),
        },
        "fit": {"rmse": rmse, "weightedRmse": wrmse, "r2": r2},
        "curve": pd.DataFrame({"lag_m": x_dense, "gamma": y_dense}),
    }


# ---------------------------------------------------------------------------
# Kriging (local neighborhood, GSTools covariance model)
# ---------------------------------------------------------------------------

_DEFAULT_MAX_NEIGHBORS = 64   # max obs per local kriging neighborhood
_DEFAULT_CHUNK_SIZE = 4096    # prediction points processed per batch


def _latlon_to_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Convert lat/lon (degrees) to 3-D Cartesian on a unit sphere for KDTree."""
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    cos_lat = np.cos(lat_r)
    return np.column_stack([cos_lat * np.cos(lon_r), cos_lat * np.sin(lon_r), np.sin(lat_r)])


def _haversine_matrix(
    lat_a: np.ndarray, lon_a: np.ndarray,
    lat_b: np.ndarray, lon_b: np.ndarray,
) -> np.ndarray:
    """Pairwise haversine distance matrix in metres (vectorized)."""
    la = np.deg2rad(lat_a)[:, None]
    lo = np.deg2rad(lon_a)[:, None]
    lb = np.deg2rad(lat_b)[None, :]
    lob = np.deg2rad(lon_b)[None, :]
    dlat = lb - la
    dlon = lob - lo
    a = np.sin(dlat / 2.0) ** 2 + np.cos(la) * np.cos(lb) * np.sin(dlon / 2.0) ** 2
    return _EARTH_RADIUS_M * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def ordinary_kriging_predict(
    obs_lat: np.ndarray, obs_lon: np.ndarray, obs_val: np.ndarray,
    grid_lat: np.ndarray, grid_lon: np.ndarray,
    model: gs.CovModel, max_neighbors: int = _DEFAULT_MAX_NEIGHBORS,
) -> tuple[np.ndarray, np.ndarray]:
    """Local neighborhood ordinary kriging (fully vectorized per chunk)."""
    n_obs = len(obs_val)
    n_grid = len(grid_lat)
    k = min(max_neighbors, n_obs)
    sill = float(model.sill)

    logger.info(
        "Local kriging: %d obs -> %d prediction points (k=%d, sill=%.6f)",
        n_obs, n_grid, k, sill,
    )

    # KDTree on unit-sphere XYZ for fast nearest-neighbor lookup
    obs_xyz = _latlon_to_xyz(obs_lat, obs_lon)
    tree = cKDTree(obs_xyz)

    pred = np.empty(n_grid, dtype=float)
    kvar = np.empty(n_grid, dtype=float)
    chunk_size = _DEFAULT_CHUNK_SIZE
    n_chunks = int(np.ceil(n_grid / chunk_size))

    for ci in range(n_chunks):
        start = ci * chunk_size
        stop = min(start + chunk_size, n_grid)
        m = stop - start
        if ci % max(1, n_chunks // 10) == 0:
            logger.info("Kriging progress: chunk %d/%d (points %d-%d)", ci + 1, n_chunks, start, stop - 1)

        g_lat = grid_lat[start:stop]
        g_lon = grid_lon[start:stop]
        g_xyz = _latlon_to_xyz(g_lat, g_lon)

        _, idx = tree.query(g_xyz, k=k)
        if k == 1:
            idx = idx[:, None]

        # Deduplicate obs indices for this chunk
        unique_obs = np.unique(idx.ravel())
        n_unique = len(unique_obs)
        local_map = np.empty(n_obs, dtype=np.int64)
        local_map[unique_obs] = np.arange(n_unique, dtype=np.int64)

        u_lat = obs_lat[unique_obs]
        u_lon = obs_lon[unique_obs]
        u_val = obs_val[unique_obs]

        # C(h) = sill - γ(h); diagonal C(0) = sill (not sill-nugget)
        d_uu = _haversine_matrix(u_lat, u_lon, u_lat, u_lon)
        c_uu = sill - model.variogram(d_uu)
        np.fill_diagonal(c_uu, sill)

        d_ug = _haversine_matrix(u_lat, u_lon, g_lat, g_lon)
        c_ug = sill - model.variogram(d_ug)

        local_idx = local_map[idx]
        c_sub = c_uu[local_idx[:, :, None], local_idx[:, None, :]]

        # Augmented kriging system: last row/col enforces Σwᵢ = 1
        kmat = np.zeros((m, k + 1, k + 1), dtype=float)
        kmat[:, :k, :k] = c_sub
        kmat[:, :k, k] = 1.0
        kmat[:, k, :k] = 1.0
        diag_idx = np.arange(k)
        kmat[:, diag_idx, diag_idx] += 1e-10

        j_range = np.arange(m)[:, None]
        c0_vals = c_ug[local_idx, j_range]

        rhs = np.zeros((m, k + 1), dtype=float)
        rhs[:, :k] = c0_vals
        rhs[:, k] = 1.0

        try:
            w = np.linalg.solve(kmat, rhs[:, :, None]).squeeze(-1)
        except np.linalg.LinAlgError:
            logger.warning("Batched solve failed on chunk %d, falling back to lstsq", ci)
            w = np.linalg.lstsq(kmat, rhs[:, :, None], rcond=None)[0].squeeze(-1)

        w_k = w[:, :k]
        nbr_vals = u_val[local_idx]

        # Z*(x₀) = Σ wᵢ·Z(xᵢ)
        pred[start:stop] = np.einsum("ij,ij->i", w_k, nbr_vals)
        # σ²(x₀) = sill - Σ wᵢ·C(xᵢ,x₀) - μ
        kvar[start:stop] = sill - np.einsum("ij,ij->i", w_k, c0_vals) - w[:, k]

    # Clamp negative variances (numerical noise) to zero
    kvar = np.maximum(kvar, 0.0)

    logger.info(
        "Kriging complete: prediction [%.4f, %.4f], std [%.4f, %.4f]",
        float(np.min(pred)), float(np.max(pred)),
        float(np.sqrt(np.min(kvar))), float(np.sqrt(np.max(kvar))),
    )
    return pred, kvar


# ---------------------------------------------------------------------------
# H3 grid helpers
# ---------------------------------------------------------------------------

def build_points_dataframe(agg_values: pd.DataFrame) -> pd.DataFrame:
    """Convert aggregated H3 cells to lat/lon points for variogram and kriging."""
    cell_str = h3r.cells_to_string(agg_values["h3_u64"].astype("uint64").tolist()).to_pylist()
    coords = [h3.cell_to_latlng(s) for s in cell_str]
    lat = np.array([c[0] for c in coords], dtype=float)
    lon = np.array([c[1] for c in coords], dtype=float)
    return pd.DataFrame({"lat": lat, "lon": lon, "value": agg_values["value"].to_numpy(dtype=float), "h3": cell_str})


def _bbox_to_latlng_poly(bbox: dict[str, float]) -> h3.LatLngPoly:
    ring = [
        (bbox["min_lat"], bbox["min_lon"]), (bbox["min_lat"], bbox["max_lon"]),
        (bbox["max_lat"], bbox["max_lon"]), (bbox["max_lat"], bbox["min_lon"]),
    ]
    return h3.LatLngPoly(ring)


def h3_cells_covering_bbox(bbox: dict[str, float], resolution: int) -> list[str]:
    return sorted(h3.polygon_to_cells(_bbox_to_latlng_poly(bbox), resolution))


# ---------------------------------------------------------------------------
# Visualization (matplotlib + contextily for OSM basemap)
# ---------------------------------------------------------------------------

def _h3_cells_to_geodataframe(df: pd.DataFrame, value_col: str) -> gpd.GeoDataFrame:
    """Convert a DataFrame with 'h3' and value columns to a GeoDataFrame of hex polygons."""
    cells = df["h3"].astype(str).tolist()
    polys = [Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(c)]) for c in cells]
    return gpd.GeoDataFrame(df[[value_col]].reset_index(drop=True), geometry=polys, crs="EPSG:4326")


def _osm_zoom_from_extent(lat_range: float, lon_range: float) -> int:
    span = max(lat_range, lon_range)
    if span <= 0:
        return 12
    zoom = int(np.log2(360.0 / span) - 0.3)
    return max(1, min(14, zoom))


def write_h3_choropleth_png(
    df: pd.DataFrame, value_col: str, title: str, out_path: Path,
    gdf: gpd.GeoDataFrame | None = None, cmap: str = "viridis",
) -> None:
    """Render H3 choropleth with OSM basemap."""
    if gdf is None:
        gdf = _h3_cells_to_geodataframe(df, value_col)
    gdf_m = gdf.to_crs("EPSG:3857")
    fig, ax = plt.subplots(figsize=(14, 9))
    gdf_m.plot(ax=ax, column=value_col, cmap=cmap, edgecolor="none", alpha=0.75,
               legend=True, legend_kwds={"label": value_col, "shrink": 0.6})
    lat_range = float(df["lat"].max() - df["lat"].min())
    lon_range = float(df["lon"].max() - df["lon"].min())
    zoom = _osm_zoom_from_extent(lat_range, lon_range)
    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik, zoom=zoom, zorder=0)
    ax.set_axis_off()
    ax.set_title(title, fontsize=13)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    logger.info("Saved: %s", out_path)


def write_variogram_plot(
    variogram_df: pd.DataFrame, fit: dict[str, Any],
    element_code: str, out_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), height_ratios=[3, 1])
    fig.suptitle(f"Variogram ({element_code})", fontsize=14)

    lag_m = variogram_df["lag_m"].values
    ax1.plot(lag_m, variogram_df["cressie_hawkins"], "o-", label="Cressie-Hawkins", markersize=5)
    ax1.plot(lag_m, variogram_df["classical"], "s-", label="Classical", markersize=4, alpha=0.7)
    ax1.plot(fit["curve"]["lag_m"], fit["curve"]["gamma"], "--", linewidth=2,
             label=f"Theoretical ({fit['modelType']})")

    nugget = float(fit["params"]["nugget"])
    sill = float(fit["params"]["sill"])
    vrange = float(fit["params"]["range"])
    range_km = vrange / 1000.0

    ax1.axhline(nugget, linestyle=":", color="grey", linewidth=0.8)
    ax1.annotate(f"nugget = {nugget:.4f}", xy=(lag_m[-1], nugget),
                 ha="right", va="top", fontsize=9, color="grey")
    ax1.axhline(sill, linestyle=":", color="grey", linewidth=0.8)
    ax1.annotate(f"sill = {sill:.4f}", xy=(lag_m[0], sill),
                 ha="left", va="bottom", fontsize=9, color="grey")
    ax1.axvline(vrange, linestyle=":", color="grey", linewidth=0.8)
    ax1.annotate(f"range = {range_km:.1f} km", xy=(vrange, ax1.get_ylim()[0]),
                 ha="left", va="bottom", fontsize=9, color="grey", rotation=90)

    ax1.set_xlabel("Lag distance (m)")
    ax1.set_ylabel("Semivariance")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.bar(lag_m / 1000.0, variogram_df["n_pairs"], width=(lag_m[1] - lag_m[0]) / 1000.0 * 0.8,
            color="steelblue", alpha=0.7)
    ax2.set_xlabel("Lag distance (km)")
    ax2.set_ylabel("Pair count")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run_workflow(config: dict[str, Any] | None = None) -> None:
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    bbox = dict(cfg["bbox"])

    _t_total = time.time()
    logger.info("=== Kriging workflow started (parquet mode) ===")
    logger.info(
        "Config: element=%s, bbox=[%.3f,%.3f,%.3f,%.3f], resolution=%d",
        cfg["element_code"],
        bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"],
        cfg["resolution"],
    )

    resolution = int(cfg["resolution"])

    # --- Step 1: Load data from Parquet, filter/normalize, aggregate ---
    _t = time.time()
    raw_df = load_parquet(cfg["parquet_path"])
    raw_df = parse_h3_column(raw_df)

    values_df = prepare_values(
        raw_df,
        sampletype_id=int(cfg["sampletype_id"]),
        element_code=str(cfg["element_code"]),
        apply_log10=bool(cfg["apply_log10"]),
    )
    agg_values = aggregate_to_resolution(values_df, resolution)
    logger.info("[TIMING] Load + prepare: %.2fs (%d obs at res %d)", time.time() - _t, len(agg_values), resolution)

    # --- Step 2: Build empirical variogram and fit theoretical model ---
    _t = time.time()
    points_df = build_points_dataframe(agg_values)
    variogram_df = build_distance_variogram(points_df)
    fit = fit_theoretical(variogram_df)
    logger.info("[TIMING] Variogram + fit: %.2fs (%s, range=%.0f m)", time.time() - _t, fit["modelType"], fit["params"]["range"])

    # --- Step 3: Build prediction grid (auto-select finest H3 res under cap) ---
    _t = time.time()
    _MAX_PREDICTION_CELLS = 50_000
    prediction_resolution = choose_prediction_resolution(
        bbox, len(points_df), max_cells=_MAX_PREDICTION_CELLS, min_res=int(cfg["min_resolution"]),
    )
    all_cells = h3_cells_covering_bbox(bbox, prediction_resolution)
    all_coords = [h3.cell_to_latlng(c) for c in all_cells]
    all_lat = np.array([c[0] for c in all_coords], dtype=float)
    all_lon = np.array([c[1] for c in all_coords], dtype=float)

    # Erode 5% of bbox edges to avoid uncertainty artifacts at borders
    lat_span = bbox["max_lat"] - bbox["min_lat"]
    lon_span = bbox["max_lon"] - bbox["min_lon"]
    margin = 0.05
    inner_bbox = {
        "min_lat": bbox["min_lat"] + lat_span * margin,
        "max_lat": bbox["max_lat"] - lat_span * margin,
        "min_lon": bbox["min_lon"] + lon_span * margin,
        "max_lon": bbox["max_lon"] - lon_span * margin,
    }
    keep = (
        (all_lat >= inner_bbox["min_lat"]) & (all_lat <= inner_bbox["max_lat"])
        & (all_lon >= inner_bbox["min_lon"]) & (all_lon <= inner_bbox["max_lon"])
    )
    all_cells = [c for c, k in zip(all_cells, keep) if k]
    all_lat = all_lat[keep]
    all_lon = all_lon[keep]
    logger.info("[TIMING] Prediction grid: %.2fs (res=%d, %d cells)", time.time() - _t, prediction_resolution, len(all_cells))

    # --- Step 4: Run ordinary kriging on the prediction grid ---
    _t = time.time()
    if len(all_cells) > 0:
        grid_pred, grid_kvar = ordinary_kriging_predict(
            obs_lat=points_df["lat"].to_numpy(),
            obs_lon=points_df["lon"].to_numpy(),
            obs_val=points_df["value"].to_numpy(),
            grid_lat=all_lat, grid_lon=all_lon,
            model=fit["gsmodel"],
            max_neighbors=int(cfg["max_neighbors"]),
        )
    else:
        grid_pred = np.array([], dtype=float)
        grid_kvar = np.array([], dtype=float)
    logger.info("[TIMING] Kriging: %.2fs (%d points)", time.time() - _t, len(all_cells))

    # Kriging std dev = sqrt(variance)
    interp_df = pd.DataFrame({
        "h3": all_cells, "lat": all_lat, "lon": all_lon,
        "prediction": grid_pred,
        "uncertainty": np.sqrt(np.maximum(grid_kvar, 0.0)),
    })

    # --- Step 5: Export PNG maps ---
    out_dir = Path(str(cfg["out_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    _t = time.time()
    write_variogram_plot(variogram_df, fit, str(cfg["element_code"]), out_dir / "variogram_plot.png")
    write_h3_choropleth_png(points_df, "value", f"Sample Values ({cfg['element_code']})", out_dir / "points_osm.png")

    # Build H3 hex polygons once, reuse for prediction + uncertainty maps
    polys = [Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(c)])
             for c in interp_df["h3"].astype(str).tolist()]

    pred_gdf = gpd.GeoDataFrame({"prediction": interp_df["prediction"].values}, geometry=polys, crs="EPSG:4326")
    write_h3_choropleth_png(
        interp_df, "prediction", f"Kriging Prediction ({cfg['element_code']})",
        out_dir / "kriging_prediction_osm.png", gdf=pred_gdf,
    )
    unc_gdf = gpd.GeoDataFrame({"uncertainty": interp_df["uncertainty"].values}, geometry=polys, crs="EPSG:4326")
    write_h3_choropleth_png(
        interp_df, "uncertainty", f"Kriging Uncertainty ({cfg['element_code']})",
        out_dir / "uncertainty_map.png", gdf=unc_gdf, cmap="YlOrRd",
    )
    logger.info("[TIMING] PNG export: %.2fs (4 files)", time.time() - _t)

    logger.info("[TIMING] Total: %.2fs", time.time() - _t_total)
    logger.info("=== Kriging workflow complete ===")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    run_workflow()
