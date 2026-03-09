"""Fetch till geochemistry data from the CH API and save as Parquet.

Queries the same bbox/resolution used by the kriging pipeline
and stores the raw API response in data/till_al.parquet for offline use.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Same parameters as kriging_osm.py DEFAULT_CONFIG
CH_API_URL = "http://ch-api.sh.local:8000"
BBOX = {
    "min_lon": 25.9786957,
    "min_lat": 62.8128557,
    "max_lon": 27.4820141,
    "max_lat": 63.6108582,
}
RESOLUTION = 7
FETCH_LIMIT = 200_000

DATA_DIR = Path(__file__).parent / "data"
OUT_PATH = DATA_DIR / "till_al.parquet"


def fetch() -> pd.DataFrame:
    payload = {
        "bbox": BBOX,
        "resolution": RESOLUTION,
        "category": "geochemistry",
        "limit": FETCH_LIMIT,
    }
    logger.info("POST %s/query  resolution=%d  limit=%d", CH_API_URL, RESOLUTION, FETCH_LIMIT)
    r = requests.post(f"{CH_API_URL}/query", json=payload, timeout=180)
    r.raise_for_status()
    body = r.json()
    rows = body.get("data", []) if isinstance(body, dict) else []
    df = pd.DataFrame(rows)
    logger.info("API returned %d rows, %d columns: %s", len(df), len(df.columns), list(df.columns))
    return df


def main() -> None:
    df = fetch()
    if df.empty:
        raise ValueError("API returned no data")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    logger.info("Saved %d rows to %s (%.1f MB)", len(df), OUT_PATH, OUT_PATH.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
