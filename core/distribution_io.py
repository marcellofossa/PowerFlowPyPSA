from __future__ import annotations

from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from config.settings import TARGET_CRS


def derive_utm_epsg(gdf: gpd.GeoDataFrame) -> int:
    """
    Derive the UTM zone EPSG (326xx north / 327xx south) from the data extent.

    Using the correct zone keeps planar distances accurate: reprojecting a
    site into a fixed zone (e.g. Keana, zone 32N, forced into 32633) inflates
    lengths systematically (~0.6-0.7% at ~750 km from the central meridian).
    """
    g = gdf if (gdf.crs is not None and gdf.crs.to_epsg() == 4326) else gdf.to_crs(epsg=4326)
    minx, miny, maxx, maxy = g.total_bounds
    lon, lat = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    zone = int((lon + 180.0) // 6) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def load_and_transform_data(file, target_crs: Optional[int] = None) -> Optional[gpd.GeoDataFrame]:
    """
    Load and transform spatial data from an uploaded file.

    Supported formats
    -----------------
    - GeoPackage (.gpkg): reprojected to `target_crs`.
    - Excel (.xlsx): expects 'latitude' and 'longitude' columns in WGS84 (EPSG:4326).

    CRS policy
    ----------
    If `target_crs` is None (default), the UTM zone is derived from the data
    extent via `derive_utm_epsg` — correct meters for any site. Passing an
    explicit `target_crs` (e.g. the legacy TARGET_CRS) preserves the old
    fixed-zone behavior.

    Returns
    -------
    GeoDataFrame in the resolved CRS or None if the file cannot be parsed.
    """
    if file is None:
        return None

    name = getattr(file, "name", "")
    if name.endswith(".gpkg"):
        gdf = gpd.read_file(file)
        if gdf.crs is None:
            # assume the legacy fixed zone if the file carries no CRS
            gdf.set_crs(epsg=TARGET_CRS, inplace=True)
        resolved = int(target_crs) if target_crs is not None else derive_utm_epsg(gdf)
        if gdf.crs.to_epsg() != resolved:
            gdf = gdf.to_crs(epsg=resolved)
        return gdf[gdf.is_valid]

    if name.endswith(".xlsx"):
        df = pd.read_excel(file)
        if "latitude" in df.columns and "longitude" in df.columns:
            geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
            gdf = gpd.GeoDataFrame(df, geometry=geometry)
            gdf.set_crs(epsg=4326, inplace=True)
            resolved = int(target_crs) if target_crs is not None else derive_utm_epsg(gdf)
            gdf = gdf.to_crs(epsg=resolved)
            return gdf[gdf.is_valid]

    # unsupported format
    return None
