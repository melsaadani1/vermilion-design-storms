"""Generate Atlas-14 and SSTmax kernel-density rasters for the Figure 7 analysis.

The analysis represents each inundated building with equal weight and applies a
2-D Gaussian kernel-density estimate (KDE) to building centroids. Buildings are
included when modeled inundation depth exceeds 0.1 ft. The resulting KDE is
converted from probability density to an expected number of inundated buildings
per raster cell by multiplying by the number of inundated buildings and the cell
area in the analysis coordinate system.

The Atlas-14 surface is evaluated on a 500 x 500 grid spanning the inundated
Atlas-14 buildings. Both surfaces are evaluated directly at the centers of this
common grid and expressed as expected inundated buildings per cell.

This script produces the analysis rasters. The final Figure 7 cartographic layout
and basemap were assembled separately in GIS software.
"""

from pathlib import Path
from _common import (
    require_configured_paths,
    validate_table,
)

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from scipy.stats import gaussian_kde


# -----------------------------------------------------------------------------
# USER CONFIGURATION
# -----------------------------------------------------------------------------
ATLAS14_INPUT = Path(
    "Shapefile containing Atlas-14 building-level inundation "
    "(required field: depth, in feet; geometry identifies each building)"
)
SST_INPUT = Path(
    "Shapefile containing SST building-level depth statistics "
    "(required field: max, in feet; geometry identifies each building)"
)

ATLAS14_OUTPUT = Path(
    "Output GeoTIFF containing Atlas-14 KDE expected buildings per cell"
)
SSTMAX_OUTPUT = Path(
    "Output GeoTIFF containing SSTmax KDE expected buildings per cell, "
    "aligned to the Atlas-14 raster grid"
)

ATLAS_DEPTH_FIELD = "depth"
SSTMAX_DEPTH_FIELD = "max"

# Study definition: buildings are treated as inundated above 0.1 ft.
DEPTH_THRESHOLD_FT = 0.1

# KDE settings used for the Figure 7 spatial-clustering analysis.
KDE_BANDWIDTH_FACTOR = 0.2
GRID_CELLS = 500


# -----------------------------------------------------------------------------
# DATA PREPARATION
# -----------------------------------------------------------------------------
def require_valid_layer(gdf: gpd.GeoDataFrame, depth_field: str, label: str) -> None:
    """Validate the minimum information required for the KDE analysis."""
    if gdf.crs is None:
        raise ValueError(f"{label} has no coordinate reference system (CRS).")
    if depth_field not in gdf.columns:
        raise ValueError(f"{label} is missing required depth field: {depth_field}")
    if not gdf.crs.is_projected:
        raise ValueError(f"{label}: use an appropriate documented projected CRS before running KDE.")
    validate_table(gdf, [depth_field], label)
    if gdf.empty:
        raise ValueError(f"{label} contains no features.")


def inundated_centroids(
    gdf: gpd.GeoDataFrame,
    depth_field: str,
    label: str,
) -> gpd.GeoDataFrame:
    """Return centroids for buildings whose depth exceeds the study threshold."""
    require_valid_layer(gdf, depth_field, label)

    depth = pd.to_numeric(gdf[depth_field], errors="coerce")
    flooded = gdf.loc[depth.notna() & (depth > DEPTH_THRESHOLD_FT)].copy()
    if flooded.geometry.isna().any() or flooded.geometry.is_empty.any() or not flooded.geometry.is_valid.all():
        raise ValueError(f"{label}: inundated buildings contain missing, empty, or invalid geometry.")

    if len(flooded) < 3:
        raise ValueError(
            f"{label} has only {len(flooded)} inundated buildings above "
            f"{DEPTH_THRESHOLD_FT} ft; at least three non-collinear points are needed."
        )

    # NSI-style inventories are commonly represented as points. For polygonal
    # inventories, use each building geometry's centroid, consistent with the
    # study's centroid-based KDE definition.
    if not flooded.geom_type.eq("Point").all():
        flooded["geometry"] = flooded.geometry.centroid
    return flooded


def xy_coordinates(points: gpd.GeoDataFrame) -> np.ndarray:
    """Return coordinates in the (2, N) shape expected by scipy gaussian_kde."""
    return np.vstack((points.geometry.x.to_numpy(), points.geometry.y.to_numpy()))


# -----------------------------------------------------------------------------
# KDE AND RASTER HELPERS
# -----------------------------------------------------------------------------
def kde_count_surface(
    points: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float] | np.ndarray,
) -> tuple[np.ndarray, rasterio.Affine]:
    """Evaluate Gaussian KDE and convert it to expected buildings per raster cell."""
    xmin, ymin, xmax, ymax = map(float, bounds)
    if not (xmax > xmin and ymax > ymin):
        raise ValueError("KDE bounds must have positive width and height.")

    coords = xy_coordinates(points)
    if not np.isfinite(coords).all() or np.linalg.matrix_rank(coords - coords.mean(axis=1, keepdims=True)) < 2:
        raise ValueError("KDE requires finite, non-collinear building coordinates.")
    kde = gaussian_kde(coords, bw_method=KDE_BANDWIDTH_FACTOR)

    # Sample cell centers so the evaluation positions match the GeoTIFF transform.
    xres = (xmax - xmin) / GRID_CELLS
    yres = (ymax - ymin) / GRID_CELLS
    x = xmin + (np.arange(GRID_CELLS) + 0.5) * xres
    y = ymax - (np.arange(GRID_CELLS) + 0.5) * yres
    xx, yy = np.meshgrid(x, y, indexing="xy")
    evaluation_points = np.vstack((xx.ravel(), yy.ravel()))
    probability_density = kde(evaluation_points).reshape(xx.shape)

    # scipy's KDE integrates to one. Multiplication by N converts probability
    # density to building-number density; multiplication by cell area converts
    # that density to expected inundated buildings per raster cell.
    cell_area = xres * yres
    expected_buildings_per_cell = probability_density * len(points) * cell_area

    # meshgrid indexes rows north-to-south and columns west-to-east.
    raster_array = expected_buildings_per_cell.astype("float32")
    transform = from_origin(xmin, ymax, xres, yres)
    return raster_array, transform


def write_geotiff(
    output_path: Path,
    data: np.ndarray,
    transform: rasterio.Affine,
    crs,
) -> None:
    """Write a single-band float32 GeoTIFF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        compress="deflate",
        nodata=np.nan,
    ) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, "Expected inundated buildings per raster cell")
        dst.update_tags(
            depth_threshold_ft=DEPTH_THRESHOLD_FT,
            kde_method="scipy.stats.gaussian_kde",
            kde_bandwidth_factor=KDE_BANDWIDTH_FACTOR,
            grid_cells=GRID_CELLS,
            value_definition="KDE probability density x inundated-building count x cell area",
        )


# -----------------------------------------------------------------------------
# MAIN ANALYSIS
# -----------------------------------------------------------------------------
def main() -> None:
    require_configured_paths(globals(), ['ATLAS14_INPUT', 'SST_INPUT'], ['ATLAS14_OUTPUT', 'SSTMAX_OUTPUT'])
    atlas = gpd.read_file(ATLAS14_INPUT)
    sst = gpd.read_file(SST_INPUT)

    require_valid_layer(atlas, ATLAS_DEPTH_FIELD, "Atlas-14 layer")
    require_valid_layer(sst, SSTMAX_DEPTH_FIELD, "SST layer")

    # Use the Atlas-14 CRS as the comparison CRS for both scenarios.
    if sst.crs != atlas.crs:
        sst = sst.to_crs(atlas.crs)

    atlas_points = inundated_centroids(atlas, ATLAS_DEPTH_FIELD, "Atlas-14 layer")
    sstmax_points = inundated_centroids(sst, SSTMAX_DEPTH_FIELD, "SSTmax layer")

    print(f"Atlas-14 inundated buildings: {len(atlas_points):,}")
    print(f"SSTmax inundated buildings:   {len(sstmax_points):,}")
    print(f"KDE bandwidth factor:         {KDE_BANDWIDTH_FACTOR}")
    print(f"Evaluation grid:              {GRID_CELLS} x {GRID_CELLS}")
    print(f"Analysis CRS:                 {atlas_points.crs}")

    # Atlas-14 defines the comparison raster grid used in Figure 7.
    atlas_surface, atlas_transform = kde_count_surface(
        atlas_points,
        atlas_points.total_bounds,
    )

    # Use the same target cell area for both expected-count rasters.
    # The Atlas-14 bounds define the comparison extent;
    # SST exposure outside that footprint is not represented by these maps.
    sst_surface_aligned, _ = kde_count_surface(sstmax_points, atlas_points.total_bounds)

    write_geotiff(
        ATLAS14_OUTPUT,
        atlas_surface,
        atlas_transform,
        atlas_points.crs,
    )
    write_geotiff(
        SSTMAX_OUTPUT,
        sst_surface_aligned,
        atlas_transform,
        atlas_points.crs,
    )

    print(f"Saved Atlas-14 KDE raster: {ATLAS14_OUTPUT}")
    print(f"Saved SSTmax KDE raster:   {SSTMAX_OUTPUT}")


if __name__ == "__main__":
    main()
