"""Generate supporting Atlas-14 versus SST depth and structure-damage statistics.

The workbook reports mean and median values for SSTmax, SST-P90, and SST-P75
for the full SST-exposed set, the Atlas-14/SST overlap, and the SST-only set.
Building records are aligned explicitly by ``fd_id`` so the result does not
depend on row order in the input files.
"""

from pathlib import Path
from _common import (
    require_configured_paths,
    validate_table,
    require_coverage,
    align_atlas,
    MISSING_ATLAS_DEFAULT,
)

import geopandas as gpd
import pandas as pd


# -----------------------------------------------------------------------------
# USER CONFIGURATION AND STUDY PARAMETERS
# -----------------------------------------------------------------------------
depth_fp = Path(
    "Shapefile containing SST building-level depth statistics "
    "(required fields: fd_id, max, p90, p75)"
)
damage_fp = Path(
    "Shapefile containing SST building-level structure-damage statistics "
    "(required fields: fd_id, max, p90, p75)"
)
atlas_fp = Path(
    "Shapefile containing Atlas-14 building-level consequences "
    "(required fields: fd_id, depth, structure)"
)
excel_out = Path("Output Excel workbook for Atlas-14/SST summary statistics")

ID_FIELD = "fd_id"
MISSING_ATLAS_IS_DRY = MISSING_ATLAS_DEFAULT
DEPTH_THRESHOLD_FT = 0.1
SST_METRICS = ["max", "p90", "p75"]
SLICES = ["Overall (SST)", "Overlap", "SST-only"]


def build_columns():
    columns = []
    for metric in SST_METRICS:
        columns.extend(
            [
                f"mean_SST_{metric}",
                f"median_SST_{metric}",
                f"mean_Atlas14_{metric}",
                f"median_Atlas14_{metric}",
            ]
        )
    return columns


def require_fields(gdf, required, label):
    missing = sorted(set(required) - set(gdf.columns))
    if missing:
        raise ValueError(f"{label} is missing required fields: {missing}")


def require_unique_ids(gdf, label):
    if gdf[ID_FIELD].duplicated().any():
        n_dup = int(gdf[ID_FIELD].duplicated().sum())
        raise ValueError(
            f"{label} contains {n_dup:,} duplicate {ID_FIELD} values. "
            "One record per building is required for identifier-based comparison."
        )


def main():
    require_configured_paths(globals(), ['depth_fp', 'damage_fp', 'atlas_fp'], ['excel_out'])
    depth_gdf = gpd.read_file(depth_fp)
    damage_gdf = gpd.read_file(damage_fp)
    atlas_gdf = gpd.read_file(atlas_fp)

    require_fields(depth_gdf, [ID_FIELD, *SST_METRICS], "SST depth layer")
    require_fields(damage_gdf, [ID_FIELD, *SST_METRICS], "SST damage layer")
    require_fields(atlas_gdf, [ID_FIELD, "depth", "structure"], "Atlas-14 consequence layer")

    require_unique_ids(depth_gdf, "SST depth layer")
    require_unique_ids(damage_gdf, "SST damage layer")
    require_unique_ids(atlas_gdf, "Atlas-14 consequence layer")

    validate_table(depth_gdf, SST_METRICS, "SST depth layer", id_field=ID_FIELD)
    validate_table(damage_gdf, SST_METRICS, "SST damage layer", id_field=ID_FIELD, nonnegative=True)
    validate_table(atlas_gdf, ["depth", "structure"], "Atlas-14 layer", id_field=ID_FIELD)
    require_coverage(depth_gdf, damage_gdf, "SST damage layer", ID_FIELD)
    aligned_atlas = align_atlas(depth_gdf, atlas_gdf, ID_FIELD, MISSING_ATLAS_IS_DRY)

    # Rename fields before joining so depth, damage, and Atlas-14 values remain unambiguous.
    depth_table = depth_gdf[[ID_FIELD, *SST_METRICS]].rename(
        columns={metric: f"sst_depth_{metric}" for metric in SST_METRICS}
    )
    damage_table = damage_gdf[[ID_FIELD, *SST_METRICS]].rename(
        columns={metric: f"sst_damage_{metric}" for metric in SST_METRICS}
    )
    atlas_table = aligned_atlas[[ID_FIELD, "depth", "structure"]].rename(
        columns={"depth": "atlas_depth", "structure": "atlas_damage"}
    )

    # The SST depth layer defines the candidate building table. Other values are joined by fd_id.
    combined = (
        depth_table
        .merge(damage_table, on=ID_FIELD, how="left", validate="one_to_one")
        .merge(atlas_table, on=ID_FIELD, how="left", validate="one_to_one")
    )

    atlas_exposed_all = atlas_gdf.loc[
        atlas_gdf["depth"] > DEPTH_THRESHOLD_FT, ["depth", "structure"]
    ]
    atlas_overall_depth = pd.to_numeric(atlas_exposed_all["depth"], errors="coerce").dropna()
    atlas_overall_damage = pd.to_numeric(atlas_exposed_all["structure"], errors="coerce").dropna()

    depth_tbl = pd.DataFrame(index=SLICES, columns=build_columns(), dtype=float)
    damage_tbl = pd.DataFrame(index=SLICES, columns=build_columns(), dtype=float)

    for metric in SST_METRICS:
        sst_depth_col = f"sst_depth_{metric}"
        sst_damage_col = f"sst_damage_{metric}"

        sst_exposed = combined[sst_depth_col] > DEPTH_THRESHOLD_FT
        atlas_exposed = combined["atlas_depth"] > DEPTH_THRESHOLD_FT
        overlap = sst_exposed & atlas_exposed
        sst_only = sst_exposed & ~atlas_exposed.fillna(False)

        masks = {
            "Overall (SST)": sst_exposed,
            "Overlap": overlap,
            "SST-only": sst_only,
        }

        for slice_name, mask in masks.items():
            sst_depth_vals = pd.to_numeric(combined.loc[mask, sst_depth_col], errors="coerce").dropna()
            sst_damage_vals = pd.to_numeric(combined.loc[mask, sst_damage_col], errors="coerce").dropna()

            depth_tbl.loc[slice_name, f"mean_SST_{metric}"] = sst_depth_vals.mean()
            depth_tbl.loc[slice_name, f"median_SST_{metric}"] = sst_depth_vals.median()
            damage_tbl.loc[slice_name, f"mean_SST_{metric}"] = sst_damage_vals.mean()
            damage_tbl.loc[slice_name, f"median_SST_{metric}"] = sst_damage_vals.median()

        # Overall Atlas-14 statistics are the same reference distribution for every SST metric.
        depth_tbl.loc["Overall (SST)", f"mean_Atlas14_{metric}"] = atlas_overall_depth.mean()
        depth_tbl.loc["Overall (SST)", f"median_Atlas14_{metric}"] = atlas_overall_depth.median()
        damage_tbl.loc["Overall (SST)", f"mean_Atlas14_{metric}"] = atlas_overall_damage.mean()
        damage_tbl.loc["Overall (SST)", f"median_Atlas14_{metric}"] = atlas_overall_damage.median()

        atlas_overlap_depth = pd.to_numeric(combined.loc[overlap, "atlas_depth"], errors="coerce").dropna()
        atlas_overlap_damage = pd.to_numeric(combined.loc[overlap, "atlas_damage"], errors="coerce").dropna()

        depth_tbl.loc["Overlap", f"mean_Atlas14_{metric}"] = atlas_overlap_depth.mean()
        depth_tbl.loc["Overlap", f"median_Atlas14_{metric}"] = atlas_overlap_depth.median()
        damage_tbl.loc["Overlap", f"mean_Atlas14_{metric}"] = atlas_overlap_damage.mean()
        damage_tbl.loc["Overlap", f"median_Atlas14_{metric}"] = atlas_overlap_damage.median()

        # Atlas-14 statistics for SST-only buildings are intentionally not reported.
        depth_tbl.loc["SST-only", f"mean_Atlas14_{metric}"] = float("nan")
        depth_tbl.loc["SST-only", f"median_Atlas14_{metric}"] = float("nan")
        damage_tbl.loc["SST-only", f"mean_Atlas14_{metric}"] = float("nan")
        damage_tbl.loc["SST-only", f"median_Atlas14_{metric}"] = float("nan")

    depth_tbl = depth_tbl.round(3)   # feet
    damage_tbl = damage_tbl.round(0)  # USD

    excel_out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(excel_out, engine="openpyxl") as writer:
        depth_tbl.to_excel(writer, sheet_name="Depth_ft")
        damage_tbl.to_excel(writer, sheet_name="Damage_USD")

    print(f"Excel workbook written -> {excel_out}")


if __name__ == "__main__":
    main()
