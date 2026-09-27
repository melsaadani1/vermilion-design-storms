"""Generate Figure 6 for the SST-exclusive exposure group.

For each building, the script counts how many SST realizations exceed 0.1 ft,
computes the mean exceedance depth, and uses the maximum realization-specific
structure damage. Buildings inundated by the Atlas-14 baseline are excluded.
The figure shows depth distributions by SST occurrence count together with
aggregate structure damage."""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from _common import (
    require_configured_paths,
    validate_table,
    require_coverage,
    realization_columns,
    align_atlas,
    MISSING_ATLAS_DEFAULT,
)


# ── USER CONFIGURATION AND STUDY PARAMETERS ───────────────────
depth_fp = Path(
    "Shapefile containing building-level SST depth statistics "
    "(required fields: fd_id and R*_depth)"
)
struct_fp = Path(
    "Shapefile containing building-level SST structure-damage statistics "
    "(building identifiers and realization-specific structure damage in USD)"
)
atlas_fp = Path(
    "Shapefile containing Atlas-14 building-level consequences "
    "(required fields: fd_id and depth)"
)

fig_png = Path("Output PNG file for Figure 6")
excel_out = Path("Output Excel workbook for Figure 6 summary statistics")

ID_FIELD = "fd_id"
DEPTH_THRESHOLD_FT = 0.1
FT_TO_M = 0.3048
MAX_OCCURRENCE_K = 9
EXPECTED_REALIZATIONS = 50
MISSING_ATLAS_IS_DRY = MISSING_ATLAS_DEFAULT
IQR_WHISKER = 1.5


def main():
    require_configured_paths(globals(), ['depth_fp', 'struct_fp', 'atlas_fp'], ['fig_png', 'excel_out'])
    # 1. Load and align the building-level scenario data.
    depth_df  = gpd.read_file(depth_fp)
    struct_df = gpd.read_file(struct_fp)
    atlas_raw = gpd.read_file(atlas_fp)

    for label, gdf in [("SST depth layer", depth_df), ("SST damage layer", struct_df), ("Atlas-14 layer", atlas_raw)]:
        if ID_FIELD not in gdf.columns:
            raise ValueError(f"{label} is missing required field: {ID_FIELD}")
        if gdf[ID_FIELD].duplicated().any():
            n_dup = int(gdf[ID_FIELD].duplicated().sum())
            raise ValueError(
                f"{label} contains {n_dup:,} duplicate {ID_FIELD} values; "
                "one record per building is required for the joins in this analysis."
            )
    if "depth" not in atlas_raw.columns:
        raise ValueError("Atlas-14 layer is missing required field: depth")

    atlas_df = atlas_raw[[ID_FIELD, "depth"]].rename(columns={"depth": "Atlas_depth"})

    # Resolve full and DBF-truncated realization fields by their run numbers.
    r_depth_map = realization_columns(depth_df.columns, "depth", EXPECTED_REALIZATIONS)
    r_struct_map = realization_columns(struct_df.columns, "damage", EXPECTED_REALIZATIONS)
    r_struct = list(r_struct_map.values())
    validate_table(depth_df, list(r_depth_map.values()), "SST depth layer", id_field=ID_FIELD)
    validate_table(struct_df, r_struct, "SST damage layer", id_field=ID_FIELD, nonnegative=True)
    validate_table(atlas_raw, ["depth"], "Atlas-14 layer", id_field=ID_FIELD)
    require_coverage(depth_df, struct_df, "SST damage layer", ID_FIELD)
    atlas_df = align_atlas(depth_df, atlas_raw, ID_FIELD, MISSING_ATLAS_IS_DRY)[[ID_FIELD, "depth"]].rename(columns={"depth": "Atlas_depth"})

    struct_numeric = struct_df[r_struct].apply(pd.to_numeric, errors="coerce")
    struct_df["damage_max"] = struct_numeric.max(axis=1)

    df = (
        depth_df
        .merge(struct_df[[ID_FIELD, "damage_max"]], on=ID_FIELD, how="left", validate="one_to_one")
        .merge(atlas_df, on=ID_FIELD, how="left", validate="one_to_one")
    )

    # 2. Count SST inundation occurrences and identify buildings not inundated by Atlas-14.
    r_depth = list(r_depth_map.values())
    if not r_depth:
        raise ValueError("No realization-specific depth fields matching R*_depth were found.")
    depth_numeric = df[r_depth].apply(pd.to_numeric, errors="coerce")
    df["n_runs"] = (depth_numeric > DEPTH_THRESHOLD_FT).sum(axis=1)
    df["mean_excess"] = depth_numeric.where(depth_numeric > DEPTH_THRESHOLD_FT).mean(axis=1)
    df["Atlas_depth"] = pd.to_numeric(df["Atlas_depth"], errors="coerce")

    mask_sst_only = (df["Atlas_depth"] <= DEPTH_THRESHOLD_FT) | df["Atlas_depth"].isna()

    plot_datasets, pos        = [], []
    n_list, mean_list         = [], []
    median_list, damage_list  = [], []
    q3_levels                 = []

    # 3. Build cumulative occurrence groups (n_runs >= k). The k=1..9 range reproduces Figure 6.
    # Apply 1.5-IQR trimming to the displayed boxplot depths; sample sizes and damage use the full selected set.
    for k in range(1, MAX_OCCURRENCE_K + 1):
        sel = (df["n_runs"] >= k) & mask_sst_only
        data_all = df.loc[sel, "mean_excess"].dropna()
        if data_all.empty:
            continue

        q1, q3 = np.percentile(data_all, [25, 75])
        iqr = q3 - q1
        low, high = q1 - IQR_WHISKER * iqr, q3 + IQR_WHISKER * iqr
        data_trim = data_all[(data_all >= low) & (data_all <= high)]

        plot_datasets.append(data_trim)   # source units: feet
        q3_levels.append(q3)              # feet
        pos.append(k)

        n_list.append(len(data_all))
        mean_list.append(data_trim.mean())         # feet; retained in the statistics workbook
        median_list.append(np.median(data_trim))   # feet; retained in the statistics workbook
        damage_list.append(df.loc[sel, "damage_max"].sum())

    if not plot_datasets:
        raise RuntimeError(
            "No SST-exclusive buildings were available in the configured k range; "
            "Figure 6 cannot be generated."
        )

    # 4. Convert plotted depths to meters.
    plot_datasets_m = [d * FT_TO_M for d in plot_datasets]
    q3_levels_m     = [q * FT_TO_M for q in q3_levels]

    # 5. Create the Figure 6 boxplot and aggregate-damage overlay.
    plt.rcParams.update({"font.size": 12})

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.boxplot(
        plot_datasets_m,
        positions=pos,
        widths=0.4,
        whis=IQR_WHISKER,
        showfliers=False,
        showmeans=True,
        boxprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        medianprops=dict(color="red"),
        meanprops=dict(
            marker="*",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=8,
        ),
    )

    max_depth_m = max(d.max() for d in plot_datasets_m)
    ax.set_ylim(0, max_depth_m * 1.1)

    ax.set_ylabel("Mean inundation depth (m)")
    ax.set_xlabel("Number of SST realizations with extra inundation (≥ k)")
    ax.set_xticks(pos)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

    # Right axis shows aggregate structure damage on a logarithmic scale.
    ax2 = ax.twinx()
    ax2.plot(pos, damage_list, color="red", marker="o", linewidth=0.8)
    if all(value > 0 for value in damage_list):
        ax2.set_yscale("log")
    ax2.set_ylabel("Total structure damage (USD)", color="red")
    ax2.tick_params(axis="y", colors="red", labelcolor="red")
    ax2.get_yaxis().get_offset_text().set_color("red")

    # Finalize axis geometry before placing sample-size annotations.
    plt.tight_layout()

    # 6. Draw sample-size annotations after layout so they remain visible.
    fig.canvas.draw()  # ensure axis limits are finalized before annotation placement
    for x, q3_val_m, n in zip(pos, q3_levels_m, n_list):
        y_lab = q3_val_m + 0.02 * max_depth_m
        ax.text(
            x,
            y_lab,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="black",
            zorder=1000,      # keep annotations above plotted artists
            clip_on=False,    # allow labels near the plot boundary
        )

    fig_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved -> {fig_png}")

    # 7. Save the supporting statistics in the source depth units (feet).
    summary_df = pd.DataFrame({
        "k":                pos,
        "n_buildings":      n_list,
        "mean_trimmed_depth_ft":    [round(v, 3) for v in mean_list],
        "median_trimmed_depth_ft":  [round(v, 3) for v in median_list],
        "total_damage_usd": [round(v, 0) for v in damage_list],
    })

    excel_out.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(excel_out, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Stats", index=False)

    print(f"Excel workbook written -> {excel_out}")


if __name__ == "__main__":
    main()
