"""Generate Figure 5: scenario-wise inundation depth, structure damage, total
structure damage, and inundated-building counts.

The Atlas-14 consequence layer is compared with SST percentile/composite
statistics at a 0.1 ft inundation threshold. Depths are converted to meters only
for plotting; structure-loss values remain in U.S. dollars."""

from pathlib import Path
from _common import (
    require_configured_paths,
    validate_table,
    require_coverage,
)

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# Figure typography.
plt.rcParams.update({"font.size": 9})

# ───────────────────────────────────────────────
# 1. USER CONFIGURATION AND STUDY PARAMETERS
# ───────────────────────────────────────────────
depth_fp = Path(
    "Shapefile containing building-level SST depth statistics "
    "(required fields: fd_id, p25, p50, p75, p90, max)"
)
struct_fp = Path(
    "Shapefile containing building-level SST structure-damage statistics "
    "(required fields: fd_id, p25, p50, p75, p90, max)"
)
atlas_fp = Path(
    "Shapefile containing Atlas-14 building-level consequences "
    "(required fields: fd_id, depth, structure)"
)

out_png = Path("Output PNG file for Figure 5")

DEPTH_THRESHOLD_FT = 0.1
FT_TO_M = 0.3048
PLOT_DEPTH_MAX_FT = 3.0

# SST composite fields shown in Figure 5.
PCT_COLS   = ["p25", "p50", "p75", "p90", "max"]

# Scenario order used on the Figure 5 y-axis.
ROW_ORDER  = ["Atlas14", "max", "p90", "p75", "p50", "p25"]


def whisker_upper(vals: np.ndarray) -> float:
    """Return Q3 + 1.5*IQR for an array, used to limit x-axis ignoring extreme outliers."""
    vals = np.asarray(vals)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return 0.0
    q1 = np.percentile(vals, 25)
    q3 = np.percentile(vals, 75)
    iqr = q3 - q1
    return q3 + 1.5 * iqr


def main():
    require_configured_paths(globals(), ['depth_fp', 'struct_fp', 'atlas_fp'], ['out_png'])
    # ───────────────────────────────────────────
    # 2. LOAD DATA
    # ───────────────────────────────────────────
    depth_gdf  = gpd.read_file(depth_fp)
    struct_gdf = gpd.read_file(struct_fp)
    atlas_gdf  = gpd.read_file(atlas_fp)

    validate_table(depth_gdf, PCT_COLS, "SST depth layer", id_field="fd_id")
    validate_table(struct_gdf, PCT_COLS, "SST damage layer", id_field="fd_id", nonnegative=True)
    validate_table(atlas_gdf, ["depth", "structure"], "Atlas-14 layer", id_field="fd_id")
    require_coverage(depth_gdf, struct_gdf, "SST damage layer")

    # ───────────────────────────────────────────
    # 3. BUILD DEPTH & DAMAGE SERIES BY SCENARIO
    # ───────────────────────────────────────────
    depth_box   = {}  # scenario -> Series of depths (ft)
    struct_box  = {}  # scenario -> Series of structure damage
    total_dam   = {}  # scenario -> total damage
    count_bldgs = {}  # scenario -> number of buildings above the depth threshold

    # Atlas-14 depths and losses are read directly from the consequence layer.
    atlas_depth = atlas_gdf["depth"]
    mask_atlas  = atlas_depth > DEPTH_THRESHOLD_FT
    depth_box["Atlas14"] = atlas_depth[mask_atlas].dropna()

    # Use losses for the same Atlas-14 buildings that exceed the depth threshold.
    atlas_dam = atlas_gdf.loc[mask_atlas, "structure"].dropna()
    struct_box["Atlas14"]   = atlas_dam
    total_dam["Atlas14"]    = float(atlas_dam.sum())
    count_bldgs["Atlas14"]  = int(mask_atlas.sum())

    # For each SST composite, define exposure from its depth-statistics field.
    for col in PCT_COLS:
        label = col  # "p25", "p50", "p75", "p90", "max"

        # Depth values remain in feet until plotting.
        d = depth_gdf[col]
        mask_depth = d > DEPTH_THRESHOLD_FT
        depths = d[mask_depth].dropna()
        depth_box[label] = depths

        # Match structure damage to the exposed building IDs.
        ids = depth_gdf.loc[mask_depth, "fd_id"]
        sub = (
            struct_gdf[["fd_id", col]]
            .merge(ids.to_frame("fd_id"), on="fd_id", how="inner")
        )
        dam_vals = sub[col].dropna()

        struct_box[label] = dam_vals
        total_dam[label] = float(dam_vals.sum())
        # Building count is defined from the depth threshold, independently of damage completeness.
        count_bldgs[label] = int(depths.shape[0])

    # ───────────────────────────────────────────
    # 4. REORDER FOR PLOTTING
    # ───────────────────────────────────────────
    labels      = ROW_ORDER
    pos         = np.arange(1, len(labels) + 1)

    # Preserve source depth units for calculations and convert to meters only for display.
    depth_data_list_ft  = [depth_box[lab]  for lab in labels]
    depth_data_list_m   = [series * FT_TO_M for series in depth_data_list_ft]

    struct_data_list = [struct_box[lab] for lab in labels]
    total_damage     = np.array([total_dam[lab]    for lab in labels], dtype=float)
    total_counts     = np.array([count_bldgs[lab]  for lab in labels], dtype=float)

    # ───────────────────────────────────────────
    # 5. CREATE FIGURE
    # ───────────────────────────────────────────
    fig, (ax_depth, ax_dam, ax_total) = plt.subplots(
        1, 3, figsize=(6, 4)   # 6 inches wide, ~4 inches tall
    )

    # Use identical scenario positions across all three panels.
    for ax in (ax_depth, ax_dam, ax_total):
        ax.set_ylim(0.5, len(labels) + 0.5)
        ax.set_yticks(pos)

    # ───────────────────────────────────────────
    # 5a. LEFT PANEL – DEPTH BOXPLOTS (meters)
    # ───────────────────────────────────────────
    boxprops     = dict(linewidth=1.2)
    whiskerprops = dict(linewidth=1.0)
    capprops     = dict(linewidth=1.0)
    medianprops  = dict(linewidth=1.5, color="red")
    meanprops    = dict(marker="*", markerfacecolor="black",
                        markeredgecolor="black", markersize=6)

    ax_depth.boxplot(
        depth_data_list_m,
        positions=pos,
        orientation="horizontal",
        patch_artist=False,
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        medianprops=medianprops,
        meanprops=meanprops,
        showmeans=True,
        showfliers=False,
    )

    ax_depth.set_xlabel("Depth (m)")
    ax_depth.set_xlim(0.0, PLOT_DEPTH_MAX_FT * FT_TO_M)  # 0–3 ft converted to meters

    # Gridlines improve comparison across scenarios.
    ax_depth.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.7)
    ax_depth.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    # Scenario labels are shown only on the left panel.
    ax_depth.set_yticks(pos)
    ax_depth.set_yticklabels(labels, fontsize=9, rotation=-90,
                             va="center", ha="center")

    # Annotate each scenario with the number of inundated buildings.
    for y, data_m in zip(pos, depth_data_list_m):
        n = len(data_m)
        if n == 0:
            continue
        x75_m = np.percentile(data_m, 75)
        ax_depth.text(
            x75_m * 1.02,
            y + 0.12,
            f"n={n:,}",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )

    # ───────────────────────────────────────────
    # 5b. MIDDLE PANEL – STRUCTURE DAMAGE BOXPLOTS
    # ───────────────────────────────────────────
    ax_dam.boxplot(
        struct_data_list,
        positions=pos,
        orientation="horizontal",
        patch_artist=False,
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        medianprops=medianprops,
        meanprops=meanprops,
        showmeans=True,
        showfliers=False,
    )

    # Set a common damage-axis range from the largest 1.5-IQR upper whisker.
    whisker_max_vals = [whisker_upper(arr.to_numpy()) for arr in struct_data_list]
    global_whisker_max = np.nanmax(whisker_max_vals) if whisker_max_vals else 0.0
    if global_whisker_max <= 0:
        x_max_dam = 1.0
    else:
        x_max_dam = global_whisker_max * 1.1

    ax_dam.set_xlabel("Structure damage (USD)")
    ax_dam.set_xlim(0.0, x_max_dam)

    # x-grid + y-grid on middle panel
    ax_dam.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.7)
    ax_dam.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    # Format per-building damage in thousands of dollars.
    def fmt_thousands(v, pos_):
        return f"{v/1000:.0f}k"

    ax_dam.xaxis.set_major_formatter(FuncFormatter(fmt_thousands))

    # Scenario names are shown only on the left panel.
    ax_dam.set_yticklabels([])

    # ───────────────────────────────────────────
    # 5c. RIGHT PANEL – TOTAL DAMAGE (BLUE) + COUNTS (RED)
    # ───────────────────────────────────────────
    # Bottom x-axis shows aggregate structure damage.
    ax_total.plot(
        total_damage,   # x in USD
        pos,            # y
        color="blue",
        marker="o",
        linestyle="-",
        label="Total damage",
    )

    ax_total.set_yticklabels([])  # scenario names only on left panel

    # Add headroom above the largest aggregate loss.
    max_total = total_damage.max()
    if max_total <= 0:
        x_max_total = 1.0
    else:
        x_max_total = max_total * 1.1
    ax_total.set_xlim(0.0, x_max_total)

    # Preserve billions for the study's scale; keep smaller user datasets legible.
    damage_scale, damage_unit = (1e9, "billion USD") if max_total >= 1e8 else (1e6, "million USD") if max_total >= 1e5 else (1e3, "thousand USD") if max_total >= 1e3 else (1, "USD")
    ax_total.set_xlabel(f"Total damage ({damage_unit})", color="blue")
    ax_total.tick_params(axis="x", colors="blue")
    ax_total.spines["bottom"].set_color("blue")

    # y-grid on right panel
    ax_total.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    # Format aggregate structure damage using the selected display unit.
    ax_total.xaxis.set_major_formatter(
        FuncFormatter(lambda v, p: f"{v/damage_scale:.1f}")
    )

    # Top x-axis shows the number of inundated buildings.
    ax_counts = ax_total.twiny()
    ax_counts.set_ylim(ax_total.get_ylim())
    ax_counts.set_yticks(pos)
    ax_counts.set_yticklabels([])

    max_count = total_counts.max()
    if max_count <= 0:
        x_max_count = 1.0
    else:
        x_max_count = max_count * 1.1
    ax_counts.set_xlim(0, x_max_count)

    ax_counts.plot(
        total_counts,   # x
        pos,            # y
        color="red",
        marker="o",
        linestyle="-",
        label="Number of buildings",
    )

    ax_counts.xaxis.set_major_formatter(
        FuncFormatter(lambda v, p: f"{v/1000:g}k" if max_count >= 1000 else f"{v:g}")
    )
    ax_counts.set_xlabel("Number of inundated\nbuildings", color="red")
    ax_counts.tick_params(axis="x", colors="red")
    ax_counts.spines["top"].set_color("red")

    # Panel labels used in the manuscript figure.
    for ax, panel_label in [
        (ax_depth, "(a)"),
        (ax_dam, "(b)"),
        (ax_total, "(c)"),
    ]:
        ax.text(
            0.97,
            0.97,
            panel_label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            fontweight="bold",
            color="black",
        )

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_png}")


if __name__ == "__main__":
    main()
