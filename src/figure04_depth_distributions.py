"""Generate the building-level inundation-depth distributions shown in Figure 4.

Inputs are the SST building-level depth-statistics shapefile and the Atlas-14
building-level consequence shapefile. Depth filtering is performed at 0.1 ft
before conversion to meters. The script produces the six-panel histogram figure
with lognormal fits and a CSV containing summary statistics."""

from pathlib import Path
from _common import (
    require_configured_paths,
    validate_table,
)

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import lognorm

# -------------------------------------------------------------
# 1. USER CONFIGURATION AND STUDY PARAMETERS
# -------------------------------------------------------------
depth_fp = Path(
    "Shapefile containing building-level SST depth statistics "
    "(required fields: p25, p50, p75, p90, max)"
)
atlas_fp = Path(
    "Shapefile containing Atlas-14 building-level consequences "
    "(required field: depth, in feet)"
)

out_png = Path("Output PNG file for Figure 4")
out_csv = Path("Output CSV file for Figure 4 distribution statistics")

FT_TO_M = 0.3048

# Study inundation threshold. The threshold is applied in the source unit (feet) before conversion.
DEPTH_THRESH_FT = 0.1

# Plotting settings used to reproduce Figure 4. These do not change which buildings are included.
MAX_DEPTH_M = 2.4
BIN_WIDTH_M = 0.06
X_TICK_INTERVAL_M = 0.3
COUNT_AXIS_ROUNDING = 1000
BINS_M = np.arange(0, MAX_DEPTH_M + BIN_WIDTH_M / 2, BIN_WIDTH_M)

# Global font size
plt.rcParams.update({"font.size": 9})

def main():
    require_configured_paths(globals(), ['depth_fp', 'atlas_fp'], ['out_png', 'out_csv'])
    # -------------------------------------------------------------
    # 2. Load data
    # -------------------------------------------------------------
    depth_gdf = gpd.read_file(depth_fp)
    atlas_gdf = gpd.read_file(atlas_fp)

    required_sst_fields = {"p25", "p50", "p75", "p90", "max"}
    missing_sst = sorted(required_sst_fields - set(depth_gdf.columns))
    if missing_sst:
        raise ValueError(f"SST depth layer is missing required fields: {missing_sst}")
    if "depth" not in atlas_gdf.columns:
        raise ValueError("Atlas-14 consequence layer is missing required field: depth")

    validate_table(depth_gdf, list(required_sst_fields), "SST depth layer")
    validate_table(atlas_gdf, ["depth"], "Atlas-14 layer")

    # -------------------------------------------------------------
    # Panel order used in Figure 4:
    # Row 1: Atlas (top-left), Max (top-right)
    # Row 2: P25, P50
    # Row 3: P75, P90
    # -------------------------------------------------------------
    panels = []
    panels.append(("(a) Atlas14", atlas_gdf["depth"].to_numpy()))
    panels.append(("(b) SSTMax", depth_gdf["max"].to_numpy()))

    for col, lab in [("p25", "(c) SST-P25"), ("p50", "(d) SST-P50"), ("p75", "(e) SST-P75"), ("p90", "(f) SST-P90")]:
        panels.append((lab, depth_gdf[col].to_numpy()))

    # -------------------------------------------------------------
    # 3. Fit lognormal parameters for depths expressed in meters
    # -------------------------------------------------------------
    def fit_lognormal(data_m):
        """
        Fit a lognormal distribution to 1D array 'data_m' (values > 0, in meters).
        Returns:
            (mu_ln, sigma_ln) where data_m ~ LogNormal(mu_ln, sigma_ln)
            i.e. ln(X) ~ N(mu_ln, sigma_ln^2)
        """
        shape_ln, loc_ln, scale_ln = lognorm.fit(data_m, floc=0)
        sigma_ln = shape_ln
        mu_ln = np.log(scale_ln)
        return mu_ln, sigma_ln

    # -------------------------------------------------------------
    # 4. Create figure and axes (3 rows x 2 columns)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(
        3, 2, figsize=(6, 6), sharex=True, sharey=True
    )
    axes = axes.flatten()

    # Use a common count-axis limit across panels.
    global_max_count = 0

    # First pass determines a shared count-axis limit.
    for (label, arr) in panels:
        # Threshold in feet, then convert to meters
        mask = arr > DEPTH_THRESH_FT
        data_m = arr[mask] * FT_TO_M
        if data_m.size == 0:
            continue
        counts, _ = np.histogram(data_m, bins=BINS_M)
        global_max_count = max(global_max_count, counts.max())

    # Round the shared upper count limit for readable axis labels.
    rounding = min(COUNT_AXIS_ROUNDING, 10 ** np.floor(np.log10(max(global_max_count, 1))))
    ymax = (
        int(np.ceil(global_max_count / rounding) * rounding)
        if global_max_count > 0 else 0
    )

    # -------------------------------------------------------------
    # 5. Plot each panel with lognormal fit and stats
    #    + collect stats for CSV
    # -------------------------------------------------------------
    from matplotlib.lines import Line2D
    mean_line = Line2D([], [], color="blue", linewidth=2)
    med_line = Line2D([], [], color="red", linestyle="--", linewidth=2)
    stats_dict = {}  # scenario -> dict of metrics

    for ax, (label, arr) in zip(axes, panels):
        # Filter depths > 0.1 ft, then convert to meters
        mask = arr > DEPTH_THRESH_FT
        data_m = arr[mask] * FT_TO_M
        n = data_m.size

        # Skip scenarios with no buildings above the inundation threshold.
        if n == 0:
            ax.set_title(label, fontsize=9)
            ax.text(0.5, 0.5, "No inundated buildings", ha="center", transform=ax.transAxes)
            stats_dict[label] = {"Mean_m": np.nan, "Median_m": np.nan, "mu_ln": np.nan, "sigma_ln": np.nan, "n": 0}
            continue

        # Histogram in meters
        counts, bin_edges, _ = ax.hist(
            data_m,
            bins=BINS_M,
            color="0.6",
            edgecolor="black",
            linewidth=0.5,
        )

        # Mean & median in meters
        mean_m = data_m.mean()
        median_m = np.median(data_m)

        mean_line = ax.axvline(mean_m, color="blue", linewidth=2)
        med_line = ax.axvline(median_m, color="red", linestyle="--", linewidth=2)

        ax.set_ylim(0, ymax)
        ax.set_xlim(0, MAX_DEPTH_M)

        ax.set_xticks(np.arange(0, MAX_DEPTH_M + 0.001, X_TICK_INTERVAL_M))

        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_title(label, fontsize=9)

        # Right-hand y-axis: percentage of inundated buildings in this panel
        ax2 = ax.twinx()
        if n > 0:
            ax2.set_ylim(0, ymax * 100.0 / n if ymax > 0 else 100)
        ax2.set_ylabel("Percent of buildings (%)", fontsize=9)
        ax2.tick_params(axis="y", labelsize=9)

        # ---------------------------------------------------------
        # Lognormal fit in meters
        # ---------------------------------------------------------
        if n >= 2 and np.ptp(data_m) > 0:
            mu_ln, sigma_ln = fit_lognormal(data_m)
        else:
            mu_ln, sigma_ln = np.nan, np.nan

        # Evaluate the fitted PDF over the displayed depth range.
        x = np.linspace(max(DEPTH_THRESH_FT * FT_TO_M, 0.001), MAX_DEPTH_M, 400)
        bin_width = BIN_WIDTH_M

        pdf_ln = lognorm.pdf(x, s=sigma_ln, loc=0, scale=np.exp(mu_ln))
        ax.plot(
            x,
            pdf_ln * n * bin_width,
            color="navy",
            linewidth=1,
            linestyle="-",
            alpha=0.9,
        )

        # ---------------------------------------------------------
        # Stats text (in meters)
        # ---------------------------------------------------------
        text_str = (
            f"Mean: {mean_m:.2f} m\n"
            f"Median: {median_m:.2f} m\n"
            f"μ: {mu_ln:.2f}\n"
            f"σ: {sigma_ln:.2f}\n"
            f"n={n:,}"
        )

        ax.text(
            0.98,
            0.98,
            text_str,
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.7,
                boxstyle="round,pad=0.3",
            ),
        )

        # ---------------------------------------------------------
        # Collect stats for CSV
        # ---------------------------------------------------------
        stats_dict[label] = {
            "Mean_m": float(mean_m),
            "Median_m": float(median_m),
            "mu_ln": float(mu_ln),
            "sigma_ln": float(sigma_ln),
            "n": int(n),
        }

    # -------------------------------------------------------------
    # 6. Shared x / y labels & legend
    # -------------------------------------------------------------
    # X-label only on bottom row
    for ax in axes[4:]:
        ax.set_xlabel("Depth (m)", fontsize=9)

    # Y-label on left column (top and bottom left)
    axes[0].set_ylabel("Count", fontsize=9)
    axes[2].set_ylabel("Count", fontsize=9)
    axes[4].set_ylabel("Count", fontsize=9)

    # Put legend in the top-right panel (axes[1], which is Max)
    legend_ax = axes[1]
    legend_ax.legend(
        [mean_line, med_line],
        ["Mean", "Median"],
        loc="lower right",
        fontsize=9,
        frameon=True,
    )

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Saved figure -> {out_png}")

    # -------------------------------------------------------------
    # 7. Build and save CSV of stats (meters)
    # -------------------------------------------------------------
    metrics = ["Mean_m", "Median_m", "mu_ln", "sigma_ln", "n"]
    df_stats = pd.DataFrame(
        {scenario: [stats_dict[scenario][m] for m in metrics] for scenario in stats_dict},
        index=metrics,
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_stats.to_csv(out_csv)
    print(f"Saved stats CSV -> {out_csv}")


if __name__ == "__main__":
    main()
