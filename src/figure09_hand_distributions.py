#!/usr/bin/env python3
"""Generate the HAND distribution comparison used in Figure 9.

The two input workbooks contain HAND values sampled at Atlas-14-inundated
buildings and SSTmax-only buildings. The script filters to nonnegative HAND,
summarizes 1 m bins, computes empirical CDFs and CDF-divergence metrics, and
plots the count distributions with both CDFs."""

import math
from pathlib import Path
from _common import (
    require_configured_paths,
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Plot formatting
plt.rcParams.update({"font.size": 18})

# ------------------ INPUT/OUTPUT CONFIGURATION ------------------
ATLAS_XLSX = Path(
    "Excel workbook containing HAND values for Atlas-14-inundated buildings"
)
DIFF_XLSX = Path(
    "Excel workbook containing HAND values for SSTmax-only buildings"
)
HAND_VALUE_COLUMN = "HAND3"  # name used in the study
SHEET_NAME = 0  # first worksheet
HAND_BIN_WIDTH_M = 1.0
HAND_LABEL_INTERVAL_M = 5.0
PLOT_X_MAX_M = 42.0
DIVERGENCE_LOW_FRACTION = 0.20
DIVERGENCE_HIGH_FRACTION = 0.80

PNG_OUT = Path("Output PNG file for Figure 9")
CSV_OUT = Path("Output CSV file containing the Figure 9 binned counts and CDFs")
# ----------------------------------------------------------------

def load_series(path: Path, col: str, sheet=0) -> pd.Series:
    df = pd.read_excel(path, sheet_name=sheet)
    cols = {str(c).lower(): c for c in df.columns}
    key = col.lower()
    if key not in cols:
        raise KeyError(f"'{col}' not found in {path.name}. Columns: {list(df.columns)}")
    s = pd.to_numeric(df[cols[key]], errors="raise").dropna()
    if not np.isfinite(s).all():
        raise ValueError(f"{path.name}: HAND values must be finite.")
    return s

def counts_per_bin(series: pd.Series, edges: np.ndarray) -> pd.Series:
    cats = pd.cut(series, bins=edges, right=False, include_lowest=True)
    return cats.value_counts().sort_index()

def cdf_percent(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    if total <= 0:
        return np.zeros_like(counts, dtype=float)
    return np.cumsum(counts.astype(float)) / float(total) * 100.0

def main():
    require_configured_paths(globals(), ['ATLAS_XLSX', 'DIFF_XLSX'], ['PNG_OUT', 'CSV_OUT'])
    PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    # ------------------ LOAD AND FILTER HAND VALUES ------------------
    atlas = load_series(ATLAS_XLSX, HAND_VALUE_COLUMN, SHEET_NAME)
    diff  = load_series(DIFF_XLSX,  HAND_VALUE_COLUMN, SHEET_NAME)

    atlas = atlas[atlas >= 0]
    diff  = diff[diff >= 0]

    if len(atlas) == 0 or len(diff) == 0:
        raise ValueError(f"No nonnegative values were found in column '{HAND_VALUE_COLUMN}' in one or both datasets; both groups are required.")

    # ------------------ 1 M HAND BINS ------------------
    global_max = max(atlas.max() if len(atlas) else 0,
                     diff.max()  if len(diff)  else 0)

    start = 0.0
    stop = max(PLOT_X_MAX_M, math.ceil(global_max) + HAND_BIN_WIDTH_M)
    bin_edges = np.arange(start, stop + HAND_BIN_WIDTH_M, HAND_BIN_WIDTH_M)

    atlas_counts = counts_per_bin(atlas, bin_edges)
    diff_counts  = counts_per_bin(diff,  bin_edges)

    intervals = atlas_counts.index
    bin_left  = np.array([iv.left for iv in intervals])

    # ------------------ BINNED CUMULATIVE DISTRIBUTIONS ------------------
    atlas_cdf_pct = cdf_percent(atlas_counts.values)
    diff_cdf_pct  = cdf_percent(diff_counts.values)

    # ------------------ SAVE BINNED COUNTS AND CDFs ------------------
    out_df = pd.DataFrame({
        "bin_left":          [iv.left for iv in intervals],
        "bin_right":         [iv.right for iv in intervals],
        "atlas_count":       atlas_counts.values,
        "diff_count":        diff_counts.values,
        "atlas_cdf_percent": atlas_cdf_pct,
        "diff_cdf_percent":  diff_cdf_pct
    })
    out_df.to_csv(CSV_OUT, index=False)
    print(f"Saved CSV: {CSV_OUT}")

    # ------------------ CDF DIVERGENCE METRICS ------------------
    delta = np.abs(atlas_cdf_pct - diff_cdf_pct)
    divergence = delta.copy()
    area_between_cdfs = divergence.sum() * HAND_BIN_WIDTH_M

    cum_div = np.cumsum(divergence)
    if cum_div[-1] > 0:
        cum_div_norm = cum_div / cum_div[-1]
        idx_low = np.where(cum_div_norm >= DIVERGENCE_LOW_FRACTION)[0][0]
        idx_high = np.where(cum_div_norm >= DIVERGENCE_HIGH_FRACTION)[0][0]
        hand_low = bin_left[idx_low]
        hand_high = bin_left[idx_high]
    else:
        hand_low = np.nan
        hand_high = np.nan

    KS = delta.max()
    KS_hand = bin_left[np.argmax(delta)]

    print("\n=== CDF Divergence Metrics ===")
    print(f"Area between CDFs: {area_between_cdfs:.4f} percentage-point*m")
    if np.isfinite(hand_low):
        middle_percent = 100 * (DIVERGENCE_HIGH_FRACTION - DIVERGENCE_LOW_FRACTION)
        print(f"Middle {middle_percent:.0f}% of divergence: HAND in [{hand_low:.1f}, {hand_high:.1f}] m")
    else:
        print("The two binned CDFs are identical; no divergence interval is defined.")
    print(f"Maximum binned CDF separation = {KS:.4f} percentage points at bin left edge {KS_hand:.1f} m\n")

    # ------------------ FIGURE ------------------
    fig, ax1 = plt.subplots(figsize=(12, 6))
    x = bin_left

    # Stacked building counts by HAND bin.
    ax1.bar(
        x, atlas_counts.values, width=HAND_BIN_WIDTH_M,
        color="grey", edgecolor="white", linewidth=1.0,
        label="Atlas-14", align="edge"
    )
    ax1.bar(
        x, diff_counts.values, width=HAND_BIN_WIDTH_M,
        bottom=atlas_counts.values, color="red",
        edgecolor="white", linewidth=1.0,
        label="SSTmax only", align="edge"
    )

    ax1.set_xlabel("HAND (m)", fontsize=18)
    ax1.set_ylabel("Number of buildings", fontsize=18)
    ax1.set_xlim(0, PLOT_X_MAX_M)

    # Place a tick at each HAND bin and label at the configured interval.
    xticks = np.arange(0, PLOT_X_MAX_M + HAND_BIN_WIDTH_M, HAND_BIN_WIDTH_M)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([f"{v:g}" if np.isclose(v % HAND_LABEL_INTERVAL_M, 0.0) else "" for v in xticks], fontsize=18)

    # Overlay the two empirical CDFs on the right axis.
    ax2 = ax1.twinx()
    (line_atlas,) = ax2.plot(
        x + HAND_BIN_WIDTH_M / 2, atlas_cdf_pct, marker="o", linewidth=2,
        color="blue", label="Atlas-14 CDF (%)"
    )
    (line_diff,)  = ax2.plot(
        x + HAND_BIN_WIDTH_M / 2, diff_cdf_pct, marker="o", linewidth=2,
        color="red", label="SSTmax-only CDF (%)"
    )

    ax2.set_ylabel("CDF (%)", fontsize=18)
    ax2.set_ylim(0, 100)

    # Highlight the configured central fraction of cumulative absolute CDF divergence.
    if np.isfinite(hand_low) and np.isfinite(hand_high):
        ax1.axvspan(hand_low, hand_high, color="orange", alpha=0.18)
        ax1.axvline(hand_low, color="orange", linestyle="--", linewidth=1.2)
        ax1.axvline(hand_high, color="orange", linestyle="--", linewidth=1.2)

    # Combine count and CDF entries in one legend.
    h1, l1 = ax1.get_legend_handles_labels()
    h2 = [line_atlas, line_diff]
    l2 = ["Atlas-14 CDF (%)", "SSTmax-only CDF (%)"]
    ax1.legend(h1 + h2, l1 + l2, loc="center right", fontsize=18)

    fig.tight_layout()
    fig.savefig(PNG_OUT, dpi=300)
    plt.close(fig)
    print(f"Saved plot: {PNG_OUT}")

if __name__ == "__main__":
    main()
