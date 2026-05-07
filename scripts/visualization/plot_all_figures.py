"""
plot_all_figures.py
--------------------
Reproduces the key figures from the paper using the processed data files.

Figures generated:
  - Fig 9a: Spatial distribution of orientation change (Δθ heatmap)
  - Fig 9b: Absolute horizontal displacement vs. orientation change (scatter)
  - Fig 9c: Force distribution heatmap
  - Supplementary: Physiological signal time series (HR, RR, SpO2)

Usage:
    python plot_all_figures.py --results ../../results/ --data ../../data/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# ── Style ───────────────────────────────────────────────────────────────── #

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "legend.fontsize": 10,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
})

PALETTE = {
    "primary":    "#1f4e79",
    "secondary":  "#2e86ab",
    "accent":     "#e84855",
    "neutral":    "#6c757d",
}


# ── Figure 9a: Δθ spatial heatmap ────────────────────────────────────────── #

def plot_delta_theta_heatmap(df: pd.DataFrame, output_path: Path):
    """Spatial distribution of orientation change across pressure sensor array."""
    fig, ax = plt.subplots(figsize=(7, 4))

    scatter = ax.scatter(
        df["sensor_x_mm"],
        df["sensor_y_mm"],
        c=df["delta_theta_deg"],
        cmap="YlOrRd",
        s=200,
        marker="s",
        vmin=20, vmax=65,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
    )
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.85)
    cbar.set_label("Δθ (degrees)", labelpad=10)
    cbar.set_ticks([20, 30, 40, 50, 60, 65])

    ax.set_xlabel("Horizontal sensor position (mm)")
    ax.set_ylabel("Vertical sensor position (mm)")
    ax.set_title("Spatial Distribution of Orientation Change Across the Pressure Sensor Array")
    ax.set_xlim(-10, 320)
    ax.set_ylim(-10, 155)
    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.set_facecolor("#f8f9fa")

    fig.tight_layout()
    fig.savefig(output_path / "fig9a_delta_theta_heatmap.png")
    fig.savefig(output_path / "fig9a_delta_theta_heatmap.pdf")
    plt.close(fig)
    print(f"  Saved: fig9a_delta_theta_heatmap")


# ── Figure 9b: Displacement vs. Orientation scatter ──────────────────────── #

def plot_displacement_vs_orientation(df: pd.DataFrame, output_path: Path):
    """Scatter plot: |ΔX| vs. Δθ with linear trend."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.scatter(
        df["abs_displacement_mm"],
        df["delta_theta_deg"],
        color=PALETTE["secondary"],
        alpha=0.7,
        s=50,
        edgecolors=PALETTE["primary"],
        linewidths=0.5,
        label="Trial measurement",
        zorder=3,
    )

    # Linear trend line
    z = np.polyfit(df["abs_displacement_mm"], df["delta_theta_deg"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df["abs_displacement_mm"].min(),
                          df["abs_displacement_mm"].max(), 100)
    ax.plot(x_line, p(x_line), "--", color=PALETTE["accent"],
            linewidth=1.5, label="Linear trend", zorder=4)

    ax.set_xlabel("Absolute Horizontal Displacement (|ΔX|) [mm]")
    ax.set_ylabel("Change in Orientation (Δθ) [degrees]")
    ax.set_title("Absolute Horizontal Displacement vs Orientation Change")
    ax.set_xlim(-5, 130)
    ax.set_ylim(-2, 75)
    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.legend()
    ax.set_facecolor("#f8f9fa")

    fig.tight_layout()
    fig.savefig(output_path / "fig9b_displacement_vs_orientation.png")
    fig.savefig(output_path / "fig9b_displacement_vs_orientation.pdf")
    plt.close(fig)
    print(f"  Saved: fig9b_displacement_vs_orientation")


# ── Figure 9c: Pressure force distribution heatmap ───────────────────────── #

def plot_pressure_heatmap(pressure_map: np.ndarray, output_path: Path):
    """2D force distribution heatmap from pressure sensor array."""
    fig, ax = plt.subplots(figsize=(6, 4))

    im = ax.imshow(
        pressure_map,
        cmap="YlOrRd",
        aspect="auto",
        interpolation="bilinear",
        vmin=0, vmax=1,
        origin="lower",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Normalized Force", labelpad=10)

    ax.set_xlabel("Horizontal position (sensor columns)")
    ax.set_ylabel("Vertical position (sensor rows)")
    ax.set_title("Force Distribution Heatmap — Pressure Sensor Array")

    fig.tight_layout()
    fig.savefig(output_path / "fig9c_pressure_heatmap.png")
    fig.savefig(output_path / "fig9c_pressure_heatmap.pdf")
    plt.close(fig)
    print(f"  Saved: fig9c_pressure_heatmap")


# ── Supplementary: Physiological signal time series ──────────────────────── #

def plot_physiological_timeseries(df: pd.DataFrame, output_path: Path,
                                   session_id: str = ""):
    """Three-panel time series plot of HR, RR, and SpO2."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    signals = [
        ("heart_rate_bpm",        "Heart Rate (bpm)",          PALETTE["accent"]),
        ("respiration_rate_brpm", "Respiration Rate (brpm)",   PALETTE["secondary"]),
        ("spo2_pct",              "SpO₂ (%)",                  PALETTE["primary"]),
    ]

    timestamps = pd.to_datetime(df["timestamp"])
    t_min = (timestamps - timestamps.iloc[0]).dt.total_seconds() / 60.0

    for ax, (col, ylabel, color) in zip(axes, signals):
        if col not in df.columns:
            ax.text(0.5, 0.5, f"'{col}' not found", transform=ax.transAxes,
                    ha="center", va="center", color="gray")
            continue

        ax.plot(t_min, df[col], color=color, linewidth=1.0, alpha=0.9)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_facecolor("#f8f9fa")

        # Mark actuation events if column present
        if "actuation_triggered" in df.columns:
            act_idx = df["actuation_triggered"] == 1
            if act_idx.any():
                ax.axvline(x=t_min[act_idx].values[0], color="green",
                           linestyle=":", linewidth=1.2, label="Actuation")

    axes[-1].set_xlabel("Time (minutes)")
    title = f"Physiological Signal Time Series"
    if session_id:
        title += f" — Session {session_id}"
    axes[0].set_title(title)

    fig.tight_layout()
    fname = f"physiological_timeseries{'_' + session_id if session_id else ''}"
    fig.savefig(output_path / f"{fname}.png")
    fig.savefig(output_path / f"{fname}.pdf")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── Main ─────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(description="Reproduce paper figures")
    parser.add_argument("--results", required=True, help="Output directory for figures")
    parser.add_argument("--data",    default="../../data", help="Root data directory")
    args = parser.parse_args()

    results_dir = Path(args.results)
    data_dir    = Path(args.data)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Generating Paper Figures ===\n")

    # Fig 9a
    delta_theta_path = data_dir / "head_rotation" / "delta_theta_array.csv"
    if delta_theta_path.exists():
        df = pd.read_csv(delta_theta_path)
        plot_delta_theta_heatmap(df, results_dir)
    else:
        print(f"  [SKIP] {delta_theta_path} not found")

    # Fig 9b
    displacement_path = data_dir / "head_rotation" / "displacement_array.csv"
    if displacement_path.exists():
        df = pd.read_csv(displacement_path)
        plot_displacement_vs_orientation(df, results_dir)
    else:
        print(f"  [SKIP] {displacement_path} not found")

    # Fig 9c — use first available pressure map
    pressure_dir = data_dir / "pressure_sensing" / "pressure_maps"
    npy_files = sorted(pressure_dir.glob("*.npy")) if pressure_dir.exists() else []
    if npy_files:
        pressure_map = np.load(npy_files[0])
        # Average across frames if 3D
        if pressure_map.ndim == 3:
            pressure_map = pressure_map.mean(axis=0)
        plot_pressure_heatmap(pressure_map, results_dir)
    else:
        print(f"  [SKIP] No pressure .npy files found in {pressure_dir}")

    # Supplementary: physiological time series
    ocr_dir = data_dir / "ocr_realtime"
    csv_files = sorted(ocr_dir.glob("realtime_log_*.csv")) if ocr_dir.exists() else []
    for csv_path in csv_files[:3]:  # plot first 3 sessions
        session_id = csv_path.stem.replace("realtime_log_", "")
        df = pd.read_csv(csv_path)
        plot_physiological_timeseries(df, results_dir, session_id)

    print(f"\nAll figures saved to: {results_dir.resolve()}")


if __name__ == "__main__":
    main()
