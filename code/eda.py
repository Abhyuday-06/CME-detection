"""
Stage 1: Exploratory Data Analysis (EDA)
========================================
Visualizes and compares parameters from all 3 data sources:
  1. Aditya-L1 ASPEX plasma (swis_moments)
  2. NOAA DSCOVR plasma (dscovr_plasma)
  3. NOAA Planetary Kp Index (geomagnetic_indices)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()
DB_URI = os.getenv('DB_URI')

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eda_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_db_connection():
    return psycopg2.connect(DB_URI)


def load_aditya_data():
    conn = get_db_connection()
    df = pd.read_sql("""
        SELECT observation_time, proton_speed, proton_density, proton_thermal_speed,
               alpha_density, alpha_speed
        FROM swis_moments
        WHERE proton_speed > 0 AND proton_speed < 2000
        ORDER BY observation_time ASC
    """, conn, parse_dates=["observation_time"])
    conn.close()
    # Replace CDF fill values (~-1e31) with NaN so they don't blow up plots
    for col in ["proton_density", "proton_thermal_speed", "alpha_density", "alpha_speed"]:
        df.loc[df[col] < -1e10, col] = np.nan
    print(f"Aditya-L1: {len(df)} records ({df['observation_time'].min()} to {df['observation_time'].max()})")
    return df


def load_dscovr_data():
    conn = get_db_connection()
    df = pd.read_sql("""
        SELECT observation_time, proton_speed, proton_density, proton_temperature
        FROM dscovr_plasma
        WHERE proton_speed IS NOT NULL
        ORDER BY observation_time ASC
    """, conn, parse_dates=["observation_time"])
    conn.close()
    print(f"DSCOVR:    {len(df)} records ({df['observation_time'].min()} to {df['observation_time'].max()})")
    return df


def load_kp_data():
    conn = get_db_connection()
    df = pd.read_sql("""
        SELECT timestamp AS observation_time, kp_index, source
        FROM geomagnetic_indices
        ORDER BY timestamp ASC
    """, conn, parse_dates=["observation_time"])
    conn.close()
    print(f"Kp Index:  {len(df)} records ({df['observation_time'].min()} to {df['observation_time'].max()})")
    return df


# ─── PLOT 1: Aditya-L1 Overview ─────────────────────────────────────────────

def plot_aditya_overview(df):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Aditya-L1 ASPEX Plasma Data Overview", fontsize=14, fontweight="bold")

    axes[0].plot(df["observation_time"], df["proton_speed"], color="tab:red", linewidth=0.4)
    axes[0].set_ylabel("Proton Speed (km/s)")
    axes[0].axhline(y=450, color="orange", linestyle="--", alpha=0.7, label="CME threshold (450 km/s)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["observation_time"], df["proton_density"], color="tab:blue", linewidth=0.4)
    axes[1].set_ylabel("Proton Density (cm⁻³)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df["observation_time"], df["proton_thermal_speed"], color="tab:green", linewidth=0.4)
    axes[2].set_ylabel("Thermal Speed (km/s)")
    axes[2].set_xlabel("Time (UTC)")
    axes[2].grid(True, alpha=0.3)

    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_aditya_overview.png"), dpi=150)
    plt.close()
    print("  Saved: 01_aditya_overview.png")


# ─── PLOT 2: DSCOVR Overview ────────────────────────────────────────────────

def plot_dscovr_overview(df):
    has_temp = df["proton_temperature"].notna().sum() > 100
    n_panels = 3 if has_temp else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels), sharex=True)
    fig.suptitle("NOAA DSCOVR / OMNI2 Plasma Data", fontsize=14, fontweight="bold")

    axes[0].plot(df["observation_time"], df["proton_speed"], color="tab:red", linewidth=0.4)
    axes[0].set_ylabel("Speed (km/s)")
    axes[0].axhline(y=450, color="orange", linestyle="--", alpha=0.7, label="CME threshold")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["observation_time"], df["proton_density"], color="tab:blue", linewidth=0.4)
    axes[1].set_ylabel("Density (cm⁻³)")
    axes[1].grid(True, alpha=0.3)

    if has_temp:
        axes[2].plot(df["observation_time"], df["proton_temperature"], color="tab:green", linewidth=0.4)
        axes[2].set_ylabel("Temperature (K)")
        axes[2].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (UTC)")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_dscovr_overview.png"), dpi=150)
    plt.close()
    print("  Saved: 02_dscovr_overview.png")


# ─── PLOT 3: Kp Index Time Series ───────────────────────────────────────────

def plot_kp_overview(df):
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle("NOAA Planetary Kp Index (Daily Max)", fontsize=14, fontweight="bold")

    # Resample to daily max for performance (15K hourly → ~600 daily bars)
    df_daily = (df.set_index("observation_time")["kp_index"]
                .resample("D").max().dropna().reset_index())
    colors = df_daily["kp_index"].apply(
        lambda k: "green" if k < 4 else ("orange" if k < 6 else "red")
    )
    span_days = (df_daily["observation_time"].max() - df_daily["observation_time"].min()).total_seconds() / 86400
    bar_width = max(span_days / len(df_daily) * 0.8, 0.5)
    ax.bar(df_daily["observation_time"], df_daily["kp_index"], width=bar_width, color=colors, alpha=0.8)
    ax.axhline(y=5, color="red", linestyle="--", alpha=0.6, label="Storm threshold (Kp≥5)")
    ax.set_ylabel("Kp Index")
    ax.set_xlabel("Time (UTC)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_kp_overview.png"), dpi=150)
    plt.close()
    print("  Saved: 03_kp_overview.png")


# ─── PLOT 4: Parameter Distributions ────────────────────────────────────────

def plot_distributions(df_aditya, df_dscovr):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Parameter Distributions: Aditya-L1 vs DSCOVR", fontsize=14, fontweight="bold")

    # Speed
    axes[0, 0].hist(df_aditya["proton_speed"].dropna(), bins=100, alpha=0.6,
                     color="tab:red", label="Aditya-L1", density=True)
    axes[0, 0].hist(df_dscovr["proton_speed"].dropna(), bins=100, alpha=0.6,
                     color="tab:blue", label="DSCOVR", density=True)
    axes[0, 0].set_xlabel("Proton Speed (km/s)")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend()
    axes[0, 0].set_title("Speed Distribution")

    # Density
    axes[0, 1].hist(df_aditya["proton_density"].dropna(), bins=100, alpha=0.6,
                     color="tab:red", label="Aditya-L1", density=True)
    axes[0, 1].hist(df_dscovr["proton_density"].dropna(), bins=100, alpha=0.6,
                     color="tab:blue", label="DSCOVR", density=True)
    axes[0, 1].set_xlabel("Proton Density (cm⁻³)")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].legend()
    axes[0, 1].set_title("Density Distribution")

    # Speed vs Density scatter
    axes[1, 0].scatter(df_aditya["proton_speed"], df_aditya["proton_density"],
                        alpha=0.01, s=1, c="tab:red", label="Aditya-L1")
    axes[1, 0].scatter(df_dscovr["proton_speed"], df_dscovr["proton_density"],
                        alpha=0.1, s=1, c="tab:blue", label="DSCOVR")
    axes[1, 0].set_xlabel("Speed (km/s)")
    axes[1, 0].set_ylabel("Density (cm⁻³)")
    axes[1, 0].set_ylim(0, 50)
    axes[1, 0].legend()
    axes[1, 0].set_title("Speed vs Density")

    # Statistical comparison table
    stats_data = []
    for name, df, col in [
        ("Aditya-L1 Speed", df_aditya, "proton_speed"),
        ("DSCOVR Speed", df_dscovr, "proton_speed"),
        ("Aditya-L1 Density", df_aditya, "proton_density"),
        ("DSCOVR Density", df_dscovr, "proton_density"),
    ]:
        s = df[col].dropna()
        stats_data.append([name, f"{s.mean():.1f}", f"{s.std():.1f}",
                           f"{s.min():.1f}", f"{s.max():.1f}", f"{s.median():.1f}"])

    axes[1, 1].axis("off")
    table = axes[1, 1].table(
        cellText=stats_data,
        colLabels=["Parameter", "Mean", "Std", "Min", "Max", "Median"],
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    axes[1, 1].set_title("Statistical Comparison")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_distributions.png"), dpi=150)
    plt.close()
    print("  Saved: 04_distributions.png")


# ─── PLOT 5: DSCOVR Plasma + Kp Overlay (The Key Correlation) ───────────────

def plot_dscovr_kp_correlation(df_dscovr, df_kp):
    overlap_start = max(df_dscovr["observation_time"].min(), df_kp["observation_time"].min())
    overlap_end = min(df_dscovr["observation_time"].max(), df_kp["observation_time"].max())

    dscovr_overlap = df_dscovr[
        (df_dscovr["observation_time"] >= overlap_start) &
        (df_dscovr["observation_time"] <= overlap_end)
    ]
    kp_overlap = df_kp[
        (df_kp["observation_time"] >= overlap_start) &
        (df_kp["observation_time"] <= overlap_end)
    ]

    if dscovr_overlap.empty or kp_overlap.empty:
        print("  No overlap between DSCOVR and Kp data for correlation plot.")
        return

    fig, ax1 = plt.subplots(figsize=(14, 6))
    fig.suptitle("DSCOVR Speed vs Kp Index (Overlap Period)\nL1 Plasma → Earth Impact (~1hr delay)",
                 fontsize=13, fontweight="bold")

    color_speed = "tab:red"
    ax1.plot(dscovr_overlap["observation_time"], dscovr_overlap["proton_speed"],
             color=color_speed, linewidth=0.5, alpha=0.7, label="DSCOVR Speed")
    ax1.set_ylabel("Proton Speed (km/s)", color=color_speed)
    ax1.tick_params(axis="y", labelcolor=color_speed)
    ax1.axhline(y=450, color="orange", linestyle="--", alpha=0.5)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    # Resample Kp to daily max for bar chart performance
    kp_daily = (kp_overlap.set_index("observation_time")["kp_index"]
                .resample("D").max().dropna().reset_index())
    colors_kp = kp_daily["kp_index"].apply(
        lambda k: "green" if k < 4 else ("orange" if k < 6 else "red")
    )
    span_days = (kp_daily["observation_time"].max() - kp_daily["observation_time"].min()).total_seconds() / 86400
    kp_bar_width = max(span_days / len(kp_daily) * 0.8, 0.5)
    ax2.bar(kp_daily["observation_time"], kp_daily["kp_index"],
            width=kp_bar_width, color=colors_kp, alpha=0.6, label="Kp Index")
    ax2.set_ylabel("Kp Index")
    ax2.axhline(y=5, color="red", linestyle="--", alpha=0.4)

    date_fmt = "%Y-%m" if span_days > 60 else "%m-%d %H:%M"
    ax1.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_dscovr_kp_correlation.png"), dpi=150)
    plt.close()
    print("  Saved: 05_dscovr_kp_correlation.png")


# ─── PLOT 6: Correlation Heatmap ────────────────────────────────────────────

def plot_correlation_matrix(df_aditya):
    numeric_cols = ["proton_speed", "proton_density", "proton_thermal_speed",
                    "alpha_density", "alpha_speed"]
    corr = df_aditya[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(numeric_cols, fontsize=9)

    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax)
    ax.set_title("Aditya-L1 Parameter Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_correlation_matrix.png"), dpi=150)
    plt.close()
    print("  Saved: 06_correlation_matrix.png")


# ─── PLOT 7: Alpha/Proton ratio analysis ────────────────────────────────────

def plot_alpha_ratio(df_aditya):
    df = df_aditya.copy()
    df["alpha_ratio"] = df["alpha_density"] / df["proton_density"].replace(0, np.nan)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Alpha/Proton Density Ratio (CME Driver Gas Indicator)", fontsize=14, fontweight="bold")

    axes[0].plot(df["observation_time"], df["alpha_ratio"], color="purple", linewidth=0.3, alpha=0.7)
    axes[0].axhline(y=0.04, color="red", linestyle="--", alpha=0.7, label="Enrichment threshold (0.04)")
    axes[0].set_ylabel("Alpha/Proton Ratio")
    axes[0].set_ylim(0, 0.15)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["observation_time"], df["proton_speed"], color="tab:red", linewidth=0.3, alpha=0.7)
    axes[1].axhline(y=450, color="orange", linestyle="--", alpha=0.5, label="Fast wind (450 km/s)")
    axes[1].set_ylabel("Proton Speed (km/s)")
    axes[1].set_xlabel("Time (UTC)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_alpha_ratio.png"), dpi=150)
    plt.close()
    print("  Saved: 07_alpha_ratio.png")


# ─── PLOT 8: Data Quality & Descriptive Statistics ──────────────────────────

def plot_data_quality(df_aditya, df_dscovr, df_kp):
    """Generate comprehensive descriptive stats, null/NaN counts, and outlier analysis."""

    datasets = [
        ("Aditya-L1", df_aditya, ["proton_speed", "proton_density", "proton_thermal_speed", "alpha_density", "alpha_speed"]),
        ("DSCOVR/OMNI2", df_dscovr, ["proton_speed", "proton_density", "proton_temperature"]),
        ("Kp Index", df_kp, ["kp_index"]),
    ]

    all_rows = []
    for src_name, df, cols in datasets:
        for col in cols:
            s = df[col]
            valid = s.dropna()
            null_count = s.isna().sum()
            pct_missing = null_count / len(s) * 100

            if len(valid) > 0:
                q1 = valid.quantile(0.25)
                q3 = valid.quantile(0.75)
                iqr = q3 - q1
                lo = q1 - 1.5 * iqr
                hi = q3 + 1.5 * iqr
                outliers = ((valid < lo) | (valid > hi)).sum()
            else:
                q1 = q3 = lo = hi = outliers = 0

            all_rows.append([
                src_name, col,
                f"{len(s)}",
                f"{null_count} ({pct_missing:.1f}%)",
                f"{valid.min():.2f}" if len(valid) else "N/A",
                f"{q1:.2f}" if len(valid) else "N/A",
                f"{valid.median():.2f}" if len(valid) else "N/A",
                f"{valid.mean():.2f}" if len(valid) else "N/A",
                f"{q3:.2f}" if len(valid) else "N/A",
                f"{valid.max():.2f}" if len(valid) else "N/A",
                f"{valid.std():.2f}" if len(valid) else "N/A",
                f"{outliers} ({outliers/len(s)*100:.1f}%)" if len(valid) else "N/A",
            ])

    # Plot as table
    fig, ax = plt.subplots(figsize=(20, 0.5 * len(all_rows) + 2))
    ax.axis("off")
    ax.set_title("Descriptive Statistics & Data Quality (All Sources)",
                 fontsize=14, fontweight="bold", pad=20)
    col_labels = ["Source", "Column", "Count", "Null/NaN", "Min", "Q1", "Median", "Mean", "Q3", "Max", "Std", "IQR Outliers"]
    table = ax.table(cellText=all_rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)

    # Color header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Alternate row colors
    for i in range(1, len(all_rows) + 1):
        color = "#D6E4F0" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(color)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_data_quality.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 08_data_quality.png")


# ─── PLOT 9: Timestamp Overlap & Coverage ────────────────────────────────────

def plot_timestamp_overlap(df_aditya, df_dscovr, df_kp):
    """Visualize temporal coverage and hourly overlap between all 3 data sources."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle("Temporal Coverage & Timestamp Overlap Analysis",
                 fontsize=14, fontweight="bold")

    # --- Panel 1: Daily record count timeline ---
    ax = axes[0]
    for name, df, color in [
        ("Aditya-L1", df_aditya, "tab:red"),
        ("DSCOVR/OMNI2", df_dscovr, "tab:blue"),
        ("Kp Index", df_kp, "tab:green"),
    ]:
        daily = df.set_index("observation_time").resample("D").size()
        ax.fill_between(daily.index, daily.values, alpha=0.3, color=color, label=name)
        ax.plot(daily.index, daily.values, color=color, linewidth=0.5, alpha=0.7)
    ax.set_ylabel("Records per Day")
    ax.set_title("Daily Record Count per Source")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # --- Panel 2: Hourly overlap Gantt-style ---
    ax = axes[1]
    aditya_hrs = set(df_aditya["observation_time"].dt.floor("h"))
    dscovr_hrs = set(df_dscovr["observation_time"].dt.floor("h"))
    kp_hrs = set(df_kp["observation_time"].dt.floor("h"))

    overlap_ad = len(aditya_hrs & dscovr_hrs)
    overlap_ak = len(aditya_hrs & kp_hrs)
    overlap_dk = len(dscovr_hrs & kp_hrs)
    overlap_all = len(aditya_hrs & dscovr_hrs & kp_hrs)

    labels = ["Aditya-L1\nunique hrs", "DSCOVR\nunique hrs", "Kp Index\nunique hrs",
              "Aditya-DSCOVR\noverlap", "Aditya-Kp\noverlap", "DSCOVR-Kp\noverlap",
              "All Three\noverlap"]
    values = [len(aditya_hrs), len(dscovr_hrs), len(kp_hrs),
              overlap_ad, overlap_ak, overlap_dk, overlap_all]
    colors = ["tab:red", "tab:blue", "tab:green",
              "#9467bd", "#d62728", "#1f77b4", "gold"]

    bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f"{val:,}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Hours")
    ax.set_title("Hourly Timestamp Overlap Between Sources")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "09_timestamp_overlap.png"), dpi=150)
    plt.close()
    print("  Saved: 09_timestamp_overlap.png")

    # Print summary
    print(f"    Aditya unique hours: {len(aditya_hrs):,}")
    print(f"    DSCOVR unique hours: {len(dscovr_hrs):,}")
    print(f"    Kp unique hours: {len(kp_hrs):,}")
    print(f"    DSCOVR-Kp overlap: {overlap_dk:,} hours")
    print(f"    All three overlap: {overlap_all:,} hours")


# ─── PLOT 10: Box Plots for Outlier Visualization ───────────────────────────

def plot_outlier_boxplots(df_aditya, df_dscovr, df_kp):
    """Box plots showing outliers for all numeric columns."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Outlier Analysis — Box Plots (All Sources)",
                 fontsize=14, fontweight="bold")

    # Aditya speed & density
    axes[0, 0].boxplot([df_aditya["proton_speed"].dropna()], vert=True, patch_artist=True,
                        boxprops=dict(facecolor="tab:red", alpha=0.5))
    axes[0, 0].set_title("Aditya-L1\nProton Speed (km/s)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].boxplot([df_aditya["proton_density"].dropna()], vert=True, patch_artist=True,
                        boxprops=dict(facecolor="tab:red", alpha=0.5))
    axes[0, 1].set_title("Aditya-L1\nProton Density (cm\u207b\u00b3)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].boxplot([df_aditya["proton_thermal_speed"].dropna()], vert=True, patch_artist=True,
                        boxprops=dict(facecolor="tab:red", alpha=0.5))
    axes[0, 2].set_title("Aditya-L1\nThermal Speed (km/s)")
    axes[0, 2].grid(True, alpha=0.3)

    # DSCOVR speed & density
    axes[1, 0].boxplot([df_dscovr["proton_speed"].dropna()], vert=True, patch_artist=True,
                        boxprops=dict(facecolor="tab:blue", alpha=0.5))
    axes[1, 0].set_title("DSCOVR/OMNI2\nProton Speed (km/s)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].boxplot([df_dscovr["proton_density"].dropna()], vert=True, patch_artist=True,
                        boxprops=dict(facecolor="tab:blue", alpha=0.5))
    axes[1, 1].set_title("DSCOVR/OMNI2\nProton Density (cm\u207b\u00b3)")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].boxplot([df_kp["kp_index"].dropna()], vert=True, patch_artist=True,
                        boxprops=dict(facecolor="tab:green", alpha=0.5))
    axes[1, 2].set_title("Kp Index")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "10_outlier_boxplots.png"), dpi=150)
    plt.close()
    print("  Saved: 10_outlier_boxplots.png")


# ─── MAIN ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  STAGE 1: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print("\n[1/3] Loading Data...")
    df_aditya = load_aditya_data()
    df_dscovr = load_dscovr_data()
    df_kp = load_kp_data()

    print(f"\n[2/3] Generating Visualizations...")

    print("\n  --- Aditya-L1 Plots ---")
    plot_aditya_overview(df_aditya)
    plot_correlation_matrix(df_aditya)
    plot_alpha_ratio(df_aditya)

    print("\n  --- DSCOVR Plots ---")
    plot_dscovr_overview(df_dscovr)

    print("\n  --- Kp Index Plots ---")
    plot_kp_overview(df_kp)

    print("\n  --- Cross-Source Analysis ---")
    plot_distributions(df_aditya, df_dscovr)
    plot_dscovr_kp_correlation(df_dscovr, df_kp)

    print("\n  --- Data Quality & Statistics ---")
    plot_data_quality(df_aditya, df_dscovr, df_kp)
    plot_timestamp_overlap(df_aditya, df_dscovr, df_kp)
    plot_outlier_boxplots(df_aditya, df_dscovr, df_kp)

    print(f"\n[3/3] Summary Statistics")
    print("-" * 60)
    print(f"{'Source':<20} {'Records':>10} {'Start Date':>15} {'End Date':>15}")
    print("-" * 60)
    for name, df, col in [
        ("Aditya-L1", df_aditya, "observation_time"),
        ("DSCOVR", df_dscovr, "observation_time"),
        ("Kp Index", df_kp, "observation_time"),
    ]:
        print(f"{name:<20} {len(df):>10} {str(df[col].min().date()):>15} {str(df[col].max().date()):>15}")
    print("-" * 60)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
