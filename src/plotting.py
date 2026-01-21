# src/plotting.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def apply_paper_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 30,
        "axes.labelsize": 30,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "figure.dpi": 120,
    })

def plot_overall_bars(summary_df: pd.DataFrame, out_png="overall_metrics_by_patient.png"):
    if summary_df is None or summary_df.empty:
        print("[INFO] Nothing to plot (summary_df empty).")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    axes[0].bar(summary_df["patient"], summary_df["TIR_%_mean"])
    axes[0].set_ylabel("TIR (%)")
    axes[0].set_title("Time in Range (70–180)")
    axes[0].set_ylim(0, 100)
    axes[0].tick_params(axis="x", labelrotation=45)

    axes[1].bar(summary_df["patient"], summary_df["Time<70_%_mean"])
    axes[1].set_ylabel("Time <70 (%)")
    axes[1].set_title("Hypoglycemia burden")
    axes[1].set_ylim(0, max(5, float(summary_df["Time<70_%_mean"].max()) * 1.2))
    axes[1].tick_params(axis="x", labelrotation=45)

    axes[2].bar(summary_df["patient"], summary_df["MeanBG_mean"])
    axes[2].set_ylabel("Mean BG (mg/dL)")
    axes[2].set_title("Average BG")
    axes[2].set_ylim(60, max(200, float(summary_df["MeanBG_mean"].max()) * 1.2))
    axes[2].tick_params(axis="x", labelrotation=45)

    for ax in axes:
        for lbl in ax.get_xticklabels():
            lbl.set_horizontalalignment("right")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_png, dpi=300)
    print(f"[Saved] {out_png}")
    plt.show()

def plot_daily_tir_strip(daily_long: pd.DataFrame, out_png="daily_TIR_strip.png"):
    if daily_long is None or daily_long.empty:
        print("No daily data to plot.")
        return

    patients = sorted(daily_long["patient"].unique())
    y_min = max(60, float(np.floor(daily_long["TIR_%"].min() / 5.0) * 5 - 5))

    fig, ax = plt.subplots(figsize=(14, 4.8))
    for i, p in enumerate(patients):
        g = daily_long[daily_long["patient"] == p]["TIR_%"].to_numpy()
        jitter = np.random.uniform(-0.12, 0.12, size=len(g))
        x = np.full_like(g, i, dtype=float) + jitter
        ax.scatter(x, g, alpha=0.75, s=48)

    ax.set_xlim(-0.5, len(patients) - 0.5)
    ax.set_xticks(range(len(patients)))
    ax.set_xticklabels(patients, rotation=0, ha="right")
    ax.set_ylabel("Per-day TIR (%)")
    ax.set_ylim(y_min, 100)
    ax.set_title("Daily TIR (post-warm) by Patient")
    ax.grid(True, which="major", axis="y", linestyle="-", alpha=0.25)

    fig.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"[Saved] {out_png}")
    plt.show()

def plot_bg_spaghetti(per_patient: dict, n_days=2, out_png="bg_spaghetti.png"):
    fig, ax = plt.subplots(figsize=(16, 6))
    for patient, blobs in per_patient.items():
        df = blobs["df_ts_only"].copy()
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["dt"]).dt.date
        last_days = sorted(df["date"].unique())[-n_days:]
        sub = df[df["date"].isin(last_days)].copy()
        sub["t0"] = pd.to_datetime(sub["date"])
        sub["t_hours"] = (pd.to_datetime(sub["dt"]) - pd.to_datetime(sub["t0"])).dt.total_seconds() / 3600.0
        ax.plot(sub["t_hours"], sub["bg"], alpha=0.7, lw=1.5, label=patient)

    ax.set_xlim(0, 24)
    ax.set_xlabel("Hours since midnight")
    ax.set_ylabel("BG (mg/dL)")
    ax.set_title(f"BG over last {n_days} TS days — overlay by patient")
    ax.legend(ncols=2, fontsize=9)
    fig.tight_layout()
    plt.savefig(out_png, dpi=160)
    print(f"[Saved] {out_png}")
    plt.show()
