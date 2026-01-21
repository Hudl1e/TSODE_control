# scripts/run_multi_patient_eval.py
import pandas as pd
import os
from pathlib import Path

from src.controllers.ts_bandit_safe import build_forecast_wrapper
from src.config import metadata
from src.experiments.runner import run_many_patients
from src.plotting import apply_paper_style, plot_overall_bars, plot_daily_tir_strip, plot_bg_spaghetti

def main():
    apply_paper_style()

    CKPT = Path(__file__).resolve().parent / "forecasting_model_checkpoint.pt"
    forecast_wrapper = build_forecast_wrapper(str(CKPT))

    # Choose patients like your notebook
    patient_ids = [1]  # edit, e.g. [1,2,3,5]
    base_seed = int(metadata.get("seed", 0))

    patients = []
    for pid in patient_ids:
        m = dict(metadata)
        m["patient"] = f"adult#{pid:03d}"
        m["seed"] = base_seed if pid == 1 else base_seed + pid * 101
        patients.append(m)

    per_patient, summary_df, daily_long = run_many_patients(
        patients,
        forecast_wrapper=forecast_wrapper,
        warm_days=30,
        train_days=30,
        sim_days=14,
    )

    summary_df.to_csv("patient_overall_summary.csv", index=False)
    if not daily_long.empty:
        daily_long.to_csv("patient_daily_metrics.csv", index=False)

    plot_overall_bars(summary_df, out_png="overall_metrics_by_patient.png")
    plot_daily_tir_strip(daily_long, out_png="daily_TIR_strip.png")
    plot_bg_spaghetti(per_patient, n_days=2, out_png="bg_spaghetti.png")

    print("\n=== Overall Means (TS-only) ===")
    print(summary_df[["patient","TIR_%_mean","Time<70_%_mean","MeanBG_mean","days"]].to_string(index=False))

if __name__ == "__main__":
    main()
