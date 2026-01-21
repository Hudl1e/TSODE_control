# scripts/run_transfer_generalization.py
import pandas as pd

from src.config import metadata
from src.plotting import apply_paper_style, plot_overall_bars, plot_daily_tir_strip
from src.experiments.controller_factory import make_ts_controller
from src.experiments.runner import run_patient_single_env

import os
from pathlib import Path

from src.controllers.ts_bandit_safe import build_forecast_wrapper

def main():
    apply_paper_style()

    CKPT = Path(__file__).resolve().parent / "forecasting_model_checkpoint.pt"
    forecast_wrapper = build_forecast_wrapper(str(CKPT))
    base_seed = int(metadata.get("seed", 0))

    def meta_for(pid: int):
        m = dict(metadata)
        m["patient"] = f"adult#{pid:03d}"
        m["seed"] = base_seed if pid == 1 else base_seed + pid * 101
        return m

    # Build ONE reusable controller object
    shared_ts = make_ts_controller(metadata, forecast_wrapper=forecast_wrapper, debug=False)

    # -------- TRAIN PHASE on patients 1 and 2 --------
    # Train only: train_days>0, warm_days=0, sim_days=0
    for pid in [1, 2]:
        print(f"\n=== Pre-training on patient {pid} ===")
        _ = run_patient_single_env(
            meta_for(pid),
            warm_days=0,
            train_days=30,
            sim_days=0,
            forecast_wrapper=forecast_wrapper,
            seed_offset=pid * 101,
            ts_controller=shared_ts,
        )

    # -------- TEST PHASE on patient 5 --------
    # Frozen eval: train_days=0 (no learning on test patient)
    print("\n=== Testing on patient 5 (frozen controller) ===")
    df_all_5, df_ts_5, daily_5, overall_5 = run_patient_single_env(
        meta_for(5),
        warm_days=30,
        train_days=0,
        sim_days=14,
        forecast_wrapper=forecast_wrapper,
        seed_offset=5 * 101,
        ts_controller=shared_ts,
    )

    summary_5 = pd.DataFrame([{
        "patient": "adult#005",
        "TIR_%_mean": overall_5["TIR_%_mean"],
        "Time<70_%_mean": overall_5["Time<70_%_mean"],
        "MeanBG_mean": overall_5["MeanBG_mean"],
        "days": overall_5["days"],
    }])

    df_all_5.to_csv("transfer_patient5_all.csv", index=False)
    df_ts_5.to_csv("transfer_patient5_eval.csv", index=False)
    if not daily_5.empty:
        daily_5.assign(patient="adult#005").to_csv("transfer_patient5_daily.csv", index=False)

    plot_overall_bars(summary_5, out_png="overall_metrics_patient5.png")
    if not daily_5.empty:
        plot_daily_tir_strip(daily_5.assign(patient="adult#005"), out_png="daily_TIR_patient5.png")

    print("\n=== Patient 5 — Overall Means (TS-only) ===")
    print(summary_5[["patient","TIR_%_mean","Time<70_%_mean","MeanBG_mean","days"]].to_string(index=False))

if __name__ == "__main__":
    main()
