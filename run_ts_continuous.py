# scripts/run_ts_continuous.py
import os
import numpy as np
import pandas as pd
import torch

from src.config import metadata
from src.forecaster_ode import NeuralODEForecast
from src.forecaster_wrapper import ForecastingWrapper
from src.metrics import daily_bg_metrics
from src.simglucose_utils import _env_step_minutes, get_true_bg, rl_shaped_reward, _get_cgm, _kovatchev_risk, _gym_step
from src.controllers.ts_bandit_safe import ThompsonBanditController, _mk_multiday_scenario
from simglucose.simulation.env import T1DSimEnv
from simglucose.sensor.cgm import CGMSensor
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.actuator.pump import InsulinPump

def load_forecaster(ckpt_path: str):
    if not os.path.exists(ckpt_path):
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    norm_params = ckpt.get("normalization", {})
    ckpt_cols = ckpt.get("feature_cols", ["bg", "insulin", "cob", "hour", "lbgi", "hbgi", "risk"])
    feature_cols_raw = [c[:-5] if isinstance(c, str) and c.endswith("_norm") else c for c in ckpt_cols]

    net = NeuralODEForecast(input_dim=7, hidden_dim=64, latent_dim=6, out_dim=2,
                            ode_hidden=128, solver="rk4", rtol=1e-3, atol=1e-4)
    if "model_state" in ckpt:
        net.load_state_dict(ckpt["model_state"], strict=False)
    net.eval()

    wrapper = ForecastingWrapper(
        base_model=net,
        norm_params=norm_params,
        feature_cols=feature_cols_raw,
        device=device,
        history_len=10,
        pred_steps=10,
        roll_with_predictions=False,
        trend_extrap="linear",
        trend_tail=3,
        delta_next=1.0,
    )
    print(f"[INFO] Loaded forecaster from {ckpt_path}; cols={feature_cols_raw}")
    return wrapper

def main():
    # 1) Controller
    forecast_wrapper = load_forecaster("forecasting_model_checkpoint.pt")

    ts = ThompsonBanditController(
        action_set=(0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2.0,2.5,3.0),
        target=(100, 160),
        basal_u_per_min=metadata["controller"]["basal_per_min"],
        refractory_min=6.0,
        min_bg_for_any_bolus=120.0,
        min_bg_for_micro=110.0,
        min_trend_for_micro=0.3,
        min_trend_for_bolus=0.6,
        max_bolus_when_flat=1.0,
        iob_halflife_min=30.0,
        iob_cap_lowbg=0.4,
        iob_cap_midbg=1.0,
        iob_cap_highbg=2.0,
        ts_prior_n=1.0,
        ts_prior_scale=25.0,
        meal_icr_g_per_u=metadata["controller"]["icr_g_per_u"],
        meal_bolus_frac=1.2,
        min_bg_for_meal_bolus=100.0,
        meal_window_min=6.0,
        meal_bolus_cap_u=3.0,
        forecastor=forecast_wrapper,
        use_forecast_in_eval=True,
        use_forecast_in_train=True,
        forecast_history_len=10,
        forecast_pred_steps=10,
        forecast_weight_decay=0.92,
        forecast_mean_threshold=120.0,
        safety_cert_mode="conformal",
        alpha=0.02,
        alpha_W=0.018,
        alpha_S=0.002,
        gamma_per_min=0.3,
        min_calib_n=80,
        enable_minimal_cap=True,
        skip_forecast_if_high_and_rising=True,
        skip_bg_high_threshold=170.0,
        skip_trend_positive_threshold=0.0,
        debug=False,
    )

    np.random.seed(metadata["seed"])

    # 2) Timeline
    sim_start_dt = pd.to_datetime(metadata["start_time"])
    train_end_dt = pd.to_datetime("2025-05-19T00:00:00")
    eval_days = 14
    sim_end_dt = train_end_dt + pd.Timedelta(days=eval_days)

    total_sim_days = (sim_end_dt.normalize() - sim_start_dt.normalize()).days
    print(f"[SETUP] {sim_start_dt.date()} -> {sim_end_dt.date()}  ({total_sim_days} days)")

    # 3) Env
    scen = _mk_multiday_scenario(sim_start_dt, metadata["meals"], days=total_sim_days)
    patient = T1DPatient.withName(metadata["patient"])
    sensor  = CGMSensor.withName(metadata["sensor"], seed=metadata["seed"])
    pump    = InsulinPump.withName("Insulet")
    env     = T1DSimEnv(patient, sensor, pump, scen)

    # 4) Loop
    results = {"Time": [], "BG": [], "CGM": [], "CHO": [], "insulin": [],
               "LBGI": [], "HBGI": [], "Risk": [], "mode": []}

    rst = env.reset()
    obs = rst[0] if isinstance(rst, (tuple, list)) else rst
    info = rst[1] if isinstance(rst, (tuple, list)) and len(rst) > 1 else {}

    now = scen.start_time
    step_min = _env_step_minutes(env)
    last_reward = 0.0
    done = False

    print(f"[RUN] step_min={step_min} min | train until {train_end_dt} | eval until {sim_end_dt}")

    while now < sim_end_dt and not done:
        is_training = now < train_end_dt
        ts.set_eval(not is_training)

        act = ts.policy(obs, last_reward, done, time=now, current_time=now, sample_time=step_min, env=env)

        bolus = float(getattr(act, "bolus", 0.0))
        basal_per_min = float(getattr(act, "basal", 0.0))
        insulin_this_step = basal_per_min * float(step_min) + bolus
        cho_grams = float(ts._grams_now(env, now))
        cgm_val = _get_cgm(obs, info, env)

        obs, _, done, info = _gym_step(env, act)

        bg_val = get_true_bg(obs, info, env)
        step_reward = rl_shaped_reward(bg_val, last_bolus=bolus)
        if is_training:
            ts.update_from_reward(step_reward)

        lbgi, hbgi, risk = _kovatchev_risk(bg_val)

        results["Time"].append(now)
        results["BG"].append(bg_val)
        results["CGM"].append(cgm_val)
        results["CHO"].append(cho_grams)
        results["insulin"].append(insulin_this_step)
        results["LBGI"].append(lbgi)
        results["HBGI"].append(hbgi)
        results["Risk"].append(risk)
        results["mode"].append("train" if is_training else "eval")

        last_reward = step_reward
        now += pd.Timedelta(minutes=step_min)

    df_all = pd.DataFrame(results)
    df_all.to_csv("ts_full_continuous.csv", index=False)
    print("[DONE] wrote ts_full_continuous.csv")

    df_eval = df_all[df_all["mode"] == "eval"].copy()
    if not df_eval.empty:
        daily, overall = daily_bg_metrics(df_eval.rename(columns={"Time": "dt", "BG": "bg"}), target=(70, 180))
        print("[EVAL]", overall)

    df_train = df_all[df_all["mode"] == "train"][["Time","BG","CGM","CHO","insulin","LBGI","HBGI","Risk"]].copy()
    df_train.to_csv("ts_training_continuous.csv", index=False, float_format="%.10g", date_format="%Y-%m-%d %H:%M:%S")
    print("[DONE] wrote ts_training_continuous.csv")

if __name__ == "__main__":
    main()
