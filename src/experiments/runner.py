# src/experiments/runner.py
import numpy as np
import pandas as pd

from simglucose.simulation.env import T1DSimEnv
from simglucose.sensor.cgm import CGMSensor
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.actuator.pump import InsulinPump

from src.controllers.ts_bandit_safe import _mk_multiday_scenario, _MealBolusOnlyController
from src.experiments.controller_factory import make_ts_controller
from src.metrics import daily_bg_metrics
from src.simglucose_utils import (
    _env_step_minutes, _gym_step, get_true_bg,
    rl_shaped_reward, _pre_step_safety_clamp
)

def run_patient_single_env(
    meta,
    warm_days=0,
    train_days=30,
    sim_days=14,
    forecast_wrapper=None,
    seed_offset=0,
    ts_controller=None,       # NEW: allow reuse for transfer experiments
):
    meta = dict(meta)
    meta["seed"] = int(meta.get("seed", 0)) + int(seed_offset)

    ts = ts_controller if ts_controller is not None else make_ts_controller(meta, forecast_wrapper=forecast_wrapper, debug=False)
    np.random.seed(meta["seed"])

    start_dt  = pd.to_datetime(meta["start_time"]).normalize()
    train_end = start_dt + pd.Timedelta(days=train_days)
    warm_end  = train_end + pd.Timedelta(days=warm_days)
    sim_end   = warm_end + pd.Timedelta(days=sim_days)

    scen = _mk_multiday_scenario(start_dt, meta["meals"], days=(train_days + warm_days + sim_days))
    patient = T1DPatient.withName(meta["patient"])
    sensor  = CGMSensor.withName(meta["sensor"], seed=meta["seed"])
    pump    = InsulinPump.withName("Insulet")
    env     = T1DSimEnv(patient, sensor, pump, scen)

    warm_mb = _MealBolusOnlyController(
        icr_g_per_u=meta["controller"]["icr_g_per_u"],
        basal_u_per_min=meta["controller"]["basal_per_min"],
        refractory_min=30.0,
    )

    rst = env.reset()
    obs = rst[0] if isinstance(rst, (tuple, list)) else rst
    info = rst[1] if isinstance(rst, (tuple, list)) and len(rst) > 1 else {}

    step_min = _env_step_minutes(env)
    now = scen.start_time
    done = False
    last_reward = 0.0

    rows_all = {"dt": [], "bg": [], "bolus": [], "basal_per_min": [], "reward": [], "phase": []}

    while (now < sim_end) and (not done):
        if now < train_end:
            ts.set_eval(False)
            ctrl = ts
        elif now < warm_end and warm_days > 0:
            ts.set_eval(True)
            ctrl = warm_mb
        else:
            ts.set_eval(True)
            ctrl = ts

        act = ctrl.policy(obs, last_reward, done, time=now, current_time=now, sample_time=step_min, env=env)
        act = _pre_step_safety_clamp(obs, info, env, ctrl, act)

        obs, _, done, info = _gym_step(env, act)

        bg_val = get_true_bg(obs, info, env)
        step_reward = rl_shaped_reward(bg_val, last_bolus=float(getattr(act, "bolus", 0.0)))

        # Learn only if TS is the active controller and we are before warm_end (matches your notebook logic)
        if (ctrl is ts) and (now < warm_end):
            ts.update_from_reward(step_reward)

        rows_all["dt"].append(now)
        rows_all["bg"].append(bg_val)
        rows_all["bolus"].append(float(getattr(act, "bolus", 0.0)))
        rows_all["basal_per_min"].append(float(getattr(act, "basal", 0.0)))
        rows_all["reward"].append(step_reward)
        rows_all["phase"].append("train" if now < train_end else ("warm" if now < warm_end else "eval"))

        last_reward = step_reward
        now += pd.Timedelta(minutes=step_min)

    df_all = pd.DataFrame(rows_all)
    df_ts_only = df_all[df_all["phase"] == "eval"].copy()

    if df_ts_only.empty:
        daily = pd.DataFrame(columns=["date", "TIR_%", "Time<70_%", "MeanBG"])
        overall = {"days": 0, "TIR_%_mean": np.nan, "Time<70_%_mean": np.nan, "MeanBG_mean": np.nan}
    else:
        daily, overall = daily_bg_metrics(df_ts_only, target=(70, 180))

    return df_all, df_ts_only, daily, overall

def run_many_patients(
    metas,
    forecast_wrapper=None,
    warm_days=30,
    train_days=30,
    sim_days=14,
    skip_if_no_eval=True,
):
    per_patient = {}
    rows = []
    daily_frames = []
    skipped = []

    for i, meta in enumerate(metas):
        patient_id = meta.get("patient", f"patient#{i:03d}")
        print(f"\n=== Running {patient_id} ===")
        try:
            df_all, df_ts, daily, overall = run_patient_single_env(
                meta,
                warm_days=warm_days,
                train_days=train_days,
                sim_days=sim_days,
                forecast_wrapper=forecast_wrapper,
                seed_offset=i * 101,
            )

            no_eval = df_ts.empty or (overall.get("days", 0) == 0)
            if skip_if_no_eval and no_eval:
                print(f"[WARN] Skipping {patient_id}: no TS eval rows.")
                skipped.append(patient_id)
                continue

            per_patient[patient_id] = {
                "df_all": df_all,
                "df_ts_only": df_ts,
                "daily": daily,
                "overall": overall,
            }

            rows.append({
                "patient": patient_id,
                "TIR_%_mean": overall["TIR_%_mean"],
                "Time<70_%_mean": overall["Time<70_%_mean"],
                "MeanBG_mean": overall["MeanBG_mean"],
                "days": overall["days"],
            })

            d2 = daily.copy()
            d2["patient"] = patient_id
            daily_frames.append(d2)

        except Exception as e:
            print(f"[ERROR] Skipping {patient_id}: {e}")
            skipped.append(patient_id)

    summary_df = pd.DataFrame(rows).sort_values("patient").reset_index(drop=True) if rows else pd.DataFrame(
        columns=["patient","TIR_%_mean","Time<70_%_mean","MeanBG_mean","days"]
    )
    daily_long = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()

    if skipped:
        print(f"[INFO] Skipped patients: {', '.join(skipped)}")

    return per_patient, summary_df, daily_long
