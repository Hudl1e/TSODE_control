import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List
from simglucose.controller.base import Action

def _env_step_minutes(env) -> float:
    from datetime import timedelta
    st = getattr(env, 'sample_time', timedelta(minutes=3))
    if isinstance(st, timedelta):
        return float(st.total_seconds()/60.0)
    try:
        return float(st) if st else 3.0
    except Exception:
        return 3.0

def _extract_now(info: dict, default=None) -> Optional[pd.Timestamp]:
    if isinstance(info, dict):
        if 'current_time' in info:
            return pd.to_datetime(info['current_time'])
        if 'time' in info:
            return pd.to_datetime(info['time'])
    return default

def get_true_bg(observation, info=None, env=None) -> float:
    if isinstance(info, dict):
        for k in ('BG', 'bg', 'PlasmaGlucose', 'plasma_glucose'):
            if k in info and info[k] is not None:
                try:
                    return float(info[k])
                except Exception:
                    pass
    for attr in ('BG', 'bg', 'glucose'):
        if hasattr(observation, attr):
            try:
                return float(getattr(observation, attr))
            except Exception:
                pass
    try:
        p = getattr(env, 'patient', None)
        st = getattr(p, '_state', None) or getattr(p, 'state', None)
        if st is not None and hasattr(st, 'G'):
            return float(getattr(st, 'G'))
    except Exception:
        pass
    if hasattr(observation, '__len__') and len(observation) > 0:
        try:
            return float(observation[0])
        except Exception:
            pass
    return float(observation)

def rl_shaped_reward(bg: float, last_bolus: float = 0.0, target=(120.0, 140.0)) -> float:
    lo, hi = target
    if bg < 60:   base = -160.0 - 4.0 * ((60 - bg) / 5.0)
    elif bg < 70: base = -100.0 - 3.0 * ((70 - bg) / 5.0)
    elif bg < 80: base = -40.0  - 1.5 * ((80 - bg) / 5.0)
    elif 80 <= bg < 100:
        d = (100 - bg) / 10.0
        base = -0.8 * (d ** 2)
    elif lo <= bg <= hi:
        base = 4.0
    elif hi < bg <= 160:
        d = (bg - hi) / 20.0
        base = 0.8 - 1.0 * (d ** 2)
    elif 160 < bg <= 200:
        base = -1.5 - 0.03 * (bg - 160)
    else:
        base = -8.0 - 0.05 * (bg - 200)
    if last_bolus > 0.0 and bg < 110:
        base -= 1.0 * float(last_bolus)
    return float(base)

def _get_cgm(observation, info=None, env=None) -> float:
    # Prefer the sensor reading in 'observation'
    try:
        for attr in ('CGM', 'cgm', 'BG', 'bg', 'glucose'):
            if hasattr(observation, attr):
                return float(getattr(observation, attr))
    except Exception:
        pass
    # If observation is array-like, take the first element
    if hasattr(observation, '__len__') and len(observation) > 0:
        try:
            return float(observation[0])
        except Exception:
            pass
    # Fallbacks: sometimes simglucose sticks CGM into info
    if isinstance(info, dict):
        for k in ('CGM', 'cgm', 'BG', 'bg'):
            if k in info and info[k] is not None:
                try:
                    return float(info[k])
                except Exception:
                    pass
    return float('nan')

def _kovatchev_risk(bg: float) -> tuple:
    """Instantaneous LBGI/HBGI/Risk from BG (mg/dL)."""
    if not np.isfinite(bg) or bg <= 0:
        return 0.0, 0.0, 0.0
    f = 1.509 * (np.log(bg)**1.084 - 5.381)
    risk = 10.0 * (f**2)
    lbgi = risk if f < 0 else 0.0
    hbgi = risk if f > 0 else 0.0
    return float(lbgi), float(hbgi), float(risk)

def _gym_step(env, action: Action):
    out = env.step(Action(basal=action.basal, bolus=action.bolus))
    if len(out) == 4:
        obs, reward_env, done, info = out
    else:
        obs, reward_env, terminated, truncated, info = out
        done = bool(terminated) or bool(truncated)
    return obs, reward_env, done, info

def _pre_step_safety_clamp(obs, info, env, ctrl, act: Action) -> Action:
    bg_pre = get_true_bg(obs, info, env)
    tr_now = ctrl._trend() if hasattr(ctrl, "_trend") else 0.0
    if (bg_pre < 85) or (bg_pre < 100 and tr_now <= 0.0) or (tr_now <= -1.0 and bg_pre < 120):
        if hasattr(ctrl, "_log_block"):
            try:
                ctrl._log_block("pre_step_clamp", bg=bg_pre, tr=tr_now)
            except Exception:
                pass
        return Action(basal=act.basal, bolus=0.0)
    return act