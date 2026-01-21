import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
from collections import deque
import os

import numpy as np
import pandas as pd

# ------------------------- Torch forecaster -------------------------
import torch
import torch.nn as nn

# ------------------------- SimGlucose pieces ------------------------
try:
    from simglucose.controller.base import Controller, Action
    from simglucose.simulation.scenario import CustomScenario
except Exception as e:
    raise RuntimeError(
        "SimGlucose is required.\nInstall with: pip install simglucose==0.2.1 gym==0.15.7\n\n" + str(e)
    )

from src.simglucose_utils import (
    _env_step_minutes,
    _extract_now,
    get_true_bg,
)

from src.forecaster_ode import NeuralODEForecast
from src.forecaster_wrapper import ForecastingWrapper


# ---------------------- Scenario helpers ----------------------
@dataclass
class DaySpec:
    start_dt: pd.Timestamp
    meals: list  # list[(hour_float, grams)]

def _day_scenario_from_metadata(day0: pd.Timestamp, meals: list) -> DaySpec:
    start_dt = day0.replace(hour=0, minute=0, second=0, microsecond=0)
    out = []
    for m in meals:
        h = float(m["hour"]) + float(m.get("minute", 0)) / 60.0
        out.append((h, float(m["grams"])) )
    return DaySpec(start_dt=start_dt, meals=out)

def _mk_custom_scenario(day_spec: DaySpec):
    return CustomScenario(
        start_time=day_spec.start_dt.to_pydatetime(),
        scenario=[(h, g) for (h, g) in day_spec.meals if g > 0]
    )

def _mk_multiday_scenario(day0: pd.Timestamp, meals: list, days: int) -> CustomScenario:
    start_dt = day0.replace(hour=0, minute=0, second=0, microsecond=0)
    events = []
    for d in range(days):
        day_hours = 24.0 * d
        for m in meals:
            h = float(m["hour"]) + float(m.get("minute", 0))/60.0 + day_hours
            g = float(m["grams"])
            if g > 0:
                events.append((h, g))
    return CustomScenario(start_time=start_dt.to_pydatetime(), scenario=events)

class _MealBolusOnlyController(Controller):
    def __init__(self, icr_g_per_u: float, basal_u_per_min: float, refractory_min: float = 30.0):
        self.icr = float(icr_g_per_u)
        self.basal = float(basal_u_per_min)
        self.refractory_min = float(refractory_min)
        self._last_bolus_time = None
    def _meals_now(self, now_dt, env) -> float:
        start = env.scenario.start_time
        meals = getattr(env.scenario, 'scenario', [])
        if not meals: return 0.0
        h_now = (now_dt - start).total_seconds()/3600.0
        step_min = _env_step_minutes(env)
        for (mh, grams) in meals:
            if abs(mh - h_now) <= step_min/60.0 + 1e-9:
                return float(grams)
        return 0.0
    def policy(self, observation, reward, done, **info):
        now_dt = _extract_now(info, default=None)
        env = info.get('env', None)
        grams = self._meals_now(now_dt, env) if (now_dt and env is not None) else 0.0
        bolus = 0.0
        if grams > 0.0:
            if self._last_bolus_time is None or (now_dt - self._last_bolus_time).total_seconds() >= self.refractory_min*60:
                bolus = grams / max(self.icr, 1e-9)
                self._last_bolus_time = now_dt
        return Action(basal=self.basal, bolus=float(bolus))

class ThompsonBanditController(Controller):
    """
    Thompson Sampling (Gaussian) bandit with the same certified predictive safety
    filter and guardrails you used in UCBBanditController.

    Reward model (per state-action):
      r | μ, σ^2 ~ Normal(μ, σ^2)
    We maintain online estimates of μ and σ^2 (via Welford). For TS we sample
    θ ~ Normal(μ_hat, sqrt( (σ_hat^2 + prior_scale) / max(1, n + prior_n) ))
    and pick argmax θ. Untried actions are explored first (like your UCB warmup).

    Notes:
      - Keeps identical knobs for safety/forecast integration.
      - update_from_reward() refit μ and variance online.
      - set_eval() toggles the forecast gate behavior exactly like UCB.

    """

    def __init__(
        self,
        action_set=(0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4),
        bg_bins=tuple(range(40, 300, 20)),
        trend_bins=(-np.inf, -2.0, -1.0, -0.3, 0.3, 1.0, 2.0, np.inf),
        target=(120.0, 140.0),
        basal_u_per_min=0.006,
        refractory_min=15.0,

        # Thompson Sampling prior/scale
        ts_prior_n: float = 1.0,          # pseudo-count for variance shrinkage
        ts_prior_scale: float = 25.0,     # mg/dL-equivalent reward scale^2 proxy (variance add-on)

        # Guardrails
        min_bg_for_any_bolus=130.0,
        min_bg_for_micro=130.0,
        min_trend_for_micro=0.2,
        min_trend_for_bolus=0.6,
        max_bolus_when_flat=1.0,
        iob_halflife_min=60.0,
        iob_cap_lowbg=1.0,
        iob_cap_midbg=2.5,
        iob_cap_highbg=4.0,

        # Meal feed-forward
        meal_icr_g_per_u=15.0,
        meal_bolus_frac=1.2,
        min_bg_for_meal_bolus=100.0,
        meal_window_min=5.0,
        meal_bolus_cap_u=10.0,

        # Forecast
        forecastor: Optional[object] = None,
        use_forecast_in_eval: bool = True,
        use_forecast_in_train: bool = False,
        forecast_history_len: int = 20,
        forecast_pred_steps: int = 20,
        forecast_weight_decay: float = 0.9,
        forecast_mean_threshold: float = 120.0,

        # Safety certification params
        safety_cert_mode: str = "conformal",  # 'conformal' | 'cantelli'
        alpha: float = 0.02,
        alpha_W: float = None,
        alpha_S: float = None,
        gamma_per_min: float = 0.3,
        min_calib_n: int = 80,
        calib_maxlen: int = 2000,
        per_h_maxlen: int = 1500,
        enable_minimal_cap: bool = True,
        safe_cap_iters: int = 7,

        # optional bypass
        skip_forecast_if_high_and_rising=True,
        skip_bg_high_threshold=160.0,
        skip_trend_positive_threshold=0.0,

        debug=False,
    ):
        self.debug = bool(debug)

        # discretization + actions
        self.actions = np.array(action_set, dtype=float)
        self.bg_bins = np.array(bg_bins, dtype=float)
        self.trend_bins = np.array(trend_bins, dtype=float)
        self.target = target
        self.basal = float(basal_u_per_min)
        self.refractory_min = float(refractory_min)

        # TS prior hyperparams
        self.ts_prior_n = float(ts_prior_n)
        self.ts_prior_scale = float(ts_prior_scale)

        # bandit stats
        self.S = (len(self.bg_bins), len(self.trend_bins) - 1, len(self.actions))
        self.N_sa = np.zeros(self.S, dtype=np.int64)   # counts
        self.M_sa = np.zeros(self.S, dtype=float)      # running means
        self.M2_sa = np.zeros(self.S, dtype=float)     # running sum of squared diffs (Welford)
        self.N_s  = np.zeros(self.S[:2], dtype=np.int64)

        # runtime state
        self._hist_bg: List[float] = []
        self._last_state: Optional[Tuple[int,int]] = None
        self._last_a_idx: Optional[int] = None
        self._last_bolus = 0.0
        self._last_bolus_time = None
        self._last_step_min = 3.0
        self._eval_mode = False

        # guardrails
        self._iob = 0.0
        self.s_min_bg_any   = float(min_bg_for_any_bolus)
        self.s_min_bg_micro = float(min_bg_for_micro)
        self.s_min_tr_micro = float(min_trend_for_micro)
        self.s_min_tr_bolus = float(min_trend_for_bolus)
        self.s_max_flat     = float(max_bolus_when_flat)
        self.s_iob_hl_min   = float(iob_halflife_min)
        self.s_iob_cap_low  = float(iob_cap_lowbg)
        self.s_iob_cap_mid  = float(iob_cap_midbg)
        self.s_iob_cap_high = float(iob_cap_highbg)

        self.meal_icr = float(meal_icr_g_per_u)
        self.meal_frac = float(meal_bolus_frac)
        self.meal_bg_min = float(min_bg_for_meal_bolus)
        self.meal_win = float(meal_window_min)
        self.meal_cap = float(meal_bolus_cap_u)

        # forecast integration
        self.forecastor = forecastor
        self.use_forecast_in_eval = bool(use_forecast_in_eval)
        self.use_forecast_in_train = bool(use_forecast_in_train)
        self.f_hist_len = int(forecast_history_len)
        self.f_pred_steps = int(forecast_pred_steps)
        self.f_decay = float(forecast_weight_decay)
        self.f_thresh = float(forecast_mean_threshold)

        # feature buffers (7 cols)
        from collections import deque
        self._buf_bg    = deque(maxlen=self.f_hist_len)
        self._buf_ins   = deque(maxlen=self.f_hist_len)
        self._buf_cob   = deque(maxlen=self.f_hist_len)
        self._buf_hour  = deque(maxlen=self.f_hist_len)
        self._buf_lbgi  = deque(maxlen=self.f_hist_len)
        self._buf_hbgi  = deque(maxlen=self.f_hist_len)
        self._buf_risk  = deque(maxlen=self.f_hist_len)

        self._cob_tau_min = 100.0
        self._cob_est = 0.0
        self._current_day: Optional[Tuple[int,int,int]] = None
        self._steps_in_day: int = 0
        self._forecast_ready: bool = False
        self._global_step: int = 0

        self.skip_forecast_if_high_and_rising = bool(skip_forecast_if_high_and_rising)
        self.skip_bg_high_threshold = float(skip_bg_high_threshold)
        self.skip_trend_positive_threshold = float(skip_trend_positive_threshold)

        # --------------- SAFETY ---------------
        self.safety_mode = safety_cert_mode.lower()
        assert self.safety_mode in ("conformal", "cantelli")
        self.alpha = float(alpha)
        if alpha_W is None and alpha_S is None:
            self.alpha_W = self.alpha * 0.5
            self.alpha_S = self.alpha - self.alpha_W
        else:
            self.alpha_W = float(alpha_W if alpha_W is not None else self.alpha * 0.5)
            self.alpha_S = float(alpha_S if alpha_S is not None else self.alpha - self.alpha_W)
            assert (self.alpha_W + self.alpha_S) <= self.alpha + 1e-12
        self.gamma = float(gamma_per_min)
        self.min_calib_n = int(min_calib_n)
        self.enable_minimal_cap = bool(enable_minimal_cap)
        self.safe_cap_iters = int(safe_cap_iters)

        # residual buffers for functionals
        self._res_W = deque(maxlen=int(calib_maxlen))
        self._res_S = deque(maxlen=int(calib_maxlen))
        self._per_h_maxlen = int(per_h_maxlen)
        self._per_h_errors: List[deque] = [deque(maxlen=self._per_h_maxlen) for _ in range(self.f_pred_steps + 1)]
        self._pending: List[Dict[str, Any]] = []

    # ----- API helpers / discretizers -----
    def _log_block(self, reason: str, **kwargs):
        """Prints a one-line reason when an insulin bolus is blocked."""
        if not getattr(self, "debug", False):
            return
        bits = []
        for k, v in kwargs.items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                bits.append(f"{k}={v:.3f}")
            else:
                bits.append(f"{k}={v}")
        print(f"[BLOCK] {reason}" + (": " + " ".join(bits) if bits else ""))

    def set_eval(self, mode: bool): self._eval_mode = bool(mode)
    def _bg_to_bin(self, bg: float) -> int:
        return int(np.clip(np.searchsorted(self.bg_bins, bg, side='right')-1, 0, len(self.bg_bins)-1))
    def _trend(self) -> float:
        if len(self._hist_bg) < 3: return 0.0
        return float(np.mean(np.diff(self._hist_bg[-3:])))
    def _trend_to_bin(self, tr: float) -> int:
        return int(np.clip(np.searchsorted(self.trend_bins, tr, side='right')-1, 0, len(self.trend_bins)-2))
    def _decay_iob(self, dt_min: float):
        lam = math.log(2.0) / max(self.s_iob_hl_min, 1e-6)
        self._iob *= math.exp(-lam * max(dt_min, 0.0))

    # --------- feature stack & forecaster calls ----------
    def _feature_matrix_ready(self) -> bool:
        return all(len(buf) >= self.f_hist_len for buf in
                   (self._buf_bg, self._buf_ins, self._buf_cob, self._buf_hour, self._buf_lbgi, self._buf_hbgi, self._buf_risk))

    def _assemble_features(self) -> np.ndarray:
        arr = np.stack([
            np.array(self._buf_bg,   dtype=float),
            np.array(self._buf_ins,  dtype=float),
            np.array(self._buf_cob,  dtype=float),
            np.array(self._buf_hour, dtype=float),
            np.array(self._buf_lbgi, dtype=float),
            np.array(self._buf_hbgi, dtype=float),
            np.array(self._buf_risk, dtype=float),
        ], axis=1)
        return arr

    def _assemble_features_with_current_bolus(self, proposed_bolus: float, include_basal: bool = False) -> np.ndarray:
        feat = self._assemble_features()
        if feat.size == 0:
            return feat
        feat = feat.copy()
        insulin_val = float(proposed_bolus)
        if include_basal:
            insulin_val += float(self.basal) * float(getattr(self, "_last_step_min", 3.0))
        feat[-1, 1] = insulin_val
        return feat

    def _call_forecastor(self, feat_hist: np.ndarray):
        H7 = np.asarray(feat_hist, dtype=np.float32)
        if H7.ndim != 2 or H7.shape[1] != 7:
            raise ValueError(f"feat_hist must be (H,7); got {H7.shape}")

        if hasattr(self.forecastor, "reset"):
            try:
                self.forecastor.reset()
            except Exception:
                pass

        out = None
        if hasattr(self.forecastor, "predict"):
            out = self.forecastor.predict(H7)
        else:
            is_module = isinstance(self.forecastor, torch.nn.Module)
            if is_module:
                device = getattr(self.forecastor, "device", None)
                with torch.no_grad():
                    x = torch.tensor(H7[None, ...], dtype=torch.float32, device=device)
                    try:
                        out = self.forecastor(x)
                    except TypeError:
                        out = self.forecastor(x, None)
            else:
                out = self.forecastor(H7)

        def _to_np(a):
            if hasattr(a, "detach"):
                a = a.detach().cpu().numpy()
            return np.asarray(a, dtype=np.float32)

        if isinstance(out, dict):
            mean = out.get('mean', out.get('mu', None))
            if mean is None:
                raise ValueError("Forecaster dict must contain 'mean' or 'mu'.")
            mean = _to_np(mean)
        elif isinstance(out, (tuple, list)):
            preds = _to_np(out[0])
            if preds.ndim == 3 and preds.shape[0] == 1:
                preds = preds[0]
            if preds.ndim == 2 and preds.shape[1] == 1:
                preds = preds[:, 0]
            mean = preds
        else:
            mean = _to_np(out)

        mean = mean[: self.f_pred_steps].astype(np.float32, copy=False)
        return {"mean": mean}

    # --------- safety math ----------
    def _weights(self, K: int) -> np.ndarray:
        k = np.arange(1, K + 1, dtype=np.float32)
        w = (self.f_decay ** k)
        w = w / (w.sum() + 1e-12)
        return w

    def _compute_W_S(self, mu: np.ndarray, last_bg: float, step_min: float) -> Tuple[float, float]:
        K = len(mu)
        if K == 0:
            return float(last_bg), 0.0
        w = self._weights(K)
        W_hat = float(np.sum(w * mu))
        S_hat = float((mu[-1] - last_bg) / (K * step_min))
        return W_hat, S_hat

    def _register_pending_validation(self, start_idx: int, mu: np.ndarray, W_hat: float, S_hat: float, step_min: float):
        """Store forecast used at decision time; we will compute residuals after K steps."""
        item = {
            "start_idx": int(start_idx),   # index of BG_t in self._hist_bg
            "K": int(len(mu)),
            "mu": np.array(mu, dtype=float),
            "W_hat": float(W_hat),
            "S_hat": float(S_hat),
            "step_min": float(step_min),
        }
        self._pending.append(item)

    def _update_calibration(self):
        """When enough future BG has accrued for any pending item, compute residuals and update buffers."""
        if not self._pending:
            return
        done_idxs = []
        for i, item in enumerate(self._pending):
            start = item["start_idx"]
            K = item["K"]
            step_min = item["step_min"]
            if len(self._hist_bg) < start + K + 1:
                continue  # not matured yet
            # realized path
            bg0 = float(self._hist_bg[start])
            real = np.array(self._hist_bg[start + 1: start + 1 + K], dtype=float)  # length K
            mu = item["mu"]
            # functionals residuals
            w = self._weights(K)
            W_real = float(np.sum(w * real))
            S_real = float((real[-1] - bg0) / (K * step_min))
            self._res_W.append(abs(W_real - item["W_hat"]))
            self._res_S.append(abs(S_real - item["S_hat"]))
            # per-horizon residuals
            for k in range(K):
                ek = float(real[k] - mu[k])
                if k + 1 < len(self._per_h_errors):
                    self._per_h_errors[k + 1].append(ek)  # horizon k+1
            done_idxs.append(i)

        # remove processed in reverse order
        for i in reversed(done_idxs):
            self._pending.pop(i)

    def _calib_ready(self) -> bool:
        return (len(self._res_W) >= self.min_calib_n) and (len(self._res_S) >= self.min_calib_n)

    def _quantile(self, arr: deque, q: float) -> float:
        if not arr:
            return np.nan
        return float(np.quantile(np.fromiter(arr, dtype=float), q))

    def _sigma_h(self, h: int) -> float:
        """Std of per-horizon residual at horizon h (>=1)."""
        if h <= 0 or h >= len(self._per_h_errors) or not self._per_h_errors[h]:
            return 12.0  # conservative default mg/dL
        x = np.fromiter(self._per_h_errors[h], dtype=float)
        return float(np.std(x, ddof=1) if x.size > 1 else np.abs(x).mean())

    # --------- certification checks ----------
    def _chance_checks_conformal(self, W_hat, S_hat, L, gamma, alpha_unused):
        qW = self._quantile(self._res_W, 1.0 - self.alpha_W)
        qS = self._quantile(self._res_S, 1.0 - self.alpha_S)
        if np.isnan(qW) or np.isnan(qS):
            return False, False, float('nan'), float('nan')
        pass_mean  = (W_hat - qW) >= L
        pass_slope = (S_hat + qS) >= (-gamma)
        return pass_mean, pass_slope, qW, qS


    def _chance_checks_cantelli(self, mu, W_hat, S_hat, L, gamma, alpha_unused, step_min):
        K = len(mu); w = self._weights(K)
        sigmas = np.array([self._sigma_h(h) for h in range(1, K+1)], dtype=float)
        sigmaW = float(np.sum(np.abs(w) * sigmas))
        zW = math.sqrt((1 - self.alpha_W) / self.alpha_W)
        mW = sigmaW * zW
        pass_mean = (W_hat - mW) >= L

        sigmaK = self._sigma_h(K)
        zS = math.sqrt((1 - self.alpha_S) / self.alpha_S)
        mS = (sigmaK / (K * max(step_min,1e-12))) * zS
        pass_slope = (S_hat + mS) >= (-gamma)
        return pass_mean, pass_slope, mW, mS


    # ----- MAIN safety filter (returns possibly reduced bolus) -----
    def _safety_filter(self, proposed_bolus: float, step_min: float) -> float:
        # if forecaster not available or not warmed up, allow through
        use_gate = (self._eval_mode and self.use_forecast_in_eval) or ((not self._eval_mode) and self.use_forecast_in_train)
        if (not use_gate) or (self.forecastor is None) or (not self._forecast_ready) or (not self._feature_matrix_ready()):
            return proposed_bolus
        
        if proposed_bolus <= 0.0:
            return proposed_bolus  # skip registering residuals for W/S
        
        # NEW: bypass NeuralODE block if high and rising
        bg_now = float(self._hist_bg[-1]) if self._hist_bg else float('nan')
        tr_now = self._trend()  # note: this is mg/dL per step; we only use its sign
        if (self.skip_forecast_if_high_and_rising 
            and np.isfinite(bg_now) 
            and (bg_now > self.skip_bg_high_threshold) 
            and (tr_now > self.skip_trend_positive_threshold)):
            if self.debug and proposed_bolus > 0.0:
                print(f"[Safety] bypass_forecast(high&rise): bg={bg_now:.1f} tr={tr_now:+.3f} "
                    f"> {self.skip_trend_positive_threshold:+.3f} → allow {proposed_bolus:.2f}U")
            return proposed_bolus

        # compute forecast under current proposed dose
        feat_hist = self._assemble_features_with_current_bolus(proposed_bolus)
        pred = self._call_forecastor(feat_hist)
        mu = np.asarray(pred['mean'], dtype=float)
        last_bg = float(self._hist_bg[-1]) if self._hist_bg else float(mu[0])
        W_hat, S_hat = self._compute_W_S(mu, last_bg, step_min)

        # chance constraints
        L = float(self.f_thresh)  # lower bound for weighted mean (e.g., 120 mg/dL)
        alpha = float(self.alpha)
        gamma = float(self.gamma)

        # Prefer conformal; fall back to Cantelli if not ready
        if self.safety_mode == "conformal" and self._calib_ready():
            pass_mean, pass_slope, qW, qS = self._chance_checks_conformal(W_hat, S_hat, L, gamma, alpha)
            margin_info = f"[Conformal] qW≈{qW:.1f}, qS≈{qS:.3f}"
        else:
            pass_mean, pass_slope, mW, mS = self._chance_checks_cantelli(mu, W_hat, S_hat, L, gamma, alpha, step_min)
            margin_info = f"[Cantelli] mW≈{mW:.1f}, mS≈{mS:.3f}"

        if self.debug:
            print(f"[Safety] dose={proposed_bolus:.2f}U | W_hat={W_hat:.1f} vs L={L:.1f}, "
                  f"S_hat={S_hat:+.3f} vs -γ={-gamma:.3f} | pass_mean={pass_mean} pass_slope={pass_slope} {margin_info}")

        # register this decision for later residual updates (calibration)
        start_idx = len(self._hist_bg) - 1  # BG_t index is the last appended
        self._register_pending_validation(start_idx, mu, W_hat, S_hat, step_min)

        # if passes, done
        if pass_mean and pass_slope:
            return proposed_bolus

        # Otherwise, optionally compute a safe cap via 1-D line search
        if not self.enable_minimal_cap:
            if proposed_bolus > 0.0:
                self._log_block("safety_gate_block_no_cap", proposed=proposed_bolus, W=W_hat, S=S_hat)
            return 0.0

        lo, hi = 0.0, proposed_bolus
        best = 0.0
        for _ in range(self.safe_cap_iters):
            mid = 0.5 * (lo + hi)
            feat_mid = self._assemble_features_with_current_bolus(mid)
            mu_mid = np.asarray(self._call_forecastor(feat_mid)['mean'], dtype=float)
            Wm, Sm = self._compute_W_S(mu_mid, last_bg, step_min)
            if self.safety_mode == "conformal" and self._calib_ready():
                ok_mean, ok_slope, *_ = self._chance_checks_conformal(Wm, Sm, L, gamma, alpha)
            else:
                ok_mean, ok_slope, *_ = self._chance_checks_cantelli(mu_mid, Wm, Sm, L, gamma, alpha, step_min)
            if ok_mean and ok_slope:
                best = mid
                lo = mid
            else:
                hi = mid

        if best == 0.0 and proposed_bolus > 0.0:
            # Helpful to echo the margins mode that applied
            if self.safety_mode == "conformal" and self._calib_ready():
                self._log_block("safety_gate_zero_cap_conformal", proposed=proposed_bolus, W=W_hat, S=S_hat, L=L, gamma=gamma, alpha=self.alpha)
            else:
                self._log_block("safety_gate_zero_cap_cantelli", proposed=proposed_bolus, W=W_hat, S=S_hat, L=L, gamma=gamma, alpha=self.alpha)

        if self.debug:
            print(f"[Safety] cap: {proposed_bolus:.2f}U -> {best:.2f}U")
        return float(best)

    # ----- Guardrails (same as before) -----
    def _apply_guardrails(self, bg: float, tr: float, bolus, is_meal: bool) -> float:
        try:
            bolus = float(bolus)
            if not np.isfinite(bolus): bolus = 0.0
        except Exception:
            bolus = 0.0

        falling = False
        if len(self._hist_bg) >= 2 and (bg + 0.5) < float(self._hist_bg[-1]):
            falling = True
        if len(self._hist_bg) >= 3:
            prev = float(self._hist_bg[-1]); prev2 = float(self._hist_bg[-2])
            if prev < prev2 and (prev - bg) > 0.5: falling = True
        if tr <= -0.05: falling = True

        if self.debug and bolus > 0.0:
            print(f"[UCB-guardrail] BG={bg:.1f}, tr={tr:+.3f}, falling={falling}, proposed={bolus:.2f}U")
        if falling and (bg < 140.0):
            self._log_block("falling_trend_low_bg", bg=bg, tr=tr, iob=self._iob)
            return 0.0

        if bg < self.s_min_bg_any and not (is_meal and bg >= self.meal_bg_min):
            self._log_block("below_min_bg", bg=bg, min_bg=self.s_min_bg_any, is_meal=is_meal)
            return 0.0
        if tr < self.s_min_tr_micro and not is_meal: bolus = min(bolus, self.s_max_flat)
        if bolus > 0.0 and not is_meal and tr < self.s_min_tr_bolus: bolus = min(bolus, self.s_max_flat)

        iob_cap = self.s_iob_cap_high if bg >= 180 else (self.s_iob_cap_mid if bg >= 140 else self.s_iob_cap_low)
        if self._iob > iob_cap:
            self._log_block("iob_cap_exceeded", bg=bg, tr=tr, iob=self._iob, cap=iob_cap)
            bolus = 0.0

        return float(max(0.0, bolus))

    # ----- Day rollover & buffers -----
    def _maybe_rollover_day(self, now_dt):
        if now_dt is None:
            return
        key = (now_dt.year, now_dt.month, now_dt.day)
        if self._current_day != key:
            self._current_day = key
            self._steps_in_day = 0
            self._forecast_ready = False
            self._buf_bg.clear(); self._buf_ins.clear(); self._buf_cob.clear()
            self._buf_hour.clear(); self._buf_lbgi.clear(); self._buf_hbgi.clear(); self._buf_risk.clear()
            self._cob_est = 0.0
            # if forecaster exposes state
            if self.forecastor is not None and hasattr(self.forecastor, "reset"):
                try:
                    self.forecastor.reset()
                except Exception:
                    pass

    def _grams_now(self, env, now_dt) -> float:
        if env is None or now_dt is None: return 0.0
        start = getattr(env.scenario, 'start_time', None)
        meals = getattr(env.scenario, 'scenario', [])
        if not start or not meals: return 0.0
        h_now = (now_dt - start).total_seconds()/3600.0
        win_h = self.meal_win/60.0
        for (mh, grams) in meals:
            if abs(mh - h_now) <= win_h + 1e-9:
                return float(grams)
        return 0.0

    def _snap_to_grid_capped(self, dose_u: float) -> float:
        a = self.actions[self.actions <= dose_u]
        return float(a.max()) if len(a) else 0.0

    # ---------------- Thompson action selection ----------------
    def _select_action_ts(self, s_bg: int, s_tr: int) -> int:
        tried = self.N_sa[s_bg, s_tr, :]

        # Ensure each action is tried at least once in this (state) bucket
        for a_idx in range(len(self.actions)):
            if tried[a_idx] == 0:
                return a_idx

        means = self.M_sa[s_bg, s_tr, :]
        M2s   = self.M2_sa[s_bg, s_tr, :]
        ns    = self.N_sa[s_bg, s_tr, :].astype(float)

        # unbiased variance estimate per (s,a)
        var_hat = np.maximum(M2s / np.maximum(ns - 1.0, 1.0), 1e-6)

        # posterior sampling std:
        #   std_post = sqrt( (var_hat + ts_prior_scale) / (ns + ts_prior_n) )
        std_post = np.sqrt((var_hat + self.ts_prior_scale) / (ns + self.ts_prior_n))

        # sample θ ~ Normal(mean, std_post) for each action
        theta = np.random.normal(loc=means, scale=std_post)
        return int(np.argmax(theta))

    # ---------------- Online reward update (Welford) -----------
    def update_from_reward(self, reward: float):
        if self._last_state is None or self._last_a_idx is None:
            return
        s_bg, s_tr = self._last_state
        a = self._last_a_idx

        # counts for the (s,a)
        self.N_s[s_bg, s_tr] += 1
        self.N_sa[s_bg, s_tr, a] += 1
        n = self.N_sa[s_bg, s_tr, a]

        # Welford updates
        mean = self.M_sa[s_bg, s_tr, a]
        delta = float(reward) - mean
        mean_new = mean + delta / max(1, n)
        delta2 = float(reward) - mean_new
        self.M_sa[s_bg, s_tr, a] = mean_new
        self.M2_sa[s_bg, s_tr, a] += delta * delta2

    # ---------------- Main policy (same flow) ------------------
    def policy(self, observation, reward, done, **info):
        env = info.get('env', None)
        now_dt = _extract_now(info, default=None)
        step_min = float(info.get('sample_time', self._last_step_min) or self._last_step_min)
        self._last_step_min = step_min

        self._maybe_rollover_day(now_dt)
        self._update_calibration()

        bg = get_true_bg(observation, info, env)
        self._hist_bg.append(bg)
        self._decay_iob(step_min)
        tr = self._trend()  # mg/dL/min

        # refractory
        can_bolus = True
        if self._last_bolus_time is not None and now_dt is not None:
            can_bolus = (now_dt - self._last_bolus_time).total_seconds() >= self.refractory_min * 60.0

        # discretized state
        s_bg = self._bg_to_bin(bg)
        s_tr = self._trend_to_bin(tr)

        # Thompson selection (greedy in eval like your UCB code)
        if self._eval_mode:
            a_idx = int(np.argmax(self.M_sa[s_bg, s_tr, :]))
        else:
            a_idx = self._select_action_ts(s_bg, s_tr)

        ts_bolus = float(self.actions[a_idx]) if can_bolus else 0.0

        # Meal feed-forward
        grams = self._grams_now(env, now_dt)
        is_meal_now = (grams > 0.0)
        meal_bolus = 0.0
        if is_meal_now and bg >= self.meal_bg_min:
            raw = (grams / max(self.meal_icr, 1e-9)) * self.meal_frac
            raw = float(min(raw, self.meal_cap))
            meal_bolus = self._snap_to_grid_capped(raw)
            if not can_bolus:
                can_bolus = True  # allow meal bolus through refractory

        # Combine + certified safety filter
        pre_gate = max(ts_bolus if can_bolus else 0.0, meal_bolus)

        if is_meal_now:
            bolus = pre_gate
        else:
            bolus = self._safety_filter(pre_gate, step_min)

        if self.debug and abs(bolus - pre_gate) > 1e-9:
            print(f"[GATE] modified dose: {pre_gate:.2f}U -> {bolus:.2f}U")

        # Guardrails after certification
        bolus = self._apply_guardrails(bg, tr, bolus, is_meal_now)
        # ---- Time-of-day cap: after 10:00pm, cap any bolus at 0.5U ----
        if now_dt is not None:
            hour_float = float(now_dt.hour) + float(now_dt.minute)/60.0
            if hour_float >= 22.0:
                if bolus > 0.5 and self.debug:
                    print(f"[Time cap] {now_dt.strftime('%H:%M')} → cap {bolus:.2f}U to 0.50U")
                bolus = min(bolus, 0.5)


        # Final coercion
        try:
            bolus = float(bolus)
            if not np.isfinite(bolus): bolus = 0.0
        except Exception:
            bolus = 0.0

        if self.debug:
            print(f"[FINAL] deliver bolus={bolus:.2f}U  basal={self.basal:.3f}U/min")

        if bolus > 0.0 and now_dt is not None:
            self._last_bolus_time = now_dt

            # optional diagnostics
            if self.forecastor is not None and self._feature_matrix_ready():
                feat_hist = self._assemble_features_with_current_bolus(bolus)
                mu = np.asarray(self._call_forecastor(feat_hist)['mean'], dtype=float)
                last_bg = float(self._hist_bg[-1]) if self._hist_bg else float(mu[0])
                W_hat, S_hat = self._compute_W_S(mu, last_bg, step_min)
                # mu_str = np.array2string(mu, precision=1, separator=' ', max_line_width=1_000)

        self._iob += bolus

        if not is_meal_now:
            bolus = self._snap_to_grid_capped(bolus)
        eff_idx = int(np.argmin(np.abs(self.actions - bolus)))
        self._last_state = (s_bg, s_tr)
        self._last_a_idx = eff_idx
        self._last_bolus = bolus

        # push features for next step
        row = self._prepare_feature_row(bg, self._last_bolus, grams, now_dt, step_min)
        self._buf_bg.append(row[0]);   self._buf_ins.append(row[1]);  self._buf_cob.append(row[2])
        self._buf_hour.append(row[3]); self._buf_lbgi.append(row[4]); self._buf_hbgi.append(row[5]); self._buf_risk.append(row[6])

        self._steps_in_day += 1
        self._global_step += 1
        if not self._forecast_ready and self._steps_in_day >= self.f_hist_len:
            self._forecast_ready = True

        if self.debug:
            tag = "MEAL" if is_meal_now else "TS"
            print(f"[TS] {tag} t={now_dt} grams={grams:.1f} BG={bg:.1f} tr={tr:+.2f} IOB={self._iob:.2f}U dose={bolus:.2f}U")

        return Action(basal=self.basal, bolus=bolus)

    # ---- feature-row assembly & simple COB model ----
    def _prepare_feature_row(self, bg: float, delivered_bolus_last: float, grams_now: float, now_dt, dt_min: float):
        # simple COB exponential decay
        decay = math.exp(-max(dt_min, 0.0) / max(self._cob_tau_min, 1e-6))
        self._cob_est = self._cob_est * decay + max(0.0, grams_now)
        hour = float(getattr(now_dt, 'hour', 0)) + float(getattr(now_dt, 'minute', 0))/60.0 if now_dt is not None else 0.0
        lbgi = 0.0; hbgi = 0.0; risk = 0.0
        return float(bg), float(delivered_bolus_last), float(self._cob_est), hour, lbgi, hbgi, risk
    

def build_forecast_wrapper(
    ckpt_path: str,
    history_len: int = 10,
    pred_steps: int = 10,
    roll_with_predictions: bool = False,
    trend_extrap: str = "linear",
    trend_tail: int = 3,
    delta_next: float = 1.0,
):
    """
    Loads forecasting_model_checkpoint.pt exactly like your notebook:
    - if file missing => returns None
    - chooses cuda if available
    - reads normalization + feature_cols
    - builds NeuralODEForecast and loads model_state (strict=False)
    - wraps with ForecastingWrapper that handles *_norm feature names
    """
    if not ckpt_path or (not os.path.exists(ckpt_path)):
        print(f"[INFO] No forecasting checkpoint found at {ckpt_path}; running without forecast safety gate.")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        norm_params = ckpt.get("normalization", {})
        ckpt_cols = ckpt.get("feature_cols", ["bg", "insulin", "cob", "hour", "lbgi", "hbgi", "risk"])

        def _to_raw_cols(cols):
            return [c[:-5] if isinstance(c, str) and c.endswith("_norm") else c for c in cols]

        feature_cols_raw = _to_raw_cols(ckpt_cols)

        # Build model (Neural ODE)
        net = NeuralODEForecast(
            input_dim=7,
            hidden_dim=64,
            latent_dim=6,
            out_dim=2,
            ode_hidden=128,
            solver="rk4",
            rtol=1e-3,
            atol=1e-4,
        )
        if "model_state" in ckpt:
            net.load_state_dict(ckpt["model_state"], strict=False)
        net.eval()

        fw = ForecastingWrapper(
            base_model=net,
            norm_params=norm_params,
            feature_cols=feature_cols_raw,
            device=device,
            history_len=history_len,
            pred_steps=pred_steps,
            roll_with_predictions=roll_with_predictions,
            trend_extrap=trend_extrap,
            trend_tail=trend_tail,
            delta_next=delta_next,
        )

        print(f"[INFO] Forecasting checkpoint loaded from {ckpt_path}; using feature cols: {feature_cols_raw}")
        return fw

    except Exception as e:
        print(f"[WARN] Could not initialize forecaster from {ckpt_path}: {e}")
        return None