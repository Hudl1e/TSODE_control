# src/experiments/controller_factory.py
from src.controllers.ts_bandit_safe import ThompsonBanditController

def make_ts_controller(meta, forecast_wrapper=None, debug=False):
    return ThompsonBanditController(
        action_set=(0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2.0,2.5,3.0),
        target=(100, 160),
        basal_u_per_min=meta["controller"]["basal_per_min"],
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

        meal_icr_g_per_u=meta["controller"]["icr_g_per_u"],
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

        debug=debug,
    )
