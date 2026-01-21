# src/config.py

metadata = {
    "seed": 42,
    "patient": "adult#001",
    "sensor": "Dexcom",
    "start_time": "2025-04-19T21:00:00",
    "days": 44,
    "native_step_minutes": 3.0,
    "output_step_minutes": 3.0,
    "meals": [
        {"hour": 8,  "minute": 0,  "grams": 50.0},
        {"hour": 12, "minute": 30, "grams": 70.0},
        {"hour": 16, "minute": 0,  "grams": 15.0},
        {"hour": 19, "minute": 0,  "grams": 60.0},
    ],
    "controller": {"type": "MealBolusOnlyController", "basal_per_min": 0.006, "icr_g_per_u": 25.0},
}
