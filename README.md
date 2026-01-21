# TS + NeuralODE + Probabilistic Safety Filter (SimGlucose)

This repository contains code for a closed-loop Type-1 diabetes simulation using:
- **Thompson Sampling** (discrete bolus bandit)
- **Neural ODE forecaster** (torchdiffeq)
- **Probabilistic safety filter** (conformal by default; Cantelli fallback)
- **SimGlucose** environment for evaluation

The main runnable entrypoint performs **one continuous simulation** over a multi-day horizon,
training until a specified cutoff time and then evaluating for a fixed number of days.

"forecasting_model_checkpoint.pt" is the after trained NeuralODE forecaster.

## Usage
```bash
# Training
python run_ts_continuous.py
# Simulation
python run_multi_patient_eval.py
# Train and test on seperate patient
python run_transfer_generalization.py
