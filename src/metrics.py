import numpy as np
import pandas as pd

def bg_metrics(df: pd.DataFrame, target=(120,140)):
    bg = pd.to_numeric(df['bg'], errors='coerce').dropna().to_numpy()
    tir = np.mean((bg>=target[0]) & (bg<=target[1]))*100.0
    below70 = np.mean(bg<70.0)*100.0
    meanbg = float(np.mean(bg)) if len(bg) else np.nan
    return {'TIR_%': round(tir,2), 'Time<70_%': round(below70,2), 'MeanBG': round(meanbg,1)}

def daily_bg_metrics(df: pd.DataFrame, target=(120,140)):
    """Per-day metrics + simple overall averages."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['dt']).dt.date
    rows = []
    for date, g in df.groupby('date', sort=True):
        m = bg_metrics(g, target=target)
        m['date'] = date
        rows.append(m)
    daily = pd.DataFrame(rows)[['date','TIR_%','Time<70_%','MeanBG']]
    overall = {
        'days': int(len(daily)),
        'TIR_%_mean': round(float(daily['TIR_%'].mean()), 2),
        'Time<70_%_mean': round(float(daily['Time<70_%'].mean()), 2),
        'MeanBG_mean': round(float(daily['MeanBG'].mean()), 1),
    }
    return daily, overall