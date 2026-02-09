from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))

def make_features(
    df: pd.DataFrame,
    horizons: List[int],
    drop_na: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Create feature matrix X and target matrix Y (multi-horizon log-returns).
    Returns:
      X: [T, D]
      Y: [T, H]
      dates: [T] datetime64
      feature_names: list[str]
    """
    d = df.copy()

    # Normalize columns (yfinance can return MultiIndex columns like ('Close','SPY'))
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
        
    # Drop duplicated column names (so d["Close"] is a Series, not a DataFrame)
    d = d.loc[:, ~d.columns.duplicated()]


    # Ensure required columns exist (Yahoo Finance naming)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in d.columns:
            raise ValueError(f"Missing column '{col}' in input dataframe.")
    
    # Force numeric (handle cached CSV / formatting issues)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        v = d[col]
        if isinstance(v, pd.DataFrame):          # если вдруг несколько колонок с одним именем
            v = v.iloc[:, 0]                    # берем первую
        v = v.astype(str).str.replace(",", "", regex=False)
        d[col] = pd.to_numeric(v, errors="coerce")



    # Log return
    d["log_ret_1"] = np.log(d["Close"] / d["Close"].shift(1))

    # Rolling stats
    for w in [5, 10, 20, 60]:
        d[f"ret_std_{w}"] = d["log_ret_1"].rolling(w).std()
        d[f"ret_mean_{w}"] = d["log_ret_1"].rolling(w).mean()
        d[f"vol_z_{w}"] = (d["Volume"] - d["Volume"].rolling(w).mean()) / (d["Volume"].rolling(w).std() + 1e-12)
        d[f"ma_{w}"] = d["Close"].rolling(w).mean()
        d[f"ema_{w}"] = d["Close"].ewm(span=w, adjust=False).mean()

    d["rsi_14"] = _rsi(d["Close"], 14)

    # Normalize price level-ish features by close to reduce scale issues
    d["hl_spread"] = (d["High"] - d["Low"]) / (d["Close"] + 1e-12)
    d["oc_change"] = (d["Close"] - d["Open"]) / (d["Open"] + 1e-12)

    # Targets: future log returns over horizons
    for h in horizons:
        d[f"y_ret_{h}"] = np.log(d["Close"].shift(-h) / d["Close"])

    feature_cols = [
        "log_ret_1",
        "hl_spread",
        "oc_change",
        "rsi_14",
    ]
    # add roll features
    for w in [5, 10, 20, 60]:
        feature_cols += [f"ret_std_{w}", f"ret_mean_{w}", f"vol_z_{w}"]
        feature_cols += [f"ma_{w}", f"ema_{w}"]

    target_cols = [f"y_ret_{h}" for h in horizons]

    out = d[feature_cols + target_cols].copy()
    if drop_na:
        out = out.dropna()

    X = out[feature_cols].to_numpy(dtype=np.float32)
    Y = out[target_cols].to_numpy(dtype=np.float32)
    dates = out.index.to_numpy()

    return X, Y, dates, feature_cols
