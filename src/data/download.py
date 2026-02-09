from __future__ import annotations
import os
from typing import Optional
import pandas as pd
import yfinance as yf
from src.utils.io import ensure_dir

def _cache_path(raw_dir: str, ticker: str, start: str, end: str, interval: str) -> str:
    safe = f"{ticker}_{start}_{end}_{interval}".replace(":", "-")
    return os.path.join(raw_dir, f"{safe}.csv")

def download_yf(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
    raw_dir: str = "data/raw",
    force: bool = False,
) -> pd.DataFrame:
    """Download OHLCV from Yahoo Finance and cache to CSV."""
    ensure_dir(raw_dir)
    path = _cache_path(raw_dir, ticker, start, end, interval)
    if os.path.exists(path) and not force:
        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.set_index("Date")
        return df

    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"Failed to download data for {ticker}. Check your internet or ticker symbol.")

    df = df.reset_index()
    df.to_csv(path, index=False)
    df = df.set_index("Date")
    return df
