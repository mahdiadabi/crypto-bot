from __future__ import annotations

import pandas as pd


def ohlcv_to_df(ohlcv: list[list], tz: str = "UTC") -> pd.DataFrame:
    """
    ccxt OHLCV format: [timestamp_ms, open, high, low, close, volume]
    Returns a DataFrame indexed by time with numeric columns.
    """
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    if tz and tz.upper() != "UTC":
        df["time"] = df["time"].dt.tz_convert(tz)
    df = df.drop(columns=["ts"]).set_index("time")
    df = df.astype(float)
    return df


def add_sma(df: pd.DataFrame, period: int, price_col: str = "close") -> pd.DataFrame:
    """
    Adds a simple moving average column: sma_{period}
    """
    out = df.copy()
    out[f"sma_{period}"] = (
        out[price_col].rolling(window=period, min_periods=period).mean()
    )
    return out


def add_ema(df: pd.DataFrame, period: int, price_col: str = "close") -> pd.DataFrame:
    """
    Adds an exponential moving average column: ema_{period}
    """
    out = df.copy()
    out[f"ema_{period}"] = out[price_col].ewm(span=period, adjust=False).mean()
    return out


def latest_row(df: pd.DataFrame) -> dict:
    """
    Convenience helper: returns last row as a plain dict (good for logging).
    """
    if df.empty:
        return {}
    row = df.iloc[-1].to_dict()
    row["time"] = df.index[-1].isoformat()
    return row
