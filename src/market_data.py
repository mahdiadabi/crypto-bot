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
    # Some exchanges return strings/None; coerce defensively to numeric.
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_sma(df: pd.DataFrame, period: int, price_col: str = "close") -> pd.DataFrame:
    """
    Adds a simple moving average column: sma_{period}
    """
    out = df.copy()
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    out[f"sma_{period}"] = (
        out[price_col].rolling(window=period, min_periods=period).mean()
    )
    return out


def add_ema(df: pd.DataFrame, period: int, price_col: str = "close") -> pd.DataFrame:
    """
    Adds an exponential moving average column: ema_{period}
    """
    out = df.copy()
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    out[f"ema_{period}"] = out[price_col].ewm(span=period, adjust=False).mean()
    return out


def add_rsi(
    df: pd.DataFrame, period: int = 14, price_col: str = "close"
) -> pd.DataFrame:
    out = df.copy()
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    delta = out[price_col].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder's smoothing approximation via ewm(alpha=1/period)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, float("nan"))
    out[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    return out


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    out = df.copy()
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    out[f"atr_{period}"] = tr.ewm(alpha=1 / period, adjust=False).mean()
    return out


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Adds ADX and directional indicators using Wilder-style smoothing (ewm alpha=1/period).
    Columns:
      - plus_di_{period}
      - minus_di_{period}
      - adx_{period}
    """
    out = df.copy()
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    tr_sm = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_dm_sm = plus_dm.ewm(alpha=1 / period, adjust=False).mean()
    minus_dm_sm = minus_dm.ewm(alpha=1 / period, adjust=False).mean()

    tr_safe = tr_sm.replace(0.0, float("nan"))
    plus_di = 100.0 * (plus_dm_sm / tr_safe)
    minus_di = 100.0 * (minus_dm_sm / tr_safe)
    dx = (
        100.0
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0.0, float("nan"))
    )
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    out[f"plus_di_{period}"] = plus_di
    out[f"minus_di_{period}"] = minus_di
    out[f"adx_{period}"] = adx
    return out


def add_bbands(
    df: pd.DataFrame,
    period: int = 20,
    std_mult: float = 2.0,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Adds Bollinger Bands:
      - bb_mid_{period}
      - bb_upper_{period}
      - bb_lower_{period}
    """
    out = df.copy()
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    mid = out[price_col].rolling(window=period, min_periods=period).mean()
    std = out[price_col].rolling(window=period, min_periods=period).std(ddof=0)
    out[f"bb_mid_{period}"] = mid
    out[f"bb_upper_{period}"] = mid + std_mult * std
    out[f"bb_lower_{period}"] = mid - std_mult * std
    return out


def add_donchian(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Adds Donchian channel bounds:
      - donchian_high_{lookback} (rolling max of high)
      - donchian_low_{lookback}  (rolling min of low)
    """
    out = df.copy()
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    out[f"donchian_high_{lookback}"] = high.rolling(
        window=lookback, min_periods=lookback
    ).max()
    out[f"donchian_low_{lookback}"] = low.rolling(
        window=lookback, min_periods=lookback
    ).min()
    return out


def add_vwap(df: pd.DataFrame, period: int | None = None) -> pd.DataFrame:
    """
    Adds VWAP using typical price (H+L+C)/3.
    If period is None, uses cumulative VWAP over the whole df; otherwise uses rolling VWAP.
    Column: vwap_{period} or vwap_cum
    """
    out = df.copy()
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")
    volume = pd.to_numeric(out["volume"], errors="coerce")
    tp = (high + low + close) / 3.0
    pv = tp * volume

    if period is None:
        denom = volume.cumsum()
        out["vwap_cum"] = pv.cumsum() / denom.replace(0.0, float("nan"))
        return out

    pv_sum = pv.rolling(window=period, min_periods=period).sum()
    vol_sum = volume.rolling(window=period, min_periods=period).sum()
    out[f"vwap_{period}"] = pv_sum / vol_sum.replace(0.0, float("nan"))
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
