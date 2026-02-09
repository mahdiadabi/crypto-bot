from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class Signal:
    action: str
    reason: str


def sma_crossover_with_rsi(
    df: pd.DataFrame,
    fast: int,
    slow: int,
    rsi_period: int,
    rsi_buy_min: float,
    rsi_sell_max: float,
    require_price_above_slow: bool = False,
    require_slow_rising: bool = False,
    sell_requires_rsi: bool = False,
) -> Signal:
    fast_col = f"sma_{fast}"
    slow_col = f"sma_{slow}"
    rsi_col = f"rsi_{rsi_period}"

    if df is None or len(df) < 2:
        return Signal("HOLD", "not enough candles")

    for col in (fast_col, slow_col, rsi_col):
        if col not in df.columns:
            return Signal("HOLD", f"missing {col}")

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    if (
        pd.isna(prev[fast_col])
        or pd.isna(prev[slow_col])
        or pd.isna(curr[fast_col])
        or pd.isna(curr[slow_col])
        or pd.isna(curr[rsi_col])
    ):
        return Signal("HOLD", "indicator not ready")

    crossed_up = prev[fast_col] <= prev[slow_col] and curr[fast_col] > curr[slow_col]
    crossed_dn = prev[fast_col] >= prev[slow_col] and curr[fast_col] < curr[slow_col]

    rsi = float(curr[rsi_col])

    if crossed_up:
        if require_price_above_slow and float(curr["close"]) <= float(curr[slow_col]):
            return Signal("HOLD", "crossover up but price <= slow SMA (trend filter)")
        if require_slow_rising and float(curr[slow_col]) <= float(prev[slow_col]):
            return Signal("HOLD", "crossover up but slow SMA not rising (trend filter)")
        if rsi >= rsi_buy_min:
            return Signal("BUY", f"crossover up + RSI {rsi:.1f} >= {rsi_buy_min}")
        return Signal("HOLD", f"crossover up but RSI {rsi:.1f} < {rsi_buy_min}")

    if crossed_dn:
        if not sell_requires_rsi:
            return Signal("SELL", "crossover down")
        if rsi <= rsi_sell_max:
            return Signal("SELL", f"crossover down + RSI {rsi:.1f} <= {rsi_sell_max}")
        return Signal("HOLD", f"crossover down but RSI {rsi:.1f} > {rsi_sell_max}")

    return Signal("HOLD", "no crossover")
