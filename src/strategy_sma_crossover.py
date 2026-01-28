from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class Signal:
    action: str  # "BUY" | "SELL" | "HOLD"
    reason: str


def sma_crossover_signal(df: pd.DataFrame, fast: int, slow: int) -> Signal:
    """
    Uses the last two rows to detect a crossover:
    BUY  if fast SMA crosses above slow SMA
    SELL if fast SMA crosses below slow SMA
    HOLD otherwise
    """
    fast_col = f"sma_{fast}"
    slow_col = f"sma_{slow}"

    # Need at least 2 candles to detect a cross
    if df is None or len(df) < 2:
        return Signal("HOLD", "not enough candles")

    # Need SMA columns present
    if fast_col not in df.columns or slow_col not in df.columns:
        return Signal("HOLD", "missing SMA columns")

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    prev_fast, prev_slow = prev[fast_col], prev[slow_col]
    curr_fast, curr_slow = curr[fast_col], curr[slow_col]

    # If SMAs aren't ready yet (NaN), hold
    if (
        pd.isna(prev_fast)
        or pd.isna(prev_slow)
        or pd.isna(curr_fast)
        or pd.isna(curr_slow)
    ):
        return Signal("HOLD", "SMA not ready yet (NaN)")

    # Crossover logic
    if prev_fast <= prev_slow and curr_fast > curr_slow:
        return Signal("BUY", f"fast SMA({fast}) crossed ABOVE slow SMA({slow})")
    if prev_fast >= prev_slow and curr_fast < curr_slow:
        return Signal("SELL", f"fast SMA({fast}) crossed BELOW slow SMA({slow})")

    return Signal("HOLD", "no crossover")
