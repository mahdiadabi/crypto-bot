from __future__ import annotations

from typing import Optional

import pandas as pd

from src.strategy_regime import Signal as RegimeSignal
from src.strategy_regime import regime_signal
from src.strategy_sma_rsi import sma_crossover_with_rsi


def compute_signal(
    df: pd.DataFrame,
    *,
    strategy_name: str,
    cfg_strategy: dict,
    cfg_regime: dict,
    sma_fast: int,
    sma_slow: int,
    rsi_period: int,
    rsi_buy_min: float,
    rsi_sell_max: float,
    atr_period: int,
    in_position: bool,
    position_profile: Optional[str],
    bars_in_position: int,
    utc_day: str,
    last_range_entry_utc_date: Optional[str],
):
    if strategy_name == "regime":
        time_stop_bars = int(cfg_regime.get("trend_time_stop_bars", 0) or 0)
        if (
            in_position
            and position_profile == "trend"
            and time_stop_bars > 0
            and bars_in_position >= time_stop_bars
        ):
            return RegimeSignal("SELL", f"time stop: bars_in_position >= {time_stop_bars}")

        entries_today = 1 if last_range_entry_utc_date == utc_day else 0
        return regime_signal(
            df,
            rsi_period=rsi_period,
            atr_period=atr_period,
            cfg=cfg_regime,
            in_position=in_position,
            position_profile=position_profile,
            range_entries_today=entries_today,
        )

    if strategy_name == "sma_rsi":
        return sma_crossover_with_rsi(
            df,
            sma_fast,
            sma_slow,
            rsi_period,
            rsi_buy_min,
            rsi_sell_max,
            require_price_above_slow=bool(
                cfg_strategy.get("require_price_above_slow", True)
            ),
            require_slow_rising=bool(cfg_strategy.get("require_slow_rising", True)),
            sell_requires_rsi=bool(cfg_strategy.get("sell_requires_rsi", False)),
        )

    raise ValueError(f"Unknown strategy '{strategy_name}'. Use 'regime' or 'sma_rsi'.")
