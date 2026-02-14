from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class Signal:
    action: str  # "BUY" | "SELL" | "HOLD"
    reason: str
    risk_profile: Optional[str] = None  # e.g. "trend" | "range"


def regime_signal(
    df: pd.DataFrame,
    *,
    rsi_period: int,
    atr_period: int,
    cfg: dict,
    in_position: bool,
    position_profile: str | None,
    range_entries_today: int = 0,
) -> Signal:
    """
    Long-only regime strategy:
      - Trend: ADX high + EMA rising + breakout (Donchian) -> BUY (profile="trend")
      - Range: ADX low + oversold (BB lower + RSI) + far from VWAP -> BUY (profile="range")
      - Trend exits (optional): break below EMA and/or Donchian low -> SELL
      - Range exits (optional): VWAP / BB mid / BB upper / RSI -> SELL

    Note: stops/trailing are handled in the risk layer / paper engine.
    """
    if df is None or len(df) < 2:
        return Signal("HOLD", "not enough candles")

    adx_period = int(cfg.get("adx_period", 14))
    ema_period = int(cfg.get("ema_period", 200))
    donchian_lookback = int(cfg.get("donchian_lookback", 96))
    bb_period = int(cfg.get("bb_period", 20))
    vwap_period = int(cfg.get("vwap_period", 96))

    trend_adx_min = float(cfg.get("trend_adx_min", 20))
    range_adx_max = float(cfg.get("range_adx_max", 18))
    trend_require_close_above_ema = bool(cfg.get("trend_require_close_above_ema", True))
    trend_require_ema_rising = bool(cfg.get("trend_require_ema_rising", True))

    enable_trend = bool(cfg.get("enable_trend", True))
    enable_range = bool(cfg.get("enable_range", True))

    # Trend exit options (default enabled to ensure the strategy can emit SELL signals in trend positions)
    trend_exit_on_close_below_ema = bool(cfg.get("trend_exit_on_close_below_ema", True))
    trend_exit_on_donchian_low_break = bool(cfg.get("trend_exit_on_donchian_low_break", False))

    range_rsi_buy_max = float(cfg.get("range_rsi_buy_max", 30))
    range_exit_on_vwap = bool(cfg.get("range_exit_on_vwap", True))
    range_exit_vwap_buffer_atr_mult = float(cfg.get("range_exit_vwap_buffer_atr_mult", 0.0) or 0.0)
    range_exit_on_bb_mid = bool(cfg.get("range_exit_on_bb_mid", False))
    range_exit_bb_mid_buffer_atr_mult = float(
        cfg.get("range_exit_bb_mid_buffer_atr_mult", 0.0) or 0.0
    )
    range_exit_on_bb_upper = bool(cfg.get("range_exit_on_bb_upper", False))
    range_rsi_sell_min = cfg.get("range_rsi_sell_min", None)
    range_rsi_sell_min = None if range_rsi_sell_min is None else float(range_rsi_sell_min)
    range_vwap_dist_atr_mult = float(cfg.get("range_vwap_dist_atr_mult", 1.2))
    range_max_entries_per_utc_day = int(cfg.get("range_max_entries_per_utc_day", 1))
    range_require_close_above_ema = bool(cfg.get("range_require_close_above_ema", False))
    range_require_ema_rising = bool(cfg.get("range_require_ema_rising", False))
    range_entry_use_wick_low = bool(cfg.get("range_entry_use_wick_low", False))
    range_exit_use_wick_high = bool(cfg.get("range_exit_use_wick_high", False))
    range_entry_require_reclaim_bb_lower = bool(
        cfg.get("range_entry_require_reclaim_bb_lower", False)
    )
    range_entry_require_rsi_rising = bool(cfg.get("range_entry_require_rsi_rising", False))

    ema_col = f"ema_{ema_period}"
    adx_col = f"adx_{adx_period}"
    don_hi_col = f"donchian_high_{donchian_lookback}"
    don_lo_col = f"donchian_low_{donchian_lookback}"
    bb_mid_col = f"bb_mid_{bb_period}"
    bb_upper_col = f"bb_upper_{bb_period}"
    bb_lower_col = f"bb_lower_{bb_period}"
    vwap_col = f"vwap_{vwap_period}"
    rsi_col = f"rsi_{rsi_period}"
    atr_col = f"atr_{atr_period}"

    needed = [
        ema_col,
        adx_col,
        don_hi_col,
        don_lo_col,
        bb_mid_col,
        bb_upper_col,
        bb_lower_col,
        vwap_col,
        rsi_col,
        atr_col,
    ]
    for col in needed:
        if col not in df.columns:
            return Signal("HOLD", f"missing {col}")

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    for col in (
        ema_col,
        adx_col,
        don_hi_col,
        don_lo_col,
        bb_mid_col,
        bb_upper_col,
        bb_lower_col,
        vwap_col,
        rsi_col,
        atr_col,
    ):
        if pd.isna(curr[col]) or pd.isna(prev[col]):
            return Signal("HOLD", "indicator not ready")

    close = float(curr["close"])
    high = float(curr["high"])
    low = float(curr["low"])
    adx = float(curr[adx_col])
    ema = float(curr[ema_col])
    ema_prev = float(prev[ema_col])

    if in_position:
        if position_profile == "range":
            vwap = float(curr[vwap_col])
            bb_mid = float(curr[bb_mid_col])
            bb_upper = float(curr[bb_upper_col])
            rsi = float(curr[rsi_col])
            atr = float(curr[atr_col])
            exit_ref = high if range_exit_use_wick_high else close
            exit_ref_name = "high" if range_exit_use_wick_high else "close"

            if range_exit_on_bb_upper and exit_ref >= bb_upper:
                return Signal(
                    "SELL",
                    f"range exit: {exit_ref_name} {exit_ref:.2f} >= BB_upper {bb_upper:.2f}",
                )
            if range_rsi_sell_min is not None and rsi >= range_rsi_sell_min:
                return Signal("SELL", f"range exit: RSI {rsi:.1f} >= {range_rsi_sell_min}")
            if range_exit_on_bb_mid:
                bb_mid_target = bb_mid
                if range_exit_bb_mid_buffer_atr_mult > 0 and atr > 0:
                    bb_mid_target = bb_mid + atr * range_exit_bb_mid_buffer_atr_mult
                if exit_ref >= bb_mid_target:
                    if bb_mid_target == bb_mid:
                        return Signal(
                            "SELL",
                            f"range exit: {exit_ref_name} {exit_ref:.2f} >= BB_mid {bb_mid:.2f}",
                        )
                    return Signal(
                        "SELL",
                        f"range exit: {exit_ref_name} {exit_ref:.2f} >= BB_mid {bb_mid:.2f} + {range_exit_bb_mid_buffer_atr_mult:.2f}*ATR",
                    )

            if range_exit_on_vwap:
                vwap_target = vwap
                if range_exit_vwap_buffer_atr_mult > 0 and atr > 0:
                    vwap_target = vwap + atr * range_exit_vwap_buffer_atr_mult
                if exit_ref >= vwap_target:
                    if vwap_target == vwap:
                        return Signal(
                            "SELL",
                            f"range exit: {exit_ref_name} {exit_ref:.2f} >= VWAP {vwap:.2f}",
                        )
                    return Signal(
                        "SELL",
                        f"range exit: {exit_ref_name} {exit_ref:.2f} >= VWAP {vwap:.2f} + {range_exit_vwap_buffer_atr_mult:.2f}*ATR",
                    )

            return Signal("HOLD", "range hold")
        if position_profile == "trend":
            if trend_exit_on_close_below_ema and close < ema:
                return Signal("SELL", f"trend exit: close {close:.2f} < EMA {ema:.2f}")
            if trend_exit_on_donchian_low_break:
                don_lo_prev = float(prev[don_lo_col])  # avoid same-bar lookahead
                if close < don_lo_prev:
                    return Signal(
                        "SELL",
                        f"trend exit: close {close:.2f} < donchian_low(prev) {don_lo_prev:.2f}",
                    )
            return Signal("HOLD", "trend hold")
        return Signal("HOLD", "in position")

    # ---- Trend regime ----
    ema_rising = ema > ema_prev
    trend_ok = adx >= trend_adx_min
    if trend_require_close_above_ema:
        trend_ok = trend_ok and close > ema
    if trend_require_ema_rising:
        trend_ok = trend_ok and ema_rising

    if enable_trend and trend_ok:
        don_hi_prev = float(prev[don_hi_col])  # use previous Donchian to avoid same-bar lookahead
        if close > don_hi_prev:
            return Signal(
                "BUY",
                f"trend breakout: close {close:.2f} > donchian_high(prev) {don_hi_prev:.2f} | ADX {adx:.1f}",
                risk_profile="trend",
            )
        return Signal("HOLD", "trend regime but no breakout")

    # ---- Range regime ----
    if enable_range and adx <= range_adx_max:
        if range_max_entries_per_utc_day > 0 and range_entries_today >= range_max_entries_per_utc_day:
            return Signal("HOLD", "range blocked: max entries reached for UTC day")

        if range_require_close_above_ema and close <= ema:
            return Signal("HOLD", "range blocked: close <= EMA (uptrend filter)")
        if range_require_ema_rising and not ema_rising:
            return Signal("HOLD", "range blocked: EMA not rising (uptrend filter)")

        bb_lower = float(curr[bb_lower_col])
        rsi = float(curr[rsi_col])
        prev_rsi = float(prev[rsi_col])
        atr = float(curr[atr_col])
        vwap = float(curr[vwap_col])
        entry_ref = low if range_entry_use_wick_low else close
        entry_ref_name = "low" if range_entry_use_wick_low else "close"
        # Reclaim mode means "wick touched below BB lower, then candle reclaimed above it".
        touch_ref = low if range_entry_require_reclaim_bb_lower else entry_ref

        vwap_far = abs(touch_ref - vwap) >= (atr * range_vwap_dist_atr_mult)
        touches_lower = touch_ref < bb_lower
        if touches_lower and range_entry_require_reclaim_bb_lower and close <= bb_lower:
            return Signal("HOLD", "range blocked: no reclaim above BB_lower")
        if touches_lower and range_entry_require_rsi_rising and rsi <= prev_rsi:
            return Signal("HOLD", "range blocked: RSI not rising")
        if touches_lower and rsi <= range_rsi_buy_max and vwap_far:
            return Signal(
                "BUY",
                f"range fade: {entry_ref_name} {entry_ref:.2f} < BB_lower {bb_lower:.2f} | RSI {rsi:.1f} <= {range_rsi_buy_max} | far from VWAP",
                risk_profile="range",
            )
        return Signal("HOLD", "range regime but no setup")

    return Signal("HOLD", "no-trade regime")
