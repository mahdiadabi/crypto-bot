from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Dict


@dataclass
class PaperState:
    cash: float
    asset_qty: float = 0.0
    in_position: bool = False
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    realized_pnl: float = 0.0
    bars_in_position: int = 0
    highest_close_since_entry: Optional[float] = None
    position_profile: Optional[str] = None
    equity_peak: Optional[float] = None
    day_start_utc_date: Optional[str] = None  # YYYY-MM-DD
    day_start_equity: Optional[float] = None
    last_range_entry_utc_date: Optional[str] = None  # YYYY-MM-DD
    last_exit_reason: Optional[str] = None
    bars_since_exit: int = 0
    trades: List[Dict] = None

    def __post_init__(self):
        if self.trades is None:
            self.trades = []


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def equity(state: PaperState, price: float) -> float:
    return state.cash + state.asset_qty * price


def unrealized_pnl(state: PaperState, price: float) -> float:
    if not state.in_position or state.entry_price is None:
        return 0.0
    return (price - state.entry_price) * state.asset_qty


def _apply_fee(notional: float, fee_rate: float) -> float:
    return notional * fee_rate


def update_trailing_stop_atr(
    state: PaperState, price: float, atr: Optional[float], trail_mult: Optional[float]
) -> Optional[float]:
    """
    Long-only trailing stop: move stop up as price increases, based on ATR.
    Returns the updated stop price, or None if no update occurred.
    """
    if not state.in_position or state.asset_qty <= 0:
        return None
    if atr is None or trail_mult is None:
        return None
    if atr <= 0 or trail_mult <= 0:
        return None

    new_stop = price - atr * trail_mult
    if state.stop_price is None or new_stop > state.stop_price:
        state.stop_price = new_stop
        return state.stop_price
    return None


def update_trailing_stop_atr_highest(
    state: PaperState, close_price: float, atr: Optional[float], trail_mult: Optional[float]
) -> Optional[float]:
    """
    Long-only trailing stop anchored to the highest close since entry.
    stop = highest_close_since_entry - atr*trail_mult
    """
    if not state.in_position or state.asset_qty <= 0:
        return None
    if atr is None or trail_mult is None:
        return None
    if atr <= 0 or trail_mult <= 0:
        return None

    if state.highest_close_since_entry is None:
        state.highest_close_since_entry = close_price
    else:
        state.highest_close_since_entry = max(state.highest_close_since_entry, close_price)

    new_stop = state.highest_close_since_entry - atr * trail_mult
    if state.stop_price is None or new_stop > state.stop_price:
        state.stop_price = new_stop
        return state.stop_price
    return None


def open_long(
    state: PaperState,
    symbol: str,
    amount: float,
    price: float,
    fee_rate: float,
    stop_price: Optional[float],
    take_profit_price: Optional[float],
    reason: str,
    position_profile: Optional[str] = None,
) -> tuple[str, Optional[Dict]]:
    if amount <= 0:
        return "OPEN_LONG skipped: amount <= 0", None
    if state.in_position:
        return "OPEN_LONG blocked: already in position", None

    notional = amount * price
    fee = _apply_fee(notional, fee_rate)
    total_cost = notional + fee

    if total_cost > state.cash + 1e-9:
        return (
            f"OPEN_LONG blocked: insufficient cash (need {total_cost:.2f}, have {state.cash:.2f})",
            None,
        )

    state.cash -= total_cost
    state.asset_qty += amount
    state.in_position = True
    state.entry_price = price
    state.stop_price = stop_price
    state.take_profit_price = take_profit_price
    state.bars_in_position = 0
    state.highest_close_since_entry = price
    state.position_profile = position_profile
    state.bars_since_exit = 0

    # state.trades.append(
    #     {
    #         "time": _utc_now_iso(),
    #         "symbol": symbol,
    #         "side": "BUY",
    #         "amount": amount,
    #         "price": price,
    #         "fee": fee,
    #         "reason": reason,
    #     }
    # )

    # return f"OPEN_LONG executed: buy {amount:.6f} @ {price:.2f} fee={fee:.2f}"
    trade = {
        "time": _utc_now_iso(),
        "symbol": symbol,
        "side": "BUY",
        "amount": amount,
        "price": price,
        "fee": fee,
        "reason": reason,
        "position_profile": position_profile,
    }
    state.trades.append(trade)

    return f"OPEN_LONG executed: buy {amount:.6f} @ {price:.2f} fee={fee:.2f}", trade


def close_long(
    state: PaperState,
    symbol: str,
    amount: float,
    price: float,
    fee_rate: float,
    reason: str,
) -> tuple[str, Optional[Dict]]:
    if not state.in_position or state.asset_qty <= 0:
        return "CLOSE_LONG skipped: no position", None

    position_profile = state.position_profile
    amount = min(amount, state.asset_qty)
    if amount <= 0:
        return "CLOSE_LONG skipped: amount <= 0", None

    notional = amount * price
    fee = _apply_fee(notional, fee_rate)
    proceeds = notional - fee

    # Realized PnL based on entry
    entry = state.entry_price if state.entry_price is not None else price
    realized = (price - entry) * amount - fee

    state.cash += proceeds
    state.asset_qty -= amount

    # If fully closed, reset position state
    if state.asset_qty <= 1e-12:
        state.asset_qty = 0.0
        state.in_position = False
        state.entry_price = None
        state.stop_price = None
        state.take_profit_price = None
        state.bars_in_position = 0
        state.highest_close_since_entry = None
        state.position_profile = None
        state.last_exit_reason = reason
        state.bars_since_exit = 0

    state.realized_pnl += realized

    trade = {
        "time": _utc_now_iso(),
        "symbol": symbol,
        "side": "SELL",
        "amount": amount,
        "price": price,
        "fee": fee,
        "reason": reason,
        "realized_pnl": realized,
        "position_profile": position_profile,
    }
    state.trades.append(trade)

    return (
        f"CLOSE_LONG executed: sell {amount:.6f} @ {price:.2f} fee={fee:.2f} realized={realized:.2f}",
        trade,
    )


def check_exits(
    state: PaperState, symbol: str, price: float, fee_rate: float
) -> Optional[tuple[str, dict]]:
    """
    If stop loss or take profit is hit, close the entire position.
    """
    if not state.in_position or state.asset_qty <= 0:
        return None

    if state.stop_price is not None and price <= state.stop_price:
        msg, trade = close_long(
            state, symbol, state.asset_qty, price, fee_rate, reason="STOP_LOSS hit"
        )
        return msg, trade

    if state.take_profit_price is not None and price >= state.take_profit_price:
        msg, trade = close_long(
            state, symbol, state.asset_qty, price, fee_rate, reason="TAKE_PROFIT hit"
        )
        return msg, trade

    return None
