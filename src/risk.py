from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OrderIntent:
    action: str  # "OPEN_LONG" | "CLOSE_LONG" | "NONE"
    reason: str
    symbol: str
    amount: float = 0.0
    price: float = 0.0
    stop_price: Optional[float] = None
    take_profit_price: Optional[float] = None


def equity(cash: float, asset_qty: float, price: float) -> float:
    return cash + asset_qty * price


def calc_buy_amount_percent_equity(
    cash: float,
    asset_qty: float,
    price: float,
    trade_pct_equity: float,
    fee_rate: float,
) -> float:
    """
    Computes how much asset to buy using trade_pct_equity of total equity.
    Caps by available cash. Includes fee estimate.
    """
    eq = equity(cash, asset_qty, price)
    target_notional = eq * trade_pct_equity

    # Can't spend more than cash (fees included)
    max_spend = cash
    spend = min(target_notional, max_spend)

    # If spend is tiny, return 0
    if spend <= 0:
        return 0.0

    # Approx fee charged on notional; buy amount uses remaining after fee
    # (simple approximation)
    spend_after_fee = spend * (1.0 - fee_rate)
    amount = spend_after_fee / price
    return max(amount, 0.0)


def compute_stop_take(
    entry_price: float, stop_loss_pct: float | None, take_profit_pct: float | None
):
    stop_price = None
    take_profit_price = None

    if stop_loss_pct is not None and stop_loss_pct > 0:
        stop_price = entry_price * (1.0 - stop_loss_pct)

    if take_profit_pct is not None and take_profit_pct > 0:
        take_profit_price = entry_price * (1.0 + take_profit_pct)

    return stop_price, take_profit_price


def risk_decision(
    symbol: str,
    signal_action: str,  # "BUY" | "SELL" | "HOLD"
    price: float,
    cash: float,
    asset_qty: float,
    in_position: bool,
    cfg_risk: dict,
    cfg_paper: dict,
) -> OrderIntent:
    """
    Turns a strategy signal into a risk-checked order intent.
    """
    trade_pct = float(cfg_risk.get("trade_pct_equity", 0.0))
    one_pos = bool(cfg_risk.get("one_position_only", True))
    sl_pct = cfg_risk.get("stop_loss_pct", None)
    tp_pct = cfg_risk.get("take_profit_pct", None)

    fee_rate = float(cfg_paper.get("fee_rate", 0.0))

    # Normalize optional values
    sl_pct = None if sl_pct is None else float(sl_pct)
    tp_pct = None if tp_pct is None else float(tp_pct)

    if signal_action == "BUY":
        if one_pos and in_position:
            return OrderIntent("NONE", "blocked: already in position", symbol)

        amount = calc_buy_amount_percent_equity(
            cash=cash,
            asset_qty=asset_qty,
            price=price,
            trade_pct_equity=trade_pct,
            fee_rate=fee_rate,
        )

        if amount <= 0:
            return OrderIntent("NONE", "blocked: insufficient cash", symbol)

        stop_price, take_profit_price = compute_stop_take(price, sl_pct, tp_pct)

        return OrderIntent(
            action="OPEN_LONG",
            reason=f"percent equity sizing: {trade_pct*100:.1f}%",
            symbol=symbol,
            amount=amount,
            price=price,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
        )

    if signal_action == "SELL":
        if not in_position or asset_qty <= 0:
            return OrderIntent("NONE", "no position to close", symbol)

        return OrderIntent(
            action="CLOSE_LONG",
            reason="strategy SELL",
            symbol=symbol,
            amount=asset_qty,
            price=price,
        )

    return OrderIntent("NONE", "strategy HOLD", symbol)
