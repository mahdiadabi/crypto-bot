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


def calc_buy_amount_risk_atr(
    cash: float,
    asset_qty: float,
    price: float,
    risk_pct_equity: float,
    atr: float,
    stop_atr_mult: float,
    fee_rate: float,
) -> float:
    """
    Risk-based sizing using ATR stop distance.
    target_risk = equity * risk_pct_equity
    stop_distance = atr * stop_atr_mult
    amount = target_risk / stop_distance
    Capped by available cash (incl. fees).
    """
    if risk_pct_equity <= 0:
        return 0.0
    if atr <= 0 or stop_atr_mult <= 0:
        return 0.0
    if price <= 0:
        return 0.0

    eq = equity(cash, asset_qty, price)
    target_risk = eq * risk_pct_equity
    stop_distance = atr * stop_atr_mult
    if stop_distance <= 0:
        return 0.0

    amount = target_risk / stop_distance
    if amount <= 0:
        return 0.0

    # Cap by cash (fees included): total_cost = amount*price*(1+fee_rate)
    cash_cap = cash / (price * (1.0 + fee_rate)) if cash > 0 else 0.0
    return max(min(amount, cash_cap), 0.0)


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


def compute_stop_take_atr(
    entry_price: float,
    atr: float,
    stop_atr_mult: float | None,
    take_profit_mult: float | None,
):
    stop_price = None
    take_profit_price = None

    if atr <= 0:
        return stop_price, take_profit_price

    if stop_atr_mult is not None and stop_atr_mult > 0:
        stop_price = entry_price - atr * stop_atr_mult

    if take_profit_mult is not None and take_profit_mult > 0:
        take_profit_price = entry_price + atr * take_profit_mult

    return stop_price, take_profit_price


def resolve_risk_profile(cfg_risk: dict, risk_profile: str | None) -> dict:
    """
    Backward-compatible way to support per-strategy/per-regime risk overrides.

    If cfg_risk contains:
      {
        ...base keys...,
        "profiles": {
          "trend": {"stop_atr_mult": 1.8, ...},
          "range": {"stop_atr_mult": 1.2, ...}
        }
      }
    then resolve_risk_profile(cfg_risk, "trend") returns base keys with the
    profile keys overlaid.
    """
    if not isinstance(cfg_risk, dict):
        return {}

    base = {k: v for k, v in cfg_risk.items() if k != "profiles"}
    profiles = cfg_risk.get("profiles", {}) if isinstance(cfg_risk, dict) else {}
    if risk_profile and isinstance(profiles, dict):
        override = profiles.get(risk_profile)
        if isinstance(override, dict):
            merged = dict(base)
            merged.update(override)
            return merged
    return base


def risk_decision(
    symbol: str,
    signal_action: str,  # "BUY" | "SELL" | "HOLD"
    price: float,
    cash: float,
    asset_qty: float,
    in_position: bool,
    cfg_risk: dict,
    cfg_paper: dict,
    atr: float | None = None,
    risk_profile: str | None = None,
) -> OrderIntent:
    """
    Turns a strategy signal into a risk-checked order intent.
    """
    r = resolve_risk_profile(cfg_risk, risk_profile)
    trade_pct = float(r.get("trade_pct_equity", 0.0))
    risk_pct = float(r.get("risk_pct_equity", 0.0) or 0.0)
    one_pos = bool(r.get("one_position_only", True))
    sl_pct = r.get("stop_loss_pct", None)
    tp_pct = r.get("take_profit_pct", None)
    stop_atr_mult = r.get("stop_atr_mult", None)
    tp_atr_mult = r.get("tp_atr_mult", None)
    min_atr_pct = r.get("min_atr_pct", None)
    max_atr_pct = r.get("max_atr_pct", None)

    fee_rate = float(cfg_paper.get("fee_rate", 0.0))

    # Normalize optional values
    sl_pct = None if sl_pct is None else float(sl_pct)
    tp_pct = None if tp_pct is None else float(tp_pct)
    stop_atr_mult = None if stop_atr_mult is None else float(stop_atr_mult)
    tp_atr_mult = None if tp_atr_mult is None else float(tp_atr_mult)
    if atr is not None and (atr <= 0 or atr != atr):
        atr = None
    min_atr_pct = None if min_atr_pct is None else float(min_atr_pct)
    max_atr_pct = None if max_atr_pct is None else float(max_atr_pct)

    if signal_action == "BUY":
        if one_pos and in_position:
            return OrderIntent("NONE", "blocked: already in position", symbol)

        if atr is not None and price > 0 and (min_atr_pct is not None or max_atr_pct is not None):
            atr_pct = atr / price
            if min_atr_pct is not None and atr_pct < min_atr_pct:
                return OrderIntent(
                    "NONE", f"blocked: atr_pct {atr_pct:.4f} < min_atr_pct {min_atr_pct}", symbol
                )
            if max_atr_pct is not None and atr_pct > max_atr_pct:
                return OrderIntent(
                    "NONE", f"blocked: atr_pct {atr_pct:.4f} > max_atr_pct {max_atr_pct}", symbol
                )

        amount = 0.0
        sizing_reason = None
        if (
            risk_pct > 0
            and atr is not None
            and stop_atr_mult is not None
            and stop_atr_mult > 0
        ):
            amount = calc_buy_amount_risk_atr(
                cash=cash,
                asset_qty=asset_qty,
                price=price,
                risk_pct_equity=risk_pct,
                atr=atr,
                stop_atr_mult=stop_atr_mult,
                fee_rate=fee_rate,
            )
            if amount > 0:
                sizing_reason = f"risk ATR sizing: {risk_pct*100:.2f}% eq"
        if amount <= 0:
            amount = calc_buy_amount_percent_equity(
                cash=cash,
                asset_qty=asset_qty,
                price=price,
                trade_pct_equity=trade_pct,
                fee_rate=fee_rate,
            )
            if amount > 0:
                sizing_reason = f"percent equity sizing: {trade_pct*100:.1f}%"

        if amount <= 0:
            return OrderIntent("NONE", "blocked: insufficient cash", symbol)

        if atr is not None and (stop_atr_mult or tp_atr_mult):
            stop_price, take_profit_price = compute_stop_take_atr(
                price, atr, stop_atr_mult, tp_atr_mult
            )
        else:
            stop_price, take_profit_price = compute_stop_take(price, sl_pct, tp_pct)

        return OrderIntent(
            action="OPEN_LONG",
            reason=(sizing_reason or "sizing") + (f" | profile={risk_profile}" if risk_profile else ""),
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
