#!/usr/bin/env python3
import sys
from pathlib import Path
from math import isclose

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.risk import risk_decision


def main() -> int:
    cfg_paper = {"fee_rate": 0.001}
    cfg_risk = {
        "trade_pct_equity": 0.10,
        "one_position_only": True,
        "atr_period": 14,
        "stop_atr_mult": 2.0,
        "tp_atr_mult": 3.0,
    }

    price = 100.0
    atr = 5.0
    cash = 1000.0
    asset_qty = 0.0

    intent = risk_decision(
        symbol="BTC/USDT",
        signal_action="BUY",
        price=price,
        cash=cash,
        asset_qty=asset_qty,
        in_position=False,
        cfg_risk=cfg_risk,
        cfg_paper=cfg_paper,
        atr=atr,
    )

    assert intent.action == "OPEN_LONG", f"unexpected action: {intent.action}"
    assert isclose(intent.stop_price, 90.0, rel_tol=1e-9), intent.stop_price
    assert isclose(intent.take_profit_price, 115.0, rel_tol=1e-9), intent.take_profit_price

    expected_amount = (cash * 0.10 * (1.0 - cfg_paper["fee_rate"])) / price
    assert isclose(intent.amount, expected_amount, rel_tol=1e-9), intent.amount

    print("OK: ATR stops and sizing sanity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
