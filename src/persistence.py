from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from src.paper_engine import PaperState


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_data_dir(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)


def state_to_dict(state: PaperState) -> Dict[str, Any]:
    return {
        "cash": state.cash,
        "asset_qty": state.asset_qty,
        "in_position": state.in_position,
        "entry_price": state.entry_price,
        "stop_price": state.stop_price,
        "take_profit_price": state.take_profit_price,
        "realized_pnl": state.realized_pnl,
        "bars_in_position": int(getattr(state, "bars_in_position", 0) or 0),
        "highest_close_since_entry": getattr(state, "highest_close_since_entry", None),
        "position_profile": getattr(state, "position_profile", None),
        "equity_peak": getattr(state, "equity_peak", None),
        "day_start_utc_date": getattr(state, "day_start_utc_date", None),
        "day_start_equity": getattr(state, "day_start_equity", None),
        "last_range_entry_utc_date": getattr(state, "last_range_entry_utc_date", None),
        "last_exit_reason": getattr(state, "last_exit_reason", None),
        "bars_since_exit": int(getattr(state, "bars_since_exit", 0) or 0),
        # We persist trades separately to CSV; keep a count here for sanity (optional)
        "trades_count": len(state.trades) if state.trades is not None else 0,
        "saved_at": _utc_now_iso(),
    }


def dict_to_state(d: Dict[str, Any]) -> PaperState:
    s = PaperState(cash=float(d.get("cash", 0.0)))
    s.asset_qty = float(d.get("asset_qty", 0.0))
    s.in_position = bool(d.get("in_position", False))
    s.entry_price = d.get("entry_price", None)
    s.stop_price = d.get("stop_price", None)
    s.take_profit_price = d.get("take_profit_price", None)
    s.realized_pnl = float(d.get("realized_pnl", 0.0))
    s.bars_in_position = int(d.get("bars_in_position", 0) or 0)
    s.highest_close_since_entry = d.get("highest_close_since_entry", None)
    s.position_profile = d.get("position_profile", None)
    s.equity_peak = d.get("equity_peak", None)
    s.day_start_utc_date = d.get("day_start_utc_date", None)
    s.day_start_equity = d.get("day_start_equity", None)
    s.last_range_entry_utc_date = d.get("last_range_entry_utc_date", None)
    s.last_exit_reason = d.get("last_exit_reason", None)
    s.bars_since_exit = int(d.get("bars_since_exit", 0) or 0)
    # trades list lives in memory; we’ll start empty on load (CSV is the journal)
    s.trades = []
    return s


def load_state(state_path: Path) -> Optional[PaperState]:
    if not state_path.exists():
        return None
    data = json.loads(state_path.read_text(encoding="utf-8"))
    return dict_to_state(data)


def save_state(state: PaperState, state_path: Path) -> None:
    state_path.write_text(json.dumps(state_to_dict(state), indent=2), encoding="utf-8")


def ensure_trades_csv(trades_csv: Path) -> None:
    if trades_csv.exists():
        return
    with trades_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "time",
                "symbol",
                "side",
                "amount",
                "price",
                "fee",
                "reason",
                "realized_pnl",
            ]
        )


def append_trade(trades_csv: Path, trade: Dict[str, Any]) -> None:
    """
    Trade dicts come from paper_engine (BUY/SELL). SELL may include realized_pnl.
    """
    ensure_trades_csv(trades_csv)
    with trades_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                trade.get("time"),
                trade.get("symbol"),
                trade.get("side"),
                trade.get("amount"),
                trade.get("price"),
                trade.get("fee"),
                trade.get("reason"),
                trade.get("realized_pnl", ""),
            ]
        )
