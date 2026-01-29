import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd

from exchange import make_exchange
from market_data import ohlcv_to_df, add_sma
from strategy_sma_crossover import sma_crossover_signal
from risk import risk_decision
from paper_engine import (
    PaperState,
    equity,
    unrealized_pnl,
    open_long,
    close_long,
    check_exits,
)


def load_config() -> dict:
    root = Path(__file__).resolve().parent.parent
    cfg_path = root / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError("Missing config.json.")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _utc(ts) -> str:
    if isinstance(ts, str):
        return ts
    if hasattr(ts, "isoformat"):
        return ts.isoformat()
    return str(ts)


def timeframe_to_minutes(tf: str) -> int:
    # Minimal mapping for common timeframes
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 60 * 24
    raise ValueError(f"Unsupported timeframe '{tf}' for this backtest helper.")


def main():
    cfg = load_config()

    symbol = cfg["symbol"]
    timeframe = cfg["timeframe"]
    sma_fast = int(cfg.get("sma_fast", 10))
    sma_slow = int(cfg.get("sma_slow", 30))

    cfg_paper = cfg.get("paper", {"starting_cash": 1000, "fee_rate": 0.0})
    cfg_risk = cfg.get("risk", {"trade_pct_equity": 0.1, "one_position_only": True})

    fee_rate = float(cfg_paper.get("fee_rate", 0.0))
    starting_cash = float(cfg_paper.get("starting_cash", 1000))

    bt_cfg = cfg.get("backtest", {"days": 30, "warmup_candles": 50})
    days = int(bt_cfg.get("days", 30))
    warmup = int(bt_cfg.get("warmup_candles", 50))

    # ---- Fetch enough candles ----
    minutes = timeframe_to_minutes(timeframe)
    candles_needed = int((days * 24 * 60) / minutes) + warmup + 5

    ex = make_exchange(
        exchange_name=cfg["exchange"],
        use_keys=bool(cfg.get("use_api_keys", False)),
        api_key=cfg.get("api_key", ""),
        api_secret=cfg.get("api_secret", ""),
    )
    ex.load_markets()

    # Many exchanges limit fetch_ohlcv to 500–1500 candles per call.
    # We'll do simple pagination using "since" in ms.
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days) - timedelta(minutes=minutes * warmup)
    since_ms = int(since.timestamp() * 1000)

    all_ohlcv = []
    max_batch = 1000  # safe default
    while len(all_ohlcv) < candles_needed:
        batch = ex.fetch_ohlcv(
            symbol, timeframe=timeframe, since=since_ms, limit=max_batch
        )
        if not batch:
            break

        # avoid duplicates if exchange returns overlapping candles
        if all_ohlcv and batch[0][0] <= all_ohlcv[-1][0]:
            batch = [c for c in batch if c[0] > all_ohlcv[-1][0]]
        if not batch:
            break

        all_ohlcv.extend(batch)
        since_ms = batch[-1][0] + minutes * 60 * 1000  # move forward one candle

        # Hard stop to prevent runaway
        if len(batch) < max_batch:
            break

    if len(all_ohlcv) < (warmup + sma_slow + 5):
        raise RuntimeError(
            f"Not enough candles fetched: {len(all_ohlcv)}. Try fewer days or smaller timeframe."
        )

    df = ohlcv_to_df(all_ohlcv)
    df = add_sma(df, sma_fast)
    df = add_sma(df, sma_slow)

    # ---- Backtest loop ----
    state = PaperState(cash=starting_cash)
    equity_curve = []
    peak_equity = starting_cash
    max_drawdown = 0.0

    # Start after warmup so indicators are ready
    start_idx = warmup
    trades = []

    for i in range(start_idx, len(df)):
        window = df.iloc[: i + 1]
        ts = window.index[-1]
        price = float(window["close"].iloc[-1])

        # 1) exits first
        exit_res = check_exits(state, symbol, price, fee_rate)
        if exit_res:
            msg, trade = exit_res
            trades.append(trade)

        # 2) signal from strategy
        sig = sma_crossover_signal(window, sma_fast, sma_slow)

        # 3) risk -> intent
        intent = risk_decision(
            symbol=symbol,
            signal_action=sig.action,
            price=price,
            cash=state.cash,
            asset_qty=state.asset_qty,
            in_position=state.in_position,
            cfg_risk=cfg_risk,
            cfg_paper=cfg_paper,
        )

        # 4) execute intent
        if intent.action == "OPEN_LONG":
            msg, trade = open_long(
                state=state,
                symbol=symbol,
                amount=float(intent.amount),
                price=price,
                fee_rate=fee_rate,
                stop_price=intent.stop_price,
                take_profit_price=intent.take_profit_price,
                reason=f"BACKTEST: {intent.reason}",
            )
            trades.append(trade)

        elif intent.action == "CLOSE_LONG":
            msg, trade = close_long(
                state=state,
                symbol=symbol,
                amount=float(intent.amount),
                price=price,
                fee_rate=fee_rate,
                reason="BACKTEST: strategy SELL",
            )
            trades.append(trade)

        # 5) record equity curve + drawdown
        eq = equity(state, price)
        equity_curve.append((ts, eq))

        if eq > peak_equity:
            peak_equity = eq
        dd = (peak_equity - eq) / peak_equity if peak_equity > 0 else 0.0
        if dd > max_drawdown:
            max_drawdown = dd

    # If still in position at end, close at last price for reporting
    if state.in_position and state.asset_qty > 0:
        last_price = float(df["close"].iloc[-1])
        msg, trade = close_long(
            state,
            symbol,
            state.asset_qty,
            last_price,
            fee_rate,
            reason="BACKTEST: end close",
        )
        trades.append(trade)
        eq = equity(state, last_price)
        equity_curve.append((df.index[-1], eq))

    # ---- Results ----
    final_equity = equity(state, float(df["close"].iloc[-1]))
    total_return = (
        (final_equity - starting_cash) / starting_cash if starting_cash > 0 else 0.0
    )

    # Trade stats (pair sells with realized pnl)
    sell_trades = [t for t in trades if t.get("side") == "SELL"]
    realized_list = [float(t.get("realized_pnl", 0.0)) for t in sell_trades]
    wins = sum(1 for x in realized_list if x > 0)
    losses = sum(1 for x in realized_list if x < 0)

    print("\n=== BACKTEST SUMMARY ===")
    print(f"Symbol: {symbol} | Timeframe: {timeframe} | Days: {days}")
    print(f"Fast SMA: {sma_fast} | Slow SMA: {sma_slow}")
    print(f"Starting cash: {starting_cash:.2f}")
    print(f"Final equity:   {final_equity:.2f}")
    print(f"Total return:   {total_return*100:.2f}%")
    print(f"Max drawdown:   {max_drawdown*100:.2f}%")
    print(f"Trades (all):   {len(trades)} | Sells(closed): {len(sell_trades)}")
    if sell_trades:
        print(
            f"Win rate:       {wins / len(sell_trades) * 100:.2f}% (wins={wins}, losses={losses})"
        )
        print(f"Avg PnL/trade:  {sum(realized_list)/len(realized_list):.2f}")
        print(f"Total realized: {sum(realized_list):.2f}")
    print("========================\n")

    # Optional: save equity curve + trades to CSV for analysis
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    eq_df = pd.DataFrame(equity_curve, columns=["time", "equity"])
    eq_df.to_csv(out_dir / "equity_curve.csv", index=False)

    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(out_dir / "backtest_trades.csv", index=False)

    print(f"Saved: data/equity_curve.csv and data/backtest_trades.csv")


if __name__ == "__main__":
    main()
