import json
import time
from pathlib import Path
from datetime import datetime, timezone

from src.exchange import make_exchange
from src.market_data import ohlcv_to_df, add_sma
from src.strategy_sma_crossover import sma_crossover_signal
from src.risk import risk_decision
from src.paper_engine import (
    PaperState,
    equity,
    unrealized_pnl,
    open_long,
    close_long,
    check_exits,
)
from src.persistence import ensure_data_dir, load_state, save_state, append_trade


def load_config() -> dict:
    config_path = Path(__file__).resolve().parent.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            "Missing config.json. Copy config.example.json to config.json and edit it."
        )
    return json.loads(config_path.read_text(encoding="utf-8"))


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    print(f"[{ts}] {msg}")


def main():
    cfg = load_config()

    # Paths
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    state_path = data_dir / "state.json"
    trades_csv = data_dir / "trades.csv"
    ensure_data_dir(data_dir)

    ex = make_exchange(
        exchange_name=cfg["exchange"],
        use_keys=bool(cfg.get("use_api_keys", False)),
        api_key=cfg.get("api_key", ""),
        api_secret=cfg.get("api_secret", ""),
    )
    ex.load_markets()

    symbol = cfg["symbol"]
    timeframe = cfg["timeframe"]
    loop_seconds = int(cfg.get("loop_seconds", 10))

    candles_limit = int(cfg.get("candles_limit", 200))
    sma_fast = int(cfg.get("sma_fast", 10))
    sma_slow = int(cfg.get("sma_slow", 30))

    cfg_paper = cfg.get("paper", {"starting_cash": 1000, "fee_rate": 0.0})
    cfg_risk = cfg.get("risk", {"trade_pct_equity": 0.1, "one_position_only": True})

    fee_rate = float(cfg_paper.get("fee_rate", 0.0))

    # Load or initialize paper state
    loaded = load_state(state_path)
    if loaded is None:
        state = PaperState(cash=float(cfg_paper.get("starting_cash", 1000)))
        save_state(state, state_path)
        log(f"State initialized -> {state_path}")
    else:
        state = loaded
        log(
            f"State loaded -> {state_path} (cash={state.cash:.2f} qty={state.asset_qty:.6f} in_pos={state.in_position})"
        )

    log(
        f"Connected | exchange={cfg['exchange']} | symbol={symbol} | timeframe={timeframe}"
    )
    log(f"Strategy: SMA Crossover | fast={sma_fast} slow={sma_slow}")
    log(f"Paper: fee_rate={fee_rate} | Trades journal -> {trades_csv}")

    while True:
        try:
            # 1) Market data + indicators
            ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=candles_limit)
            df = ohlcv_to_df(ohlcv)
            df = add_sma(df, sma_fast)
            df = add_sma(df, sma_slow)

            price = float(df["close"].iloc[-1])
            time_idx = df.index[-1].isoformat()

            # 2) Check exits first (SL/TP)
            exit_res = check_exits(state, symbol, price, fee_rate)
            if exit_res:
                exit_msg, exit_trade = exit_res
                log(f"{time_idx} | EXIT: {exit_msg}")
                append_trade(trades_csv, exit_trade)
                save_state(state, state_path)

            # 3) Strategy signal
            sig = sma_crossover_signal(df, sma_fast, sma_slow)

            # 4) Risk -> intent
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

            # 5) Execute intent (paper)
            exec_msg = None
            exec_trade = None

            if intent.action == "OPEN_LONG":
                exec_msg, exec_trade = open_long(
                    state=state,
                    symbol=symbol,
                    amount=float(intent.amount),
                    price=price,
                    fee_rate=fee_rate,
                    stop_price=intent.stop_price,
                    take_profit_price=intent.take_profit_price,
                    reason=intent.reason,
                )
            elif intent.action == "CLOSE_LONG":
                exec_msg, exec_trade = close_long(
                    state=state,
                    symbol=symbol,
                    amount=float(intent.amount),
                    price=price,
                    fee_rate=fee_rate,
                    reason=intent.reason,
                )

            if exec_msg:
                log(f"{time_idx} | EXEC: {exec_msg}")

            if exec_trade:
                append_trade(trades_csv, exec_trade)
                save_state(state, state_path)

            # 6) Status line
            eq = equity(state, price)
            upnl = unrealized_pnl(state, price)

            log(
                f"{time_idx} price={price:.2f} equity={eq:.2f} cash={state.cash:.2f} qty={state.asset_qty:.6f} "
                f"in_pos={state.in_position} entry={state.entry_price} "
                f"uPnL={upnl:.2f} rPnL={state.realized_pnl:.2f} "
                f"| SIGNAL={sig.action} | INTENT={intent.action}"
            )

        except Exception as e:
            log(f"ERROR: {type(e).__name__}: {e}")

        time.sleep(loop_seconds)


if __name__ == "__main__":
    main()
