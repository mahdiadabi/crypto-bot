import json
import time
from pathlib import Path
from datetime import datetime, timezone
import traceback

import pandas as pd

from src.exchange import make_exchange
from src.market_data import (
    ohlcv_to_df,
    add_sma,
    add_ema,
    add_atr,
    add_rsi,
    add_adx,
    add_bbands,
    add_donchian,
    add_vwap,
)
from src.strategy_sma_rsi import sma_crossover_with_rsi
from src.strategy_regime import regime_signal, Signal as RegimeSignal
from src.risk import risk_decision
from src.paper_engine import (
    PaperState,
    equity,
    unrealized_pnl,
    open_long,
    close_long,
    check_exits,
    update_trailing_stop_atr,
    update_trailing_stop_atr_highest,
)
from src.persistence import ensure_data_dir, load_state, save_state, append_trade


def load_config() -> dict:
    config_path = Path(__file__).resolve().parent / "config.json"
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
    root = Path(__file__).resolve().parent
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
    rsi_period = int(cfg.get("rsi_period", 14))
    rsi_buy_min = float(cfg.get("rsi_buy_min", 55))
    rsi_sell_max = float(cfg.get("rsi_sell_max", 45))
    cfg_strategy = cfg.get("strategy", {})
    strategy_name = str(cfg_strategy.get("name", "sma_rsi")).strip().lower()
    cfg_regime = cfg_strategy.get("regime", {}) if isinstance(cfg_strategy, dict) else {}

    cfg_paper = cfg.get("paper", {"starting_cash": 1000, "fee_rate": 0.0})
    cfg_risk = cfg.get("risk", {"trade_pct_equity": 0.1, "one_position_only": True})
    atr_period = int(cfg_risk.get("atr_period", 14))
    if strategy_name == "regime":
        ema_period = int(cfg_regime.get("ema_period", 200))
        adx_period = int(cfg_regime.get("adx_period", 14))
        donchian_lookback = int(cfg_regime.get("donchian_lookback", 96))
        bb_period = int(cfg_regime.get("bb_period", 20))
        vwap_period = int(cfg_regime.get("vwap_period", 96))
        required_bars = (
            max(
                ema_period,
                adx_period * 3,
                donchian_lookback,
                bb_period,
                vwap_period,
                rsi_period,
                atr_period,
            )
            + 2
        )
    else:
        required_bars = max(sma_fast, sma_slow, rsi_period, atr_period) + 2
    if candles_limit < required_bars + 1:
        log(
            f"candles_limit={candles_limit} too small; bumping to {required_bars + 1}"
        )
        candles_limit = required_bars + 1

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
    if strategy_name == "regime":
        log(
            "Strategy: Regime Switch (trend breakout + range mean-reversion) | "
            f"EMA={int(cfg_regime.get('ema_period', 200))} "
            f"ADX={int(cfg_regime.get('adx_period', 14))} "
            f"Donchian={int(cfg_regime.get('donchian_lookback', 96))} "
            f"BB={int(cfg_regime.get('bb_period', 20))} "
            f"VWAP={int(cfg_regime.get('vwap_period', 96))}"
        )
    else:
        log(
            "Strategy: SMA+RSI Crossover | "
            f"fast={sma_fast} slow={sma_slow} rsi={rsi_period} "
            f"buy_min={rsi_buy_min} sell_max={rsi_sell_max} "
            f"| filters: above_slow={bool(cfg_strategy.get('require_price_above_slow', True))} "
            f"slow_rising={bool(cfg_strategy.get('require_slow_rising', True))} "
            f"sell_requires_rsi={bool(cfg_strategy.get('sell_requires_rsi', False))}"
        )
    log(f"Paper: fee_rate={fee_rate} | Trades journal -> {trades_csv}")

    while True:
        try:
            state_dirty = False
            # 1) Market data + indicators
            ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=candles_limit)
            df = ohlcv_to_df(ohlcv)
            if strategy_name == "regime":
                ema_period = int(cfg_regime.get("ema_period", 200))
                adx_period = int(cfg_regime.get("adx_period", 14))
                donchian_lookback = int(cfg_regime.get("donchian_lookback", 96))
                bb_period = int(cfg_regime.get("bb_period", 20))
                bb_std = float(cfg_regime.get("bb_std", 2.0))
                vwap_period = int(cfg_regime.get("vwap_period", 96))

                df = add_ema(df, ema_period)
                df = add_adx(df, adx_period)
                df = add_donchian(df, donchian_lookback)
                df = add_bbands(df, bb_period, bb_std)
                df = add_vwap(df, vwap_period)
                df = add_rsi(df, rsi_period)
                df = add_atr(df, atr_period)
            else:
                df = add_sma(df, sma_fast)
                df = add_sma(df, sma_slow)
                df = add_rsi(df, rsi_period)
                df = add_atr(df, atr_period)

            if len(df) < required_bars + 1:
                log(
                    f"Waiting for more candles: have {len(df)} need {required_bars + 1}"
                )
                time.sleep(loop_seconds)
                continue

            # Use last closed candle for signals/decisions
            df = df.iloc[:-1]
            if len(df) < required_bars:
                time.sleep(loop_seconds)
                continue

            price = float(df["close"].iloc[-1])
            time_idx = df.index[-1].isoformat()
            utc_day = df.index[-1].date().isoformat()
            atr_col = f"atr_{atr_period}"
            atr_val = df[atr_col].iloc[-1] if atr_col in df.columns else None
            atr = None if atr_val is None or pd.isna(atr_val) else float(atr_val)

            # 2) Book-keeping + trailing stop, then check exits (SL/TP)
            if state.in_position:
                state.bars_in_position = int(getattr(state, "bars_in_position", 0) or 0) + 1
            else:
                state.bars_in_position = 0
                state.highest_close_since_entry = None
                state.position_profile = None

            trail_mult = cfg_risk.get("trail_stop_atr_mult", cfg_risk.get("stop_atr_mult", None))
            if state.in_position and state.position_profile and isinstance(cfg_risk.get("profiles", None), dict):
                prof = cfg_risk.get("profiles", {}).get(state.position_profile, {})
                if isinstance(prof, dict) and prof.get("trail_stop_atr_mult", None) is not None:
                    trail_mult = prof.get("trail_stop_atr_mult")

            trail_use_highest = bool(cfg_risk.get("trail_use_highest_close", False)) or state.position_profile == "trend"
            if trail_mult is not None and atr is not None and state.in_position:
                try:
                    if trail_use_highest:
                        update_trailing_stop_atr_highest(state, price, atr, float(trail_mult))
                    else:
                        update_trailing_stop_atr(state, price, atr, float(trail_mult))
                    state_dirty = True
                except Exception:
                    pass

            exit_res = check_exits(state, symbol, price, fee_rate)
            if exit_res:
                exit_msg, exit_trade = exit_res
                log(f"{time_idx} | EXIT: {exit_msg}")
                append_trade(trades_csv, exit_trade)
                state_dirty = True

            # 3) Strategy signal
            if strategy_name == "regime":
                # Trend time-stop (risk control)
                time_stop_bars = int(cfg_regime.get("trend_time_stop_bars", 0) or 0)
                if (
                    state.in_position
                    and state.position_profile == "trend"
                    and time_stop_bars > 0
                    and int(getattr(state, "bars_in_position", 0) or 0) >= time_stop_bars
                ):
                    sig = RegimeSignal("SELL", f"time stop: bars_in_position >= {time_stop_bars}")
                else:
                    entries_today = 1 if getattr(state, "last_range_entry_utc_date", None) == utc_day else 0
                    sig = regime_signal(
                        df,
                        rsi_period=rsi_period,
                        atr_period=atr_period,
                        cfg=cfg_regime,
                        in_position=state.in_position,
                        position_profile=state.position_profile,
                        range_entries_today=entries_today,
                    )
            else:
                sig = sma_crossover_with_rsi(
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

            # 3.5) Risk controls (daily loss + max drawdown) for new entries
            eq_now = equity(state, price)
            if getattr(state, "equity_peak", None) is None:
                state.equity_peak = eq_now
                state_dirty = True
            elif eq_now > float(state.equity_peak):
                state.equity_peak = eq_now
                state_dirty = True

            if getattr(state, "day_start_utc_date", None) != utc_day:
                state.day_start_utc_date = utc_day
                state.day_start_equity = eq_now
                state_dirty = True

            day_start_eq = float(getattr(state, "day_start_equity", eq_now) or eq_now)
            daily_ret = (eq_now - day_start_eq) / day_start_eq if day_start_eq > 0 else 0.0
            peak_eq = float(getattr(state, "equity_peak", eq_now) or eq_now)
            dd = (peak_eq - eq_now) / peak_eq if peak_eq > 0 else 0.0

            daily_loss_limit = cfg_risk.get("daily_loss_limit_pct", None)
            dd_reduce_pct = cfg_risk.get("dd_reduce_pct", None)
            dd_reduce_mult = float(cfg_risk.get("dd_reduce_mult", 1.0) or 1.0)
            dd_stop_pct = cfg_risk.get("dd_stop_pct", None)

            entry_blocked_reason = None
            size_mult = 1.0
            if daily_loss_limit is not None and daily_ret <= -float(daily_loss_limit):
                entry_blocked_reason = f"daily loss limit hit: {daily_ret*100:.2f}% <= -{float(daily_loss_limit)*100:.2f}%"
            if dd_stop_pct is not None and dd >= float(dd_stop_pct):
                entry_blocked_reason = f"max drawdown stop: {dd*100:.2f}% >= {float(dd_stop_pct)*100:.2f}%"
            elif dd_reduce_pct is not None and dd >= float(dd_reduce_pct):
                size_mult = max(min(dd_reduce_mult, 1.0), 0.0)

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
                atr=atr,
                risk_profile=getattr(sig, "risk_profile", None),
            )

            # 5) Execute intent (paper)
            exec_msg = None
            exec_trade = None

            if intent.action == "OPEN_LONG":
                if entry_blocked_reason is not None:
                    exec_msg = f"OPEN_LONG blocked by risk controls: {entry_blocked_reason}"
                else:
                    amount = float(intent.amount) * float(size_mult)
                    if amount <= 0:
                        exec_msg = "OPEN_LONG blocked: size multiplier reduced amount to 0"
                    else:
                        # Ensure we don't exceed available cash (incl. fee)
                        cash_cap = state.cash / (price * (1.0 + fee_rate)) if price > 0 else 0.0
                        amount = min(amount, cash_cap)
                        exec_msg, exec_trade = open_long(
                            state=state,
                            symbol=symbol,
                            amount=amount,
                            price=price,
                            fee_rate=fee_rate,
                            stop_price=intent.stop_price,
                            take_profit_price=intent.take_profit_price,
                            reason=intent.reason + (f" | scaled={size_mult:.2f}" if size_mult != 1.0 else ""),
                            position_profile=getattr(sig, "risk_profile", None),
                        )
                        state_dirty = True

                        if getattr(sig, "risk_profile", None) == "range":
                            state.last_range_entry_utc_date = utc_day
                            state_dirty = True
            elif intent.action == "CLOSE_LONG":
                exec_msg, exec_trade = close_long(
                    state=state,
                    symbol=symbol,
                    amount=float(intent.amount),
                    price=price,
                    fee_rate=fee_rate,
                    reason=intent.reason,
                )
                state_dirty = True

            if exec_msg:
                log(f"{time_idx} | EXEC: {exec_msg}")

            if exec_trade:
                append_trade(trades_csv, exec_trade)
            if state_dirty:
                save_state(state, state_path)

            # 6) Status line
            eq = equity(state, price)
            upnl = unrealized_pnl(state, price)

            log(
                f"{time_idx} price={price:.2f} equity={eq:.2f} cash={state.cash:.2f} qty={state.asset_qty:.6f} "
                f"in_pos={state.in_position} entry={state.entry_price} "
                f"uPnL={upnl:.2f} rPnL={state.realized_pnl:.2f} "
                f"| SIGNAL={sig.action}({getattr(sig, 'risk_profile', None)}) | INTENT={intent.action} dd={dd*100:.2f}%"
            )

        except Exception as e:
            log(f"ERROR: {type(e).__name__}: {e}")
            print(traceback.format_exc())

        time.sleep(loop_seconds)


if __name__ == "__main__":
    main()
