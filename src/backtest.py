import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.exchange import make_exchange
from src.market_data import (
    add_rsi,
    add_sma,
    add_ema,
    add_atr,
    add_adx,
    add_bbands,
    add_donchian,
    add_vwap,
    ohlcv_to_df,
)
from src.paper_engine import (
    PaperState,
    check_exits,
    close_long,
    equity,
    open_long,
    unrealized_pnl,
    update_trailing_stop_atr,
    update_trailing_stop_atr_highest,
)
from src.risk import risk_decision
from src.strategy_runner import compute_signal


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


def _parse_end_utc(raw_end_utc) -> datetime:
    if raw_end_utc is None:
        return datetime.now(timezone.utc)
    if not isinstance(raw_end_utc, str) or not raw_end_utc.strip():
        raise ValueError("backtest.end_utc must be an ISO8601 string when provided.")

    txt = raw_end_utc.strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    dt = datetime.fromisoformat(txt)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


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


def required_warmup_bars(
    sma_fast: int, sma_slow: int, rsi_period: int, atr_period: int, extra: int = 2
) -> int:
    return max(sma_fast, sma_slow, rsi_period, atr_period) + extra


def main():
    cfg = load_config()

    symbol = cfg["symbol"]
    timeframe = cfg["timeframe"]
    sma_fast = int(cfg.get("sma_fast", 10))
    sma_slow = int(cfg.get("sma_slow", 30))
    rsi_period = int(cfg.get("rsi_period", 14))
    rsi_buy_min = float(cfg.get("rsi_buy_min", 55))
    rsi_sell_max = float(cfg.get("rsi_sell_max", 45))

    cfg_paper = cfg.get("paper", {"starting_cash": 1000, "fee_rate": 0.0})
    cfg_risk = cfg.get("risk", {"trade_pct_equity": 0.1, "one_position_only": True})
    atr_period = int(cfg_risk.get("atr_period", 14))
    cfg_strategy = cfg.get("strategy", {})
    strategy_name = str(cfg_strategy.get("name", "sma_rsi")).strip().lower()
    cfg_regime = (
        cfg_strategy.get("regime", {}) if isinstance(cfg_strategy, dict) else {}
    )
    if strategy_name not in ("regime", "sma_rsi"):
        raise ValueError(f"Unknown strategy '{strategy_name}'. Use 'regime' or 'sma_rsi'.")

    fee_rate = float(cfg_paper.get("fee_rate", 0.0))
    starting_cash = float(cfg_paper.get("starting_cash", 1000))

    bt_cfg = cfg.get("backtest", {"days": 30, "warmup_candles": 50})
    days = int(bt_cfg.get("days", 30))
    warmup = int(bt_cfg.get("warmup_candles", 50))
    end_utc = _parse_end_utc(bt_cfg.get("end_utc", None))
    print_candle_range = bool(bt_cfg.get("print_candle_range", True))
    if strategy_name == "regime":
        ema_period = int(cfg_regime.get("ema_period", 200))
        adx_period = int(cfg_regime.get("adx_period", 14))
        donchian_lookback = int(cfg_regime.get("donchian_lookback", 96))
        bb_period = int(cfg_regime.get("bb_period", 20))
        vwap_period = int(cfg_regime.get("vwap_period", 96))
        required = (
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
        warmup = max(warmup, required)
    else:
        warmup = max(
            warmup, required_warmup_bars(sma_fast, sma_slow, rsi_period, atr_period)
        )

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
    since = end_utc - timedelta(days=days) - timedelta(minutes=minutes * warmup)
    since_ms = int(since.timestamp() * 1000)
    end_ms = int(end_utc.timestamp() * 1000)

    all_ohlcv = []
    max_batch = 1000  # safe default
    while len(all_ohlcv) < candles_needed:
        batch = ex.fetch_ohlcv(
            symbol, timeframe=timeframe, since=since_ms, limit=max_batch
        )
        if not batch:
            break

        # Deterministic run end: ignore candles that start after configured end_utc.
        batch = [c for c in batch if int(c[0]) <= end_ms]
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

    df = ohlcv_to_df(all_ohlcv)
    if len(df) > 0:
        df = df[df.index <= end_utc]
    if len(df) > 0:
        last_open = df.index[-1].to_pydatetime()
        if last_open + timedelta(minutes=minutes) > end_utc:
            df = df.iloc[:-1]

    if len(df) < (warmup + 5):
        raise RuntimeError(
            f"Not enough candles after end_utc/open-candle filtering: {len(df)}. "
            "Try increasing days or using an earlier end_utc."
        )

    if print_candle_range and len(df) > 0:
        first_ts = df.index[0]
        last_ts = df.index[-1]
        print(f"Requested window: since={_utc(since)} | end={_utc(end_utc)}")
        print(
            f"Fetched candles: n={len(df)} | first={_utc(first_ts)} | last={_utc(last_ts)}"
        )
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
    # ---- Backtest loop ----
    state = PaperState(cash=starting_cash)
    equity_curve = []
    peak_equity = starting_cash
    max_drawdown = 0.0
    day_start_utc = None
    day_start_equity = starting_cash
    last_range_entry_utc = None
    signal_action_counts: Counter[str] = Counter()
    hold_reason_counts: Counter[str] = Counter()
    blocked_buy_reason_counts: Counter[str] = Counter()

    # Start after warmup so indicators are ready
    start_idx = warmup
    if print_candle_range and len(df) > 0 and 0 <= start_idx < len(df):
        trade_first_ts = df.index[start_idx]
        trade_last_ts = df.index[-1]
        print(
            f"Trade window (after warmup): first={_utc(trade_first_ts)} | last={_utc(trade_last_ts)}"
        )
    trades = []

    for i in range(start_idx, len(df)):
        window = df.iloc[: i + 1]
        ts = window.index[-1]
        price = float(window["close"].iloc[-1])
        utc_day = ts.date().isoformat()
        atr_col = f"atr_{atr_period}"
        atr_val = window[atr_col].iloc[-1] if atr_col in window.columns else None
        atr = None if atr_val is None or pd.isna(atr_val) else float(atr_val)

        # 0) bars-in-position book-keeping
        if state.in_position:
            state.bars_in_position = int(getattr(state, "bars_in_position", 0) or 0) + 1
            state.bars_since_exit = 0
        else:
            state.bars_in_position = 0
            state.highest_close_since_entry = None
            state.position_profile = None
            state.bars_since_exit = int(getattr(state, "bars_since_exit", 0) or 0) + 1

        # 1) update trailing stop, then exits
        trail_mult = cfg_risk.get("trail_stop_atr_mult", cfg_risk.get("stop_atr_mult"))
        if (
            state.in_position
            and state.position_profile
            and isinstance(cfg_risk.get("profiles", None), dict)
        ):
            prof = cfg_risk.get("profiles", {}).get(state.position_profile, {})
            if (
                isinstance(prof, dict)
                and prof.get("trail_stop_atr_mult", None) is not None
            ):
                trail_mult = prof.get("trail_stop_atr_mult")

        trail_use_highest = (
            bool(cfg_risk.get("trail_use_highest_close", False))
            or state.position_profile == "trend"
        )
        if trail_mult is not None and atr is not None and state.in_position:
            try:
                if trail_use_highest:
                    update_trailing_stop_atr_highest(
                        state, price, atr, float(trail_mult)
                    )
                else:
                    update_trailing_stop_atr(state, price, atr, float(trail_mult))
            except Exception:
                pass

        exit_res = check_exits(state, symbol, price, fee_rate)
        exited_this_bar = False
        if exit_res:
            msg, trade = exit_res
            # Backtests should record the candle timestamp, not wall-clock time.
            if isinstance(trade, dict):
                trade["time"] = _utc(ts)
            trades.append(trade)
            exited_this_bar = True

        # 2) signal from strategy
        sig = compute_signal(
            window,
            strategy_name=strategy_name,
            cfg_strategy=cfg_strategy,
            cfg_regime=cfg_regime,
            sma_fast=sma_fast,
            sma_slow=sma_slow,
            rsi_period=rsi_period,
            rsi_buy_min=rsi_buy_min,
            rsi_sell_max=rsi_sell_max,
            atr_period=atr_period,
            in_position=state.in_position,
            position_profile=state.position_profile,
            bars_in_position=int(getattr(state, "bars_in_position", 0) or 0),
            utc_day=utc_day,
            last_range_entry_utc_date=last_range_entry_utc,
        )

        signal_action_counts[str(sig.action)] += 1
        if str(sig.action) == "HOLD":
            hold_reason_counts[str(sig.reason)] += 1

        # 2.5) daily loss + drawdown controls (entries only)
        eq_pre = equity(state, price)
        if day_start_utc != utc_day:
            day_start_utc = utc_day
            day_start_equity = eq_pre
        daily_ret = (
            (eq_pre - day_start_equity) / day_start_equity
            if day_start_equity > 0
            else 0.0
        )
        dd = (peak_equity - eq_pre) / peak_equity if peak_equity > 0 else 0.0

        daily_loss_limit = cfg_risk.get("daily_loss_limit_pct", None)
        dd_reduce_pct = cfg_risk.get("dd_reduce_pct", None)
        dd_reduce_mult = float(cfg_risk.get("dd_reduce_mult", 1.0) or 1.0)
        dd_stop_pct = cfg_risk.get("dd_stop_pct", None)

        entry_blocked = False
        entry_block_reason = None
        size_mult = 1.0
        if daily_loss_limit is not None and daily_ret <= -float(daily_loss_limit):
            entry_blocked = True
            entry_block_reason = "blocked: daily_loss_limit_pct"
        if dd_stop_pct is not None and dd >= float(dd_stop_pct):
            entry_blocked = True
            entry_block_reason = "blocked: dd_stop_pct"
        elif dd_reduce_pct is not None and dd >= float(dd_reduce_pct):
            size_mult = max(min(dd_reduce_mult, 1.0), 0.0)

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
            atr=atr,
            risk_profile=getattr(sig, "risk_profile", None),
        )

        # 4) execute intent
        opened_this_bar = False
        if intent.action == "OPEN_LONG":
            # Prevent immediate re-entry on the same candle after an exit (common churn pattern).
            if exited_this_bar:
                pass
            elif not entry_blocked:
                amount = float(intent.amount) * float(size_mult)
                cash_cap = state.cash / (price * (1.0 + fee_rate)) if price > 0 else 0.0
                amount = min(amount, cash_cap)
                if amount > 0:
                    msg, trade = open_long(
                        state=state,
                        symbol=symbol,
                        amount=amount,
                        price=price,
                        fee_rate=fee_rate,
                        stop_price=intent.stop_price,
                        take_profit_price=intent.take_profit_price,
                        reason=f"BACKTEST: {intent.reason}",
                        position_profile=getattr(sig, "risk_profile", None),
                    )
                    if isinstance(trade, dict):
                        trade["time"] = _utc(ts)
                    trades.append(trade)
                    opened_this_bar = True
                    if getattr(sig, "risk_profile", None) == "range":
                        last_range_entry_utc = utc_day

        elif intent.action == "CLOSE_LONG":
            strategy_exit_reason = str(getattr(sig, "reason", "") or "strategy SELL")
            msg, trade = close_long(
                state=state,
                symbol=symbol,
                amount=float(intent.amount),
                price=price,
                fee_rate=fee_rate,
                reason=f"BACKTEST: {strategy_exit_reason}",
            )
            if isinstance(trade, dict):
                trade["time"] = _utc(ts)
            trades.append(trade)

        if sig.action == "BUY":
            if intent.action != "OPEN_LONG":
                blocked_buy_reason_counts[str(intent.reason)] += 1
            elif exited_this_bar:
                blocked_buy_reason_counts["blocked: same-candle reentry"] += 1
            elif entry_blocked:
                blocked_buy_reason_counts[str(entry_block_reason or "blocked: risk guard")] += 1
            elif not opened_this_bar:
                blocked_buy_reason_counts["blocked: amount <= 0 after caps"] += 1

        # 5) record equity curve + drawdown stats
        eq_post = equity(state, price)
        equity_curve.append((ts, eq_post))
        if eq_post > peak_equity:
            peak_equity = eq_post
        dd_post = (peak_equity - eq_post) / peak_equity if peak_equity > 0 else 0.0
        if dd_post > max_drawdown:
            max_drawdown = dd_post

    # If still in position at end, close at last price for reporting
    if state.in_position and state.asset_qty > 0:
        last_price = float(df["close"].iloc[-1])
        last_ts = df.index[-1]
        msg, trade = close_long(
            state,
            symbol,
            state.asset_qty,
            last_price,
            fee_rate,
            reason="BACKTEST: end close",
        )
        if isinstance(trade, dict):
            trade["time"] = _utc(last_ts)
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
    print(f"Symbol: {symbol} | Timeframe: {timeframe} | Days: {days} | End={_utc(end_utc)}")
    print(f"Strategy: {strategy_name}")
    if strategy_name == "regime":
        print(
            "Regime: "
            f"trend={'on' if bool(cfg_regime.get('enable_trend', True)) else 'off'} "
            f"range={'on' if bool(cfg_regime.get('enable_range', True)) else 'off'} | "
            f"ADX(trend>={float(cfg_regime.get('trend_adx_min', 20)):.1f}, range<={float(cfg_regime.get('range_adx_max', 18)):.1f})"
        )
        print(
            "Range exits: "
            f"vwap={bool(cfg_regime.get('range_exit_on_vwap', True))} "
            f"vwap_buf_atr={float(cfg_regime.get('range_exit_vwap_buffer_atr_mult', 0.0) or 0.0):.2f} "
            f"bb_mid={bool(cfg_regime.get('range_exit_on_bb_mid', False))} "
            f"bb_mid_buf_atr={float(cfg_regime.get('range_exit_bb_mid_buffer_atr_mult', 0.0) or 0.0):.2f} "
            f"bb_upper={bool(cfg_regime.get('range_exit_on_bb_upper', False))} "
            f"rsi_sell_min={cfg_regime.get('range_rsi_sell_min', None)} "
            f"wick_high={bool(cfg_regime.get('range_exit_use_wick_high', False))}"
        )
        print(
            "Range entry filters: "
            f"close_above_ema={bool(cfg_regime.get('range_require_close_above_ema', False))} "
            f"ema_rising={bool(cfg_regime.get('range_require_ema_rising', False))} "
            f"wick_low={bool(cfg_regime.get('range_entry_use_wick_low', False))} "
            f"reclaim_bb_lower={bool(cfg_regime.get('range_entry_require_reclaim_bb_lower', False))} "
            f"rsi_rising={bool(cfg_regime.get('range_entry_require_rsi_rising', False))}"
        )
    else:
        print(f"SMA: fast={sma_fast} slow={sma_slow} | RSI={rsi_period}")

    profiles = cfg_risk.get("profiles", {}) if isinstance(cfg_risk, dict) else {}
    print(
        "Risk exits: "
        f"stop_atr_mult={cfg_risk.get('stop_atr_mult', None)} "
        f"trail_stop_atr_mult={cfg_risk.get('trail_stop_atr_mult', None)} "
        f"tp_atr_mult={cfg_risk.get('tp_atr_mult', None)}"
    )
    if isinstance(profiles, dict) and profiles:
        for prof_name in ("trend", "range"):
            if prof_name in profiles and isinstance(profiles.get(prof_name), dict):
                p = profiles[prof_name]
                print(
                    f"Risk profile '{prof_name}': "
                    f"stop_atr_mult={p.get('stop_atr_mult', 'inherit')} "
                    f"trail_stop_atr_mult={p.get('trail_stop_atr_mult', 'inherit')} "
                    f"tp_atr_mult={p.get('tp_atr_mult', 'inherit')}"
                )
    print(f"Starting cash: {starting_cash:.2f}")
    print(f"Final equity:   {final_equity:.2f}")
    print(f"Total return:   {total_return*100:.2f}%")
    print(f"Max drawdown:   {max_drawdown*100:.2f}%")
    print(
        "Signals: "
        f"BUY={signal_action_counts.get('BUY', 0)} "
        f"SELL={signal_action_counts.get('SELL', 0)} "
        f"HOLD={signal_action_counts.get('HOLD', 0)}"
    )
    print(f"Trades (all):   {len(trades)} | Sells(closed): {len(sell_trades)}")
    if hold_reason_counts:
        top_hold = hold_reason_counts.most_common(6)
        print(
            "Top HOLD reasons: "
            + " | ".join(f"{reason}={count}" for reason, count in top_hold)
        )
    if blocked_buy_reason_counts:
        print(
            "Blocked BUY reasons: "
            + " | ".join(
                f"{reason}={count}"
                for reason, count in blocked_buy_reason_counts.most_common(8)
            )
        )
    if sell_trades:
        exit_counts = {
            "TAKE_PROFIT hit": 0,
            "STOP_LOSS hit": 0,
            "strategy": 0,
            "BACKTEST: end close": 0,
            "other": 0,
        }
        strategy_exit_details: Counter[str] = Counter()
        for t in sell_trades:
            reason = str(t.get("reason", "") or "")
            if reason in ("TAKE_PROFIT hit", "STOP_LOSS hit", "BACKTEST: end close"):
                exit_counts[reason] += 1
            elif reason.startswith("BACKTEST:"):
                exit_counts["strategy"] += 1
                strategy_exit_details[reason] += 1
            else:
                exit_counts["other"] += 1

        print(
            "Exit reasons: "
            f"TP={exit_counts['TAKE_PROFIT hit']} "
            f"SL={exit_counts['STOP_LOSS hit']} "
            f"strategy={exit_counts['strategy']} "
            f"end={exit_counts['BACKTEST: end close']} "
            f"other={exit_counts['other']}"
        )
        if strategy_exit_details:
            print(
                "Strategy exit details: "
                + " | ".join(
                    f"{reason.replace('BACKTEST: ', '', 1)}={count}"
                    for reason, count in strategy_exit_details.most_common(8)
                )
            )

        # Per-position-profile breakdown (trend vs range)
        prof_stats = {}
        for t in sell_trades:
            prof = str(t.get("position_profile", "") or "unknown")
            realized = float(t.get("realized_pnl", 0.0) or 0.0)
            s = prof_stats.setdefault(prof, {"n": 0, "wins": 0, "losses": 0, "realized": 0.0})
            s["n"] += 1
            s["realized"] += realized
            if realized > 0:
                s["wins"] += 1
            elif realized < 0:
                s["losses"] += 1

        if prof_stats:
            parts = []
            for prof in sorted(prof_stats.keys()):
                s = prof_stats[prof]
                wr = (s["wins"] / s["n"] * 100.0) if s["n"] > 0 else 0.0
                parts.append(f"{prof}: n={s['n']} win%={wr:.1f} realized={s['realized']:.2f}")
            print("By profile: " + " | ".join(parts))

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
