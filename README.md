# crypto-bot

Lightweight crypto paper-trading bot with a simple SMA crossover strategy and an
offline backtest runner.

## Setup
- Create and activate your virtualenv.
- Install deps: `pip install -r requirements.txt`
- Edit `config.json` for your exchange, symbol, and risk settings.

## Config overview (config.json)
- Market:
  - `exchange`, `symbol`, `timeframe`, `loop_seconds`
  - `candles_limit` (minimum will be enforced based on indicator periods)
- Strategy:
  - Select with `strategy.name`:
    - `sma_rsi` (default/legacy): SMA crossover with RSI confirmation (`sma_fast`, `sma_slow`, `rsi_*`)
    - `regime`: trend breakout + range mean-reversion (computed locally)
  - Regime params live under `strategy.regime.*` (ADX/EMA/Donchian/BB/VWAP + thresholds)
    - Enable/disable modules: `enable_trend`, `enable_range`
    - Trend exits (optional): `trend_exit_on_close_below_ema`, `trend_exit_on_donchian_low_break`
    - Range exits (optional): `range_exit_on_vwap`, `range_exit_vwap_buffer_atr_mult`, `range_exit_on_bb_mid`, `range_exit_on_bb_upper`, `range_rsi_sell_min`
- Paper:
  - `paper.starting_cash`, `paper.fee_rate`
- Risk (ATR-based stops):
  - `risk.trade_pct_equity`, `risk.one_position_only`
  - `risk.atr_period`, `risk.stop_atr_mult`, `risk.tp_atr_mult`
  - Optional: `risk.trail_stop_atr_mult`, `risk.risk_pct_equity`, `risk.min_atr_pct`, `risk.max_atr_pct`
  - Optional risk controls: `risk.daily_loss_limit_pct`, `risk.dd_reduce_pct`, `risk.dd_reduce_mult`, `risk.dd_stop_pct`
  - Optional per-regime overrides: `risk.profiles.trend.*`, `risk.profiles.range.*`
- Backtest:
  - `backtest.days`, `backtest.warmup_candles`

## Run live (paper)
```
python -m main
```
Notes:
- Signals are evaluated on the last closed candle to avoid look-ahead bias.
- Trades and state are saved under `data/`.

## Run backtest
```
python -m src.backtest
```
Outputs:
- `data/equity_curve.csv`
- `data/backtest_trades.csv`

## Sanity check (ATR sizing/stops)
```
python scripts/sanity_atr.py
```
