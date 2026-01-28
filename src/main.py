import json
import time
from pathlib import Path
import datetime

from exchange import make_exchange
from market_data import ohlcv_to_df, add_sma
from strategy_sma_crossover import sma_crossover_signal


def load_config() -> dict:
    config_path = Path(__file__).resolve().parent.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            "Missing config.json. Copy config.example.json to config.json and edit it."
        )
    return json.loads(config_path.read_text(encoding="utf-8"))


def log(msg: str) -> None:
    print(f"[{datetime.datetime.now(datetime.UTC)}Z] {msg}")


def main():
    cfg = load_config()

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

    log(
        f"Connected | exchange={cfg['exchange']} | symbol={symbol} | timeframe={timeframe}"
    )
    log(f"Strategy: SMA Crossover | fast={sma_fast} slow={sma_slow}")

    while True:
        try:
            ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=candles_limit)
            df = ohlcv_to_df(ohlcv)
            df = add_sma(df, sma_fast)
            df = add_sma(df, sma_slow)

            sig = sma_crossover_signal(df, sma_fast, sma_slow)

            close = df["close"].iloc[-1]
            fast_val = df[f"sma_{sma_fast}"].iloc[-1]
            slow_val = df[f"sma_{sma_slow}"].iloc[-1]
            time_idx = df.index[-1].isoformat()

            log(
                f"time={time_idx} close={close:.2f} "
                f"sma_{sma_fast}={fast_val if fast_val==fast_val else None} "
                f"sma_{sma_slow}={slow_val if slow_val==slow_val else None} | "
                f"SIGNAL={sig.action} ({sig.reason})"
            )

        except Exception as e:
            log(f"ERROR: {type(e).__name__}: {e}")

        time.sleep(loop_seconds)


if __name__ == "__main__":
    main()
