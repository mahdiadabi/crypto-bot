import ccxt


def make_exchange(
    exchange_name: str, use_keys: bool, api_key: str = "", api_secret: str = ""
):
    if not hasattr(ccxt, exchange_name):
        raise ValueError(f"Unknown exchange '{exchange_name}' in ccxt.")

    exchange_class = getattr(ccxt, exchange_name)

    params = {"enableRateLimit": True}

    if use_keys:
        params.update({"apiKey": api_key, "secret": api_secret})

    ex = exchange_class(params)

    # Optional: set defaults / hardening
    ex.options["adjustForTimeDifference"] = True

    return ex
