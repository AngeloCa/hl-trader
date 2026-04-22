"""
Market data fetcher — pulls 4h OHLCV from Hyperliquid public API.
No authentication required.
"""
import time
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from config import BASE_URL, COIN, INTERVAL, LOOKBACK_BARS


def fetch_candles(lookback_bars: int = LOOKBACK_BARS) -> pd.DataFrame:
    """
    Fetch the last `lookback_bars` completed 4h candles.
    Adds a 1-bar safety margin so the last row is always a closed bar.
    """
    end_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    # Go back far enough to get all bars (4h = 14400s)
    start_ms = end_ms - (lookback_bars + 2) * 4 * 3600 * 1000

    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin":      COIN,
            "interval":  INTERVAL,
            "startTime": start_ms,
            "endTime":   end_ms,
        },
    }
    for attempt in range(3):
        try:
            resp = requests.post(BASE_URL + "/info", json=payload, timeout=15)
            resp.raise_for_status()
            candles = resp.json()
            break
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(f"Failed to fetch candles after 3 attempts: {e}")
            time.sleep(3)

    if not candles:
        raise RuntimeError("Empty candle response from Hyperliquid")

    df = pd.DataFrame(candles).rename(columns={
        "t": "timestamp", "o": "open", "h": "high",
        "l": "low",       "c": "close", "v": "volume",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df = (df.sort_values("timestamp")
            .drop_duplicates("timestamp")
            .reset_index(drop=True))

    # Drop the last (potentially incomplete) bar
    now_utc  = datetime.now(timezone.utc)
    bar_secs = 4 * 3600
    bar_open = datetime.fromtimestamp(
        (now_utc.timestamp() // bar_secs) * bar_secs, tz=timezone.utc
    )
    df = df[df["timestamp"] < bar_open].copy()

    return df.tail(lookback_bars).reset_index(drop=True)


def get_mid_price() -> float:
    """Fetch current mid price for HYPE/USDC."""
    resp = requests.post(BASE_URL + "/info",
                         json={"type": "allMids"}, timeout=10)
    resp.raise_for_status()
    mids = resp.json()
    return float(mids[COIN])


def get_spot_balance(wallet: str) -> dict[str, float]:
    """
    Returns spot balances: {"USDC": float, "HYPE": float}
    """
    resp = requests.post(BASE_URL + "/info",
                         json={"type": "spotClearinghouseState",
                               "user": wallet}, timeout=10)
    resp.raise_for_status()
    state   = resp.json()
    balances = {}
    for b in state.get("balances", []):
        balances[b["coin"]] = float(b["total"])
    return {
        "USDC": balances.get("USDC", 0.0),
        "HYPE": balances.get("HYPE", 0.0),
    }
