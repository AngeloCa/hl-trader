"""
Hyperliquid HYPE/USDC Data Indexer
Fetches 5m, 1h and 4h candles from HYPE genesis and computes RSI(14).
Uses 30-day time chunks to bypass the API's ~5000 candle-per-call limit.
"""

import time
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

API_URL    = "https://api.hyperliquid.xyz/info"
COIN       = "HYPE"
RSI_PERIOD = 14
OUTPUT_DIR = Path("data")

INTERVALS = {
    "5m": "5m",
    "1h": "1h",
    "4h": "4h",
}

# 30-day chunk in milliseconds
CHUNK_MS = 30 * 24 * 60 * 60 * 1000


def fetch_candles(coin: str, interval: str, start_ms: int, end_ms: int) -> list[dict]:
    """
    Walk forward in 30-day chunks to retrieve all candles.
    The Hyperliquid API caps responses at ~5000 candles anchored at endTime,
    so fixed-window chunking ensures full coverage.
    """
    seen: set[int] = set()
    all_candles: list[dict] = []
    chunk_start = start_ms

    while chunk_start < end_ms:
        chunk_end = min(chunk_start + CHUNK_MS, end_ms)
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin":      coin,
                "interval":  interval,
                "startTime": chunk_start,
                "endTime":   chunk_end,
            },
        }
        resp = requests.post(API_URL, json=payload, timeout=30)
        resp.raise_for_status()
        batch = resp.json()

        new = [c for c in batch if c["t"] not in seen]
        for c in new:
            seen.add(c["t"])
        all_candles.extend(new)

        print(f"    {pd.Timestamp(chunk_start, unit='ms').date()} → "
              f"{pd.Timestamp(chunk_end,   unit='ms').date()} : "
              f"{len(new)} new candles  (total {len(all_candles)})", flush=True)

        chunk_start = chunk_end + 1
        time.sleep(0.3)

    return all_candles


def candles_to_df(candles: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(candles)
    df = df.rename(columns={"t": "timestamp", "o": "open", "h": "high",
                             "l": "low",       "c": "close", "v": "volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    now     = datetime.now(timezone.utc)
    genesis = datetime(2024, 11, 29, tzinfo=timezone.utc)
    start_ms = int(genesis.timestamp() * 1000)
    end_ms   = int(now.timestamp() * 1000)

    print(f"Fetching {COIN} candles from {genesis.date()} to {now.date()}\n")

    for label, interval in INTERVALS.items():
        print(f"[{label}] Fetching...")
        raw = fetch_candles(COIN, interval, start_ms, end_ms)
        print(f"  → {len(raw)} total candles\n")

        df = candles_to_df(raw)
        df[f"rsi_{RSI_PERIOD}"] = compute_rsi(df["close"], RSI_PERIOD)

        out_path = OUTPUT_DIR / f"HYPE_USDC_{label}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved {len(df)} rows to {out_path}")
        print(df[["timestamp", "close", f"rsi_{RSI_PERIOD}"]].tail(3).to_string(index=False))
        print()

    print("Done.")


if __name__ == "__main__":
    main()
