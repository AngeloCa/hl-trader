"""
Signal engine — Strategy 3: Parabolic SAR + MACD
Identical math to the backtested strategy3.py.

Input : DataFrame with columns [open, high, low, close, volume]
Output: 'buy' | 'sell' | 'hold'

BUY  : PSAR flips bullish (was above price → flips below)
        AND MACD histogram > 0
SELL : PSAR flips bearish (was below price → flips above)
        OR MACD histogram turns negative (crosses below zero)
"""
import numpy as np
import pandas as pd
from config import PSAR_START, PSAR_STEP, PSAR_MAX, MACD_FAST, MACD_SLOW, MACD_SIG


def _psar(df: pd.DataFrame) -> np.ndarray:
    """
    Wilder Parabolic SAR.
    Returns direction array: +1 bullish, -1 bearish, 0 warmup.
    """
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    n     = len(close)

    dire = np.zeros(n, dtype=int)

    if n < 3:
        return dire

    # Initialise on bar 1 using first 2 bars
    if close[1] >= close[0]:
        dire[1] = 1
        sar      = min(low[0], low[1])
        ep       = max(high[0], high[1])
    else:
        dire[1] = -1
        sar      = max(high[0], high[1])
        ep       = min(low[0], low[1])

    af = PSAR_START

    for i in range(2, n):
        prev_dir = dire[i - 1]

        if prev_dir == 1:                          # ── Bullish ──────────────
            new_sar = sar + af * (ep - sar)
            new_sar = min(new_sar, low[i - 1], low[i - 2])

            if low[i] < new_sar:                   # reversal → bearish
                dire[i] = -1
                sar      = ep
                ep       = low[i]
                af       = PSAR_START
            else:
                dire[i] = 1
                sar      = new_sar
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + PSAR_STEP, PSAR_MAX)

        else:                                      # ── Bearish ──────────────
            new_sar = sar - af * (sar - ep)
            new_sar = max(new_sar, high[i - 1], high[i - 2])

            if high[i] > new_sar:                  # reversal → bullish
                dire[i] = 1
                sar      = ep
                ep       = high[i]
                af       = PSAR_START
            else:
                dire[i] = -1
                sar      = new_sar
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + PSAR_STEP, PSAR_MAX)

    return dire


def _macd_hist(close: np.ndarray) -> np.ndarray:
    def ema(x, p):
        out = np.full(len(x), np.nan)
        out[p - 1] = x[:p].mean()
        a = 2 / (p + 1)
        for i in range(p, len(x)):
            out[i] = out[i - 1] * (1 - a) + x[i] * a
        return out

    macd_line = ema(close, MACD_FAST) - ema(close, MACD_SLOW)
    sig_line  = ema(np.nan_to_num(macd_line, nan=0.0), MACD_SIG)
    return macd_line - sig_line


def compute_signal(df: pd.DataFrame) -> tuple[str, dict]:
    """
    Compute trading signal on the last completed bar.

    Returns
    -------
    signal : 'buy' | 'sell' | 'hold'
    debug  : dict of indicator values for logging
    """
    min_bars = max(PSAR_START and 10, MACD_SLOW + MACD_SIG + 5)
    if len(df) < min_bars:
        return "hold", {"reason": "insufficient data"}

    dire = _psar(df)
    hist = _macd_hist(df["close"].values)

    i  = len(df) - 1      # last completed bar
    ip = i - 1            # previous bar

    d_now,  d_prev  = dire[i],  dire[ip]
    h_now,  h_prev  = hist[i],  hist[ip]

    psar_flip_bull = (d_prev != 1)  and (d_now == 1)
    psar_flip_bear = (d_prev != -1) and (d_now == -1)
    macd_positive  = (not np.isnan(h_now)) and (h_now > 0)
    macd_flip_neg  = (not np.isnan(h_now)) and (not np.isnan(h_prev)) \
                     and (h_prev >= 0) and (h_now < 0)

    debug = {
        "close":          float(df["close"].iloc[-1]),
        "psar_direction": int(d_now),
        "psar_flip_bull": bool(psar_flip_bull),
        "psar_flip_bear": bool(psar_flip_bear),
        "macd_hist":      float(h_now) if not np.isnan(h_now) else None,
        "macd_flip_neg":  bool(macd_flip_neg),
        "macd_positive":  bool(macd_positive),
    }

    if psar_flip_bull and macd_positive:
        return "buy", {**debug, "reason": "PSAR bullish flip + MACD positive"}

    if psar_flip_bear or macd_flip_neg:
        reason = "PSAR bearish flip" if psar_flip_bear else "MACD turned negative"
        return "sell", {**debug, "reason": reason}

    return "hold", {**debug, "reason": "no signal"}
