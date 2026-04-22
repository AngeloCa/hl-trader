"""
Signal engine — identical math to the backtested strategy.
Input : DataFrame with columns [open, high, low, close, volume]
Output: 'buy' | 'sell' | 'hold'
"""
import numpy as np
import pandas as pd
from config import ATR_PERIOD, ST_MULT, MACD_FAST, MACD_SLOW, MACD_SIG


def _atr(df: pd.DataFrame, period: int) -> np.ndarray:
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    pc  = np.concatenate([[c[0]], c[:-1]])
    tr  = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    out = np.full(len(tr), np.nan)
    out[period - 1] = tr[:period].mean()
    a   = 1.0 / period
    for i in range(period, len(tr)):
        out[i] = out[i-1] * (1 - a) + tr[i] * a
    return out


def _supertrend(df: pd.DataFrame, period: int, mult: float) -> np.ndarray:
    """Returns direction array: +1 bullish, -1 bearish, 0 undefined."""
    c   = df["close"].values
    hl2 = (df["high"].values + df["low"].values) / 2
    atr = _atr(df, period)
    n   = len(c)
    fu  = (hl2 + mult * atr).copy()
    fl  = (hl2 - mult * atr).copy()
    d   = np.zeros(n, dtype=int)

    for i in range(1, n):
        if np.isnan(atr[i]):
            continue
        fu[i] = min(fu[i], fu[i-1]) if c[i-1] <= fu[i-1] else fu[i]
        fl[i] = max(fl[i], fl[i-1]) if c[i-1] >= fl[i-1] else fl[i]
        if d[i-1] == -1:
            d[i] = 1  if c[i] > fu[i-1] else -1
        else:
            d[i] = -1 if c[i] < fl[i-1] else 1

    return d


def _macd_hist(close: np.ndarray, fast: int, slow: int, sig: int) -> np.ndarray:
    def ema(x, p):
        out = np.full(len(x), np.nan)
        out[p-1] = x[:p].mean()
        a = 2 / (p + 1)
        for i in range(p, len(x)):
            out[i] = out[i-1] * (1 - a) + x[i] * a
        return out
    m = ema(close, fast) - ema(close, slow)
    s = ema(np.nan_to_num(m, nan=0.0), sig)
    return m - s


def compute_signal(df: pd.DataFrame) -> tuple[str, dict]:
    """
    Compute trading signal on the last completed bar.

    Returns
    -------
    signal : 'buy' | 'sell' | 'hold'
    debug  : dict of indicator values for logging
    """
    if len(df) < max(ATR_PERIOD + 20, MACD_SLOW + MACD_SIG + 5):
        return "hold", {"reason": "insufficient data"}

    st   = _supertrend(df, ATR_PERIOD, ST_MULT)
    hist = _macd_hist(df["close"].values, MACD_FAST, MACD_SLOW, MACD_SIG)

    i    = len(df) - 1          # last completed bar index
    ip   = i - 1                # previous bar

    st_now,  st_prev  = st[i],   st[ip]
    h_now,   h_prev   = hist[i], hist[ip]

    st_flip_bull = (st_prev != 1)  and (st_now == 1)
    st_flip_bear = (st_prev != -1) and (st_now == -1)
    macd_flip_neg = (not np.isnan(h_now)) and (not np.isnan(h_prev)) and (h_prev >= 0) and (h_now < 0)
    macd_positive = (not np.isnan(h_now)) and (h_now > 0)

    debug = {
        "close":         float(df["close"].iloc[-1]),
        "st_direction":  int(st_now),
        "st_flip_bull":  st_flip_bull,
        "st_flip_bear":  st_flip_bear,
        "macd_hist":     float(h_now) if not np.isnan(h_now) else None,
        "macd_flip_neg": macd_flip_neg,
        "macd_positive": macd_positive,
    }

    if st_flip_bull and macd_positive:
        return "buy", {**debug, "reason": "ST bullish flip + MACD positive"}

    if st_flip_bear or macd_flip_neg:
        reason = "ST bearish flip" if st_flip_bear else "MACD turned negative"
        return "sell", {**debug, "reason": reason}

    return "hold", {**debug, "reason": "no signal"}
