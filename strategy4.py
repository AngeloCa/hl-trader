"""
Strategy 4: Parabolic SAR + MACD + ADX Trend Filter
====================================================
Extends Strategy 3 by gating BUY signals through an ADX filter.
ADX measures trend *strength* (not direction) — values above the
threshold mean the market is trending; below = sideways/choppy.

This directly targets two weaknesses found in Strategy 3:
  1. Permutation test p=0.558 — strategy could not be distinguished
     from random trading on shuffled price series.
  2. Prolonged sideways markets — PSAR whipsaws when no clear trend.

ADX filter logic:
  BUY  : PSAR flips bullish AND MACD hist > 0 AND ADX > threshold
  SELL : PSAR flips bearish OR MACD hist turns negative
         (sell triggers are NOT filtered — we always allow exits)

ADX computation (Wilder, 1978):
  TR   = max(H-L, |H-Cp|, |L-Cp|)
  +DM  = max(H-pH, 0) if (H-pH) > (pL-L) else 0
  -DM  = max(pL-L, 0) if (pL-L) > (H-pH) else 0
  Smooth TR, +DM, -DM with Wilder's EMA (period=ADX_PERIOD)
  +DI  = 100 × sDM+ / sTR
  -DI  = 100 × sDM- / sTR
  DX   = 100 × |+DI − −DI| / (+DI + −DI)
  ADX  = Wilder EMA of DX (period=ADX_PERIOD)

Grid:
  psar_step       : [0.01, 0.02, 0.03]
  psar_max        : [0.1, 0.2, 0.3]
  macd_fast/slow/sig : same as Strategy 3
  adx_threshold   : [0, 15, 20, 25]   (0 = no filter, baseline)

Total: 3×3×4×2×2×2 = 288 combinations

Validation plan:
  A) Fitted params on BTC Binance 2019-2024 (cold OOS)
  B) ETH Binance 2018-2023 (cold OOS — includes brutal 2018-2020 sideways)
  C) Permutation test re-run — expect p < 0.05 if ADX filters real noise
"""

import numpy as np
import pandas as pd
import requests
import time as _time
from scipy import stats
from itertools import product
from pathlib import Path

np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
ANNUALIZE    = np.sqrt(8760 / 4)      # 4h bars
ROUND_TRIP   = 0.0016                 # 0.16% spot (taker 0.05% × 2 + slippage 0.10%)
MIN_TRADES   = 5
TRAIN_BARS   = 720
TEST_BARS    = 360
STEP_BARS    = 360
PSAR_START   = 0.02

PSAR_STEPS      = [0.01, 0.02, 0.03]
PSAR_MAXS       = [0.1,  0.2,  0.3]
MACD_FASTS      = [8, 12]
MACD_SLOWS      = [21, 26]
MACD_SIGS       = [7, 9]
ADX_PERIOD      = 14
ADX_THRESHOLDS  = [0, 15, 20, 25]   # 0 = disabled (Strategy 3 baseline)


# ── Indicators ────────────────────────────────────────────────────────────────

def compute_psar(df: pd.DataFrame, af_start: float, af_step: float, af_max: float):
    """Wilder Parabolic SAR. Returns direction array: +1 bullish, -1 bearish."""
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    n     = len(close)

    sar  = np.full(n, np.nan)
    dire = np.zeros(n, dtype=int)

    if close[1] >= close[0]:
        dire[1] = 1
        sar[1]  = min(low[0], low[1])
        ep      = max(high[0], high[1])
    else:
        dire[1] = -1
        sar[1]  = max(high[0], high[1])
        ep      = min(low[0], low[1])

    af = af_start

    for i in range(2, n):
        prev_dir = dire[i-1]
        prev_sar = sar[i-1]

        if prev_dir == 1:
            new_sar = prev_sar + af * (ep - prev_sar)
            new_sar = min(new_sar, low[i-1], low[i-2])
            if low[i] < new_sar:
                dire[i] = -1
                sar[i]  = ep
                ep       = low[i]
                af       = af_start
            else:
                dire[i] = 1
                sar[i]  = new_sar
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            new_sar = prev_sar - af * (prev_sar - ep)
            new_sar = max(new_sar, high[i-1], high[i-2])
            if high[i] > new_sar:
                dire[i] = 1
                sar[i]  = ep
                ep       = high[i]
                af       = af_start
            else:
                dire[i] = -1
                sar[i]  = new_sar
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)

    return sar, dire


def compute_macd(close: np.ndarray, fast: int, slow: int, signal: int) -> np.ndarray:
    def ema(x, p):
        out = np.full(len(x), np.nan)
        out[p-1] = x[:p].mean()
        a = 2 / (p + 1)
        for i in range(p, len(x)):
            out[i] = out[i-1] * (1 - a) + x[i] * a
        return out
    macd_line = ema(close, fast) - ema(close, slow)
    sig_line  = ema(np.where(np.isnan(macd_line), 0, macd_line), signal)
    return macd_line - sig_line


def compute_adx(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """
    Wilder's ADX. Returns adx array aligned to df index.
    Values < period*2 are NaN (warmup).
    """
    high  = df["high"].values.astype(float)
    low   = df["low"].values.astype(float)
    close = df["close"].values.astype(float)
    n     = len(close)

    tr    = np.zeros(n)
    dm_p  = np.zeros(n)
    dm_m  = np.zeros(n)

    for i in range(1, n):
        h, l, pc = high[i], low[i], close[i-1]
        ph, pl   = high[i-1], low[i-1]
        tr[i]    = max(h - l, abs(h - pc), abs(l - pc))
        up_move  = h - ph
        dn_move  = pl - l
        dm_p[i]  = up_move if (up_move > dn_move and up_move > 0) else 0.0
        dm_m[i]  = dn_move if (dn_move > up_move and dn_move > 0) else 0.0

    # Wilder smoothing: initial = sum of first `period` bars; then rolling
    def wilder_smooth(x, p):
        out = np.full(n, np.nan)
        out[p] = x[1:p+1].sum()
        for i in range(p+1, n):
            out[i] = out[i-1] - out[i-1] / p + x[i]
        return out

    s_tr   = wilder_smooth(tr,   period)
    s_dm_p = wilder_smooth(dm_p, period)
    s_dm_m = wilder_smooth(dm_m, period)

    di_p = np.where(s_tr > 0, 100 * s_dm_p / s_tr, 0.0)
    di_m = np.where(s_tr > 0, 100 * s_dm_m / s_tr, 0.0)

    di_sum  = di_p + di_m
    dx      = np.where(di_sum > 0, 100 * np.abs(di_p - di_m) / di_sum, 0.0)

    # ADX = Wilder smooth of DX, starting from bar period*2
    adx = np.full(n, np.nan)
    start = period * 2
    if start < n:
        adx[start] = dx[period:start+1].mean()
        for i in range(start + 1, n):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period

    return adx


# ── Backtest ──────────────────────────────────────────────────────────────────

def backtest(
    df: pd.DataFrame,
    psar_step:     float = 0.02,
    psar_max:      float = 0.2,
    macd_fast:     int   = 8,
    macd_slow:     int   = 21,
    macd_sig:      int   = 9,
    adx_threshold: float = 0.0,
    fee_rt:        float = 0.0,
) -> dict:
    close = df["close"].values
    n     = len(df)

    _, dire = compute_psar(df, PSAR_START, psar_step, psar_max)
    hist    = compute_macd(close, macd_fast, macd_slow, macd_sig)
    adx     = compute_adx(df, ADX_PERIOD)

    position    = 0
    entry_price = 0.0
    entry_bar   = 0
    trades      = []
    equity      = np.ones(n)
    bars_in     = 0

    for i in range(2, n):
        d   = dire[i]
        dp  = dire[i-1]
        h   = hist[i]
        hp  = hist[i-1]
        p   = close[i]
        adx_ok = (adx_threshold <= 0) or (not np.isnan(adx[i]) and adx[i] > adx_threshold)

        psar_flip_bull = (dp != 1)  and (d == 1)
        psar_flip_bear = (dp != -1) and (d == -1)
        macd_positive  = not np.isnan(h) and (h > 0)
        macd_bear      = (not np.isnan(h)) and (not np.isnan(hp)) and (hp >= 0) and (h < 0)

        buy_sig  = psar_flip_bull and macd_positive and adx_ok
        sell_sig = (position == 1) and (psar_flip_bear or macd_bear)

        if position == 0 and buy_sig:
            position    = 1
            entry_price = p * (1 + fee_rt / 2)
            entry_bar   = i
        elif sell_sig:
            exit_p = p * (1 - fee_rt / 2)
            ret    = (exit_p - entry_price) / entry_price
            trades.append({
                "entry_bar":  entry_bar,
                "exit_bar":   i,
                "entry":      close[entry_bar],
                "exit":       close[i],
                "return":     ret,
                "duration":   i - entry_bar,
                "exit_cause": "psar_bear" if psar_flip_bear else "macd_bear",
            })
            position = 0

        if position == 1:
            equity[i] = equity[i-1] * (p / close[i-1])
            bars_in  += 1
        else:
            equity[i] = equity[i-1]

    if position == 1:
        exit_p = close[-1] * (1 - fee_rt / 2)
        trades.append({
            "entry_bar": entry_bar, "exit_bar": n-1,
            "entry": close[entry_bar], "exit": close[-1],
            "return": (exit_p - entry_price) / entry_price,
            "duration": n-1-entry_bar, "exit_cause": "eod",
        })

    bar_ret    = np.diff(equity) / equity[:-1]
    trade_rets = np.array([t["return"] for t in trades])
    peak       = np.maximum.accumulate(equity)
    max_dd     = float(((equity - peak) / peak).min())

    sharpe = (
        float(bar_ret.mean() / bar_ret.std() * ANNUALIZE)
        if len(trades) >= MIN_TRADES and bar_ret.std() > 0
        else -np.inf
    )

    return {
        "sharpe":     sharpe,
        "total_ret":  float(equity[-1] - 1),
        "max_dd":     max_dd,
        "n_trades":   len(trades),
        "trade_rets": trade_rets,
        "equity":     equity,
        "trades":     trades,
        "bar_ret":    bar_ret,
        "bars_in":    bars_in,
    }


# ── Walk-Forward CV ───────────────────────────────────────────────────────────

def walk_forward_cv(df: pd.DataFrame, params: dict) -> dict:
    n      = len(df)
    splits = []
    start  = 0
    while start + TRAIN_BARS + TEST_BARS <= n:
        tr_df  = df.iloc[start : start + TRAIN_BARS].reset_index(drop=True)
        te_df  = df.iloc[start + TRAIN_BARS : start + TRAIN_BARS + TEST_BARS].reset_index(drop=True)
        tr_res = backtest(tr_df, **params)
        te_res = backtest(te_df, **params)
        splits.append({
            "train_sharpe": tr_res["sharpe"],
            "test_sharpe":  te_res["sharpe"],
            "test_ret":     te_res["total_ret"],
            "test_dd":      te_res["max_dd"],
            "test_trades":  te_res["n_trades"],
        })
        start += STEP_BARS
    if not splits:
        return {"mean_oos_sharpe": -np.inf, "splits": []}
    oos_sharpes = [s["test_sharpe"] for s in splits if s["test_sharpe"] > -np.inf]
    return {
        "mean_oos_sharpe": np.mean(oos_sharpes) if oos_sharpes else -np.inf,
        "splits": splits,
    }


# ── Grid Search ──────────────────────────────────────────────────────────────

def grid_search(df: pd.DataFrame):
    best_score  = -np.inf
    best_params = None
    best_cv     = None
    results     = []

    combos = list(product(PSAR_STEPS, PSAR_MAXS, MACD_FASTS, MACD_SLOWS, MACD_SIGS, ADX_THRESHOLDS))
    combos = [(s, mx, mf, ms, msig, adx)
              for s, mx, mf, ms, msig, adx in combos if mf < ms]
    print(f"Grid search: {len(combos)} combinations × walk-forward CV...")

    for i, (step, mx, mf, ms, msig, adx_thr) in enumerate(combos):
        params = dict(psar_step=step, psar_max=mx,
                      macd_fast=mf, macd_slow=ms, macd_sig=msig,
                      adx_threshold=adx_thr)
        cv    = walk_forward_cv(df, params)
        score = cv["mean_oos_sharpe"]
        results.append({**params, "oos_sharpe": score, "n_splits": len(cv["splits"])})
        if score > best_score:
            best_score  = score
            best_params = params
            best_cv     = cv
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(combos)}  best so far: {best_score:.3f}")

    results_df = pd.DataFrame(results).sort_values("oos_sharpe", ascending=False)
    print(f"\nTop 10 parameter sets (by OOS Sharpe):")
    print(results_df.head(10).to_string(index=False))
    return best_params, best_cv, results_df


# ── Permutation Test ──────────────────────────────────────────────────────────

def permutation_test(df: pd.DataFrame, params: dict, n_perms: int = 1000, label: str = ""):
    """
    Shuffle the price series (returns, not prices) and measure how often
    the strategy beats the real strategy's Sharpe.
    A low p-value means the strategy genuinely exploits structure in the data.
    """
    print(f"\n[PERMUTATION TEST] {label}  (n={n_perms} shuffles)")
    real_res    = backtest(df, **params, fee_rt=ROUND_TRIP)
    real_sharpe = real_res["sharpe"]
    print(f"  Real Sharpe: {real_sharpe:.3f}")

    close  = df["close"].values.astype(float)
    log_ret = np.diff(np.log(close))

    perm_sharpes = []
    for _ in range(n_perms):
        shuffled_ret   = np.random.permutation(log_ret)
        shuffled_price = np.exp(np.concatenate([[np.log(close[0])],
                                                np.log(close[0]) + np.cumsum(shuffled_ret)]))
        df_perm = df.copy()
        # Rebuild OHLC from shuffled close (keep H/L/O proportional to close)
        ratio = shuffled_price / close
        df_perm = df_perm.copy()
        df_perm["close"] = shuffled_price
        df_perm["open"]  = df["open"].values  * ratio
        df_perm["high"]  = df["high"].values  * ratio
        df_perm["low"]   = df["low"].values   * ratio
        res = backtest(df_perm, **params, fee_rt=ROUND_TRIP)
        perm_sharpes.append(res["sharpe"] if res["sharpe"] > -np.inf else -99.0)

    perm_sharpes = np.array(perm_sharpes)
    p_value = float((perm_sharpes >= real_sharpe).mean())
    print(f"  Perm Sharpe: mean={perm_sharpes.mean():.3f}  "
          f"p95={np.percentile(perm_sharpes, 95):.3f}")
    print(f"  p-value: {p_value:.4f}  "
          f"({'✅ significant p<0.05' if p_value < 0.05 else '⚠️  not significant'})")
    return p_value, real_sharpe


# ── Data Fetcher ──────────────────────────────────────────────────────────────

def fetch_binance_4h(symbol: str, start_str: str, end_str: str,
                     out_path: Path) -> pd.DataFrame:
    """Download Binance 4h OHLCV, cache to CSV."""
    if out_path.exists():
        print(f"  Loading cached {out_path}")
        df = pd.read_csv(out_path, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    print(f"  Fetching {symbol} 4h from Binance ({start_str} → {end_str})...")
    url    = "https://api.binance.com/api/v3/klines"
    start_ms = int(pd.Timestamp(start_str, tz="UTC").timestamp() * 1000)
    end_ms   = int(pd.Timestamp(end_str,   tz="UTC").timestamp() * 1000)
    rows = []

    while start_ms < end_ms:
        params = {
            "symbol":    symbol,
            "interval":  "4h",
            "startTime": start_ms,
            "endTime":   end_ms,
            "limit":     1000,
        }
        try:
            resp = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()
            batch = resp.json()
        except Exception as e:
            print(f"    Fetch error: {e} — retrying in 5s")
            _time.sleep(5)
            continue
        if not batch:
            break
        rows.extend(batch)
        start_ms = batch[-1][0] + 1
        print(f"    {len(rows)} bars so far…")
        _time.sleep(0.25)

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_base", "taker_quote", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} bars → {out_path}")
    return df


# ── OOS Validation ────────────────────────────────────────────────────────────

def oos_validate(df: pd.DataFrame, params: dict, label: str,
                 fee_rt: float = ROUND_TRIP):
    """
    Apply fixed params to a completely unseen dataset.
    Walk-forward CV + full-period stats.
    """
    print(f"\n{'='*65}")
    print(f"  OOS VALIDATION — {label}")
    params_str = (f"PSAR(step={params['psar_step']}, max={params['psar_max']})"
                  f" + MACD({params['macd_fast']},{params['macd_slow']},{params['macd_sig']})"
                  f" + ADX>{params['adx_threshold']}")
    print(f"  {params_str}")
    print(f"  {len(df)} bars  "
          f"{df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()}")
    print(f"{'='*65}")

    res = backtest(df, **params, fee_rt=fee_rt)
    bh  = df["close"].iloc[-1] / df["close"].iloc[0] - 1

    print(f"\n  Full-period performance:")
    print(f"    Strategy  : {res['total_ret']:+.2%}")
    print(f"    Buy & Hold: {bh:+.2%}")
    print(f"    Sharpe    : {res['sharpe']:.3f}")
    print(f"    Max DD    : {res['max_dd']:.2%}")
    print(f"    Trades    : {res['n_trades']}")
    print(f"    Time in   : {res['bars_in']/len(df)*100:.0f}%")

    cv = walk_forward_cv(df, params)
    oos_sharpes = [s["test_sharpe"] for s in cv["splits"] if s["test_sharpe"] > -np.inf]
    profitable  = sum(1 for s in cv["splits"] if s["test_ret"] > 0)
    total_splits = len(cv["splits"])

    print(f"\n  Walk-forward OOS splits:")
    cum_oos = 1.0
    for i, s in enumerate(cv["splits"], 1):
        te_sh = f"{s['test_sharpe']:.2f}" if s["test_sharpe"] > -np.inf else "n/a"
        print(f"    Split {i}: oos_sharpe={te_sh:>6}  "
              f"oos_ret={s['test_ret']:+.2%}  dd={s['test_dd']:.2%}  "
              f"trades={s['test_trades']}")
        cum_oos *= (1 + s["test_ret"])
    print(f"    Profitable splits  : {profitable}/{total_splits}")
    print(f"    Mean OOS Sharpe    : {np.mean(oos_sharpes):.3f}" if oos_sharpes else "")
    print(f"    OOS stacked return : {cum_oos-1:+.2%}")

    return res, cv


# ── Regime Analysis ───────────────────────────────────────────────────────────

def regime_analysis(df: pd.DataFrame, params: dict, label: str):
    print(f"\n  Regime breakdown ({label}):")
    close = df["close"].values
    ma50  = pd.Series(close).rolling(50).mean().values
    regimes = []
    for i in range(len(close)):
        if np.isnan(ma50[i]):
            regimes.append("warmup")
        elif close[i] > ma50[i] * 1.05:
            regimes.append("bull")
        elif close[i] < ma50[i] * 0.95:
            regimes.append("bear")
        else:
            regimes.append("sideways")

    for regime in ["bull", "sideways", "bear"]:
        idx = [i for i, r in enumerate(regimes) if r == regime]
        if not idx:
            continue
        regime_df = df.iloc[idx].reset_index(drop=True)
        try:
            rr = backtest(regime_df, **params, fee_rt=ROUND_TRIP)
            print(f"    {regime:9s}: ret={rr['total_ret']:+.2%}  "
                  f"sharpe={rr['sharpe'] if rr['sharpe']>-np.inf else 'n/a':>6}  "
                  f"trades={rr['n_trades']}")
        except Exception as e:
            print(f"    {regime:9s}: error ({e})")


# ── ADX threshold sweep ───────────────────────────────────────────────────────

def adx_threshold_sweep(df: pd.DataFrame, base_params: dict):
    """Show how ADX threshold affects core metrics — useful for picking the right gate."""
    print(f"\n  ADX threshold sensitivity (base PSAR+MACD fixed):")
    print(f"  {'ADX_thr':>8} {'Sharpe':>8} {'Return':>10} {'MaxDD':>8} "
          f"{'Trades':>8} {'TimeIn%':>9}")
    for thr in [0, 10, 15, 20, 25, 30]:
        p = {**base_params, "adx_threshold": thr}
        r = backtest(df, **p, fee_rt=ROUND_TRIP)
        pct_in = r["bars_in"] / len(df) * 100
        print(f"  {thr:>8}  {r['sharpe']:>8.3f}  {r['total_ret']:>+10.2%}  "
              f"{r['max_dd']:>8.2%}  {r['n_trades']:>8}  {pct_in:>9.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # ── Load HYPE in-sample data for grid search ──────────────────────────────
    hype_path = data_dir / "HYPE_USDC_4h.csv"
    if not hype_path.exists():
        print(f"ERROR: {hype_path} not found. Run indexer.py first.")
        raise SystemExit(1)

    print("=" * 65)
    print("  STRATEGY 4: PSAR + MACD + ADX TREND FILTER")
    print("=" * 65)

    df_hype = pd.read_csv(hype_path, parse_dates=["timestamp"])
    df_hype = df_hype.sort_values("timestamp").reset_index(drop=True)
    print(f"\nHYPE: {len(df_hype)} bars  "
          f"{df_hype['timestamp'].iloc[0].date()} → {df_hype['timestamp'].iloc[-1].date()}")

    # ── Grid search on HYPE ───────────────────────────────────────────────────
    print("\n[STEP 1] Grid search on HYPE (in-sample parameter fitting)")
    best_params, best_cv, grid_df = grid_search(df_hype)
    grid_df.to_csv(data_dir / "strategy4_grid.csv", index=False)

    print(f"\n{'='*65}")
    print(f"  WINNER: PSAR(step={best_params['psar_step']}, max={best_params['psar_max']})"
          f"  MACD({best_params['macd_fast']},{best_params['macd_slow']},{best_params['macd_sig']})"
          f"  ADX>{best_params['adx_threshold']}")
    print(f"  Mean OOS Sharpe (HYPE): {best_cv['mean_oos_sharpe']:.3f}")
    print(f"{'='*65}")

    # ── ADX threshold sensitivity on winner PSAR+MACD ────────────────────────
    base_no_adx = {k: v for k, v in best_params.items() if k != "adx_threshold"}
    adx_threshold_sweep(df_hype, base_no_adx)

    # ── HYPE full-period performance ──────────────────────────────────────────
    print("\n[STEP 2] HYPE full-period validation")
    oos_validate(df_hype, best_params, "HYPE (in-sample, full period)")
    regime_analysis(df_hype, best_params, "HYPE")

    # ── BTC Binance OOS ───────────────────────────────────────────────────────
    print("\n[STEP 3] BTC Binance 2019-2024 (cold OOS)")
    btc_path = data_dir / "BTC_binance_4h_2019_2024.csv"
    df_btc   = fetch_binance_4h("BTCUSDT", "2019-01-01", "2024-01-01", btc_path)
    print(f"BTC: {len(df_btc)} bars  "
          f"{df_btc['timestamp'].iloc[0].date()} → {df_btc['timestamp'].iloc[-1].date()}")
    oos_validate(df_btc, best_params, "BTC Binance 2019-2024 (cold OOS)",
                 fee_rt=0.0017)  # 0.035%×2 taker + 0.10% slippage
    regime_analysis(df_btc, best_params, "BTC 2019-2024")

    # ── ETH Binance 2018-2023 OOS (worst-case sideways) ───────────────────────
    print("\n[STEP 4] ETH Binance 2018-2023 (sideways stress test OOS)")
    eth_path = data_dir / "ETH_binance_4h_2018_2023.csv"
    df_eth   = fetch_binance_4h("ETHUSDT", "2018-01-01", "2023-01-01", eth_path)
    print(f"ETH: {len(df_eth)} bars  "
          f"{df_eth['timestamp'].iloc[0].date()} → {df_eth['timestamp'].iloc[-1].date()}")
    oos_validate(df_eth, best_params, "ETH Binance 2018-2023 (cold OOS)",
                 fee_rt=0.0017)
    regime_analysis(df_eth, best_params, "ETH 2018-2023")

    # ── Permutation test: Strategy 3 baseline (no ADX) ───────────────────────
    print("\n[STEP 5] Permutation tests (250 shuffles each — statistical significance)")
    params_no_adx = {**best_params, "adx_threshold": 0}
    p_btc_s3, sh_btc_s3 = permutation_test(
        df_btc, params_no_adx, n_perms=250,
        label="BTC — Strategy 3 (no ADX filter)"
    )

    # ── Permutation test: Strategy 4 (with ADX) ───────────────────────────────
    p_btc_s4, sh_btc_s4 = permutation_test(
        df_btc, best_params, n_perms=250,
        label=f"BTC — Strategy 4 (ADX>{best_params['adx_threshold']})"
    )

    # ── Permutation test on ETH ───────────────────────────────────────────────
    p_eth_s4, sh_eth_s4 = permutation_test(
        df_eth, best_params, n_perms=250,
        label=f"ETH — Strategy 4 (ADX>{best_params['adx_threshold']})"
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  STRATEGY 4 SUMMARY")
    print(f"{'='*65}")
    print(f"  Params   : PSAR(step={best_params['psar_step']}, max={best_params['psar_max']})"
          f" + MACD({best_params['macd_fast']},{best_params['macd_slow']},{best_params['macd_sig']})"
          f" + ADX>{best_params['adx_threshold']}")
    print(f"\n  Permutation test results:")
    print(f"    BTC S3 (no ADX) : p={p_btc_s3:.4f}  "
          f"{'✅ significant' if p_btc_s3 < 0.05 else '⚠️  not significant'}")
    print(f"    BTC S4 (ADX)    : p={p_btc_s4:.4f}  "
          f"{'✅ significant' if p_btc_s4 < 0.05 else '⚠️  not significant'}")
    print(f"    ETH S4 (ADX)    : p={p_eth_s4:.4f}  "
          f"{'✅ significant' if p_eth_s4 < 0.05 else '⚠️  not significant'}")

    improvement = "IMPROVED" if p_btc_s4 < p_btc_s3 else "no improvement"
    print(f"\n  ADX filter on permutation test: {improvement}")
    print(f"  (S3 p={p_btc_s3:.4f} → S4 p={p_btc_s4:.4f})")
    print(f"{'='*65}")
