"""
Strategy 3: Parabolic SAR + MACD Confirmation
==============================================
Parabolic SAR is a CTA-grade adaptive trailing stop that accelerates
as the trend matures — capturing more of a strong move but exiting
quickly when momentum fades.

Signal logic:
  BUY  : PSAR flips bullish (SAR was above price → flips below)
           AND MACD histogram > 0  (momentum confirmation)
  SELL : PSAR flips bearish (SAR was below price → flips above)
           OR MACD histogram turns negative (early momentum exit)

Why this is different from Strategy 2 (Supertrend):
  - ST uses fixed ATR bands around midpoint; adapts only to volatility
  - PSAR uses an acceleration factor that SPEEDS UP as new highs are set
    → catches trends that accelerate (e.g. parabolic HYPE rallies)
    → exits faster when momentum stalls even before full reversal

Grid:
  psar_step : [0.01, 0.02, 0.03]   acceleration step per new extreme
  psar_max  : [0.1, 0.2, 0.3]      max acceleration factor
  macd_fast : [8, 12]
  macd_slow : [21, 26]
  macd_sig  : [7, 9]

Total: 3 × 3 × 2 × 2 × 2 = 72 parameter sets
Walk-forward CV: same 6 OOS splits as strategies 1 & 2
Full quant suite: Sharpe, Calmar, bootstrap CI, Monte Carlo,
                  regime analysis, Kelly, VaR, Sortino
"""

import numpy as np
import pandas as pd
from scipy import stats
from itertools import product
from pathlib import Path

np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = Path("data/HYPE_USDC_4h.csv")
BAR_HOURS    = 4
ANNUALIZE    = np.sqrt(8760 / BAR_HOURS)
ROUND_TRIP   = 0.0016
MIN_TRADES   = 5
TRAIN_BARS   = 720
TEST_BARS    = 360
STEP_BARS    = 360

PSAR_START   = 0.02                    # fixed AF initial value (Wilder standard)
PSAR_STEPS   = [0.01, 0.02, 0.03]
PSAR_MAXS    = [0.1,  0.2,  0.3]
MACD_FASTS   = [8, 12]
MACD_SLOWS   = [21, 26]
MACD_SIGS    = [7, 9]


# ── Indicators ────────────────────────────────────────────────────────────────
def compute_psar(df: pd.DataFrame, af_start: float, af_step: float, af_max: float):
    """
    Wilder Parabolic SAR.
    Returns arrays: sar (float), direction (+1 bullish / -1 bearish)
    """
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    n     = len(close)

    sar  = np.full(n, np.nan)
    dire = np.zeros(n, dtype=int)

    # Initialise on bar 1 using first 2 bars
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

        if prev_dir == 1:                          # ── Bullish ──────────────
            new_sar = prev_sar + af * (ep - prev_sar)
            # SAR must not be above the two prior lows
            new_sar = min(new_sar, low[i-1], low[i-2])

            if low[i] < new_sar:                   # reversal
                dire[i] = -1
                sar[i]  = ep                       # SAR flips to prior EP
                ep       = low[i]
                af       = af_start
            else:
                dire[i] = 1
                sar[i]  = new_sar
                if high[i] > ep:                   # new extreme → accelerate
                    ep = high[i]
                    af = min(af + af_step, af_max)

        else:                                      # ── Bearish ──────────────
            new_sar = prev_sar - af * (prev_sar - ep)
            # SAR must not be below the two prior highs
            new_sar = max(new_sar, high[i-1], high[i-2])

            if high[i] > new_sar:                  # reversal
                dire[i] = 1
                sar[i]  = ep
                ep       = high[i]
                af       = af_start
            else:
                dire[i] = -1
                sar[i]  = new_sar
                if low[i] < ep:                    # new extreme → accelerate
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


# ── Backtest ──────────────────────────────────────────────────────────────────
def backtest(
    df: pd.DataFrame,
    psar_step: float = 0.02,
    psar_max:  float = 0.2,
    macd_fast: int   = 8,
    macd_slow: int   = 21,
    macd_sig:  int   = 9,
    fee_rt:    float = 0.0,
) -> dict:
    close = df["close"].values
    n     = len(df)

    _, dire = compute_psar(df, PSAR_START, psar_step, psar_max)
    hist    = compute_macd(close, macd_fast, macd_slow, macd_sig)

    position    = 0
    entry_price = 0.0
    entry_bar   = 0
    trades      = []
    equity      = np.ones(n)           # all bars initialised to 1.0
    bars_in     = 0

    for i in range(2, n):
        d   = dire[i]
        dp  = dire[i-1]
        h   = hist[i]
        hp  = hist[i-1]
        p   = close[i]

        psar_flip_bull = (dp != 1)  and (d == 1)
        psar_flip_bear = (dp != -1) and (d == -1)
        macd_positive  = not np.isnan(h) and (h > 0)
        macd_bear      = (not np.isnan(h)) and (not np.isnan(hp)) and (hp >= 0) and (h < 0)

        buy_sig  = psar_flip_bull and macd_positive
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

    # Close any open position at end
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
    n       = len(df)
    splits  = []
    start   = 0

    while start + TRAIN_BARS + TEST_BARS <= n:
        tr_df   = df.iloc[start : start + TRAIN_BARS].reset_index(drop=True)
        te_df   = df.iloc[start + TRAIN_BARS : start + TRAIN_BARS + TEST_BARS].reset_index(drop=True)
        tr_res  = backtest(tr_df, **params)
        te_res  = backtest(te_df, **params)
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

    combos = list(product(PSAR_STEPS, PSAR_MAXS, MACD_FASTS, MACD_SLOWS, MACD_SIGS))
    print(f"Grid search: {len(combos)} combinations × walk-forward CV...")

    for step, mx, mf, ms, msig in combos:
        if mf >= ms:
            continue
        params = dict(psar_step=step, psar_max=mx,
                      macd_fast=mf, macd_slow=ms, macd_sig=msig)
        cv = walk_forward_cv(df, params)
        score = cv["mean_oos_sharpe"]
        results.append({**params, "oos_sharpe": score,
                        "n_splits": len(cv["splits"])})
        if score > best_score:
            best_score  = score
            best_params = params
            best_cv     = cv

    results_df = pd.DataFrame(results).sort_values("oos_sharpe", ascending=False)
    print(f"\nTop 10 parameter sets (by OOS Sharpe):")
    print(results_df.head(10).to_string(index=False))
    return best_params, best_cv, results_df


# ── Full Quant Suite ──────────────────────────────────────────────────────────
def quant_suite(df: pd.DataFrame, params: dict, label: str):
    print(f"\n{'='*65}")
    print(f"  QUANT VALIDATION — {label}")
    print(f"  PSAR(start={PSAR_START}, step={params['psar_step']}, max={params['psar_max']})"
          f"  MACD({params['macd_fast']},{params['macd_slow']},{params['macd_sig']})")
    print(f"{'='*65}")

    res = backtest(df, **params, fee_rt=ROUND_TRIP)
    bh  = df["close"].iloc[-1] / df["close"].iloc[0] - 1

    print(f"\n[1] Hold-out Performance vs Buy & Hold")
    print(f"    Strategy total return : {res['total_ret']:+.2%}")
    print(f"    Buy & Hold            : {bh:+.2%}")
    print(f"    Sharpe (ann.)         : {res['sharpe']:.3f}")
    ann_ret = (1 + res['total_ret']) ** (ANNUALIZE**2 / len(df)) - 1
    calmar  = ann_ret / abs(res['max_dd']) if res['max_dd'] < 0 else np.inf
    print(f"    Calmar ratio          : {calmar:.2f}")
    print(f"    Max drawdown          : {res['max_dd']:.2%}")
    print(f"    Trades                : {res['n_trades']}")
    market_pct = res['bars_in'] / len(df) * 100
    print(f"    Time in market        : {market_pct:.0f}%")

    exit_causes = {}
    for t in res['trades']:
        cause = t.get('exit_cause', 'unknown')
        exit_causes[cause] = exit_causes.get(cause, 0) + 1
    print(f"    Exit causes           : {exit_causes}")

    # ── Fee sensitivity ───────────────────────────────────────────────────────
    print(f"\n[2] Fee Sensitivity")
    for fee in [0.0, 0.0016, 0.003, 0.005]:
        r = backtest(df, **params, fee_rt=fee)
        print(f"    fee={fee:.1%}  ret={r['total_ret']:+.2%}  sharpe={r['sharpe']:.2f}  dd={r['max_dd']:.2%}")

    # ── Statistical significance ──────────────────────────────────────────────
    tr = res["trade_rets"]
    if len(tr) >= 2:
        t_stat, p_val = stats.ttest_1samp(tr, 0)
        print(f"\n[3] Statistical Significance")
        print(f"    Trades         : {len(tr)}")
        print(f"    Mean trade ret : {tr.mean():+.4f}  ({tr.mean()*100:+.2f}%)")
        print(f"    t-stat         : {t_stat:.3f}   p-value: {p_val:.4f}")
        n_needed = int(np.ceil((1.96 / (tr.mean() / tr.std()))**2)) if tr.std() > 0 else "∞"
        print(f"    Trades for p<0.05: {n_needed}")

    # ── Bootstrap Sharpe CI ───────────────────────────────────────────────────
    print(f"\n[4] Bootstrap Sharpe 95% CI  (10,000 resamples)")
    br = res["bar_ret"]
    boot_sharpes = []
    for _ in range(10_000):
        s = np.random.choice(br, size=len(br), replace=True)
        if s.std() > 0:
            boot_sharpes.append(s.mean() / s.std() * ANNUALIZE)
    boot_sharpes = np.array(boot_sharpes)
    lo, hi = np.percentile(boot_sharpes, [2.5, 97.5])
    print(f"    95% CI : [{lo:.2f}, {hi:.2f}]")
    print(f"    P(Sharpe > 1) : {(boot_sharpes > 1).mean():.1%}")
    print(f"    P(Sharpe > 2) : {(boot_sharpes > 2).mean():.1%}")

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    print(f"\n[5] Monte Carlo  (5,000 trade-order shuffles)")
    if len(tr) >= 2:
        mc_final = []
        for _ in range(5_000):
            shuffled = np.random.permutation(tr)
            mc_final.append(np.prod(1 + shuffled) - 1)
        mc_final = np.array(mc_final)
        print(f"    P(profitable) : {(mc_final > 0).mean():.1%}")
        print(f"    Median return : {np.median(mc_final):+.2%}")
        print(f"    5th pct       : {np.percentile(mc_final, 5):+.2%}")

    # ── Regime analysis ───────────────────────────────────────────────────────
    print(f"\n[6] Regime Analysis")
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

    # ── Kelly sizing ──────────────────────────────────────────────────────────
    print(f"\n[7] Kelly Criterion")
    if len(tr) >= 2:
        wins  = tr[tr > 0]
        losses= tr[tr < 0]
        if len(wins) and len(losses):
            win_rate = len(wins) / len(tr)
            avg_win  = wins.mean()
            avg_loss = abs(losses.mean())
            kelly    = win_rate - (1 - win_rate) / (avg_win / avg_loss)
            print(f"    Win rate  : {win_rate:.1%}  avg win={avg_win:+.3f}  avg loss={avg_loss:.3f}")
            print(f"    Full Kelly: {kelly:.1%}  Half Kelly: {kelly/2:.1%}")
            print(f"    → ${kelly/2*1000:.0f} per $1,000 account (half-Kelly)")

    # ── Risk metrics ──────────────────────────────────────────────────────────
    print(f"\n[8] Risk Metrics")
    if len(br) > 10:
        var95   = np.percentile(br, 5)
        neg_ret = br[br < 0]
        sortino = br.mean() / neg_ret.std() * ANNUALIZE if len(neg_ret) > 1 else np.inf
        consec_losses = 0; max_cl = 0; cur = 0
        for t in tr:
            cur = cur + 1 if t < 0 else 0
            max_cl = max(max_cl, cur)
        print(f"    VaR 95% (1-bar) : {var95:.4f}  ({var95*100:.2f}%)")
        print(f"    Sortino ratio   : {sortino:.3f}")
        print(f"    Max consec loss : {max_cl}")

    # ── Walk-Forward OOS detail ───────────────────────────────────────────────
    print(f"\n[9] Walk-Forward OOS Splits")
    cv = walk_forward_cv(df, params)
    cum_oos = 1.0
    for i, s in enumerate(cv["splits"], 1):
        tr_sh = f"{s['train_sharpe']:.2f}" if s['train_sharpe'] > -np.inf else "n/a"
        te_sh = f"{s['test_sharpe']:.2f}"  if s['test_sharpe']  > -np.inf else "n/a"
        print(f"    Split {i}: train_sharpe={tr_sh:>6}  "
              f"oos_sharpe={te_sh:>6}  oos_ret={s['test_ret']:+.2%}  "
              f"dd={s['test_dd']:.2%}  trades={s['test_trades']}")
        cum_oos *= (1 + s["test_ret"])
    print(f"    OOS stacked return : {cum_oos-1:+.2%}")
    print(f"    Mean OOS Sharpe    : {cv['mean_oos_sharpe']:.3f}")

    return res, cv


# ── Strategy 2 comparison baseline ───────────────────────────────────────────
def strategy2_baseline(df):
    """Quick re-run of ST+MACD winner for apples-to-apples comparison."""
    from strategy2 import compute_supertrend, compute_macd as s2_macd, add_features, backtest as s2_bt

    df2 = add_features(df.copy())
    res = s2_bt(df2, atr_period=7, mult=1.5, macd_fast=8, macd_slow=21,
                macd_sig=9, vol_min=0.0, fee_rt=ROUND_TRIP)
    bh  = df["close"].iloc[-1] / df["close"].iloc[0] - 1
    print(f"\n{'─'*65}")
    print(f"  BASELINE — Strategy 2 (ST ATR7/×1.5 + MACD 8,21,9)")
    print(f"  Total return: {res['total_return']:+.2%}  "
          f"Sharpe: {res['sharpe']:.2f}  "
          f"MaxDD: {res['max_dd']:.2%}  "
          f"Trades: {res['n_trades']}")
    print(f"  Buy & Hold : {bh:+.2%}")
    print(f"{'─'*65}")
    return res


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading 4h HYPE data...")
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  {len(df)} bars  {df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()}")

    # ── Baseline ──────────────────────────────────────────────────────────────
    strategy2_baseline(df)

    # ── Grid search ───────────────────────────────────────────────────────────
    best_params, best_cv, grid_df = grid_search(df)
    grid_df.to_csv("data/strategy3_grid.csv", index=False)

    print(f"\n{'='*65}")
    print(f"  WINNER: PSAR(start={PSAR_START}, step={best_params['psar_step']}, max={best_params['psar_max']})"
          f"  MACD({best_params['macd_fast']},{best_params['macd_slow']},{best_params['macd_sig']})")
    print(f"  Mean OOS Sharpe: {best_cv['mean_oos_sharpe']:.3f}")
    print(f"{'='*65}")

    # ── Full quant suite on winner ────────────────────────────────────────────
    label = (f"PSAR(step={best_params['psar_step']}, max={best_params['psar_max']})"
             f" + MACD({best_params['macd_fast']},{best_params['macd_slow']},{best_params['macd_sig']})")
    quant_suite(df, best_params, label)
