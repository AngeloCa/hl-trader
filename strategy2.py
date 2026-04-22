"""
Strategy 2: Supertrend + MACD Confirmation
===========================================
Supertrend is an ATR-based trend-following indicator that adapts to volatility.
It flips bullish/bearish when price closes on the opposite side of the band.

Signal logic:
  BUY  : Supertrend flips to BULLISH  (close > upper band)
           AND MACD histogram turns positive (momentum confirmation)
           AND volume ratio >= vol_min
  SELL : Supertrend flips to BEARISH  (close < lower band)
           OR MACD histogram turns negative  (early momentum exit)

Grid:
  atr_period  : [7, 10, 14]
  multiplier  : [1.5, 2.0, 2.5, 3.0, 3.5]
  macd_fast   : [8, 12]     (MACD fast EMA)
  macd_slow   : [21, 26]    (MACD slow EMA)
  macd_signal : [7, 9]      (MACD signal EMA)
  vol_min     : [0.0, 1.0]

Validation: same walk-forward CV + full quant suite as strategy 1.
Benchmark:  Momentum RSI B50/S55/MA5-20 (strategy 1 winner).
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
ROUND_TRIP   = 0.0016          # 0.05% taker + 0.03% slippage, each side
MIN_TRADES   = 5
TRAIN_BARS   = 720             # 120 days
TEST_BARS    = 360             # 60 days
STEP_BARS    = 360

ATR_PERIODS  = [7, 10, 14]
MULTIPLIERS  = [1.5, 2.0, 2.5, 3.0, 3.5]
MACD_FAST    = [8, 12]
MACD_SLOW    = [21, 26]
MACD_SIG     = [7, 9]
VOL_MINS     = [0.0, 1.0]


# ── Indicators ────────────────────────────────────────────────────────────────
def compute_atr(df: pd.DataFrame, period: int) -> np.ndarray:
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low,
         np.maximum(np.abs(high - prev_close),
                    np.abs(low  - prev_close)))
    atr = np.full(len(tr), np.nan)
    atr[period - 1] = tr[:period].mean()
    alpha = 1.0 / period
    for i in range(period, len(tr)):
        atr[i] = atr[i-1] * (1 - alpha) + tr[i] * alpha
    return atr


def compute_supertrend(df: pd.DataFrame, atr_period: int, mult: float):
    """Returns direction array: +1 bullish, -1 bearish."""
    close = df["close"].values
    hl2   = (df["high"].values + df["low"].values) / 2
    atr   = compute_atr(df, atr_period)

    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    n     = len(close)

    final_upper = upper.copy()
    final_lower = lower.copy()
    direction   = np.zeros(n, dtype=int)

    for i in range(1, n):
        if np.isnan(atr[i]):
            direction[i] = 0
            continue
        final_upper[i] = min(upper[i], final_upper[i-1]) if close[i-1] <= final_upper[i-1] else upper[i]
        final_lower[i] = max(lower[i], final_lower[i-1]) if close[i-1] >= final_lower[i-1] else lower[i]

        if direction[i-1] == -1:
            direction[i] = 1 if close[i] > final_upper[i-1] else -1
        else:
            direction[i] = -1 if close[i] < final_lower[i-1] else 1

    return direction, final_lower, final_upper


def compute_macd(close: np.ndarray, fast: int, slow: int, signal: int):
    def ema(x, p):
        out = np.full(len(x), np.nan)
        out[p-1] = x[:p].mean()
        alpha = 2 / (p + 1)
        for i in range(p, len(x)):
            out[i] = out[i-1] * (1 - alpha) + x[i] * alpha
        return out
    macd_line = ema(close, fast) - ema(close, slow)
    sig_line  = ema(np.where(np.isnan(macd_line), 0, macd_line), signal)
    histogram = macd_line - sig_line
    return histogram


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["vol_ma20"]  = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma20"]
    df["ma50"]      = df["close"].rolling(50).mean()
    return df


# ── Backtest ──────────────────────────────────────────────────────────────────
def backtest(
    df: pd.DataFrame,
    atr_period: int   = 10,
    mult: float       = 2.0,
    macd_fast: int    = 12,
    macd_slow: int    = 26,
    macd_sig: int     = 9,
    vol_min: float    = 0.0,
    fee_rt: float     = 0.0,
) -> dict:
    close     = df["close"].values
    vol_ratio = df["vol_ratio"].values
    n         = len(df)

    direction, _, _ = compute_supertrend(df, atr_period, mult)
    hist = compute_macd(close, macd_fast, macd_slow, macd_sig)

    position    = 0
    entry_price = 0.0
    entry_bar   = 0
    trades      = []
    equity      = np.empty(n); equity[0] = 1.0
    bars_in     = 0

    for i in range(1, n):
        p   = close[i]
        d   = direction[i]
        dp  = direction[i-1]
        h   = hist[i]
        hp  = hist[i-1]
        vr  = vol_ratio[i]

        st_flip_bull = (dp != 1) and (d == 1)     # supertrend turns bullish
        st_flip_bear = (dp != -1) and (d == -1)   # supertrend turns bearish
        macd_bull    = not np.isnan(h) and not np.isnan(hp) and (hp <= 0) and (h > 0)
        macd_bear    = not np.isnan(h) and not np.isnan(hp) and (hp >= 0) and (h < 0)
        vol_ok       = (vol_min == 0.0) or (not np.isnan(vr) and vr >= vol_min)

        # Entry: supertrend flips bullish AND MACD confirms (recent cross or already positive)
        macd_pos = not np.isnan(h) and (h > 0)
        buy_sig  = st_flip_bull and macd_pos and vol_ok

        # Exit: supertrend flips bearish OR MACD turns negative
        sell_sig = (position == 1) and (st_flip_bear or macd_bear)

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
                "exit_cause": "st_bear" if st_flip_bear else "macd_bear",
            })
            position = 0

        if position == 1:
            equity[i] = equity[i-1] * (p / close[i-1])
            bars_in  += 1
        else:
            equity[i] = equity[i-1]

    if position == 1:
        exit_p = close[-1] * (1 - fee_rt / 2)
        trades.append({"entry_bar": entry_bar, "exit_bar": n-1,
                       "entry": close[entry_bar], "exit": close[-1],
                       "return": (exit_p - entry_price) / entry_price,
                       "duration": n-1-entry_bar, "exit_cause": "eod"})

    bar_ret    = np.diff(equity) / equity[:-1]
    trade_rets = np.array([t["return"] for t in trades])
    peak       = np.maximum.accumulate(equity)
    max_dd     = float(((equity - peak) / peak).min())

    if len(trades) < MIN_TRADES or bar_ret.std() == 0:
        sharpe = -np.inf
    else:
        sharpe = bar_ret.mean() / bar_ret.std() * ANNUALIZE

    ann    = equity[-1] ** ((8760 / BAR_HOURS) / n) - 1
    calmar = ann / abs(max_dd) if max_dd != 0 else 0.0

    return {
        "trades":       trades,
        "trade_rets":   trade_rets,
        "equity":       equity,
        "bar_ret":      bar_ret,
        "sharpe":       sharpe,
        "ann_return":   ann,
        "total_return": equity[-1] - 1,
        "max_dd":       max_dd,
        "calmar":       calmar,
        "n_trades":     len(trades),
        "win_rate":     float(np.mean(trade_rets > 0)) if len(trade_rets) else 0.0,
        "avg_trade":    float(trade_rets.mean() * 100) if len(trade_rets) else 0.0,
        "in_market":    bars_in / n * 100,
    }


# ── Grid search ───────────────────────────────────────────────────────────────
def grid_search(df: pd.DataFrame, fee_rt: float = 0.0) -> pd.DataFrame:
    records = []
    for atr_p, mult, mf, ms, msig, vol in product(
        ATR_PERIODS, MULTIPLIERS, MACD_FAST, MACD_SLOW, MACD_SIG, VOL_MINS
    ):
        if mf >= ms:
            continue
        res = backtest(df, atr_p, mult, mf, ms, msig, vol, fee_rt)
        records.append({
            "atr_period": atr_p, "mult": mult,
            "macd_fast": mf, "macd_slow": ms, "macd_sig": msig, "vol_min": vol,
            **{k: v for k, v in res.items() if k not in ("trades","equity","bar_ret","trade_rets")},
        })
    return (pd.DataFrame(records)
            .replace([np.inf, -np.inf], np.nan)
            .sort_values("sharpe", ascending=False, na_position="last"))


def plabel(row) -> str:
    return (f"ST(ATR{int(row['atr_period'])}/x{row['mult']})"
            f"+MACD({int(row['macd_fast'])},{int(row['macd_slow'])},{int(row['macd_sig'])})"
            f"/V{row['vol_min']:.1f}")


# ── Walk-forward CV ───────────────────────────────────────────────────────────
def walk_forward_cv(df: pd.DataFrame) -> pd.DataFrame:
    n, rows, k = len(df), [], 0
    while True:
        train_end = TRAIN_BARS + k * STEP_BARS
        test_end  = train_end + TEST_BARS
        if test_end > n:
            break
        train_df = df.iloc[:train_end].reset_index(drop=True)
        test_df  = df.iloc[train_end:test_end].reset_index(drop=True)

        grid  = grid_search(train_df, 0.0)          # optimise without fee (cleaner signal)
        valid = grid.dropna(subset=["sharpe"])
        best  = valid.iloc[0] if not valid.empty else grid.iloc[0]

        oos = backtest(test_df,
                       int(best["atr_period"]), float(best["mult"]),
                       int(best["macd_fast"]),  int(best["macd_slow"]),
                       int(best["macd_sig"]),   float(best["vol_min"]),
                       ROUND_TRIP)

        bah_ret = test_df["close"].iloc[-1] / test_df["close"].iloc[0] - 1
        bah_eq  = test_df["close"].values / test_df["close"].values[0]
        bah_dd  = float(((bah_eq - np.maximum.accumulate(bah_eq)) / np.maximum.accumulate(bah_eq)).min())

        t0_tr = df["timestamp"].iloc[0].date()
        t1_tr = df["timestamp"].iloc[train_end - 1].date()
        t0_te = df["timestamp"].iloc[train_end].date()
        t1_te = df["timestamp"].iloc[test_end - 1].date()

        oos_sharpe = f"{oos['sharpe']:.3f}" if np.isfinite(oos['sharpe']) else "n/a"
        rows.append({
            "split":        k + 1,
            "train":        f"{t0_tr}→{t1_tr}",
            "test":         f"{t0_te}→{t1_te}",
            "params":       plabel(best),
            "tr_sharpe":    f"{best['sharpe']:.3f}" if pd.notna(best['sharpe']) else "n/a",
            "oos_sharpe":   oos_sharpe,
            "oos_calmar":   f"{oos['calmar']:.2f}",
            "oos_ret_%":    round(oos["total_return"] * 100, 2),
            "oos_dd_%":     round(oos["max_dd"] * 100, 2),
            "bah_ret_%":    round(bah_ret * 100, 2),
            "bah_dd_%":     round(bah_dd * 100, 2),
            "trades":       oos["n_trades"],
            "win_%":        round(oos["win_rate"] * 100, 1),
            "in_mkt_%":     round(oos["in_market"], 1),
        })
        k += 1

    return pd.DataFrame(rows)


# ── Quant validation ──────────────────────────────────────────────────────────
def sep(title="", w=68):
    if title:
        print(f"\n{'=' * w}\n  {title}\n{'=' * w}")
    else:
        print("─" * w)


def quant_validation(df_oos_all: pd.DataFrame, all_trades: list,
                     all_bar_rets: np.ndarray, best_params: dict):
    trade_rets = np.array([t["return"] for t in all_trades])

    # Fee sensitivity
    sep("FEE & SLIPPAGE SENSITIVITY")
    print(f"  {'Scenario':<34} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>7}")
    print("  " + "─" * 66)
    for label, rt in [("Zero cost", 0.0), ("Maker only (0.04%)", 0.0008),
                       ("Taker+slip (0.16%)", ROUND_TRIP), ("High cost (0.30%)", 0.006)]:
        r = backtest(df_oos_all, **best_params, fee_rt=rt)
        mk = " ◄" if rt == ROUND_TRIP else ""
        print(f"  {label:<34} {r['total_return']*100:>7.1f}%"
              f" {r['sharpe']:>8.3f} {r['max_dd']*100:>7.1f}% {r['n_trades']:>7}{mk}")

    # Statistical significance
    sep("STATISTICAL SIGNIFICANCE")
    n = len(trade_rets)
    if n >= 3:
        t_stat, p_val = stats.ttest_1samp(trade_rets, 0)
        mean, std = trade_rets.mean(), trade_rets.std(ddof=1)
        print(f"  Trades          : {n}")
        print(f"  Mean per trade  : {mean*100:+.3f}%")
        print(f"  Std dev         : {std*100:.3f}%")
        print(f"  t-stat / p-val  : {t_stat:.3f} / {p_val:.4f}  "
              f"{'✓ significant' if p_val < 0.05 else '✗ not significant at 5%'}")
        min_n = int(np.ceil(3.84 * (1 + 0.5*(mean/std)**2) / (mean/std)**2)) if std > 0 else 999
        print(f"  Min trades for 95% significance: {min_n}")

    # Bootstrap Sharpe
    sep("BOOTSTRAP SHARPE CI  (10,000 resamples)")
    bs = []
    for _ in range(10_000):
        s = np.random.choice(all_bar_rets, len(all_bar_rets), replace=True)
        if s.std() > 0:
            bs.append(s.mean() / s.std() * ANNUALIZE)
    bs = np.array(bs)
    actual_sh = all_bar_rets.mean() / all_bar_rets.std() * ANNUALIZE if all_bar_rets.std() > 0 else 0
    print(f"  Actual Sharpe   : {actual_sh:.3f}")
    print(f"  95% CI          : [{np.percentile(bs, 2.5):.3f},  {np.percentile(bs, 97.5):.3f}]")
    print(f"  P(Sharpe > 0)   : {(bs > 0).mean()*100:.1f}%")
    print(f"  P(Sharpe > 1)   : {(bs > 1).mean()*100:.1f}%")
    print(f"  P(Sharpe > 2)   : {(bs > 2).mean()*100:.1f}%")

    # Monte Carlo
    sep("MONTE CARLO  (10,000 shuffles)")
    frs, mds = [], []
    for _ in range(10_000):
        sh = np.random.permutation(trade_rets)
        eq = np.concatenate([[1.0], (1 + sh).cumprod()])
        pk = np.maximum.accumulate(eq)
        frs.append(eq[-1] - 1)
        mds.append(((eq - pk) / pk).min())
    frs, mds = np.array(frs), np.array(mds)
    print(f"  Actual return   : {(1+trade_rets).prod()-1:.1%}")
    print(f"  MC median       : {np.median(frs):.1%}")
    print(f"  MC 5th pct      : {np.percentile(frs, 5):.1%}")
    print(f"  MC 95th pct     : {np.percentile(frs, 95):.1%}")
    print(f"  P(profit)       : {(frs>0).mean()*100:.1f}%")
    print(f"  Median max DD   : {np.median(mds)*100:.1f}%")
    print(f"  Worst 5% DD     : {np.percentile(mds,5)*100:.1f}%")

    # Regime breakdown
    sep("REGIME BREAKDOWN")
    df_oos_all["slope"] = df_oos_all["close"].diff(20) / df_oos_all["close"].shift(20)
    def regime(row):
        if pd.isna(row.get("ma50")) or pd.isna(row.get("slope")):
            return "unknown"
        if row["close"] > row["ma50"] and row["slope"] > 0.05:
            return "bull"
        elif row["close"] < row["ma50"] and row["slope"] < -0.05:
            return "bear"
        return "sideways"
    df_oos_all["regime"] = df_oos_all.apply(regime, axis=1)
    rows = []
    for t in all_trades:
        eb = t["entry_bar"]
        if eb < len(df_oos_all):
            rows.append({"regime": df_oos_all["regime"].iloc[eb], "return": t["return"]})
    if rows:
        tdf = pd.DataFrame(rows)
        print(f"  {'Regime':<10} {'Trades':>7} {'Win%':>7} {'AvgRet':>9} {'TotalRet':>10}")
        print("  " + "─" * 48)
        for reg in ["bull", "sideways", "bear"]:
            sub = tdf[tdf["regime"] == reg]["return"]
            if sub.empty:
                print(f"  {reg:<10} {'0':>7}")
                continue
            print(f"  {reg:<10} {len(sub):>7} {(sub>0).mean()*100:>6.1f}%"
                  f" {sub.mean()*100:>8.2f}% {((1+sub).prod()-1)*100:>9.1f}%")

    # Kelly + risk
    sep("KELLY & RISK METRICS")
    if len(trade_rets) >= 5:
        wins   = trade_rets[trade_rets > 0]
        losses = trade_rets[trade_rets < 0]
        p  = len(wins) / len(trade_rets)
        b  = wins.mean() / abs(losses.mean()) if len(losses) else 10.0
        kelly = (p * b - (1-p)) / b
        print(f"  Win rate / b-ratio   : {p*100:.1f}% / {b:.2f}x")
        print(f"  Kelly fraction       : {kelly*100:.1f}%")
        print(f"  Half-Kelly (rec.)    : {kelly*50:.1f}%  → "
              f"${kelly*50*100:,.0f} per $10k capital")
        var95  = np.percentile(trade_rets, 5)
        cvar95 = trade_rets[trade_rets <= var95].mean()
        neg    = all_bar_rets[all_bar_rets < 0]
        sortino = all_bar_rets.mean() / neg.std() * ANNUALIZE if len(neg) > 1 else 0
        consec = 0; cur_c = 0
        for w in (trade_rets > 0):
            cur_c = 0 if w else cur_c + 1
            consec = max(consec, cur_c)
        print(f"  VaR 95% / CVaR 95%  : {var95*100:.2f}% / {cvar95*100:.2f}%")
        print(f"  Sortino ratio        : {sortino:.3f}")
        print(f"  Profit factor        : "
              f"{wins.sum()/abs(losses.sum()):.2f}x" if len(losses) else "  Profit factor: ∞")
        print(f"  Max consecutive loss : {consec}")
        print(f"  Avg hold time        : "
              f"{np.mean([t['duration'] for t in all_trades])*BAR_HOURS/24:.1f} days")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.dropna(subset=["rsi_14"]).reset_index(drop=True)
    df = add_features(df)
    bah = df["close"].iloc[-1] / df["close"].iloc[0] - 1

    print("=" * 68)
    print("  STRATEGY 2: Supertrend + MACD  |  HYPE/USDC 4h")
    print("=" * 68)
    print(f"  Dataset : {len(df)} bars · {(df['timestamp'].iloc[-1]-df['timestamp'].iloc[0]).days}d")
    print(f"  B&H     : {bah*100:+.1f}%  |  Friction: {ROUND_TRIP*100:.2f}% RT")

    # ── In-sample grid search ────────────────────────────────────────────────
    sep("IN-SAMPLE GRID SEARCH  (top 10, no fees)")
    full = grid_search(df, fee_rt=0.0)
    show = ["atr_period","mult","macd_fast","macd_slow","macd_sig","vol_min",
            "sharpe","calmar","total_return","max_dd","n_trades","win_rate","avg_trade","in_market"]
    disp = full[show].dropna(subset=["sharpe"]).head(10).copy()
    disp["sharpe"]       = disp["sharpe"].map(lambda x: f"{x:.3f}")
    disp["calmar"]       = disp["calmar"].map(lambda x: f"{x:.2f}")
    disp["total_return"] = disp["total_return"].map(lambda x: f"{x*100:+.1f}%")
    disp["max_dd"]       = disp["max_dd"].map(lambda x: f"{x*100:.1f}%")
    disp["win_rate"]     = disp["win_rate"].map(lambda x: f"{x*100:.0f}%")
    disp["avg_trade"]    = disp["avg_trade"].map(lambda x: f"{x:.2f}%")
    disp["in_market"]    = disp["in_market"].map(lambda x: f"{x:.0f}%")
    print(disp.to_string(index=False))
    full["param_label"] = full.apply(plabel, axis=1)
    full.to_csv("data/strategy2_grid.csv", index=False)

    # ── Walk-forward CV ──────────────────────────────────────────────────────
    sep(f"WALK-FORWARD CV  (train={TRAIN_BARS} bars={TRAIN_BARS*BAR_HOURS//24}d"
        f" · test={TEST_BARS} bars={TEST_BARS*BAR_HOURS//24}d · step={STEP_BARS})")
    cv = walk_forward_cv(df)
    print(cv.to_string(index=False))
    cv.to_csv("data/strategy2_cv.csv", index=False)

    # ── Collect OOS data ─────────────────────────────────────────────────────
    n_df = len(df)
    oos_dfs, oos_trades, oos_bar_rets = [], [], []
    k = 0
    while True:
        train_end = TRAIN_BARS + k * STEP_BARS
        test_end  = train_end + TEST_BARS
        if test_end > n_df:
            break
        best_row  = grid_search(df.iloc[:train_end].reset_index(drop=True)).dropna(subset=["sharpe"]).iloc[0]
        bp        = dict(atr_period=int(best_row["atr_period"]),
                         mult=float(best_row["mult"]),
                         macd_fast=int(best_row["macd_fast"]),
                         macd_slow=int(best_row["macd_slow"]),
                         macd_sig=int(best_row["macd_sig"]),
                         vol_min=float(best_row["vol_min"]))
        test_df   = df.iloc[train_end:test_end].reset_index(drop=True)
        r         = backtest(test_df, **bp, fee_rt=ROUND_TRIP)
        oos_dfs.append(test_df)
        oos_trades.extend(r["trades"])
        oos_bar_rets.extend(r["bar_ret"].tolist())
        k += 1

    oos_all = pd.concat(oos_dfs, ignore_index=True)

    # Best overall in-sample params (for sensitivity display)
    best_is = full.dropna(subset=["sharpe"]).iloc[0]
    best_params = dict(atr_period=int(best_is["atr_period"]),
                       mult=float(best_is["mult"]),
                       macd_fast=int(best_is["macd_fast"]),
                       macd_slow=int(best_is["macd_slow"]),
                       macd_sig=int(best_is["macd_sig"]),
                       vol_min=float(best_is["vol_min"]))

    # ── Full quant suite ─────────────────────────────────────────────────────
    sep("QUANT VALIDATION SUITE")
    print(f"  OOS: {len(oos_all)} bars · {len(oos_trades)} trades across {k} splits\n")
    quant_validation(oos_all, oos_trades, np.array(oos_bar_rets), best_params)

    # ── Summary + comparison with Strategy 1 ────────────────────────────────
    sep("SUMMARY — STRATEGY 2 vs STRATEGY 1 (momentum RSI)")
    strat_ret = (1 + cv["oos_ret_%"] / 100).prod() - 1
    bah_ret   = (1 + cv["bah_ret_%"] / 100).prod() - 1
    print(f"  {'Metric':<28} {'Strat 2 (ST+MACD)':>20} {'Strat 1 (Mom RSI)':>20}")
    print("  " + "─" * 68)
    tr = np.array([t["return"] for t in oos_trades])
    pval = stats.ttest_1samp(tr, 0).pvalue if len(tr) >= 3 else 1.0
    print(f"  {'Stacked OOS return':<28} {strat_ret*100:>19.1f}% {'747.6%':>20}")
    print(f"  {'Avg OOS Sharpe':<28} {pd.to_numeric(cv['oos_sharpe'], errors='coerce').mean():>20.3f} {'5.408':>20}")
    print(f"  {'Worst OOS drawdown':<28} {cv['oos_dd_%'].min():>19.1f}% {'-9.9%':>20}")
    print(f"  {'Total OOS trades':<28} {len(oos_trades):>20} {'34':>20}")
    print(f"  {'Avg return/trade':<28} {tr.mean()*100:>19.2f}% {'2.11%':>20}")
    print(f"  {'Win rate':<28} {(tr>0).mean()*100:>19.1f}% {'52.9%':>20}")
    print(f"  {'p-value (significance)':<28} {pval:>20.4f} {'0.1277':>20}")
    print(f"  {'B&H (same OOS windows)':<28} {bah_ret*100:>19.1f}% {'270.4%':>20}")
    print()


if __name__ == "__main__":
    main()
