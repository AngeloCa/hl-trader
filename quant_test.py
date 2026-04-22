"""
Professional Quant Validation Suite
====================================
Strategy : Momentum RSI  B50/S55/MA5-20
Data     : HYPE/USDC 4h bars (Dec 2024 → Apr 2026)

Tests performed:
  1. Fee & slippage sensitivity
  2. Hold-out test (last 60 days, never seen by optimizer)
  3. Statistical significance  (t-test + Sharpe SE)
  4. Bootstrap Sharpe CI        (10 000 resamples)
  5. Monte Carlo simulation     (10 000 trade-order shuffles)
  6. Parameter sensitivity      (neighbourhood grid)
  7. Regime breakdown           (bull / bear / sideways)
  8. Kelly criterion            (optimal position sizing)
  9. Risk metrics               (VaR, CVaR, consecutive losses)
"""

import numpy as np
import pandas as pd
from scipy import stats
from itertools import product
from pathlib import Path

np.random.seed(42)

# ── Strategy params ───────────────────────────────────────────────────────────
MODE     = "momentum"
RSI_BUY  = 50
RSI_SELL = 55
MA_FAST  = 5
MA_SLOW  = 20
VOL_MIN  = 0.0

# ── Market microstructure ────────────────────────────────────────────────────
TAKER_FEE  = 0.0005   # 0.05% per side (Hyperliquid spot taker)
SLIPPAGE   = 0.0003   # 0.03% per side (conservative estimate)
ROUND_TRIP = (TAKER_FEE + SLIPPAGE) * 2   # total friction per trade

BAR_HOURS  = 4
ANNUALIZE  = np.sqrt(8760 / BAR_HOURS)

# Walk-forward splits (must match optimize.py)
TRAIN_BARS = 720
TEST_BARS  = 360
STEP_BARS  = 360

DATA_PATH  = Path("data/HYPE_USDC_4h.csv")


# ── Feature engineering ───────────────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for p in [5, 10, 20, 50, 100]:
        df[f"ma_{p}"] = df["close"].rolling(p).mean()
    df["vol_ma20"]  = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma20"]
    sign = np.sign(df["close"].diff().fillna(0))
    df["obv"] = (sign * df["volume"]).cumsum()
    return df


# ── Core backtest (returns individual trades + equity) ────────────────────────
def run_strategy(df: pd.DataFrame, fee_rt: float = 0.0) -> dict:
    close  = df["close"].values
    rsi    = df["rsi_14"].values
    ma_f   = df[f"ma_{MA_FAST}"].values
    ma_s   = df[f"ma_{MA_SLOW}"].values
    n      = len(df)

    position = 0; entry_price = 0.0; entry_bar = 0
    trades   = []
    equity   = np.empty(n); equity[0] = 1.0
    bars_in  = 0

    for i in range(1, n):
        p, r       = close[i], rsi[i]
        mfv, msv   = ma_f[i], ma_s[i]
        mfp, msp   = ma_f[i-1], ma_s[i-1]
        nan_ma     = any(np.isnan(x) for x in [mfv, msv, mfp, msp])
        uptrend    = (not nan_ma) and (mfv > msv)
        death      = (not nan_ma) and (mfp >= msp) and (mfv < msv)
        rsi_up     = (rsi[i-1] < RSI_BUY)  and (r >= RSI_BUY)
        rsi_dn     = (rsi[i-1] > RSI_SELL) and (r <= RSI_SELL)

        if position == 0 and rsi_up and uptrend:
            position    = 1
            entry_price = p * (1 + fee_rt / 2)   # entry cost
            entry_bar   = i

        elif position == 1 and (rsi_dn or death):
            exit_price = p * (1 - fee_rt / 2)    # exit cost
            ret        = (exit_price - entry_price) / entry_price
            trades.append({
                "entry_bar":  entry_bar,
                "exit_bar":   i,
                "entry":      close[entry_bar],
                "exit":       close[i],
                "return":     ret,
                "duration":   i - entry_bar,
                "exit_cause": "rsi" if rsi_dn else "death_cross",
            })
            position = 0

        if position == 1:
            equity[i] = equity[i-1] * (p / close[i-1])
            bars_in  += 1
        else:
            equity[i] = equity[i-1]

    if position == 1:
        exit_price = close[-1] * (1 - fee_rt / 2)
        ret        = (exit_price - entry_price) / entry_price
        trades.append({"entry_bar": entry_bar, "exit_bar": n-1,
                       "entry": close[entry_bar], "exit": close[-1],
                       "return": ret, "duration": n-1-entry_bar,
                       "exit_cause": "eod"})

    bar_ret    = np.diff(equity) / equity[:-1]
    trade_rets = np.array([t["return"] for t in trades])
    peak       = np.maximum.accumulate(equity)
    max_dd     = float(((equity - peak) / peak).min())
    sharpe     = (bar_ret.mean() / bar_ret.std() * ANNUALIZE
                  if bar_ret.std() > 0 else 0.0)
    ann        = equity[-1] ** ((8760 / BAR_HOURS) / n) - 1

    return {
        "trades":       trades,
        "trade_rets":   trade_rets,
        "equity":       equity,
        "bar_ret":      bar_ret,
        "sharpe":       sharpe,
        "ann_return":   ann,
        "total_return": equity[-1] - 1,
        "max_dd":       max_dd,
        "calmar":       ann / abs(max_dd) if max_dd != 0 else 0,
        "n_trades":     len(trades),
        "win_rate":     float(np.mean(trade_rets > 0)) if len(trade_rets) else 0,
        "bars_in":      bars_in,
    }


def sep(title="", w=68):
    if title:
        print(f"\n{'=' * w}")
        print(f"  {title}")
        print('=' * w)
    else:
        print("─" * w)


# ── 1. Fee & slippage sensitivity ─────────────────────────────────────────────
def test_fees(df_oos: pd.DataFrame):
    sep("1. FEE & SLIPPAGE SENSITIVITY")
    scenarios = [
        ("Zero cost (ideal)",         0.0000),
        ("Maker 0.02% + slip 0.02%",  0.0008),
        ("Taker 0.05% + slip 0.03%",  0.0016),   # ← realistic (ROUND_TRIP)
        ("Taker 0.1%  + slip 0.05%",  0.0030),
        ("Worst case  0.2%  + 0.1%",  0.0060),
    ]
    print(f"  {'Scenario':<34} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>7}")
    print("  " + "─" * 66)
    for label, rt in scenarios:
        r = run_strategy(df_oos, rt)
        marker = " ◄ realistic" if rt == 0.0016 else ""
        print(f"  {label:<34} {r['total_return']*100:>7.1f}% "
              f"{r['sharpe']:>8.3f} {r['max_dd']*100:>7.1f}% "
              f"{r['n_trades']:>7}{marker}")


# ── 2. Hold-out test ──────────────────────────────────────────────────────────
def test_holdout(df: pd.DataFrame):
    sep("2. HOLD-OUT TEST  (last 60 days — never seen by optimizer)")
    holdout = df.iloc[-360:].reset_index(drop=True)
    bah_ret = (holdout["close"].iloc[-1] / holdout["close"].iloc[0] - 1) * 100
    bah_eq  = holdout["close"].values / holdout["close"].values[0]
    bah_pk  = np.maximum.accumulate(bah_eq)
    bah_dd  = ((bah_eq - bah_pk) / bah_pk).min() * 100

    r = run_strategy(holdout, ROUND_TRIP)
    print(f"  Period : {holdout['timestamp'].iloc[0].date()} → {holdout['timestamp'].iloc[-1].date()}")
    print(f"  {'':30} {'Strategy':>12} {'B&H':>12}")
    print("  " + "─" * 54)
    print(f"  {'Total return':<30} {r['total_return']*100:>11.2f}% {bah_ret:>11.2f}%")
    print(f"  {'Max drawdown':<30} {r['max_dd']*100:>11.2f}% {bah_dd:>11.2f}%")
    print(f"  {'Sharpe (annualised)':<30} {r['sharpe']:>12.3f} {'n/a':>12}")
    print(f"  {'Calmar':<30} {r['calmar']:>12.3f} {'n/a':>12}")
    print(f"  {'Trades':<30} {r['n_trades']:>12} {'1':>12}")
    if r["n_trades"]:
        tr = r["trade_rets"] * 100
        print(f"  {'Avg return/trade':<30} {tr.mean():>11.2f}%")
        print(f"  {'Win rate':<30} {r['win_rate']*100:>11.1f}%")


# ── 3. Statistical significance ───────────────────────────────────────────────
def test_significance(trade_rets: np.ndarray):
    sep("3. STATISTICAL SIGNIFICANCE")
    if len(trade_rets) < 5:
        print("  Insufficient trades."); return

    t_stat, p_val = stats.ttest_1samp(trade_rets, 0)
    n = len(trade_rets)
    mean, std = trade_rets.mean(), trade_rets.std(ddof=1)
    se_sharpe  = np.sqrt((1 + 0.5 * (mean/std)**2) / n)   # Lo (2002) approx
    sharpe_tr  = mean / std * np.sqrt(n)                    # per-trade Sharpe

    print(f"  Trades                : {n}")
    print(f"  Mean return/trade     : {mean*100:+.3f}%")
    print(f"  Std dev               : {std*100:.3f}%")
    print(f"  t-statistic           : {t_stat:.3f}")
    print(f"  p-value (H0: mean=0)  : {p_val:.4f}  {'✓ significant' if p_val < 0.05 else '✗ NOT significant at 5%'}")
    print(f"  Per-trade Sharpe      : {sharpe_tr:.3f}")
    print(f"  Sharpe std error      : ±{se_sharpe:.3f}")
    print(f"  Min trades for 95% CI : {int(np.ceil(3.84 * (1 + 0.5*(mean/std)**2) / (mean/std)**2))}")


# ── 4. Bootstrap Sharpe CI ────────────────────────────────────────────────────
def test_bootstrap(bar_ret: np.ndarray, n_boot: int = 10_000):
    sep(f"4. BOOTSTRAP SHARPE CI  ({n_boot:,} resamples)")
    bootstraps = []
    for _ in range(n_boot):
        sample = np.random.choice(bar_ret, size=len(bar_ret), replace=True)
        if sample.std() > 0:
            bootstraps.append(sample.mean() / sample.std() * ANNUALIZE)

    bs = np.array(bootstraps)
    actual_sharpe = bar_ret.mean() / bar_ret.std() * ANNUALIZE if bar_ret.std() > 0 else 0
    print(f"  Actual Sharpe         : {actual_sharpe:.3f}")
    print(f"  Bootstrap mean        : {bs.mean():.3f}")
    print(f"  95% CI                : [{np.percentile(bs, 2.5):.3f},  {np.percentile(bs, 97.5):.3f}]")
    print(f"  99% CI                : [{np.percentile(bs, 0.5):.3f},  {np.percentile(bs, 99.5):.3f}]")
    print(f"  P(Sharpe > 0)         : {(bs > 0).mean()*100:.1f}%")
    print(f"  P(Sharpe > 1)         : {(bs > 1).mean()*100:.1f}%")
    print(f"  P(Sharpe > 2)         : {(bs > 2).mean()*100:.1f}%")


# ── 5. Monte Carlo ────────────────────────────────────────────────────────────
def test_montecarlo(trade_rets: np.ndarray, n_sim: int = 10_000):
    sep(f"5. MONTE CARLO  ({n_sim:,} trade-order shuffles)")
    final_returns, max_dds = [], []
    for _ in range(n_sim):
        shuffled = np.random.permutation(trade_rets)
        eq       = (1 + shuffled).cumprod()
        peak     = np.maximum.accumulate(np.concatenate([[1.0], eq]))
        dd       = ((np.concatenate([[1.0], eq]) - peak) / peak).min()
        final_returns.append(eq[-1] - 1)
        max_dds.append(dd)

    fr, md = np.array(final_returns), np.array(max_dds)
    actual_ret = (1 + trade_rets).prod() - 1

    print(f"  Actual total return   : {actual_ret*100:+.1f}%")
    print(f"  MC median return      : {np.median(fr)*100:+.1f}%")
    print(f"  MC 5th  percentile    : {np.percentile(fr, 5)*100:+.1f}%")
    print(f"  MC 95th percentile    : {np.percentile(fr, 95)*100:+.1f}%")
    print(f"  P(positive return)    : {(fr > 0).mean()*100:.1f}%")
    print(f"  Median max drawdown   : {np.median(md)*100:.1f}%")
    print(f"  Worst 5% drawdown     : {np.percentile(md, 5)*100:.1f}%")


# ── helpers for sensitivity (avoids patching globals) ────────────────────────
def _run_with_params(df, fee_rt, buy, sell, maf, mas):
    close  = df["close"].values
    rsi    = df["rsi_14"].values
    ma_f   = df[f"ma_{maf}"].values
    ma_s   = df[f"ma_{mas}"].values
    n      = len(df)
    position = 0; entry_price = 0.0; trades = []; equity = np.empty(n); equity[0] = 1.0
    for i in range(1, n):
        p, r = close[i], rsi[i]
        mfv, msv, mfp, msp = ma_f[i], ma_s[i], ma_f[i-1], ma_s[i-1]
        nan_ma  = any(np.isnan(x) for x in [mfv, msv, mfp, msp])
        uptrend = (not nan_ma) and (mfv > msv)
        death   = (not nan_ma) and (mfp >= msp) and (mfv < msv)
        rsi_up  = (rsi[i-1] < buy)  and (r >= buy)
        rsi_dn  = (rsi[i-1] > sell) and (r <= sell)
        if position == 0 and rsi_up and uptrend:
            position = 1; entry_price = p * (1 + fee_rt / 2)
        elif position == 1 and (rsi_dn or death):
            trades.append((entry_price, p * (1 - fee_rt / 2)))
            position = 0
        equity[i] = equity[i-1] * (p / close[i-1]) if position == 1 else equity[i-1]
    if position == 1:
        trades.append((entry_price, close[-1] * (1 - fee_rt / 2)))
    bar_ret = np.diff(equity) / equity[:-1]
    trets   = np.array([(ex - en) / en for en, ex in trades]) if trades else np.array([0.0])
    peak    = np.maximum.accumulate(equity)
    max_dd  = float(((equity - peak) / peak).min())
    sharpe  = bar_ret.mean() / bar_ret.std() * ANNUALIZE if bar_ret.std() > 0 else 0.0
    return {"sharpe": sharpe, "total_return": equity[-1]-1, "max_dd": max_dd, "n_trades": len(trades)}


# ── 6. Parameter sensitivity ──────────────────────────────────────────────────
def test_sensitivity(df_oos: pd.DataFrame):
    sep("6. PARAMETER SENSITIVITY  (OOS neighbourhood)")
    base_buy, base_sell, base_maf, base_mas = RSI_BUY, RSI_SELL, MA_FAST, MA_SLOW
    print(f"  Base: RSI_BUY={base_buy} RSI_SELL={base_sell} MA_FAST={base_maf} MA_SLOW={base_mas}")
    print()

    base = (RSI_BUY, RSI_SELL, MA_FAST, MA_SLOW)

    results = []
    for buy, sell, maf, mas in product(
        [RSI_BUY - 5, RSI_BUY, RSI_BUY + 5],
        [RSI_SELL - 5, RSI_SELL, RSI_SELL + 5],
        [MA_FAST - 2, MA_FAST, MA_FAST + 3],
        [MA_SLOW - 5, MA_SLOW, MA_SLOW + 10],
    ):
        if buy >= sell or maf <= 0 or mas <= maf or buy <= 0 or sell > 100:
            continue
        df_tmp = df_oos.copy()
        for p in [maf, mas]:
            df_tmp[f"ma_{p}"] = df_tmp["close"].rolling(p).mean()
        r = _run_with_params(df_tmp, ROUND_TRIP, buy, sell, maf, mas)
        tag = " ◄ BASE" if (buy, sell, maf, mas) == base else ""
        results.append((buy, sell, maf, mas,
                        r["sharpe"], r["total_return"]*100,
                        r["max_dd"]*100, r["n_trades"], tag))

    results.sort(key=lambda x: -x[4])

    print(f"  {'buy':>4} {'sell':>4} {'maf':>4} {'mas':>4} "
          f"{'sharpe':>8} {'ret%':>8} {'dd%':>8} {'trades':>7}")
    print("  " + "─" * 60)
    for row in results[:15]:
        buy, sell, maf, mas, sh, ret, dd, nt, tag = row
        print(f"  {buy:>4} {sell:>4} {maf:>4} {mas:>4} "
              f"{sh:>8.3f} {ret:>7.1f}% {dd:>7.1f}% {nt:>7}{tag}")


# ── 7. Regime breakdown ───────────────────────────────────────────────────────
def test_regimes(df_oos: pd.DataFrame, trades: list):
    sep("7. REGIME BREAKDOWN")
    df = df_oos.copy()
    df["ma50"]  = df["close"].rolling(50).mean()
    df["ma100"] = df["close"].rolling(100).mean()
    df["slope"] = df["close"].diff(20) / df["close"].shift(20)  # 20-bar momentum

    # Classify each bar
    def regime(row):
        if pd.isna(row["ma50"]) or pd.isna(row["slope"]):
            return "unknown"
        if row["close"] > row["ma50"] and row["slope"] > 0.05:
            return "bull"
        elif row["close"] < row["ma50"] and row["slope"] < -0.05:
            return "bear"
        else:
            return "sideways"

    df["regime"] = df.apply(regime, axis=1)

    rows = []
    for t in trades:
        eb = t["entry_bar"]
        if eb >= len(df):
            continue
        reg = df["regime"].iloc[eb]
        rows.append({"regime": reg, "return": t["return"]})

    if not rows:
        print("  No trades to classify."); return

    tdf = pd.DataFrame(rows)
    print(f"  {'Regime':<12} {'Trades':>7} {'Win%':>7} {'AvgRet':>9} {'TotalRet':>10}")
    print("  " + "─" * 50)
    for reg in ["bull", "sideways", "bear"]:
        sub = tdf[tdf["regime"] == reg]["return"]
        if sub.empty:
            print(f"  {reg:<12} {'0':>7} {'—':>7} {'—':>9} {'—':>10}")
            continue
        wr  = (sub > 0).mean() * 100
        avg = sub.mean() * 100
        tot = ((1 + sub).prod() - 1) * 100
        print(f"  {reg:<12} {len(sub):>7} {wr:>6.1f}% {avg:>8.2f}% {tot:>9.1f}%")


# ── 8. Kelly criterion ────────────────────────────────────────────────────────
def test_kelly(trade_rets: np.ndarray):
    sep("8. KELLY CRITERION & POSITION SIZING")
    if len(trade_rets) < 5:
        print("  Insufficient trades."); return

    wins   = trade_rets[trade_rets > 0]
    losses = trade_rets[trade_rets < 0]
    p  = len(wins) / len(trade_rets)
    q  = 1 - p
    b  = wins.mean() / abs(losses.mean()) if len(losses) else np.inf

    kelly      = (p * b - q) / b if b != np.inf else p
    half_kelly = kelly / 2
    quarter_kelly = kelly / 4

    print(f"  Win rate (p)          : {p*100:.1f}%")
    print(f"  Avg win / avg loss (b): {b:.3f}x")
    print(f"  Full Kelly fraction   : {kelly*100:.1f}%  (theoretically optimal, very aggressive)")
    print(f"  Half Kelly            : {half_kelly*100:.1f}%  (recommended for live trading)")
    print(f"  Quarter Kelly         : {quarter_kelly*100:.1f}%  (conservative)")
    print()

    capital = 10_000
    print(f"  Position sizes on ${capital:,} capital:")
    print(f"    Full Kelly   → ${kelly * capital:,.0f} per trade")
    print(f"    Half Kelly   → ${half_kelly * capital:,.0f} per trade  ← recommended")
    print(f"    Quarter Kelly→ ${quarter_kelly * capital:,.0f} per trade")


# ── 9. Risk metrics ───────────────────────────────────────────────────────────
def test_risk(trade_rets: np.ndarray, bar_ret: np.ndarray):
    sep("9. RISK METRICS")
    if len(trade_rets) < 5:
        print("  Insufficient trades."); return

    # VaR / CVaR on per-trade returns
    var95  = np.percentile(trade_rets, 5)
    cvar95 = trade_rets[trade_rets <= var95].mean()
    var99  = np.percentile(trade_rets, 1)

    # Consecutive losses
    wins_seq = (trade_rets > 0).astype(int)
    max_consec_loss = 0; cur = 0
    for w in wins_seq:
        cur = 0 if w else cur + 1
        max_consec_loss = max(max_consec_loss, cur)

    print(f"  VaR 95% (per trade)   : {var95*100:.2f}%")
    print(f"  CVaR 95% (per trade)  : {cvar95*100:.2f}%  (expected loss in worst 5%)")
    print(f"  VaR 99% (per trade)   : {var99*100:.2f}%")
    print()
    print(f"  Max consecutive losses: {max_consec_loss}")
    print(f"  Avg bars between trades: {len(bar_ret) / max(len(trade_rets), 1):.0f} "
          f"({len(bar_ret) / max(len(trade_rets), 1) * BAR_HOURS / 24:.1f} days)")

    # Annualised vol
    ann_vol = bar_ret.std() * ANNUALIZE * 100
    print(f"  Annualised volatility : {ann_vol:.1f}%")

    # Sortino (downside std only)
    neg = bar_ret[bar_ret < 0]
    sortino = (bar_ret.mean() / neg.std() * ANNUALIZE) if len(neg) > 1 else 0
    print(f"  Sortino ratio         : {sortino:.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.dropna(subset=["rsi_14"]).reset_index(drop=True)
    df = add_features(df)

    n_days = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days
    print("=" * 68)
    print("  QUANT VALIDATION SUITE — Momentum RSI B50/S55/MA5-20")
    print("=" * 68)
    print(f"  Dataset  : {len(df)} bars · {n_days} days "
          f"({df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()})")
    print(f"  B&H      : {(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:+.1f}%")
    print(f"  Friction : {ROUND_TRIP*100:.2f}% round-trip (taker fee + slippage)")

    # Collect all OOS bars & trades across CV splits
    oos_bars, oos_trades, oos_bar_rets = [], [], []
    n, k = len(df), 0
    while True:
        train_end = TRAIN_BARS + k * STEP_BARS
        test_end  = train_end + TEST_BARS
        if test_end > n:
            break
        test_df = df.iloc[train_end:test_end].reset_index(drop=True)
        r       = run_strategy(test_df, ROUND_TRIP)
        oos_trades.extend(r["trades"])
        oos_bar_rets.extend(r["bar_ret"].tolist())
        oos_bars.append(test_df)
        k += 1

    oos_trade_rets = np.array([t["return"] for t in oos_trades])
    oos_bar_arr    = np.array(oos_bar_rets)
    oos_all        = pd.concat(oos_bars, ignore_index=True)

    print(f"  OOS total: {len(oos_all)} bars across {k} splits · {len(oos_trades)} trades")

    # Run all tests
    test_fees(oos_all)
    test_holdout(df)
    test_significance(oos_trade_rets)
    test_bootstrap(oos_bar_arr)
    test_montecarlo(oos_trade_rets)
    test_sensitivity(oos_all)
    test_regimes(oos_all, oos_trades)
    test_kelly(oos_trade_rets)
    test_risk(oos_trade_rets, oos_bar_arr)

    sep()
    print("VERDICT")
    sep()
    pval = stats.ttest_1samp(oos_trade_rets, 0).pvalue if len(oos_trade_rets) >= 5 else 1.0
    bs   = [np.random.choice(oos_bar_arr, len(oos_bar_arr), replace=True) for _ in range(5000)]
    bs_s = [s.mean()/s.std()*ANNUALIZE for s in bs if s.std() > 0]
    prob_pos_sharpe = (np.array(bs_s) > 0).mean()

    print(f"  Edge significance   : p={pval:.4f}  {'✓' if pval < 0.05 else '✗'}")
    print(f"  P(Sharpe > 0)       : {prob_pos_sharpe*100:.1f}%")
    print(f"  Profit factor       : {oos_trade_rets[oos_trade_rets>0].sum() / abs(oos_trade_rets[oos_trade_rets<0].sum()):.2f}x" if len(oos_trade_rets[oos_trade_rets<0]) else "  Profit factor : ∞")
    print(f"  Recommended sizing  : Half-Kelly")
    print()
    if pval < 0.05 and prob_pos_sharpe > 0.90:
        print("  ✓ STRATEGY PASSES quant validation.")
        print("    Edge is statistically detectable, Sharpe is robustly positive.")
        print("    Proceed with paper trading before live deployment.")
    else:
        print("  ✗ STRATEGY DOES NOT pass full quant validation.")
        print("    Collect more trades before going live.")


if __name__ == "__main__":
    main()
