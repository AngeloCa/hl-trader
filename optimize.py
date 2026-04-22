"""
RSI + Trend-Following Hybrid Optimizer with Walk-Forward CV
===========================================================
Three strategy modes compared in a unified grid search:

  "rsi"     — pure mean-reversion
              BUY:  RSI < rsi_buy
              SELL: RSI > rsi_sell

  "trend"   — pure trend-following (MA crossover)
              BUY:  MA_fast crosses above MA_slow  (golden cross)
              SELL: MA_fast crosses below MA_slow  (death cross)

  "hybrid"  — trend filter + RSI dip entry
              BUY:  MA_fast > MA_slow  AND  RSI < rsi_buy
              SELL: RSI > rsi_sell  OR  MA_fast < MA_slow

Volume confirmation (optional): require vol_ratio >= vol_min on entry.

Grid:
  rsi_buy   : [25, 30, 35, 40, 45, 50]
  rsi_sell  : [60, 65, 70, 75, 80]
  ma_fast   : [10, 20]
  ma_slow   : [50, 100]
  vol_min   : [0.0, 1.0, 1.5]
  mode      : ["rsi", "trend", "hybrid"]

Objective : Sharpe (annualised), min MIN_TRADES per window.
CV        : expanding walk-forward.
"""

import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = Path("data/HYPE_USDC_4h.csv")
RSI_COL     = "rsi_14"
BAR_HOURS   = 4                    # 4h bars
ANNUALIZE   = np.sqrt(8760 / BAR_HOURS)   # bars → annual Sharpe

BUY_LEVELS  = [30, 35, 40, 45, 50]
SELL_LEVELS = [55, 60, 65, 70, 75]
MA_FAST     = [5, 10, 20]          # faster → more crossovers per window
MA_SLOW     = [20, 50]
VOL_MINS    = [0.0, 1.0, 1.5]
# "rsi"    : buy RSI dip, sell RSI peak (mean-reversion)
# "trend"  : MA golden/death cross (trend-following)
# "hybrid" : uptrend filter (MA_fast > MA_slow) + RSI dip entry + double exit
# "momentum": RSI crosses above rsi_buy (momentum) while above MA; exit on RSI cross below rsi_sell
MODES       = ["rsi", "trend", "hybrid", "momentum"]

MIN_TRADES  = 5

# Walk-forward: 120-day train, 60-day test, 60-day step → ~6 splits on 500d dataset
TRAIN_BARS = 720    # 120 days
TEST_BARS  = 360    # 60 days
STEP_BARS  = 360


# ── Feature engineering ───────────────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for p in [5, 10, 20, 50, 100]:
        df[f"ma_{p}"] = df["close"].rolling(p).mean()
    df["vol_ma20"]  = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma20"]
    sign = np.sign(df["close"].diff().fillna(0))
    df["obv"]       = (sign * df["volume"]).cumsum()
    df["obv_ma20"]  = df["obv"].rolling(20).mean()
    return df


# ── Backtest ──────────────────────────────────────────────────────────────────
def backtest(
    df: pd.DataFrame,
    mode: str,
    rsi_buy: float  = 40,
    rsi_sell: float = 70,
    maf: int        = 10,
    mas: int        = 50,
    vol_min: float  = 0.0,
) -> dict:
    close     = df["close"].values
    rsi       = df[RSI_COL].values
    ma_f      = df[f"ma_{maf}"].values
    ma_s      = df[f"ma_{mas}"].values
    vol_ratio = df["vol_ratio"].values
    n         = len(df)

    position    = 0
    entry_price = 0.0
    trades      = []
    equity      = np.empty(n)
    equity[0]   = 1.0
    bars_in     = 0

    for i in range(1, n):
        p   = close[i]
        r   = rsi[i]
        mfv = ma_f[i]
        msv = ma_s[i]
        mfp = ma_f[i - 1]
        msp = ma_s[i - 1]
        vr  = vol_ratio[i]

        nan_ma  = np.isnan(mfv) or np.isnan(msv) or np.isnan(mfp) or np.isnan(msp)
        uptrend = (not nan_ma) and (mfv > msv)
        golden  = (not nan_ma) and (mfp <= msp) and (mfv > msv)   # fresh crossover
        death   = (not nan_ma) and (mfp >= msp) and (mfv < msv)

        vol_ok  = (vol_min == 0.0) or (not np.isnan(vr) and vr >= vol_min)

        # Previous bar RSI for crossover detection
        rsi_prev = rsi[i - 1]

        # ── Entry logic ────────────────────────────────────────────────────
        buy_signal = False
        if mode == "rsi":
            # Mean-reversion: enter on oversold dip
            buy_signal = (r < rsi_buy) and vol_ok
        elif mode == "trend":
            # Pure trend: enter on MA golden cross
            buy_signal = golden and vol_ok
        elif mode == "hybrid":
            # Trend filter + RSI dip: buy the dip only while in uptrend
            buy_signal = uptrend and (r < rsi_buy) and vol_ok
        elif mode == "momentum":
            # RSI momentum cross: RSI crosses UP through rsi_buy while above MA
            rsi_cross_up = (rsi_prev < rsi_buy) and (r >= rsi_buy)
            buy_signal = rsi_cross_up and uptrend and vol_ok

        # ── Exit logic ─────────────────────────────────────────────────────
        sell_signal = False
        if position == 1:
            if mode == "rsi":
                sell_signal = r > rsi_sell
            elif mode == "trend":
                sell_signal = death
            elif mode == "hybrid":
                sell_signal = (r > rsi_sell) or death
            elif mode == "momentum":
                # Exit when RSI crosses DOWN through rsi_sell, or trend reverses
                rsi_cross_dn = (rsi_prev > rsi_sell) and (r <= rsi_sell)
                sell_signal = rsi_cross_dn or death

        if position == 0 and buy_signal:
            position    = 1
            entry_price = p
        elif position == 1 and sell_signal:
            trades.append((entry_price, p))
            position = 0

        if position == 1:
            equity[i] = equity[i - 1] * (p / close[i - 1])
            bars_in  += 1
        else:
            equity[i] = equity[i - 1]

    if position == 1:
        trades.append((entry_price, close[-1]))
        bars_in += 1

    bar_ret    = np.diff(equity) / equity[:-1]
    trade_rets = np.array([(ex - en) / en for en, ex in trades]) if trades else np.array([])

    if len(trades) < MIN_TRADES or bar_ret.std() == 0:
        sharpe = -np.inf
    else:
        sharpe = bar_ret.mean() / bar_ret.std() * ANNUALIZE

    peak   = np.maximum.accumulate(equity)
    max_dd = float(((equity - peak) / peak).min())
    ann    = equity[-1] ** ((8760 / BAR_HOURS) / n) - 1
    calmar = ann / abs(max_dd) if max_dd != 0 else 0.0

    return {
        "sharpe":        sharpe,
        "calmar":        calmar,
        "total_return":  equity[-1] - 1,
        "ann_return":    ann,
        "max_drawdown":  max_dd,
        "num_trades":    len(trades),
        "win_rate":      float(np.mean(trade_rets > 0)) if len(trade_rets) else 0.0,
        "avg_trade_pct": float(trade_rets.mean() * 100) if len(trade_rets) else 0.0,
        "in_market_pct": bars_in / n * 100,
        "equity":        equity,
    }


# ── Grid search ───────────────────────────────────────────────────────────────
def grid_search(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for mode, buy, sell, maf, mas, vol in product(
        MODES, BUY_LEVELS, SELL_LEVELS, MA_FAST, MA_SLOW, VOL_MINS
    ):
        if mode != "trend" and buy >= sell:
            continue
        if maf >= mas:
            continue
        res = backtest(df, mode, buy, sell, maf, mas, vol)
        records.append({
            "mode": mode, "rsi_buy": buy, "rsi_sell": sell,
            "ma_fast": maf, "ma_slow": mas, "vol_min": vol,
            **{k: v for k, v in res.items() if k != "equity"},
        })
    return (
        pd.DataFrame(records)
        .replace([np.inf, -np.inf], np.nan)
        .sort_values("sharpe", ascending=False, na_position="last")
    )


# ── Helpers ───────────────────────────────────────────────────────────────────
def bah(df):
    return df["close"].iloc[-1] / df["close"].iloc[0] - 1

def bah_dd(df):
    eq = df["close"].values / df["close"].values[0]
    pk = np.maximum.accumulate(eq)
    return float(((eq - pk) / pk).min())

def fmt(v, d=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{d}f}"

def plabel(row):
    mode = row["mode"]
    if mode == "trend":
        return f"trend/MA{int(row['ma_fast'])}-{int(row['ma_slow'])}/V{row['vol_min']:.1f}"
    return (f"{mode}/B{int(row['rsi_buy'])}/S{int(row['rsi_sell'])}"
            f"/MA{int(row['ma_fast'])}-{int(row['ma_slow'])}/V{row['vol_min']:.1f}")


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

        grid  = grid_search(train_df)
        valid = grid.dropna(subset=["sharpe"])
        best  = valid.iloc[0] if not valid.empty else grid.sort_values("num_trades", ascending=False).iloc[0]

        oos = backtest(
            test_df,
            best["mode"],
            float(best["rsi_buy"]), float(best["rsi_sell"]),
            int(best["ma_fast"]),   int(best["ma_slow"]),
            float(best["vol_min"]),
        )

        t0_tr = df["timestamp"].iloc[0].date()
        t1_tr = df["timestamp"].iloc[train_end - 1].date()
        t0_te = df["timestamp"].iloc[train_end].date()
        t1_te = df["timestamp"].iloc[test_end - 1].date()

        rows.append({
            "split":        k + 1,
            "train":        f"{t0_tr}→{t1_tr}",
            "test":         f"{t0_te}→{t1_te}",
            "params":       plabel(best),
            "tr_sharpe":    fmt(best["sharpe"]),
            "oos_sharpe":   fmt(oos["sharpe"]),
            "oos_calmar":   fmt(oos["calmar"]),
            "oos_ret_%":    round(oos["total_return"] * 100, 2),
            "oos_dd_%":     round(oos["max_drawdown"] * 100, 2),
            "bah_ret_%":    round(bah(test_df) * 100, 2),
            "bah_dd_%":     round(bah_dd(test_df) * 100, 2),
            "trades":       oos["num_trades"],
            "win_%":        round(oos["win_rate"] * 100, 1),
            "in_mkt_%":     round(oos["in_market_pct"], 1),
        })
        k += 1

    return pd.DataFrame(rows)


# ── Mode comparison ───────────────────────────────────────────────────────────
def mode_comparison(full: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode in MODES:
        sub = full[full["mode"] == mode].dropna(subset=["sharpe"])
        if sub.empty:
            continue
        best = sub.iloc[0]
        rows.append({
            "mode":       mode,
            "best_config": plabel(best),
            "sharpe":     round(best["sharpe"], 3),
            "calmar":     round(best["calmar"], 2),
            "return_%":   round(best["total_return"] * 100, 1),
            "max_dd_%":   round(best["max_drawdown"] * 100, 1),
            "trades":     int(best["num_trades"]),
            "win_%":      round(best["win_rate"] * 100, 0),
            "in_mkt_%":   round(best["in_market_pct"], 0),
        })
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.dropna(subset=[RSI_COL]).reset_index(drop=True)
    df = add_features(df)

    n_days = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days
    print(f"Dataset : {len(df)} hourly bars  ({df['timestamp'].iloc[0].date()} → "
          f"{df['timestamp'].iloc[-1].date()})  [{n_days} days]")
    print(f"Price   : {df['close'].iloc[0]:.3f} → {df['close'].iloc[-1]:.3f}  "
          f"(B&H {bah(df)*100:+.1f}%,  MaxDD {bah_dd(df)*100:.2f}%)\n")

    # ── 1. Full-sample grid ──────────────────────────────────────────────────
    print("=" * 76)
    print(f"IN-SAMPLE GRID SEARCH  (MIN_TRADES={MIN_TRADES})")
    print("=" * 76)
    full = grid_search(df)

    # Top 5 per mode
    print("\nTop 5 per strategy mode (sorted by Sharpe):")
    show = ["mode", "rsi_buy", "rsi_sell", "ma_fast", "ma_slow", "vol_min",
            "sharpe", "calmar", "total_return", "max_drawdown", "num_trades", "win_rate", "in_market_pct"]
    for mode in MODES:
        sub = full[full["mode"] == mode].dropna(subset=["sharpe"]).head(5)[show].copy()
        if sub.empty:
            print(f"\n  [{mode.upper()}] — no valid params")
            continue
        sub["sharpe"]       = sub["sharpe"].map(lambda x: f"{x:.3f}")
        sub["calmar"]       = sub["calmar"].map(lambda x: f"{x:.2f}")
        sub["total_return"] = sub["total_return"].map(lambda x: f"{x*100:+.1f}%")
        sub["max_drawdown"] = sub["max_drawdown"].map(lambda x: f"{x*100:.1f}%")
        sub["win_rate"]     = sub["win_rate"].map(lambda x: f"{x*100:.0f}%")
        sub["in_market_pct"]= sub["in_market_pct"].map(lambda x: f"{x:.0f}%")
        print(f"\n  [{mode.upper()}]")
        print(sub.to_string(index=False))

    # ── 2. Mode comparison ───────────────────────────────────────────────────
    print(f"\n{'─' * 76}")
    print("BEST-OF-MODE COMPARISON  (in-sample)")
    print("─" * 76)
    mc = mode_comparison(full)
    print(mc.to_string(index=False))

    full["param_label"] = full.apply(plabel, axis=1)
    full.to_csv("data/grid_search_full.csv", index=False)

    # ── 3. Walk-forward CV ───────────────────────────────────────────────────
    print(f"\n{'=' * 76}")
    print(f"WALK-FORWARD CV  (train={TRAIN_BARS} bars={TRAIN_BARS*BAR_HOURS//24}d · "
          f"test={TEST_BARS} bars={TEST_BARS*BAR_HOURS//24}d · step={STEP_BARS} bars)")
    print("=" * 76)
    cv = walk_forward_cv(df)
    print(cv.to_string(index=False))
    cv.to_csv("data/cv_results.csv", index=False)

    # ── 4. Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 76}")
    print("SUMMARY")
    print("=" * 76)
    strat_ret  = (1 + cv["oos_ret_%"] / 100).prod() - 1
    bah_ret_cv = (1 + cv["bah_ret_%"]  / 100).prod() - 1
    worst_dd   = cv["oos_dd_%"].min()
    bah_wdd    = cv["bah_dd_%"].min()

    sharpe_n = pd.to_numeric(cv["oos_sharpe"], errors="coerce")
    calmar_n = pd.to_numeric(cv["oos_calmar"], errors="coerce")

    print(f"  Stacked OOS return  : {strat_ret*100:+.2f}%  vs B&H {bah_ret_cv*100:+.2f}%")
    print(f"  Avg OOS Sharpe      : {fmt(sharpe_n.mean())}")
    print(f"  Avg OOS Calmar      : {fmt(calmar_n.mean())}")
    print(f"  Worst OOS drawdown  : {worst_dd:.2f}%  vs B&H {bah_wdd:.2f}%")
    print(f"  Avg trades/window   : {cv['trades'].mean():.1f}")
    print(f"  Avg win rate        : {cv['win_%'].mean():.1f}%")
    print(f"  Avg time in market  : {cv['in_mkt_%'].mean():.1f}%")
    print(f"\n  Full period B&H     : {bah(df)*100:+.2f}%  /  MaxDD {bah_dd(df)*100:.2f}%")

    print(f"\n  Param frequency across CV splits:")
    freq = cv["params"].value_counts().reset_index()
    freq.columns = ["params", "count"]
    print(freq.to_string(index=False))

    print(f"\nOutputs → data/grid_search_full.csv  data/cv_results.csv")


if __name__ == "__main__":
    main()
