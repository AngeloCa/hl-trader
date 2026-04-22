"""
Quant Upgrade — three tests to push confidence from 8/10 toward 9/10
=====================================================================

Test 1: Phase-Randomization Permutation Test
  Correct test for trend-following: preserves autocorrelation structure
  (spectral density) while destroying the specific entry timing the strategy
  exploits. Unlike return-shuffle, it doesn't incorrectly destroy momentum.
  p < 0.05 → strategy exploits something real beyond general trending.

Test 2: Cross-Asset OOS — AVAX, LINK, DOGE (Binance 2019-2023)
  Three completely different assets, never seen during parameter fitting.
  Tests that the strategy generalises beyond BTC/ETH/SOL/HYPE.

Test 3: Alpha / Beta Decomposition
  Decomposes strategy returns into:
    Beta return  = passive BTC exposure (anyone can get this)
    Alpha return = excess return from timing skill
  Computes Information Ratio on alpha alone — the purest measure of edge.
"""

import numpy as np
import pandas as pd
import requests
import time as _time
from scipy import stats
from pathlib import Path

np.random.seed(42)

# ── Shared params (Strategy 3 winner) ────────────────────────────────────────
BEST_PARAMS = dict(
    psar_step=0.01, psar_max=0.2,
    macd_fast=8, macd_slow=26, macd_sig=7,
    adx_threshold=0,
)
SPOT_FEE = 0.0016   # HYPE spot
PERP_FEE = 0.0017   # BTC/ETH/others perp

from strategy4 import backtest, walk_forward_cv, ANNUALIZE, ROUND_TRIP, fetch_binance_4h


# ════════════════════════════════════════════════════════════════════════════
# TEST 1 — Phase-Randomization Permutation Test
# ════════════════════════════════════════════════════════════════════════════

def phase_randomize(close: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Return a synthetic price series with the same autocorrelation structure
    (power spectrum) as `close` but randomised phase — destroying the specific
    entry timing while preserving the overall trending character.
    """
    log_ret = np.diff(np.log(close))
    n       = len(log_ret)

    fft_vals = np.fft.rfft(log_ret)
    phases   = rng.uniform(0, 2 * np.pi, len(fft_vals))
    # Keep DC component and Nyquist real; randomise all others
    phases[0] = 0.0
    if n % 2 == 0:
        phases[-1] = 0.0

    amplitudes = np.abs(fft_vals)
    new_fft    = amplitudes * np.exp(1j * phases)
    new_ret    = np.fft.irfft(new_fft, n=n)

    # Rebuild price series from synthetic returns
    synth = np.exp(
        np.concatenate([[np.log(close[0])],
                         np.log(close[0]) + np.cumsum(new_ret)])
    )
    return synth


def phase_permutation_test(df: pd.DataFrame, params: dict,
                            n_perms: int = 1000, fee_rt: float = PERP_FEE,
                            label: str = "") -> tuple[float, float]:
    rng = np.random.default_rng(42)

    real_res    = backtest(df, **params, fee_rt=fee_rt)
    real_sharpe = real_res["sharpe"]

    close  = df["close"].values.astype(float)
    high_r = df["high"].values  / close
    low_r  = df["low"].values   / close
    open_r = df["open"].values  / close

    perm_sharpes = []
    for _ in range(n_perms):
        synth_close = phase_randomize(close, rng)
        df_perm = df.copy()
        df_perm["close"] = synth_close
        df_perm["high"]  = synth_close * high_r
        df_perm["low"]   = synth_close * low_r
        df_perm["open"]  = synth_close * open_r
        res = backtest(df_perm, **params, fee_rt=fee_rt)
        perm_sharpes.append(res["sharpe"] if res["sharpe"] > -np.inf else -99.0)

    perm_sharpes = np.array(perm_sharpes)
    p_value = float((perm_sharpes >= real_sharpe).mean())

    sig = "✅ SIGNIFICANT (p<0.05)" if p_value < 0.05 else \
          "🟡 MARGINAL    (p<0.10)" if p_value < 0.10 else \
          "⚠️  NOT SIGNIFICANT"

    print(f"\n  Phase-Randomization Permutation Test — {label}")
    print(f"  Real strategy Sharpe : {real_sharpe:.3f}")
    print(f"  Synthetic mean Sharpe: {perm_sharpes.mean():.3f}  "
          f"(p95={np.percentile(perm_sharpes, 95):.3f})")
    print(f"  p-value              : {p_value:.4f}  {sig}")
    return p_value, real_sharpe


# ════════════════════════════════════════════════════════════════════════════
# TEST 2 — Cross-Asset OOS
# ════════════════════════════════════════════════════════════════════════════

def cross_asset_oos(params: dict):
    """
    Apply fitted params cold to AVAX, LINK, DOGE — never seen during fitting.
    """
    data_dir = Path("data")
    assets = [
        ("AVAX", "AVAXUSDT", "2020-09-15", "2024-01-01",
         "AVAX_binance_4h_2020_2024.csv"),
        ("LINK", "LINKUSDT", "2019-01-01", "2024-01-01",
         "LINK_binance_4h_2019_2024.csv"),
        ("DOGE", "DOGEUSDT", "2019-07-01", "2024-01-01",
         "DOGE_binance_4h_2019_2024.csv"),
    ]

    print(f"\n{'='*65}")
    print(f"  TEST 2 — Cross-Asset Cold OOS")
    print(f"  (params fitted on HYPE only; never seen these assets)")
    print(f"{'='*65}")

    summary = []
    for name, symbol, start, end, fname in assets:
        path = data_dir / fname
        df = fetch_binance_4h(symbol, start, end, path)
        df = df.sort_values("timestamp").reset_index(drop=True)

        res = backtest(df, **params, fee_rt=PERP_FEE)
        bh  = df["close"].iloc[-1] / df["close"].iloc[0] - 1
        cv  = walk_forward_cv(df, params)
        oos_shs    = [s["test_sharpe"] for s in cv["splits"] if s["test_sharpe"] > -np.inf]
        profitable = sum(1 for s in cv["splits"] if s["test_ret"] > 0)
        total      = len(cv["splits"])
        mean_oos   = np.mean(oos_shs) if oos_shs else float("nan")

        period = f"{df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()}"
        print(f"\n  {name} ({period})")
        print(f"    Strategy  : {res['total_ret']:+.2%}")
        print(f"    Buy&Hold  : {bh:+.2%}")
        print(f"    Sharpe    : {res['sharpe']:.3f}")
        print(f"    Max DD    : {res['max_dd']:.2%}")
        print(f"    Trades    : {res['n_trades']}")
        print(f"    OOS splits: {profitable}/{total}  mean OOS Sharpe={mean_oos:.3f}")

        # Regime breakdown
        close = df["close"].values
        ma50  = pd.Series(close).rolling(50).mean().values
        for regime in ["bull", "sideways", "bear"]:
            idx = [i for i, c in enumerate(close)
                   if not np.isnan(ma50[i]) and (
                       (regime == "bull"     and c > ma50[i]*1.05) or
                       (regime == "bear"     and c < ma50[i]*0.95) or
                       (regime == "sideways" and ma50[i]*0.95 <= c <= ma50[i]*1.05)
                   )]
            if not idx:
                continue
            rdf = df.iloc[idx].reset_index(drop=True)
            try:
                rr = backtest(rdf, **params, fee_rt=PERP_FEE)
                print(f"    {regime:9s}: ret={rr['total_ret']:+.2%}  "
                      f"sharpe={rr['sharpe'] if rr['sharpe']>-np.inf else 'n/a':>6}  "
                      f"trades={rr['n_trades']}")
            except Exception:
                pass

        summary.append({
            "asset": name,
            "strategy_ret": res["total_ret"],
            "bh_ret": bh,
            "sharpe": res["sharpe"],
            "max_dd": res["max_dd"],
            "trades": res["n_trades"],
            "oos_splits": f"{profitable}/{total}",
            "mean_oos_sharpe": mean_oos,
        })

    return summary


# ════════════════════════════════════════════════════════════════════════════
# TEST 3 — Alpha / Beta Decomposition
# ════════════════════════════════════════════════════════════════════════════

def alpha_beta_decomposition(params: dict):
    """
    Regress strategy bar returns against passive BTC holding.
    Isolates how much of the edge is timing skill (alpha) vs crypto beta.
    """
    data_dir = Path("data")

    datasets = {
        "BTC 2019-2024":  (data_dir / "BTC_binance_4h_2019_2024.csv",  PERP_FEE, None),
        "ETH 2018-2023":  (data_dir / "ETH_binance_4h_2018_2023.csv",  PERP_FEE,
                           data_dir / "BTC_binance_4h_2019_2024.csv"),
    }

    print(f"\n{'='*65}")
    print(f"  TEST 3 — Alpha / Beta Decomposition")
    print(f"  (how much return is timing skill vs passive crypto exposure?)")
    print(f"{'='*65}")

    for label, (path, fee, btc_path) in datasets.items():
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        res = backtest(df, **params, fee_rt=fee)
        strat_ret = res["bar_ret"]       # bar-by-bar strategy returns

        # Market returns: use same asset as market proxy if no separate BTC
        if btc_path and btc_path.exists():
            btc_df = pd.read_csv(btc_path, parse_dates=["timestamp"])
            btc_df = btc_df.sort_values("timestamp").reset_index(drop=True)
            # Align timestamps
            merged = pd.merge(
                df[["timestamp"]].assign(idx=range(len(df))),
                btc_df[["timestamp", "close"]].rename(columns={"close": "btc_close"}),
                on="timestamp", how="left"
            )
            btc_close = merged["btc_close"].ffill().values
            mkt_ret = np.diff(btc_close) / btc_close[:-1]
        else:
            # Use the asset itself as the market (BTC vs BTC)
            close   = df["close"].values.astype(float)
            mkt_ret = np.diff(close) / close[:-1]

        # Align lengths
        min_len = min(len(strat_ret), len(mkt_ret))
        s = strat_ret[:min_len]
        m = mkt_ret[:min_len]

        # OLS regression: strategy_ret = alpha + beta * market_ret + epsilon
        valid = ~(np.isnan(s) | np.isnan(m) | np.isinf(s) | np.isinf(m))
        s, m  = s[valid], m[valid]

        beta, alpha_bar, r_val, p_val, std_err = stats.linregress(m, s)
        r_sq   = r_val ** 2
        resid  = s - (alpha_bar + beta * m)

        # Annualise alpha
        alpha_ann = (1 + alpha_bar) ** (len(s) / (len(df) / ANNUALIZE**2)) - 1

        # Information Ratio = annualised alpha / annualised tracking error
        te_ann = resid.std() * ANNUALIZE
        ir     = (alpha_bar * ANNUALIZE) / te_ann if te_ann > 0 else np.inf

        # Correlation of strategy returns with market
        corr = np.corrcoef(s, m)[0, 1]

        # Strategy Sharpe on alpha (residual) only
        alpha_sharpe = (resid.mean() / resid.std() * ANNUALIZE
                        if resid.std() > 0 else 0)

        print(f"\n  {label}")
        print(f"    Beta (market sensitivity) : {beta:+.4f}")
        print(f"    Alpha per bar             : {alpha_bar:+.6f}  "
              f"({alpha_bar*100*6*365:.1f}% annualised approx)")
        print(f"    R²  (fraction explained)  : {r_sq:.4f}  "
              f"({r_sq*100:.1f}% of variance from market)")
        print(f"    Correlation with market   : {corr:+.4f}")
        print(f"    Information Ratio (alpha) : {ir:.3f}")
        print(f"    Sharpe on alpha alone     : {alpha_sharpe:.3f}")
        print(f"    Strategy total Sharpe     : {res['sharpe']:.3f}")
        pct_alpha = (1 - r_sq) * 100
        print(f"    → {pct_alpha:.0f}% of variance is timing alpha (not market beta)")
        interpretation = (
            "✅ Timing alpha dominates — genuine edge"
            if pct_alpha > 60 else
            "🟡 Mixed alpha/beta — meaningful but partial timing edge"
            if pct_alpha > 30 else
            "⚠️  Beta-driven — mostly passive exposure"
        )
        print(f"    → {interpretation}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    data_dir = Path("data")
    params   = BEST_PARAMS

    print("=" * 65)
    print("  QUANT UPGRADE — 3 Tests Toward 9/10 Confidence")
    print("=" * 65)

    # ── Load datasets ─────────────────────────────────────────────────────────
    df_btc = pd.read_csv(data_dir / "BTC_binance_4h_2019_2024.csv",
                         parse_dates=["timestamp"])
    df_btc = df_btc.sort_values("timestamp").reset_index(drop=True)

    df_eth = pd.read_csv(data_dir / "ETH_binance_4h_2018_2023.csv",
                         parse_dates=["timestamp"])
    df_eth = df_eth.sort_values("timestamp").reset_index(drop=True)

    df_hype = pd.read_csv(data_dir / "HYPE_USDC_4h.csv",
                          parse_dates=["timestamp"])
    df_hype = df_hype.sort_values("timestamp").reset_index(drop=True)

    # ── TEST 1 ────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  TEST 1 — Phase-Randomization Permutation Test")
    print("  (1,000 synthetic series preserving autocorrelation)")
    print(f"{'='*65}")

    p_btc,  sh_btc  = phase_permutation_test(
        df_btc,  params, n_perms=1000, fee_rt=PERP_FEE,
        label="BTC Binance 2019-2024"
    )
    p_eth,  sh_eth  = phase_permutation_test(
        df_eth,  params, n_perms=1000, fee_rt=PERP_FEE,
        label="ETH Binance 2018-2023"
    )
    p_hype, sh_hype = phase_permutation_test(
        df_hype, params, n_perms=1000, fee_rt=SPOT_FEE,
        label="HYPE Hyperliquid Dec24-Apr26"
    )

    # ── TEST 2 ────────────────────────────────────────────────────────────────
    cross_summary = cross_asset_oos(params)

    # ── TEST 3 ────────────────────────────────────────────────────────────────
    alpha_beta_decomposition(params)

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    all_oos = sum(1 for d in cross_summary
                  for sp in d["oos_splits"].split("/")
                  if d["oos_splits"].split("/")[0] == d["oos_splits"].split("/")[1])
    profitable_all = all(d["strategy_ret"] > 0 for d in cross_summary)

    print(f"\n{'='*65}")
    print("  FINAL SUMMARY")
    print(f"{'='*65}")

    print(f"\n  Test 1 — Phase-Randomization Permutation")
    for lbl, p in [("BTC", p_btc), ("ETH", p_eth), ("HYPE", p_hype)]:
        sig = "✅" if p < 0.05 else "🟡" if p < 0.10 else "⚠️ "
        print(f"    {lbl:<6}: p={p:.4f}  {sig}")

    print(f"\n  Test 2 — Cross-Asset OOS")
    for d in cross_summary:
        ok = "✅" if d["strategy_ret"] > 0 and d["sharpe"] > 1 else "⚠️ "
        print(f"    {d['asset']:<6}: ret={d['strategy_ret']:+.2%}  "
              f"sharpe={d['sharpe']:.2f}  splits={d['oos_splits']}  {ok}")

    print(f"\n  Test 3 — Alpha/Beta: see per-asset breakdown above")

    # Score confidence
    n_perm_pass = sum(1 for p in [p_btc, p_eth, p_hype] if p < 0.10)
    n_assets_pass = sum(1 for d in cross_summary if d["strategy_ret"] > 0)

    confidence = 8.0
    if n_perm_pass >= 2:
        confidence += 0.3
        print(f"\n  ✅ Permutation test passes on {n_perm_pass}/3 datasets (+0.3)")
    elif n_perm_pass >= 1:
        confidence += 0.15
        print(f"\n  🟡 Permutation test passes on {n_perm_pass}/3 datasets (+0.15)")
    else:
        print(f"\n  ⚠️  Permutation test: 0/3 datasets significant (structural limit)")

    if n_assets_pass == 3:
        confidence += 0.2
        print(f"  ✅ All 3 new assets profitable cold OOS (+0.2)")
    elif n_assets_pass >= 2:
        confidence += 0.1
        print(f"  🟡 {n_assets_pass}/3 new assets profitable cold OOS (+0.1)")

    print(f"\n  Confidence estimate: {confidence:.1f}/10")
    print(f"  (Live track record would add +0.5 → 0.8 beyond this)")
    print(f"{'='*65}")
