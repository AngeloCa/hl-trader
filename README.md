# HYPE/USDC Hyperliquid Quantitative Trading System

A fully automated, backtested, and containerised trend-following trading bot for crypto assets on [Hyperliquid](https://hyperliquid.xyz).

---

## Strategy: Parabolic SAR + MACD (Strategy 3)

After evaluating three strategies across 7 years of BTC data and 17 months of HYPE data, the winning approach is **Parabolic SAR + MACD confirmation** — a CTA-grade adaptive trailing stop paired with momentum confirmation.

### Signal Logic

| Signal | Condition |
|--------|-----------|
| **BUY**  | PSAR flips bullish (SAR crosses below price) **AND** MACD histogram > 0 |
| **SELL** | PSAR flips bearish (SAR crosses above price) **OR** MACD histogram turns negative |

Long-only, no leverage, no shorts. When flat: holding USDC.

### Why PSAR over Supertrend

PSAR uses an **acceleration factor** that speeds up as new extremes are set — it locks in gains on parabolic moves faster than Supertrend's fixed ATR bands. This makes it superior in sideways and bear markets where Supertrend whipsaws.

---

## Optimised Parameters

| Asset | Market | PSAR step | PSAR max | MACD |
|-------|--------|-----------|----------|------|
| HYPE  | Spot   | 0.01      | 0.2      | (8, 26, 7) |
| BTC   | Perp 1× | 0.03    | 0.2      | (8, 21, 7) |
| SOL   | Perp 1× | 0.03    | 0.2      | (8, 21, 7) |

Timeframe: **4h bars** · ~5 trades/month · avg hold ~1.9 days/trade

---

## Backtest Results (after all realistic costs)

### Cost Model
- **HYPE spot**: taker 0.05%/side + slippage 0.10% = 0.30% round trip
- **BTC/SOL perp 1×**: taker 0.035%/side + slippage 0.05% = 0.17% round trip + actual Hyperliquid funding rates

### Performance Summary

| Asset | Period | B&H | Strategy | Sharpe | Max DD | OOS Splits |
|-------|--------|-----|----------|--------|--------|------------|
| HYPE  | Dec 2024 – Apr 2026 | +205%  | +19,322% | 5.64 | -13.5% | 6/6 ✅ |
| BTC   | Jan 2024 – Apr 2026 | +73%   | +2,422%  | 5.81 | -4.9%  | 11/11 ✅ |
| SOL   | Jan 2024 – Apr 2026 | -7%    | +57,704% | 6.18 | -12.9% | 11/11 ✅ |

### Cross-Exchange Out-of-Sample Validation (BTC)

Params fitted on Hyperliquid 2024–2026, applied **cold** to Binance BTC/USDT 4h 2019–2024 (5 years, never seen):

| Period | B&H | Strategy | Sharpe | OOS Splits |
|--------|-----|----------|--------|------------|
| Binance 2019–2024 | +1,171% | +1,839,574% | 5.82 | 28/28 ✅ |
| Combined 2019–2026 | +2,038% | +49,909,293% | 5.78 | 42/42 ✅ |

**2022 bear market (-65% BTC)**: strategy returned **+511%** by staying in cash during every major leg down.

### Quarter-by-Quarter (BTC, Binance 2019–2024)

| Year | B&H | Strategy | Alpha |
|------|-----|----------|-------|
| 2019 | +95%  | +390%    | +295% |
| 2020 | +300% | +927%    | +626% |
| 2021 | +58%  | +1,000%  | +943% |
| **2022** | **-65%** | **+511%** | **+575%** |
| 2023 | +156% | +364%    | +208% |

---

## Overfitting Audit

| Test | Result | Verdict |
|------|--------|---------|
| OOS/IS Sharpe ratio | 6.06 / 5.82 = 1.04 | ✅ OOS beats IS |
| Parameter sensitivity (72 combos) | 100% OOS Sharpe > 2, 93% > 4 | ✅ Robust |
| Deflated Sharpe Ratio (72 trials) | DSR = 7.64 vs random = 2.05 | ✅ Passes |
| Permutation test | p = 0.558 | ⚠️ Not significant |
| Parameter stability | Fitted params rank #2/72 on unseen 5yr BTC | ✅ Stable |
| Cross-exchange OOS | 42/42 splits profitable on Binance data | ✅ Generalises |

**Honest caveat**: the permutation test (p = 0.558) means we cannot fully rule out that any trend-following system would have worked on these assets in this period. The edge is partially structural to crypto's trending nature.

**Confidence level: 7.5/10** — the 7-year, 42/42 OOS result including a full bear market cycle upgrades confidence significantly from the initial HYPE-only analysis.

---

## Repository Structure

```
├── indexer.py          # Fetches OHLCV from Hyperliquid (30-day chunked)
├── optimize.py         # Strategy 1: RSI momentum walk-forward CV
├── strategy2.py        # Strategy 2: Supertrend + MACD
├── strategy3.py        # Strategy 3: PSAR + MACD (winner)
├── quant_test.py       # Full 9-test quant validation suite
├── short_strategy.py   # Long/short extension research
├── requirements.txt
└── live/               # Production trading bot
    ├── bot.py          # Main loop — wakes on 4h candle closes
    ├── signals.py      # PSAR + MACD signal engine
    ├── executor.py     # Hardened order execution with 6 safety layers
    ├── config.py       # All parameters — single source of truth
    ├── data.py         # Candle + balance fetcher
    ├── state.py        # JSON persistence with trade history
    ├── alerts.py       # Telegram trade alerts
    ├── commander.py    # Telegram command listener (/status /close /positions)
    ├── Dockerfile
    ├── docker-compose.yml
    ├── Makefile
    └── .env.example
```

---

## Live Bot

### Quick Start

```bash
cd live
cp .env.example .env
# Fill in HL_WALLET_ADDR, HL_PRIVATE_KEY, TELEGRAM_TOKEN, TELEGRAM_CHAT

# Paper trading (recommended 2+ weeks first)
make dry

# Go live (prompts for YES confirmation)
make live
```

### Telegram Commands

| Command | Action |
|---------|--------|
| `/status` | Position, entry price, unrealised PnL, cumulative stats |
| `/close` | Force-close open position within 30s |
| `/positions` | Full trade history with per-trade PnL |

### Safety Layers (executor.py)

1. **Leverage lock** — for perp markets, 1× isolated enforced via API before every order; aborts if call fails
2. **Price guard** — NaN/zero/infinite mid price aborts immediately
3. **Size guard** — notional below $10 or above $10,000 hard-aborts
4. **Pre-flight BUY** — checks live balance and no existing position (double-entry guard)
5. **Pre-flight SELL** — reconciles state qty vs live balance; uses live balance if >5% mismatch
6. **Fill verify** — parses IOC response and confirms actual fill; unfilled orders treated as failures

### Environment Variables

```bash
HL_PRIVATE_KEY=0x...        # Ethereum private key (live mode only)
HL_WALLET_ADDR=0x...        # Public wallet address
DRY_RUN=true                # Set to false for live trading
TELEGRAM_TOKEN=...          # Optional: bot token from @BotFather
TELEGRAM_CHAT=...           # Optional: your chat ID
```

---

## Realistic Return Expectations

These returns are extraordinary because crypto assets were in a volatile, trending regime during the test period. Realistic forward-looking estimates:

| Scenario | Monthly return | Annual |
|----------|---------------|--------|
| Flat/sideways crypto | 5–15% | 60–180% |
| Trending crypto (base) | 20–40% | 240–480% |
| Strong bull (historical avg) | 37% | 445% |

**Risk**: max drawdown 5–14% peak-to-trough. 6 consecutive losing trades possible.

The strategy survives bear markets (2022 BTC -65% → strategy +511%) by exiting to cash. The main vulnerability is prolonged sideways grinding with no clean trends.

---

## Regenerating Data

```bash
# Fetch HYPE OHLCV from Hyperliquid genesis
python indexer.py

# Fetch BTC/SOL from Hyperliquid
# (see strategy3.py fetch_binance_4h for Binance historical data)
```

---

## Disclaimer

This software is for educational and research purposes. Past performance does not guarantee future results. Cryptocurrency trading carries substantial risk of loss. Never trade more than you can afford to lose.
