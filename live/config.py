"""
Live trading configuration.
All secrets come from environment variables — never hardcode keys.
"""
import os
from pathlib import Path

# ── Identity ─────────────────────────────────────────────────────────────────
PRIVATE_KEY   = os.getenv("HL_PRIVATE_KEY", "")          # Ethereum private key
WALLET_ADDR   = os.getenv("HL_WALLET_ADDR", "")          # Public address
BASE_URL      = "https://api.hyperliquid.xyz"

# ── Instrument ────────────────────────────────────────────────────────────────
COIN          = "HYPE"
INTERVAL      = "4h"
LOOKBACK_BARS = 150                    # bars to fetch for indicator warmup

# ── Market type & leverage guard ──────────────────────────────────────────────
# "spot"  → HYPE/USDC spot, no leverage, no liquidation
# "perp"  → perpetual futures; PERP_LEVERAGE is ENFORCED before every order
MARKET_TYPE   = "spot"
PERP_LEVERAGE = 1                      # NEVER change this — 1x only, no liquidation risk

# ── Strategy params — PSAR + MACD (optimised OOS Sharpe 6.06) ────────────────
PSAR_START    = 0.02                  # AF initial value (Wilder standard)
PSAR_STEP     = 0.01                  # AF increment per new extreme
PSAR_MAX      = 0.2                   # AF ceiling
MACD_FAST     = 8
MACD_SLOW     = 26
MACD_SIG      = 7

# ── Risk & sizing ─────────────────────────────────────────────────────────────
TOTAL_CAPITAL_USDC  = 1_000.0         # total allocated capital
DEPLOY_FRACTION     = 0.95            # % of capital to deploy per trade
FEE_BUFFER_USDC     = 5.0             # keep aside for fees
MAX_POSITION_USDC   = TOTAL_CAPITAL_USDC * DEPLOY_FRACTION - FEE_BUFFER_USDC
MAX_ORDER_USDC      = 10_000.0        # hard cap — order rejected if notional exceeds this
SLIPPAGE_PCT        = 0.003           # 0.3% aggressive limit to ensure fill
MIN_ORDER_USDC      = 10.0            # don't trade below this (dust)

# ── Execution ─────────────────────────────────────────────────────────────────
DRY_RUN       = os.getenv("DRY_RUN", "true").lower() != "false"   # safe default
ORDER_RETRIES = 3
RETRY_DELAY_S = 5

# ── Scheduler ─────────────────────────────────────────────────────────────────
BAR_SECONDS   = 4 * 3600              # 4h in seconds
CHECK_OFFSET_S = 90                   # run 90s after candle close (data lag)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DATA_DIR      = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
DATA_DIR.mkdir(exist_ok=True)
STATE_FILE    = DATA_DIR / "state.json"
LOG_FILE      = DATA_DIR / "bot.log"

# ── Alerts (optional) ────────────────────────────────────────────────────────
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT   = os.getenv("TELEGRAM_CHAT",  "")
