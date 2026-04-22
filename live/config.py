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

# ── Strategy params (must match optimised values exactly) ─────────────────────
ATR_PERIOD    = 7
ST_MULT       = 1.5
MACD_FAST     = 8
MACD_SLOW     = 21
MACD_SIG      = 9

# ── Risk & sizing ─────────────────────────────────────────────────────────────
TOTAL_CAPITAL_USDC  = 1_000.0         # total allocated capital
DEPLOY_FRACTION     = 0.95            # % of capital to deploy per trade
FEE_BUFFER_USDC     = 5.0            # keep aside for fees
MAX_POSITION_USDC   = TOTAL_CAPITAL_USDC * DEPLOY_FRACTION - FEE_BUFFER_USDC
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
