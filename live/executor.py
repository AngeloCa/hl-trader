"""
Order executor — wraps Hyperliquid SDK with dry-run support.

In DRY_RUN mode (default): logs orders but never sends them.
In LIVE mode: places IOC limit orders slightly inside spread to ensure fill.

Position sizing:
  BUY  → spend up to MAX_POSITION_USDC of available USDC
  SELL → sell 100% of HYPE balance
"""
import time
import logging
from config import (
    PRIVATE_KEY, BASE_URL, COIN, DRY_RUN,
    MAX_POSITION_USDC, MIN_ORDER_USDC,
    SLIPPAGE_PCT, ORDER_RETRIES, RETRY_DELAY_S,
)
from data import get_mid_price, get_spot_balance

log = logging.getLogger("executor")


# ── Dry-run order simulation ──────────────────────────────────────────────────
class DryRunExchange:
    def order(self, coin, is_buy, size, price, order_type):
        side = "BUY" if is_buy else "SELL"
        log.info(f"[DRY-RUN] {side} {size:.4f} {coin} @ {price:.4f}  "
                 f"(notional ${size*price:.2f})")
        return {"status": "ok", "response": {"type": "order",
                "data": {"statuses": [{"filled": {"totalSz": str(size),
                                                   "avgPx": str(price)}}]}}}


# ── Live exchange (lazy-loaded to avoid import errors without SDK) ─────────────
def _build_live_exchange():
    try:
        from hyperliquid.exchange import Exchange
        from eth_account import Account
    except ImportError:
        raise RuntimeError(
            "hyperliquid-python-sdk not installed. "
            "Run: pip install hyperliquid-python-sdk eth-account"
        )
    if not PRIVATE_KEY:
        raise RuntimeError("HL_PRIVATE_KEY environment variable not set")
    wallet = Account.from_key(PRIVATE_KEY)
    return Exchange(wallet, base_url=BASE_URL)


def _get_exchange():
    if DRY_RUN:
        return DryRunExchange()
    return _build_live_exchange()


# ── Core order functions ──────────────────────────────────────────────────────
def _place_ioc(exchange, is_buy: bool, qty: float, mid: float) -> dict:
    """
    Place an IOC limit order priced aggressively to ensure fill.
    BUY  at mid * (1 + slippage)  — willing to pay slightly more
    SELL at mid * (1 - slippage)  — willing to accept slightly less
    """
    price = round(mid * (1 + SLIPPAGE_PCT) if is_buy else mid * (1 - SLIPPAGE_PCT), 5)
    qty   = round(qty, 4)

    for attempt in range(1, ORDER_RETRIES + 1):
        try:
            result = exchange.order(
                COIN, is_buy, qty, price,
                {"limit": {"tif": "Ioc"}},
            )
            return result
        except Exception as e:
            log.warning(f"Order attempt {attempt} failed: {e}")
            if attempt < ORDER_RETRIES:
                time.sleep(RETRY_DELAY_S)
                mid   = get_mid_price()
                price = round(mid * (1 + SLIPPAGE_PCT) if is_buy else mid * (1 - SLIPPAGE_PCT), 5)

    raise RuntimeError(f"Order failed after {ORDER_RETRIES} attempts")


def execute_buy(wallet_addr: str) -> dict:
    """
    Buy HYPE with available USDC (up to MAX_POSITION_USDC).
    Returns order result + execution details.
    """
    mid      = get_mid_price()
    balances = get_spot_balance(wallet_addr) if not DRY_RUN else {"USDC": 1000.0, "HYPE": 0.0}
    usdc_bal = balances["USDC"]

    usdc_to_spend = min(usdc_bal, MAX_POSITION_USDC)

    if usdc_to_spend < MIN_ORDER_USDC:
        log.warning(f"USDC balance too low to trade: ${usdc_bal:.2f}")
        return {"status": "skipped", "reason": "insufficient_usdc", "balance": usdc_bal}

    qty = usdc_to_spend / mid

    log.info(f"BUY signal | mid={mid:.4f} | spending ${usdc_to_spend:.2f} → {qty:.4f} HYPE")

    exchange = _get_exchange()
    result   = _place_ioc(exchange, True, qty, mid)

    return {
        "status":     "ok",
        "side":       "buy",
        "qty_hype":   qty,
        "usdc_spent": usdc_to_spend,
        "price":      mid,
        "dry_run":    DRY_RUN,
        "raw":        result,
    }


def execute_sell(wallet_addr: str, hype_qty: float | None = None) -> dict:
    """
    Sell HYPE position.
    If hype_qty is None, sell full balance.
    """
    mid      = get_mid_price()
    balances = get_spot_balance(wallet_addr) if not DRY_RUN else {"USDC": 0.0, "HYPE": hype_qty or 50.0}
    hype_bal = balances["HYPE"]

    qty = hype_qty or hype_bal

    if qty * mid < MIN_ORDER_USDC:
        log.warning(f"HYPE balance too small to sell: {hype_bal:.4f} HYPE (${hype_bal*mid:.2f})")
        return {"status": "skipped", "reason": "insufficient_hype", "balance": hype_bal}

    log.info(f"SELL signal | mid={mid:.4f} | selling {qty:.4f} HYPE (${qty*mid:.2f})")

    exchange = _get_exchange()
    result   = _place_ioc(exchange, False, qty, mid)

    return {
        "status":       "ok",
        "side":         "sell",
        "qty_hype":     qty,
        "usdc_received": qty * mid,
        "price":        mid,
        "dry_run":      DRY_RUN,
        "raw":          result,
    }
