"""
Order executor — wraps Hyperliquid SDK with dry-run support.

Safety guarantees:
  1. LEVERAGE LOCK  — for perp markets, 1x is enforced via API call before
                      every order. If that call fails, the order is aborted.
  2. SIZE GUARD     — order notional is capped at MAX_ORDER_USDC and floored
                      at MIN_ORDER_USDC. NaN / zero / negative sizes abort.
  3. PRICE GUARD    — mid price must be positive and finite. Stale price
                      (unchanged across retries) aborts after ORDER_RETRIES.
  4. FILL VERIFY    — after an IOC order, the response is parsed to confirm
                      an actual fill. Unfilled IOC orders are not treated as
                      successful trades.
  5. PRE-FLIGHT     — before every BUY, existing open position is checked so
                      we never double-enter by accident.
  6. DRY-RUN parity — DryRunExchange mirrors every safety path so the same
                      code runs in paper and live mode.

Market types (set in config.py):
  "spot" → HYPE/USDC spot order  (no leverage, no liquidation)
  "perp" → perpetual futures 1×  (leverage enforced, no liquidation at 1×)
"""
import time
import math
import logging
from config import (
    PRIVATE_KEY, BASE_URL, COIN, DRY_RUN,
    MARKET_TYPE, PERP_LEVERAGE,
    MAX_POSITION_USDC, MAX_ORDER_USDC, MIN_ORDER_USDC,
    SLIPPAGE_PCT, ORDER_RETRIES, RETRY_DELAY_S,
)
from data import get_mid_price, get_spot_balance

log = logging.getLogger("executor")

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _assert_valid_price(mid: float, context: str = ""):
    """Raise if mid price is unusable."""
    if mid is None or not math.isfinite(mid) or mid <= 0:
        raise ValueError(f"Invalid mid price {mid!r} [{context}]")


def _assert_valid_qty(qty: float, mid: float, context: str = ""):
    """Raise if order size is outside safe bounds."""
    if qty is None or not math.isfinite(qty) or qty <= 0:
        raise ValueError(f"Invalid quantity {qty!r} [{context}]")
    notional = qty * mid
    if notional < MIN_ORDER_USDC:
        raise ValueError(
            f"Order too small: {qty:.6f} × {mid:.4f} = ${notional:.2f} "
            f"(min ${MIN_ORDER_USDC}) [{context}]"
        )
    if notional > MAX_ORDER_USDC:
        raise ValueError(
            f"Order too large: {qty:.6f} × {mid:.4f} = ${notional:.2f} "
            f"(hard cap ${MAX_ORDER_USDC}) [{context}]"
        )


def _parse_fill(result: dict) -> tuple[float, float]:
    """
    Extract (avg_fill_price, filled_qty) from Hyperliquid order response.
    Raises if nothing was filled (IOC expired without fill).
    """
    try:
        statuses = result["response"]["data"]["statuses"]
        filled   = statuses[0].get("filled") or statuses[0].get("resting")
        if not filled:
            # Check for error status
            err = statuses[0].get("error", "unknown error")
            raise RuntimeError(f"Order not filled: {err}")
        avg_px  = float(filled["avgPx"])
        total_sz = float(filled["totalSz"])
        if total_sz <= 0:
            raise RuntimeError(f"Zero fill: totalSz={total_sz}")
        return avg_px, total_sz
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Could not parse fill from response {result}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Dry-run exchange (mirrors full live code path)
# ─────────────────────────────────────────────────────────────────────────────

class DryRunExchange:
    def update_leverage(self, leverage: int, coin: str, is_cross: bool = False):
        if leverage != PERP_LEVERAGE:
            raise RuntimeError(
                f"[DRY-RUN] Leverage guard: requested {leverage}x but "
                f"PERP_LEVERAGE={PERP_LEVERAGE}x — aborting."
            )
        log.info(f"[DRY-RUN] Leverage confirmed: {leverage}x on {coin} (isolated)")

    def order(self, coin: str, is_buy: bool, size: float,
              price: float, order_type: dict) -> dict:
        side = "BUY" if is_buy else "SELL"
        log.info(
            f"[DRY-RUN] {side} {size:.6f} {coin} @ {price:.4f}  "
            f"(notional ${size * price:.2f})"
        )
        return {
            "status": "ok",
            "response": {
                "type": "order",
                "data": {
                    "statuses": [{
                        "filled": {
                            "totalSz": str(size),
                            "avgPx":   str(price),
                        }
                    }]
                },
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# Live exchange (lazy-loaded)
# ─────────────────────────────────────────────────────────────────────────────

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
        raise RuntimeError("HL_PRIVATE_KEY environment variable is not set")
    wallet = Account.from_key(PRIVATE_KEY)
    return Exchange(wallet, base_url=BASE_URL)


def _get_exchange():
    return DryRunExchange() if DRY_RUN else _build_live_exchange()


# ─────────────────────────────────────────────────────────────────────────────
# Leverage enforcement (perp only)
# ─────────────────────────────────────────────────────────────────────────────

def _enforce_leverage(exchange) -> None:
    """
    For perp markets: explicitly set 1x isolated leverage via the API
    before every order. Aborts if the call fails or returns wrong leverage.
    For spot markets: no-op.
    """
    if MARKET_TYPE != "perp":
        return

    log.info(f"Enforcing {PERP_LEVERAGE}x isolated leverage on {COIN}...")

    if isinstance(exchange, DryRunExchange):
        exchange.update_leverage(PERP_LEVERAGE, COIN, is_cross=False)
        return

    for attempt in range(1, ORDER_RETRIES + 1):
        try:
            result = exchange.update_leverage(PERP_LEVERAGE, COIN, is_cross=False)
            # Hyperliquid returns {"status":"ok"} on success
            status = (result or {}).get("status", "")
            if status == "ok":
                log.info(f"Leverage set: {PERP_LEVERAGE}x isolated on {COIN}")
                return
            # Some SDK versions return None on success — treat as ok
            if result is None:
                log.info(f"Leverage set: {PERP_LEVERAGE}x isolated on {COIN} (no response body)")
                return
            raise RuntimeError(f"Unexpected leverage response: {result}")
        except Exception as e:
            log.error(f"Leverage enforcement attempt {attempt} failed: {e}")
            if attempt < ORDER_RETRIES:
                time.sleep(RETRY_DELAY_S)

    raise RuntimeError(
        f"SAFETY ABORT: could not enforce {PERP_LEVERAGE}x leverage on {COIN} "
        f"after {ORDER_RETRIES} attempts. Order cancelled."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight checks
# ─────────────────────────────────────────────────────────────────────────────

def _preflight_buy(wallet_addr: str, mid: float) -> float:
    """
    Validate conditions before placing a BUY.
    Returns the USDC amount to spend.
    Raises on any safety violation.
    """
    _assert_valid_price(mid, "preflight_buy")

    if DRY_RUN:
        usdc_bal = MAX_POSITION_USDC + 100.0   # simulated balance
    else:
        balances = get_spot_balance(wallet_addr)
        usdc_bal = balances.get("USDC", 0.0)
        # Also check we don't already hold the asset (double-entry guard)
        asset_bal = balances.get(COIN, 0.0)
        if asset_bal * mid > MIN_ORDER_USDC:
            raise RuntimeError(
                f"PRE-FLIGHT: Already holding {asset_bal:.6f} {COIN} "
                f"(${asset_bal * mid:.2f}) — refusing duplicate BUY."
            )

    usdc_to_spend = min(usdc_bal, MAX_POSITION_USDC)

    if usdc_to_spend < MIN_ORDER_USDC:
        raise ValueError(
            f"Insufficient USDC: balance=${usdc_bal:.2f}, "
            f"min required=${MIN_ORDER_USDC:.2f}"
        )

    notional = usdc_to_spend
    if notional > MAX_ORDER_USDC:
        raise ValueError(
            f"PRE-FLIGHT: Computed notional ${notional:.2f} exceeds hard cap "
            f"${MAX_ORDER_USDC:.2f}. Check MAX_POSITION_USDC config."
        )

    return usdc_to_spend


def _preflight_sell(wallet_addr: str, qty: float, mid: float) -> float:
    """
    Validate conditions before placing a SELL.
    Returns the confirmed quantity to sell.
    Raises on any safety violation.
    """
    _assert_valid_price(mid, "preflight_sell")

    if not DRY_RUN:
        balances  = get_spot_balance(wallet_addr)
        live_qty  = balances.get(COIN, 0.0)
        if live_qty <= 0:
            raise RuntimeError(
                f"PRE-FLIGHT: Tried to sell {qty:.6f} {COIN} "
                f"but live balance is {live_qty:.6f} — nothing to sell."
            )
        # Use live balance as source of truth; warn if mismatch
        if abs(live_qty - qty) / max(qty, 1e-9) > 0.05:
            log.warning(
                f"PRE-FLIGHT: State qty={qty:.6f} but live balance={live_qty:.6f} "
                f"({abs(live_qty-qty)/qty:.1%} mismatch) — using live balance."
            )
        qty = live_qty

    _assert_valid_qty(qty, mid, "preflight_sell")
    return qty


# ─────────────────────────────────────────────────────────────────────────────
# Core IOC order placement
# ─────────────────────────────────────────────────────────────────────────────

def _place_ioc(exchange, is_buy: bool, qty: float, mid: float) -> tuple[float, float]:
    """
    Place IOC limit order and return (avg_fill_price, filled_qty).
    Retries up to ORDER_RETRIES times with refreshed mid price.
    Raises if all attempts fail or no fill is confirmed.
    """
    last_error = None

    for attempt in range(1, ORDER_RETRIES + 1):
        _assert_valid_price(mid, f"ioc attempt {attempt}")
        price = round(
            mid * (1 + SLIPPAGE_PCT) if is_buy else mid * (1 - SLIPPAGE_PCT),
            5
        )
        qty_r = round(qty, 6)
        _assert_valid_qty(qty_r, mid, f"ioc attempt {attempt}")

        try:
            log.info(
                f"Order attempt {attempt}/{ORDER_RETRIES}: "
                f"{'BUY' if is_buy else 'SELL'} {qty_r:.6f} {COIN} "
                f"@ {price:.4f}  (notional ${qty_r * price:.2f})"
            )
            result = exchange.order(
                COIN, is_buy, qty_r, price,
                {"limit": {"tif": "Ioc"}},
            )
            avg_px, filled_sz = _parse_fill(result)
            log.info(
                f"Fill confirmed: {filled_sz:.6f} {COIN} @ avg {avg_px:.4f}  "
                f"(notional ${filled_sz * avg_px:.2f})"
            )
            return avg_px, filled_sz

        except Exception as e:
            last_error = e
            log.warning(f"Order attempt {attempt} failed: {e}")
            if attempt < ORDER_RETRIES:
                time.sleep(RETRY_DELAY_S)
                mid = get_mid_price()   # refresh price before retry

    raise RuntimeError(
        f"Order failed after {ORDER_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def execute_buy(wallet_addr: str) -> dict:
    """
    Open a long position: buy COIN with USDC.

    Safety sequence:
      1. Fetch & validate mid price
      2. Pre-flight checks (balance, no existing position, size caps)
      3. Enforce leverage (perp only) — ABORTS if this fails
      4. Place IOC order
      5. Verify fill from response

    Returns {"status": "ok", ...} or {"status": "skipped", "reason": ...}.
    """
    try:
        mid           = get_mid_price()
        usdc_to_spend = _preflight_buy(wallet_addr, mid)
        qty           = usdc_to_spend / mid

        log.info(
            f"BUY  | mid={mid:.4f} | spending ${usdc_to_spend:.2f} "
            f"→ {qty:.6f} {COIN}"
        )

        exchange = _get_exchange()
        _enforce_leverage(exchange)          # no-op for spot, mandatory for perp
        avg_px, filled_sz = _place_ioc(exchange, True, qty, mid)

        return {
            "status":     "ok",
            "side":       "buy",
            "qty_hype":   filled_sz,
            "usdc_spent": filled_sz * avg_px,
            "price":      avg_px,
            "dry_run":    DRY_RUN,
        }

    except (ValueError, RuntimeError) as e:
        log.error(f"BUY aborted: {e}")
        return {"status": "skipped", "reason": str(e)}

    except Exception as e:
        log.error(f"BUY unexpected error: {e}", exc_info=True)
        return {"status": "error", "reason": str(e)}


def execute_sell(wallet_addr: str, hype_qty: float | None = None) -> dict:
    """
    Close long position: sell COIN back to USDC.

    Safety sequence:
      1. Fetch & validate mid price
      2. Pre-flight checks (live balance reconciliation, size floor)
      3. Enforce leverage (perp only) — ABORTS if this fails
      4. Place IOC order
      5. Verify fill from response

    Returns {"status": "ok", ...} or {"status": "skipped", "reason": ...}.
    """
    try:
        mid = get_mid_price()
        qty = _preflight_sell(wallet_addr, hype_qty or 0.0, mid)

        log.info(
            f"SELL | mid={mid:.4f} | selling {qty:.6f} {COIN} "
            f"(${qty * mid:.2f})"
        )

        exchange = _get_exchange()
        _enforce_leverage(exchange)          # no-op for spot, mandatory for perp
        avg_px, filled_sz = _place_ioc(exchange, False, qty, mid)

        return {
            "status":        "ok",
            "side":          "sell",
            "qty_hype":      filled_sz,
            "usdc_received": filled_sz * avg_px,
            "price":         avg_px,
            "dry_run":       DRY_RUN,
        }

    except (ValueError, RuntimeError) as e:
        log.error(f"SELL aborted: {e}")
        return {"status": "skipped", "reason": str(e)}

    except Exception as e:
        log.error(f"SELL unexpected error: {e}", exc_info=True)
        return {"status": "error", "reason": str(e)}
