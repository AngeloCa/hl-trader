"""
Main trading bot — runs continuously, wakes on 4h candle closes.

Usage:
  DRY_RUN=true  python bot.py      # paper trading (default)
  DRY_RUN=false python bot.py      # live trading

Logs to both stdout and bot.log.
"""
import sys
import time
import logging
import traceback
from datetime import datetime, timezone

import config
import state as st
import alerts
import commander
from data    import fetch_candles, get_mid_price
from signals import compute_signal, assert_strategy_parity
from executor import execute_buy, execute_sell

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE),
    ],
)
log = logging.getLogger("bot")


# ── Timing helpers ────────────────────────────────────────────────────────────
def next_bar_close_ts() -> float:
    """Unix timestamp of the next 4h bar close + offset."""
    now      = datetime.now(timezone.utc).timestamp()
    bar_secs = config.BAR_SECONDS
    # Next multiple of bar_secs after now
    next_bar = (now // bar_secs + 1) * bar_secs
    return next_bar + config.CHECK_OFFSET_S


def sleep_until(target_ts: float):
    """Sleep until target_ts, waking every 30 s to check for Telegram /close."""
    delta = target_ts - datetime.now(timezone.utc).timestamp()
    if delta <= 0:
        return
    log.info(f"Sleeping {delta/60:.1f} min until next candle check "
             f"({datetime.fromtimestamp(target_ts, tz=timezone.utc).strftime('%H:%M UTC')})")
    while True:
        remaining = target_ts - datetime.now(timezone.utc).timestamp()
        if remaining <= 0:
            break
        if commander.close_event.is_set():
            log.info("Close command detected — waking from sleep early")
            break
        time.sleep(min(30, remaining))


# ── Core cycle ────────────────────────────────────────────────────────────────
def run_cycle(position_state: dict) -> dict:
    """
    One decision cycle:
      1. Fetch data
      2. Compute signal
      3. Execute if needed
      4. Return updated state
    """
    # ── Data ─────────────────────────────────────────────────────────────────
    log.info("Fetching candles...")
    df = fetch_candles()
    log.info(f"Got {len(df)} bars  "
             f"({df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()})")

    # ── Signal ───────────────────────────────────────────────────────────────
    signal, debug = compute_signal(df)
    log.info(f"Signal: {signal.upper():4s}  |  {debug}")

    pos = position_state["position"]

    # ── Execute ───────────────────────────────────────────────────────────────
    if signal == "buy" and pos == "none":
        log.info("Opening LONG position...")
        result = execute_buy(config.WALLET_ADDR)

        if result["status"] == "ok":
            entry_price = result["price"]
            qty         = result["qty_hype"]
            usdc        = result["usdc_spent"]
            position_state = st.enter_long(position_state, entry_price, qty, usdc)

            log.info(f"LONG opened: {qty:.4f} HYPE @ ${entry_price:.4f}  (${usdc:.2f})")
            alerts.trade_alert("buy", entry_price, qty, dry_run=config.DRY_RUN)
        else:
            log.warning(f"Buy skipped: {result.get('reason')}")

    elif signal == "sell" and pos == "long":
        log.info("Closing LONG position...")
        qty    = position_state.get("qty_hype")
        result = execute_sell(config.WALLET_ADDR, qty)

        if result["status"] == "ok":
            exit_price = result["price"]
            entry_price = position_state["entry_price"]
            pnl_pct    = (exit_price - entry_price) / entry_price * 100 if entry_price else None
            position_state = st.exit_long(position_state, exit_price)

            log.info(f"LONG closed @ ${exit_price:.4f}  P&L: {pnl_pct:+.2f}%")
            alerts.trade_alert("sell", exit_price, qty or 0, pnl_pct, config.DRY_RUN)
        else:
            log.warning(f"Sell skipped: {result.get('reason')}")

    elif signal == "buy" and pos == "long":
        log.info("BUY signal but already long — hold")

    elif signal == "sell" and pos == "none":
        log.info("SELL signal but no position — ignore")

    # ── Status summary ───────────────────────────────────────────────────────
    mid = get_mid_price()
    if pos == "long" and position_state["entry_price"]:
        unrealised = (mid - position_state["entry_price"]) / position_state["entry_price"] * 100
        log.info(f"Portfolio: LONG {position_state['qty_hype']:.4f} HYPE  "
                 f"entry=${position_state['entry_price']:.4f}  "
                 f"mid=${mid:.4f}  unrealised={unrealised:+.2f}%")
    else:
        log.info(f"Portfolio: CASH  |  HYPE mid=${mid:.4f}")

    return position_state


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    mode = "DRY-RUN" if config.DRY_RUN else "⚠️  LIVE TRADING"
    log.info("=" * 60)
    log.info(f"  HYPE/USDC Bot  |  {mode}")
    log.info(f"  Strategy : PSAR(start={config.PSAR_START}, step={config.PSAR_STEP}, max={config.PSAR_MAX})"
             f" + MACD({config.MACD_FAST},{config.MACD_SLOW},{config.MACD_SIG})")
    log.info(f"  Capital  : ${config.TOTAL_CAPITAL_USDC:,.0f} USDC"
             f"  |  Max position: ${config.MAX_POSITION_USDC:,.0f}")
    log.info("=" * 60)

    if not config.DRY_RUN and not config.PRIVATE_KEY:
        log.error("LIVE mode requires HL_PRIVATE_KEY to be set. Aborting.")
        sys.exit(1)

    # ── Strategy parity guard ─────────────────────────────────────────────────
    # Raises immediately if config.py params deviate from the backtested values.
    # This prevents silent drift between live and research code.
    try:
        assert_strategy_parity()
        log.info("Strategy parity check PASSED — config matches backtested params")
    except RuntimeError as e:
        log.error(str(e))
        sys.exit(1)

    # ── Warmup guard ──────────────────────────────────────────────────────────
    min_required = config.MACD_SLOW + config.MACD_SIG + 5
    if config.LOOKBACK_BARS < min_required:
        log.error(
            f"LOOKBACK_BARS={config.LOOKBACK_BARS} is too small — "
            f"need at least {min_required} bars for MACD warmup."
        )
        sys.exit(1)

    position_state = st.load()

    # Start Telegram command listener
    commander.start()

    # ── Startup protective sell ───────────────────────────────────────────────
    # On restart, immediately check for a SELL signal to protect any open
    # position — a sell that fired at the last bar close may be hours overdue.
    # BUY signals are suppressed on startup: we never enter on a signal that is
    # up to 4h stale (the backtest assumes entry at bar close, not hours later).
    log.info("Startup: checking for overdue SELL signal...")
    try:
        from data import fetch_candles
        df_startup = fetch_candles()
        sig_startup, dbg_startup = compute_signal(df_startup)
        log.info(f"Startup signal: {sig_startup.upper()}  |  {dbg_startup}")
        if sig_startup == "sell" and position_state["position"] == "long":
            log.info("Startup SELL: closing overdue position...")
            qty    = position_state.get("qty_hype")
            result = execute_sell(config.WALLET_ADDR, qty)
            if result["status"] == "ok":
                exit_price  = result["price"]
                entry_price = position_state["entry_price"]
                pnl_pct     = (exit_price - entry_price) / entry_price * 100 if entry_price else None
                position_state = st.exit_long(position_state, exit_price)
                log.info(f"Startup SELL closed @ ${exit_price:.4f}  P&L: {pnl_pct:+.2f}%")
                alerts.trade_alert("sell", exit_price, qty or 0, pnl_pct, config.DRY_RUN)
                alerts.send("⚠️ <b>Startup SELL</b>: closed overdue position on restart.")
            else:
                log.warning(f"Startup SELL failed: {result.get('reason')}")
        elif sig_startup == "buy" and position_state["position"] == "none":
            log.info("Startup BUY signal suppressed — waiting for next bar close "
                     "(signal may be up to 4h stale; backtest assumes entry at bar close)")
        else:
            log.info("Startup: no action needed — sleeping until next bar close")
    except Exception as e:
        log.warning(f"Startup check failed (non-fatal): {e}")

    while True:
        try:
            sleep_until(next_bar_close_ts())

            # ── Handle Telegram /close command ───────────────────────────────
            if commander.close_event.is_set():
                commander.close_event.clear()
                if position_state["position"] == "long":
                    log.info("FORCE CLOSE triggered by Telegram /close command")
                    qty    = position_state.get("qty_hype")
                    result = execute_sell(config.WALLET_ADDR, qty)
                    if result["status"] == "ok":
                        exit_price  = result["price"]
                        entry_price = position_state["entry_price"]
                        pnl_pct     = (exit_price - entry_price) / entry_price * 100 if entry_price else None
                        position_state = st.exit_long(position_state, exit_price)
                        log.info(f"Force closed: {qty:.4f} HYPE @ ${exit_price:.4f}  P&L: {pnl_pct:+.2f}%")
                        alerts.trade_alert("sell", exit_price, qty or 0, pnl_pct, config.DRY_RUN)
                        alerts.send("✅ <b>Position closed</b> via /close command.")
                    else:
                        log.warning(f"Force close failed: {result.get('reason')}")
                        alerts.send(f"⚠️ Force close failed: {result.get('reason')}")
                    first_run = False
                    continue
                else:
                    log.info("/close received but no position open — ignoring")

            log.info(f"─── Cycle at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ───")
            position_state = run_cycle(position_state)

        except KeyboardInterrupt:
            log.info("Interrupted by user — shutting down.")
            break

        except Exception as e:
            log.error(f"Cycle error: {e}\n{traceback.format_exc()}")
            alerts.error_alert(str(e))
            time.sleep(60)   # wait 1 min then retry


if __name__ == "__main__":
    main()
