"""
Optional Telegram alerts — set TELEGRAM_TOKEN and TELEGRAM_CHAT env vars.
Silently skips if not configured.
"""
import requests
import logging
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT

log = logging.getLogger("alerts")


def send(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        log.warning(f"Telegram alert failed: {e}")


def trade_alert(side: str, price: float, qty: float, pnl_pct: float | None = None,
                dry_run: bool = True):
    tag  = "🔵 DRY RUN" if dry_run else "🟢 LIVE"
    side_emoji = "📈 BUY" if side == "buy" else "📉 SELL"
    pnl_str = f"\nP&L: <b>{pnl_pct:+.2f}%</b>" if pnl_pct is not None else ""
    send(
        f"{tag} | HYPE/USDC\n"
        f"{side_emoji}  {qty:.4f} HYPE @ <b>${price:.4f}</b>"
        f"  (${qty*price:.1f}){pnl_str}"
    )


def error_alert(msg: str):
    send(f"🔴 BOT ERROR\n{msg}")
