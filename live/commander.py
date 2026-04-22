"""
Telegram command listener — runs in a background daemon thread.

Commands (only accepted from your TELEGRAM_CHAT):
  /status     — current position, unrealised PnL, cumulative stats
  /close      — force-close open position on next wake cycle (≤ 30 s)
  /positions  — full trade history with per-trade PnL
"""
import threading
import logging
import time
import requests
from datetime import datetime, timezone

from config import TELEGRAM_TOKEN, TELEGRAM_CHAT, DRY_RUN

log = logging.getLogger("commander")

# ── Shared signal — main loop watches this ────────────────────────────────────
close_event = threading.Event()


# ── Telegram helpers ──────────────────────────────────────────────────────────
def _send(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        log.warning(f"Telegram reply failed: {e}")


# ── Command handlers ──────────────────────────────────────────────────────────
def _cmd_status():
    from data import get_mid_price
    import state as st

    s = st.load()
    pos = s["position"]
    mode = "🔵 DRY-RUN" if DRY_RUN else "🟢 LIVE"

    try:
        mid = get_mid_price()
    except Exception:
        mid = 0.0

    if pos == "long" and s.get("entry_price"):
        entry  = s["entry_price"]
        qty    = s["qty_hype"] or 0
        unreal = (mid - entry) / entry * 100
        value  = qty * mid
        etime  = (s.get("entry_time") or "")[:16].replace("T", " ")
        msg = (
            f"📊 <b>HYPE Bot</b> [{mode}]\n\n"
            f"Position : <b>LONG</b>\n"
            f"Entry    : ${entry:.4f}\n"
            f"Qty      : {qty:.4f} HYPE\n"
            f"Mid      : ${mid:.4f}\n"
            f"Value    : ${value:.2f}\n"
            f"Unrealised PnL : <b>{unreal:+.2f}%</b>\n"
            f"Opened   : {etime} UTC\n\n"
            f"Closed trades : {s.get('trade_count', 0)}\n"
            f"Cumulative PnL: <b>{s.get('total_pnl_pct', 0):+.2f}%</b>"
        )
    else:
        msg = (
            f"📊 <b>HYPE Bot</b> [{mode}]\n\n"
            f"Position : <b>CASH</b>\n"
            f"HYPE mid : ${mid:.4f}\n\n"
            f"Closed trades : {s.get('trade_count', 0)}\n"
            f"Cumulative PnL: <b>{s.get('total_pnl_pct', 0):+.2f}%</b>"
        )
    _send(msg)


def _cmd_close():
    import state as st
    s = st.load()
    if s["position"] == "none":
        _send("ℹ️ No open position to close.")
        return
    close_event.set()
    _send(
        "🔴 <b>CLOSE command received.</b>\n"
        "Selling position within 30 s — watch /status for confirmation."
    )


def _cmd_positions():
    import state as st
    s = st.load()
    history = s.get("trade_history", [])

    if not history:
        _send("📋 No closed trades yet.")
        return

    lines = [f"📋 <b>Trade History</b>  ({len(history)} trades)\n"]
    running_pnl = 0.0
    for i, t in enumerate(history[-20:], 1):   # show last 20
        pnl = t.get("pnl_pct", 0)
        running_pnl += pnl
        emoji = "✅" if pnl >= 0 else "❌"
        entry_d = (t.get("entry_time") or "")[:10]
        exit_d  = (t.get("exit_time")  or "")[:10]
        lines.append(
            f"{emoji} <b>#{i}</b>  {entry_d} → {exit_d}\n"
            f"   ${t.get('entry_price', 0):.4f} → ${t.get('exit_price', 0):.4f}"
            f"  ({t.get('qty', 0):.4f} HYPE)\n"
            f"   PnL: <b>{pnl:+.2f}%</b>\n"
        )

    lines.append(f"Cumulative (shown): <b>{running_pnl:+.2f}%</b>")
    _send("\n".join(lines))


def _dispatch(text: str):
    cmd = text.strip().lower().split()[0]
    if cmd == "/status":
        _cmd_status()
    elif cmd == "/close":
        _cmd_close()
    elif cmd == "/positions":
        _cmd_positions()
    else:
        _send(
            f"❓ Unknown command: <code>{text}</code>\n\n"
            "Available commands:\n"
            "/status    — position &amp; PnL\n"
            "/close     — force-close position\n"
            "/positions — trade history"
        )


# ── Long-poll loop ────────────────────────────────────────────────────────────
def _poll_loop():
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        log.info("Telegram not configured — command listener disabled")
        return

    offset = 0
    mode = "DRY-RUN" if DRY_RUN else "LIVE"
    log.info("Telegram command listener started")
    _send(
        f"🤖 <b>HYPE Bot online</b>  [{mode}]\n\n"
        "Commands:\n"
        "/status    — position &amp; PnL\n"
        "/close     — force-close position\n"
        "/positions — trade history"
    )

    while True:
        try:
            resp = requests.get(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
                params={"timeout": 30, "offset": offset, "allowed_updates": ["message"]},
                timeout=40,
            )
            updates = resp.json().get("result", [])
            for upd in updates:
                offset = upd["update_id"] + 1
                msg     = upd.get("message", {})
                chat_id = str(msg.get("chat", {}).get("id", ""))
                text    = msg.get("text", "")

                if chat_id != str(TELEGRAM_CHAT):
                    log.warning(f"Ignoring message from unauthorized chat {chat_id}")
                    continue

                if text.startswith("/"):
                    log.info(f"TG command: {text}")
                    _dispatch(text)

        except Exception as e:
            log.warning(f"Poll error: {e}")
            time.sleep(5)


def start():
    """Start the Telegram command listener in a background daemon thread."""
    t = threading.Thread(target=_poll_loop, name="tg-commander", daemon=True)
    t.start()
    return t
