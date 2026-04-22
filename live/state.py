"""
Persistent position state — survives bot restarts.
Stored as JSON: state.json
"""
import json
import logging
from datetime import datetime, timezone
from config import STATE_FILE

log = logging.getLogger("state")

EMPTY_STATE = {
    "position":    "none",       # "none" | "long"
    "entry_price": None,
    "entry_time":  None,
    "qty_hype":    None,
    "usdc_spent":  None,
    "trade_count": 0,
    "total_pnl_pct": 0.0,
    "trade_history": [],         # list of closed trade dicts
}


def load() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                s = json.load(f)
            log.info(f"State loaded: position={s['position']} "
                     f"entry={s.get('entry_price')} qty={s.get('qty_hype')}")
            return s
        except Exception as e:
            log.error(f"State file corrupt, resetting: {e}")
    return EMPTY_STATE.copy()


def save(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)
    log.debug("State saved")


def enter_long(state: dict, price: float, qty: float, usdc: float) -> dict:
    state.update({
        "position":   "long",
        "entry_price": price,
        "entry_time":  datetime.now(timezone.utc).isoformat(),
        "qty_hype":    qty,
        "usdc_spent":  usdc,
    })
    save(state)
    return state


def exit_long(state: dict, exit_price: float) -> dict:
    if state["entry_price"]:
        pnl_pct = (exit_price - state["entry_price"]) / state["entry_price"] * 100
        state["total_pnl_pct"] = round(state.get("total_pnl_pct", 0) + pnl_pct, 4)
        state["trade_count"]   = state.get("trade_count", 0) + 1

        # Append to trade history
        trade = {
            "entry_price": state["entry_price"],
            "exit_price":  exit_price,
            "qty":         state["qty_hype"],
            "pnl_pct":     round(pnl_pct, 4),
            "entry_time":  state.get("entry_time"),
            "exit_time":   datetime.now(timezone.utc).isoformat(),
        }
        if "trade_history" not in state:
            state["trade_history"] = []
        state["trade_history"].append(trade)

        log.info(f"Trade closed: {pnl_pct:+.2f}%  |  "
                 f"total trades={state['trade_count']}  "
                 f"cumulative={state['total_pnl_pct']:+.2f}%")
    state.update({
        "position":    "none",
        "entry_price": None,
        "entry_time":  None,
        "qty_hype":    None,
        "usdc_spent":  None,
    })
    save(state)
    return state
