.PHONY: build up down logs status shell dry live

# ── Build ─────────────────────────────────────────────────────────────────────
build:
	docker compose build

# ── Run (dry-run, default) ────────────────────────────────────────────────────
dry: build
	@echo "Starting in DRY-RUN mode (no real orders)"
	DRY_RUN=true docker compose up -d
	docker compose logs -f

# ── Run (live) ────────────────────────────────────────────────────────────────
live: build
	@echo "⚠️  LIVE MODE — real orders will be placed"
	@read -p "Type YES to confirm: " c && [ "$$c" = "YES" ] || exit 1
	DRY_RUN=false docker compose up -d
	docker compose logs -f

# ── Management ────────────────────────────────────────────────────────────────
up:
	docker compose up -d

down:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f --tail=100

status:
	@echo "=== Container ==="
	@docker compose ps
	@echo ""
	@echo "=== Position state ==="
	@cat data/state.json 2>/dev/null || echo "No state file yet"
	@echo ""
	@echo "=== Last 20 log lines ==="
	@tail -20 data/bot.log 2>/dev/null || docker compose logs --tail=20

shell:
	docker compose exec bot python -c "\
import sys; sys.path.insert(0,'.'); \
import state, config; \
s = state.load(); \
print('Position:', s['position']); \
print('Entry:',    s.get('entry_price')); \
print('Trades:',   s.get('trade_count')); \
print('Cum PnL:',  s.get('total_pnl_pct'), '%'); \
"

# ── One-shot signal check (no trade) ──────────────────────────────────────────
check:
	docker compose exec bot python -c "\
import sys; sys.path.insert(0,'.'); \
from data import fetch_candles, get_mid_price; \
from signals import compute_signal; \
df = fetch_candles(); \
sig, dbg = compute_signal(df); \
print('Signal:', sig.upper()); \
[print(' ', k, ':', v) for k,v in dbg.items()]; \
print('Mid price: \$$%.4f' % get_mid_price()); \
"
