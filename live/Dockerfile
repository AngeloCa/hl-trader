FROM python:3.12-slim

# Security: run as non-root
RUN useradd -m -u 1000 bot
WORKDIR /app

# Install deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY *.py ./

# State and logs live on mounted volumes — not baked into image
VOLUME ["/app/data"]

USER bot

HEALTHCHECK --interval=5m --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import requests; requests.post('https://api.hyperliquid.xyz/info', json={'type':'allMids'}, timeout=5)" || exit 1

CMD ["python", "bot.py"]
