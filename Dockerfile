# Quant Portfolio-Kaizen — production container
# Multi-stage build: slim runtime, no build tools in final image.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install system deps required by pandas/pyarrow/numpy/matplotlib at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       libgomp1 \
       curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps separately to leverage Docker layer cache.
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy the application source.
COPY . .

# Streamlit listens on 8501 by default; expose it.
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# Use 0.0.0.0 so the container is reachable from outside.
ENTRYPOINT ["python", "-m", "streamlit", "run", "stockpicker_app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]
