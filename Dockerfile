# syntax=docker/dockerfile:1.7
# ---- web map -------------------------------------------------------------------------------
FROM node:22-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# ---- python runtime ------------------------------------------------------------------------
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HIMSAT_DATA_DIR=/data HIMSAT_CONFIG_DIR=/app/config HIMSAT_WEB_DIST_DIR=/app/web/dist
ARG EXTRAS="postgres"
RUN apt-get update && apt-get install -y --no-install-recommends curl tini && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 himsat
WORKDIR /app
COPY pyproject.toml README.md ./
COPY himsat/ himsat/
RUN pip install ".[${EXTRAS}]"
COPY config/ config/
COPY data/assets/ /app/seed/assets/
COPY data/osm/ /app/seed/osm/
COPY --from=web /web/dist web/dist
COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && mkdir -p /data && chown himsat:himsat /data
USER himsat
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD curl -fs http://localhost:8000/api/health || exit 1
ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]
CMD ["himsat", "serve", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
