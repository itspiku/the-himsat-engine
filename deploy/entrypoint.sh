#!/bin/sh
set -e
# seed bundled reference data (curated assets, cached OSM extracts) into the data volume once
mkdir -p "$HIMSAT_DATA_DIR/assets" "$HIMSAT_DATA_DIR/osm"
cp -rn /app/seed/assets/. "$HIMSAT_DATA_DIR/assets/" 2>/dev/null || true
cp -rn /app/seed/osm/. "$HIMSAT_DATA_DIR/osm/" 2>/dev/null || true
himsat db upgrade
exec "$@"
