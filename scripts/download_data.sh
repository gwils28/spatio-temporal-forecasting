#!/usr/bin/env bash
# Télécharge METR-LA (Google Drive du dépôt DCRNN) et le graphe capteurs (GitHub).
set -euo pipefail
cd "$(dirname "$0")/.."
RAW=data/raw
mkdir -p "$RAW"
BASE=https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph
for f in adj_mx.pkl graph_sensor_ids.txt graph_sensor_locations.csv distances_la_2012.csv; do
  curl -sSfL -o "$RAW/$f" "$BASE/$f"
done
# Dossier Drive référencé dans le README de DCRNN (metr-la.h5, pems-bay.h5)
tmp=$(mktemp -d)
uv run gdown --folder "https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX" -O "$tmp"
mv "$tmp/metr-la.h5" "$RAW/" && rm -rf "$tmp"
echo "OK : $(ls "$RAW")"
