#!/usr/bin/env bash
# Lance toutes les expériences sur deux files GPU parallèles.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
run() { for c in "$@"; do uv run python -m stforecast.train --config "configs/$c.yaml" > "logs/$c.log" 2>&1; done; }
run gwnet gwnet_road &
run stid stid_no_node stid_no_time stid_no_ids gwnet_adaptive gwnet_nograph &
wait
echo "ALL DONE"
