#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT_DIR/scripts/_run_python.sh" DATASET=amazon WEIGHTED_LOSS=1 scripts/run_10seed_supervised_rdl_weighted.py "$@"