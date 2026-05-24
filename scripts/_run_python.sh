#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 [KEY=VALUE ...] path/to/script.py [args ...]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/rad/bin/activate" ]]; then
  source "$ROOT_DIR/rad/bin/activate"
fi

env_args=()
while [[ $# -gt 0 && "$1" == *=* ]]; do
  env_args+=("$1")
  shift
done

target_script="$1"
shift

if [[ ${#env_args[@]} -gt 0 ]]; then
  exec env "${env_args[@]}" python "$ROOT_DIR/$target_script" "$@"
fi

exec python "$ROOT_DIR/$target_script" "$@"