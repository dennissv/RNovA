#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./RNovA.sh <in_dir> <useunimod:true|false> <k>

Arguments:
  in_dir     Directory containing input .mgf files
  useunimod  Whether to use UniMod annotation: true or false
  k          Number of PTMs to consider for top-k selection

Options:
  -h, --help Show this help and exit
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 3 ]]; then
  usage
  exit 1
fi

in_dir="$1"
useunimod="$2"
k="$3"

if [[ "$useunimod" != "true" && "$useunimod" != "false" ]]; then
  echo "error: useunimod must be true or false" >&2
  exit 1
fi

args=(rnova run "$in_dir" --top-k-ptms "$k")
if [[ "$useunimod" == "true" ]]; then
  args+=(--use-unimod)
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required. Install uv, then run: uv sync" >&2
  exit 1
fi

uv run "${args[@]}"
