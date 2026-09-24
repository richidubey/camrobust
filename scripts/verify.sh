#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"

.venv/bin/python -m compileall -q src
.venv/bin/pytest

run_count="$(find results/runs -name metrics.json -type f | wc -l)"
if [[ "$run_count" -ne 66 ]]; then
  echo "Expected 66 definitive run records, found $run_count" >&2
  exit 1
fi

.venv/bin/camrobust-analyze

for figure in overview.pdf diagnostics.pdf qualitative_main.pdf; do
  if [[ ! -s "results/aggregate/$figure" ]]; then
    echo "Missing generated paper figure: results/aggregate/$figure" >&2
    exit 1
  fi
done

(
  cd paper
  ../tools/bin/tectonic -X compile main.tex --keep-logs
  ../tools/bin/tectonic -X compile supplementary.tex --keep-logs
)

if rg -n 'LaTeX Error|Undefined control sequence|Missing .* inserted|Emergency stop' paper/*.log; then
  echo "LaTeX validation found a fatal diagnostic" >&2
  exit 1
fi

echo "Verification complete."
