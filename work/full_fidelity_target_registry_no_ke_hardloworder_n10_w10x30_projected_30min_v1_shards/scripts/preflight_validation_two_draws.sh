#!/usr/bin/env bash
set -euo pipefail

# Run from a compute-appropriate environment, not a login/head node.
# This validates one ordinary target draw and one likely stress target draw.
ROOT="work/full_fidelity_target_registry_no_ke_hardloworder_n10_w10x30_projected_30min_v1_shards"
PREFLIGHT_ROOT="${PREFLIGHT_ROOT:-${ROOT}/preflight_validation_results}"
TMP_MANIFEST="${ROOT}/validation_two_draw_manifest.csv"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" - <<'PY'
import csv
from pathlib import Path

root = Path("work/full_fidelity_target_registry_no_ke_hardloworder_n10_w10x30_projected_30min_v1_shards")
rows = []
for target in ("61_CYG", "HR_2667_2668"):
    manifest = root / target / "shard_manifest.csv"
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["draw_index"] == "0":
                rows.append(row)
                break
out = root / "validation_two_draw_manifest.csv"
with out.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(out)
PY

mkdir -p "${PREFLIGHT_ROOT}"
PYTHONPATH=src "${PYTHON_BIN}" examples/scripts/check_full_fidelity_campaign_shards.py preflight \
  --manifest "${TMP_MANIFEST}" \
  --results-root "${PREFLIGHT_ROOT}"
