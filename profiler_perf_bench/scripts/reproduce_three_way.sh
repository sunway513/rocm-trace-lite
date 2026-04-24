#!/usr/bin/env bash
# reproduce_three_way.sh — one-click reproducer for the three-way profiler overhead benchmark.
#
# Usage:
#   bash profiler_perf_bench/scripts/reproduce_three_way.sh [OUTPUT_DIR]
#
# Default OUTPUT_DIR: $HOME
# Output file: $OUTPUT_DIR/three_way_result.json
#
# Requirements (pre-installed in ROCm docker image):
#   rocprofv3, rocprof, hipcc, python3 >= 3.10, /dev/kfd accessible
#
# Reference: https://github.com/sunway513/rocm-trace-lite/issues
#   See the "Three-way profiler overhead benchmark" issue for expected numbers,
#   tolerance bands, and troubleshooting guidance.
#
# Total runtime target: < 6 minutes on any modern MI-series GPU.

set -euo pipefail

# ── Output directory (optional $1 override) ────────────────────────────────
OUTPUT_DIR="${1:-$HOME}"
OUTPUT_JSON="${OUTPUT_DIR}/three_way_result.json"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PRESET="${REPO_ROOT}/profiler_perf_bench/presets/three_way.yaml"

# ── Color helpers ───────────────────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*" >&2; exit 1; }

echo ""
echo "======================================================="
echo "  Three-Way Profiler Overhead Benchmark — Reproducer"
echo "  Repo: ${REPO_ROOT}"
echo "  Output: ${OUTPUT_JSON}"
echo "======================================================="
echo ""

# ── STEP 1: Preflight checks ────────────────────────────────────────────────
info "Step 1/6: Preflight checks"

# /dev/kfd accessible
if [[ ! -r /dev/kfd ]]; then
    fail "/dev/kfd not readable. Run container with --device /dev/kfd --device /dev/dri or equivalent."
fi
info "  /dev/kfd: accessible"

# rocprofv3 on PATH
if ! command -v rocprofv3 &>/dev/null; then
    fail "rocprofv3 not found on PATH. Install ROCm >= 6.0 or use rocm/dev-ubuntu-24.04:7.2 image."
fi
ROCPROFV3_VER="$(rocprofv3 --version 2>&1 | head -1 || true)"
info "  rocprofv3: ${ROCPROFV3_VER}"

# rocprof on PATH
if ! command -v rocprof &>/dev/null; then
    fail "rocprof not found on PATH. Install ROCm or use the reference docker image."
fi
ROCPROF_VER="$(rocprof --version 2>&1 | head -1 || true)"
info "  rocprof: ${ROCPROF_VER}"

# hipcc on PATH
if ! command -v hipcc &>/dev/null; then
    fail "hipcc not found on PATH. Install ROCm HIP SDK."
fi
HIPCC_VER="$(hipcc --version 2>&1 | head -1 || true)"
info "  hipcc: ${HIPCC_VER}"

# python3 >= 3.10
if ! command -v python3 &>/dev/null; then
    fail "python3 not found on PATH."
fi
PY_VERSION="$(python3 -c 'import sys; print(".".join(map(str,sys.version_info[:3])))')"
PY_MAJOR="$(python3 -c 'import sys; print(sys.version_info.major)')"
PY_MINOR="$(python3 -c 'import sys; print(sys.version_info.minor)')"
if [[ "${PY_MAJOR}" -lt 3 ]] || { [[ "${PY_MAJOR}" -eq 3 ]] && [[ "${PY_MINOR}" -lt 10 ]]; }; then
    fail "python3 >= 3.10 required, found ${PY_VERSION}."
fi
info "  python3: ${PY_VERSION}"

# GPU present
GPU_INFO="$(rocminfo 2>/dev/null | grep -i 'gfx' | head -3 || true)"
info "  GPU: ${GPU_INFO:-<rocminfo not available>}"

echo ""

# ── STEP 2: Build librtl.so ─────────────────────────────────────────────────
info "Step 2/6: Build librtl.so (make clean && make)"
cd "${REPO_ROOT}"
make clean
make
LIBRTL="${REPO_ROOT}/librtl.so"
if [[ ! -f "${LIBRTL}" ]]; then
    fail "librtl.so not found after build at ${LIBRTL}"
fi
BUILT_MD5="$(md5sum "${LIBRTL}" | awk '{print $1}')"
info "  librtl.so md5: ${BUILT_MD5}"
info "  Reference md5: 6c09b15e12fc25c2a64ad9bcb154573a (MI355X ROCm 6.16.13 build)"
if [[ "${BUILT_MD5}" != "6c09b15e12fc25c2a64ad9bcb154573a" ]]; then
    warn "  md5 mismatch — expected for a different ROCm/compiler version. Numbers may differ slightly."
fi

# ── STEP 3: Build HIP gpu_workload binary ───────────────────────────────────
info "Step 3/6: Build tests/gpu_workload (hipcc)"
make tests/gpu_workload
GPU_WORKLOAD="${REPO_ROOT}/tests/gpu_workload"
if [[ ! -x "${GPU_WORKLOAD}" ]]; then
    fail "tests/gpu_workload not built at ${GPU_WORKLOAD}"
fi
info "  gpu_workload binary: OK"

# Copy librtl.so into package lib dir
cp "${LIBRTL}" "${REPO_ROOT}/rocm_trace_lite/lib/librtl.so"
info "  librtl.so copied to rocm_trace_lite/lib/"

echo ""

# ── STEP 4: Install Python package ─────────────────────────────────────────
info "Step 4/6: pip install -e . (editable install)"
if pip install --break-system-packages -e . -q 2>/dev/null; then
    info "  pip install (--break-system-packages): OK"
elif pip install -e . -q; then
    info "  pip install (no flag): OK"
else
    fail "pip install -e . failed. Check pip and Python environment."
fi

echo ""

# ── STEP 5: Adapter sanity check ────────────────────────────────────────────
info "Step 5/6: Adapter list sanity"
ADAPTER_OUT="$(python3 -m profiler_perf_bench.cli adapter-list 2>&1)"
echo "${ADAPTER_OUT}"

REQUIRED_ADAPTERS=("none" "rtl" "rtl_standard" "rocprofv3" "rocprof")
for adapter in "${REQUIRED_ADAPTERS[@]}"; do
    if ! echo "${ADAPTER_OUT}" | grep -q "^${adapter}"; then
        fail "Required adapter '${adapter}' not found in adapter-list output."
    fi
done
info "  All 5 required adapters present: ${REQUIRED_ADAPTERS[*]}"

echo ""

# ── STEP 6: Run benchmark sweep ─────────────────────────────────────────────
info "Step 6/6: Running three-way benchmark sweep (~3-6 minutes)"
info "  Config:  ${PRESET}"
info "  Output:  ${OUTPUT_JSON}"
info "  Rounds:  3 (per adapter × workload pair)"
echo ""

mkdir -p "${OUTPUT_DIR}"
python3 -m profiler_perf_bench.cli run \
    --config "${PRESET}" \
    --rounds 3 \
    --output "${OUTPUT_JSON}"

if [[ ! -f "${OUTPUT_JSON}" ]]; then
    fail "Output JSON not found at ${OUTPUT_JSON} after benchmark run."
fi
info "Benchmark complete. Results written to: ${OUTPUT_JSON}"

echo ""
echo "======================================================="
echo "  ANALYSIS — Median wall-time overhead table"
echo "======================================================="

export _THREE_WAY_JSON="${OUTPUT_JSON}"
python3 - <<'PYEOF'
import json
import sys
import os
import statistics
from pathlib import Path

result_path = os.environ.get("_THREE_WAY_JSON", "")
if not result_path or not Path(result_path).exists():
    print("[WARN] Cannot find result JSON for analysis — set _THREE_WAY_JSON env.")
    sys.exit(0)

with open(result_path) as f:
    data = json.load(f)

runs = data.get("runs", [])

# Group by adapter + workload
from collections import defaultdict
groups = defaultdict(list)
for r in runs:
    if r.get("run_succeeded"):
        key = (r["adapter"], r["workload"])
        groups[key].append(r["metrics"]["wall_s"])

# Build baseline medians per workload
baseline = {}
for (adapter, workload), vals in groups.items():
    if adapter == "none":
        baseline[workload] = statistics.median(vals)

# Adapter display order: none first, then rtl, rtl_standard, rocprofv3, rocprof
ADAPTER_ORDER = ["none", "rtl", "rtl_standard", "rocprofv3", "rocprof"]
WORKLOADS_OF_INTEREST = ["L1-gemm-steady", "L1-short-kernels"]

print(f"\n{'Adapter':<16} | ", end="")
for wl in WORKLOADS_OF_INTEREST:
    short = wl.replace("L1-", "")
    print(f"{'Δms '+short:<18} {'Δ% '+short:<14}", end=" | ")
print()
print("-" * 90)

all_rtl_pcts = []
all_rocprofv3_pcts = []
all_rocprof_pcts = []

for adapter in ADAPTER_ORDER:
    row = f"{adapter:<16} | "
    for wl in WORKLOADS_OF_INTEREST:
        bl = baseline.get(wl)
        vals = groups.get((adapter, wl), [])
        if not vals or bl is None:
            row += f"{'N/A':<18} {'N/A':<14}   "
            continue
        med = statistics.median(vals)
        delta_ms = (med - bl) * 1000.0
        delta_pct = ((med - bl) / bl * 100.0) if bl > 0 else float("nan")
        row += f"{delta_ms:>+10.1f} ms    {delta_pct:>+7.1f}%       "
        if adapter in ("rtl", "rtl_standard"):
            all_rtl_pcts.append(delta_pct)
        elif adapter == "rocprofv3":
            all_rocprofv3_pcts.append(delta_pct)
        elif adapter == "rocprof":
            all_rocprof_pcts.append(delta_pct)
    print(row)

print("-" * 90)
print()

# Sanity gate: RTL < 20%, rocprofv3 > 100%, rocprof > 200%
rtl_max = max(all_rtl_pcts) if all_rtl_pcts else float("nan")
rocprofv3_min = min(all_rocprofv3_pcts) if all_rocprofv3_pcts else float("nan")
rocprof_min = min(all_rocprof_pcts) if all_rocprof_pcts else float("nan")

gate_rtl = rtl_max < 20.0
gate_rocprofv3 = rocprofv3_min > 100.0
gate_rocprof = rocprof_min > 200.0

print(f"Sanity gate:")
print(f"  RTL max overhead       = {rtl_max:+.1f}%  (must be < 20%)    → {'PASS' if gate_rtl else 'FAIL'}")
print(f"  rocprofv3 min overhead = {rocprofv3_min:+.1f}% (must be > 100%)  → {'PASS' if gate_rocprofv3 else 'FAIL'}")
print(f"  rocprof min overhead   = {rocprof_min:+.1f}% (must be > 200%)  → {'PASS' if gate_rocprof else 'FAIL'}")
print()

if gate_rtl and gate_rocprofv3 and gate_rocprof:
    print("\033[0;32m[SANITY PASS] Results are in expected order of magnitude.\033[0m")
    print("  RTL overhead is ≪ rocprofv3 ≪ rocprof — consistent with reference MI355X run.")
else:
    print("\033[0;31m[SANITY FAIL] One or more gates outside expected range.\033[0m")
    print("  See GitHub Issue for troubleshooting: https://github.com/sunway513/rocm-trace-lite/issues")
    print("  Checklist:")
    print("  (a) librtl.so md5 mismatch → rebuild")
    print("  (b) rocprofv3 version drift → rocprofv3 --version should be >= 0.7")
    print("  (c) gfx model mismatch → rocminfo | grep gfx")
    print("  (d) concurrent GPU users → rocm-smi --showpids should be clean")
PYEOF

echo ""
info "Reproduce complete. Please attach ${OUTPUT_JSON} to the GitHub Issue."
info "Issue: https://github.com/sunway513/rocm-trace-lite/issues"
