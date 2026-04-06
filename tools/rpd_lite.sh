#!/bin/bash
# rpd_lite.sh — Launch a command with rpd_lite profiling enabled
#
# Usage: rpd_lite.sh [options] command [args...]
#   -o FILE   Output trace file (default: trace.db)
#
# Example:
#   rpd_lite.sh python my_model.py
#   rpd_lite.sh -o model_trace.db python -m atom.serve ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB="${SCRIPT_DIR}/librpd_lite.so"

if [ ! -f "$LIB" ]; then
    # Try installed location
    LIB="/usr/local/lib/librpd_lite.so"
fi

if [ ! -f "$LIB" ]; then
    echo "Error: librpd_lite.so not found" >&2
    exit 1
fi

OUTPUT="trace.db"

while getopts "o:" opt; do
    case $opt in
        o) OUTPUT="$OPTARG" ;;
        *) echo "Usage: $0 [-o output.db] command [args...]" >&2; exit 1 ;;
    esac
done
shift $((OPTIND-1))

if [ $# -eq 0 ]; then
    echo "Usage: $0 [-o output.db] command [args...]" >&2
    exit 1
fi

rm -f "$OUTPUT"
echo "rpd_lite: tracing to $OUTPUT"

RPD_LITE_OUTPUT="$OUTPUT" \
HSA_TOOLS_LIB="$LIB" \
"$@"

EXIT_CODE=$?

if [ -f "$OUTPUT" ]; then
    echo ""
    echo "=== Trace Summary ==="
    sqlite3 "$OUTPUT" "SELECT '  API calls: ' || COUNT(*) FROM rocpd_api;" 2>/dev/null
    sqlite3 "$OUTPUT" "SELECT '  GPU ops:   ' || COUNT(*) FROM rocpd_op;" 2>/dev/null
    echo ""
    echo "=== Top GPU Kernels ==="
    sqlite3 -header -column "$OUTPUT" "SELECT * FROM top LIMIT 10;" 2>/dev/null
    echo ""
    echo "Trace saved to: $OUTPUT"
fi

exit $EXIT_CODE
