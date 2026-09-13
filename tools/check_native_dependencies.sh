#!/usr/bin/env bash
# Reject missing dependencies, forbidden tracing libraries, and ldd failures.
set -euo pipefail
dependencies="$(ldd "${1:?usage: check_native_dependencies.sh LIBRARY}")"
printf '%s\n' "$dependencies"
if printf '%s\n' "$dependencies" | grep -E 'not found|roctracer|rocprofiler-sdk|libamdhip64|libroctx64'; then
    echo 'Missing or forbidden native dependency' >&2
    exit 1
fi
