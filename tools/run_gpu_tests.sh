#!/usr/bin/env bash
# Preserve pytest's status through tee and require actual non-skipped tests.
set -euo pipefail
log=$1
minimum=$2
shift 2
python3 -m pytest "$@" -v --junitxml="${log}.xml" 2>&1 | tee "$log"
python3 - "${log}.xml" "$minimum" <<'PY'
import sys
import xml.etree.ElementTree as ET
cases = list(ET.parse(sys.argv[1]).iter('testcase'))
passed = sum(not any(c.tag in ('failure', 'error', 'skipped') for c in case) for case in cases)
if passed < int(sys.argv[2]):
    raise SystemExit(f'GPU coverage gate: {passed} passed, need {sys.argv[2]}')
print(f'GPU coverage gate: {passed} passed')
PY
