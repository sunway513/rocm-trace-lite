#!/usr/bin/env python3
"""Require lossless full-mode capture of the native graph stress workload.

Requires the ROCm 10 runtime from benchmarks/e2e/Dockerfile. Each selected
physical GPU runs an isolated process; an empty trace is always a failure.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--devices', default='0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    executable = args.output / 'graph_stress'
    subprocess.run(['hipcc', '-O2', '--offload-arch=gfx950', '-o', str(executable),
                    str(repo / 'repro/repro_hipgraph_stress.hip')], check=True)

    def run(device):
        db = args.output / f'gpu-{device}.db'
        env = dict(os.environ, HIP_VISIBLE_DEVICES=str(device),
                   PYTHONPATH=str(repo))
        cmd = [sys.executable, '-m', 'rocm_trace_lite.cli', 'trace', '--mode', 'full',
               '-o', str(db), '--', str(executable)]
        log_path = args.output / f'gpu-{device}.log'
        with log_path.open('w') as log:
            result = subprocess.run(cmd, env=env, stdout=log, stderr=log, timeout=180)
        if result.returncode:
            raise RuntimeError(f'GPU {device}: workload failed, see {log_path}')
        with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as conn:
            count = conn.execute(
                "SELECT count(*) FROM rocpd_op o JOIN rocpd_string s "
                "ON s.id=o.description_id WHERE s.string LIKE '%vec_add%'"
            ).fetchone()[0]
            integrity = conn.execute('PRAGMA integrity_check').fetchone()[0]
        # Keep synchronized with NUM_KERNELS and BATCH_SIZES in the workload.
        expected = 256 * sum([10, 20, 50, 50, 50, 50, 50, 100, 100, 100])
        return dict(device=device, kernels=count, expected=expected,
                    integrity=integrity, passed=count == expected and integrity == 'ok')

    devices = [int(d) for d in args.devices.split(',')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        results = list(pool.map(run, devices))
    (args.output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))
    return 0 if all(r['passed'] for r in results) else 1


if __name__ == '__main__':
    sys.exit(main())
