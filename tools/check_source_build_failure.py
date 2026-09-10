#!/usr/bin/env python3
"""A poisoned prebuilt library must not hide a failed source build."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sdist', type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='rtl-source-failure-') as directory:
        root = Path(directory)
        with tarfile.open(args.sdist.resolve()) as archive:
            archive.extractall(root, filter='data')
        source = next(p for p in root.iterdir() if p.is_dir())
        # Both stale locations existed in the previous packaging implementation.
        for path in [source / 'librtl.so', source / 'rocm_trace_lite/lib/librtl.so']:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'stale binary must not be packaged')
        result = subprocess.run(
            [sys.executable, '-m', 'build', '--wheel', '--no-isolation', '--outdir', str(root / 'out')],
            cwd=source, env=dict(os.environ, CXX='false'), capture_output=True, text=True,
        )
        output = result.stdout + result.stderr
        if result.returncode == 0 or 'Cannot build the native profiler' not in output:
            raise RuntimeError('Source build did not fail clearly after forced compiler failure:\n' + output[-4000:])
        if list((root / 'out').glob('*.whl')):
            raise RuntimeError('Failed source build produced a wheel')
        print('PASS: compiler failure rejected despite stale root/package libraries; no wheel emitted')


if __name__ == '__main__':
    main()
