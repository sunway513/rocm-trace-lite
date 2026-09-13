#!/usr/bin/env python3
"""Require all six benchmark-integrity cases, including the workload trace check."""
import sys
import xml.etree.ElementTree as ET


def check(path):
    cases = ET.parse(path).getroot().findall('.//testcase')
    assert len(cases) >= 6, f'Expected at least six executed cases, got {len(cases)}'
    assert all(not list(case) or all(child.tag not in {'skipped', 'failure', 'error'}
                                   for child in case) for case in cases), 'Non-passing test case'
    assert any(case.get('name') == 'test_torch_profiles_workload_process' for case in cases), 'Missing workload profiler check'
    print(f'Benchmark gate passed: {len(cases)} executed, no skips/failures/errors')


if __name__ == '__main__':
    check(sys.argv[1])
