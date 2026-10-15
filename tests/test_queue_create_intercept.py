"""Conformance tests for RTL's HSA queue-creation interception.

Drives the compiled librtl.so through tests/mock_hsa_table.cpp, a synthetic
HsaApiTable whose entries are recording stubs. No GPU and no real HSA runtime
are involved, so these run anywhere the ROCm headers are installed.

Each CHECK line the harness prints becomes one test case here. Skips cleanly
when the headers predate ROCm 10 (the descriptor API does not exist) or when
librtl.so has not been built.
"""
import os
import shutil
import subprocess

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARNESS_SRC = os.path.join(REPO_ROOT, "tests", "mock_hsa_table.cpp")
LIBRTL = os.path.join(REPO_ROOT, "librtl.so")
ROCM = os.environ.get("HIP_PATH", "/opt/rocm")

# Keep these in sync with the check() calls in mock_hsa_table.cpp. Listing them
# means a harness that dies early shows up as missing results, not as a pass.

# Always expected: these exercise the HSA API table ABI, which every build has.
ABI_CHECKS = [
    "abi_hook_install_guarded",
    "abi_table_copy_bounded",
]

# Only meaningful once the build intercepts hsa_amd_queue_create.
DESCRIPTOR_CHECKS = [
    "hook_installed",
    "queue_size_in_packets",
    "cu_mask_forwarded",
    "nonzero_flags_not_dropped",
    "unknown_desc_version_rejected",
    "zero_queue_size_rejected",
    "default_private_segment_preserved",
    "sdma_passthrough",
    "priority_applied",
    "profiling_enabled",
    "batch_mixed_engines",
    "null_and_empty_args",
    "queue_out_field_owned",
    "intercept_failure_fallback",
]

EXPECTED_CHECKS = ABI_CHECKS + DESCRIPTOR_CHECKS


def _requirements():
    if not shutil.which("g++"):
        return "g++ not available"
    if not os.path.exists(os.path.join(ROCM, "include", "hsa", "hsa_api_trace.h")):
        return f"ROCm HSA headers not found under {ROCM}"
    if not os.path.exists(LIBRTL):
        return "librtl.so not built (run `make` first)"
    return None


@pytest.fixture(scope="module")
def harness_results(tmp_path_factory):
    reason = _requirements()
    if reason:
        pytest.skip(reason)

    binary = str(tmp_path_factory.mktemp("mock_hsa") / "mock_hsa_table")
    build = subprocess.run(
        ["g++", "-std=c++17", "-g", "-O0", "-rdynamic",
         f"-I{ROCM}/include", f"-I{ROCM}/include/hsa",
         "-D__HIP_PLATFORM_AMD__", "-DAMD_INTERNAL_BUILD",
         "-o", binary, HARNESS_SRC,
         f"-L{REPO_ROOT}", f"-Wl,-rpath,{REPO_ROOT}", "-lrtl",
         f"-L{ROCM}/lib", f"-Wl,-rpath,{ROCM}/lib", "-lhsa-runtime64"],
        capture_output=True, text=True,
    )
    if build.returncode != 0:
        pytest.skip(f"harness did not build:\n{build.stderr[-2000:]}")

    env = dict(os.environ, RTL_OUTPUT=str(tmp_path_factory.mktemp("rtl") / "trace.db"))
    run = subprocess.run([binary], capture_output=True, text=True, timeout=120, env=env)

    results = {}
    for line in run.stdout.splitlines():
        if not line.startswith("CHECK "):
            continue
        _, name, verdict, detail = line.split(" ", 3)
        results[name] = (verdict, detail)

    if results.get("rocm10_headers", ("",))[0] == "SKIP":
        pytest.skip("ROCm < 10 headers: hsa_amd_queue_create descriptor API absent")
    if results.get("hook_installed", ("",))[0] == "SKIP":
        # Pre-ROCm-10 build: the ABI checks still apply, the descriptor ones do not.
        note = "this librtl.so does not intercept hsa_amd_queue_create"
        for name in DESCRIPTOR_CHECKS:
            results.setdefault(name, ("SKIP", note))
        results["hook_installed"] = ("SKIP", note)
    if not results:
        pytest.fail(
            f"harness produced no CHECK lines (rc={run.returncode})\n"
            f"stdout:\n{run.stdout[-2000:]}\nstderr:\n{run.stderr[-2000:]}"
        )
    return results


@pytest.mark.parametrize("name", EXPECTED_CHECKS)
def test_check(harness_results, name):
    assert name in harness_results, (
        f"harness did not report {name}; it likely aborted earlier"
    )
    verdict, detail = harness_results[name]
    if verdict == "SKIP":
        pytest.skip(detail)
    if verdict == "XFAIL":
        # Known gap, tracked upstream; the harness carries the reason.
        pytest.xfail(detail)
    assert verdict == "PASS", detail
