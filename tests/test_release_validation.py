"""Pre-release artifact validation (CPU only).

GPU E2E and microbench tests are in test_gpu_hip.py (HIP-native, no torch).
"""
import os
import re
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_PATH = os.path.join(REPO_ROOT, "rocm_trace_lite", "lib", "librtl.so")


class TestReleaseArtifacts:
    """Validate built artifacts."""

    def test_no_forbidden_deps(self):
        """librtl.so must not directly link roctracer/rocprofiler/libroctx64."""
        if not os.path.exists(LIB_PATH):
            pytest.skip("librtl.so not built")
        r = subprocess.run(["readelf", "-d", LIB_PATH], stdout=subprocess.PIPE,
                           universal_newlines=True)
        for dep in ["roctracer", "rocprofiler", "libroctx64"]:
            assert dep not in r.stdout, "Direct dep on {}".format(dep)

    def test_exports_onload(self):
        if not os.path.exists(LIB_PATH):
            pytest.skip("librtl.so not built")
        r = subprocess.run(["nm", "-D", LIB_PATH], stdout=subprocess.PIPE,
                           universal_newlines=True)
        assert "T OnLoad" in r.stdout
        assert "T OnUnload" in r.stdout

    def test_exports_all_roctx(self):
        if not os.path.exists(LIB_PATH):
            pytest.skip("librtl.so not built")
        r = subprocess.run(["nm", "-D", LIB_PATH], stdout=subprocess.PIPE,
                           universal_newlines=True)
        for sym in ["roctxRangePushA", "roctxRangePop", "roctxMarkA",
                    "roctxRangeStartA", "roctxRangeStop"]:
            assert sym in r.stdout, "Missing: {}".format(sym)

    def test_version_consistency(self):
        from rocm_trace_lite import __version__
        pyproject = os.path.join(REPO_ROOT, "pyproject.toml")
        with open(pyproject) as f:
            match = re.search(r'version = "([^"]+)"', f.read())
        assert match
        assert __version__ == match.group(1)

    def test_cli_version(self):
        from rocm_trace_lite import __version__
        env = os.environ.copy()
        env["PYTHONPATH"] = "{}:{}".format(REPO_ROOT, env.get("PYTHONPATH", ""))
        r = subprocess.run([sys.executable, "-m", "rocm_trace_lite.cli", "--version"],
                           stdout=subprocess.PIPE, universal_newlines=True, env=env)
        assert __version__ in r.stdout
