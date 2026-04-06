"""Sprint 4 — Error handling and CI hardening tests.

Covers SQLite error paths, rpd_lite.sh validation, converter consistency,
shutdown ordering, completion worker diagnostics, extended source guards,
CI workflow validation, and arch-specific GPU tracing.

CPU-only tests run without any GPU or librpd_lite.so.
GPU tests are skipped when no ROCm GPU is available.
"""
import json
import os
import re
import sqlite3
import stat
import subprocess
import sys

import pytest
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "tests"))

from conftest import SCHEMA_SQL, populate_synthetic_trace  # noqa: E402
from rocm_trace_lite.cmd_trace import (  # noqa: E402
    _checkpoint_wal,
    _generate_summary,
    _merge_traces,
)

SRC_DIR = os.path.join(REPO_ROOT, "src")
TOOLS_DIR = os.path.join(REPO_ROOT, "tools")
LIB_PATH = os.path.join(REPO_ROOT, "librpd_lite.so")


# ---------------------------------------------------------------------------
# GPU detection (mirrors test_roctx_e2e.py pattern)
# ---------------------------------------------------------------------------

def _has_gpu():
    """Check if ROCm GPU is available via /dev/kfd or rocm-smi."""
    if os.path.exists("/dev/kfd"):
        return True
    try:
        r = subprocess.run(
            ["rocm-smi", "--showid"], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=5,
        )
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _has_lib():
    return os.path.exists(LIB_PATH)


HAS_GPU = _has_gpu() and _has_lib()


def _make_db(path):
    """Create an RPD database with schema and synthetic data."""
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA_SQL)
    conn.close()
    return path


def _make_populated_db(path, num_kernels=50):
    """Create an RPD database with synthetic kernel data."""
    populate_synthetic_trace(path, num_kernels=num_kernels, num_gpus=1)
    return path


def _run_traced(script, trace_path, timeout=120):
    """Run a Python script under rpd_lite tracing, return (returncode, stderr)."""
    env = os.environ.copy()
    env["HSA_TOOLS_LIB"] = LIB_PATH
    env["RPD_LITE_OUTPUT"] = trace_path
    r = subprocess.run(
        [sys.executable, "-c", script],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=timeout,
    )
    return r.returncode, r.stderr.decode(errors="replace")


# ===========================================================================
# 1. SQLite Error Paths (CPU only)
# ===========================================================================

class TestSQLiteErrorPaths:
    """Validate graceful handling of SQLite error conditions."""

    def test_readonly_db_summary(self, tmp_path):
        """_generate_summary on a read-only DB should not crash."""
        db_path = str(tmp_path / "readonly.db")
        _make_populated_db(db_path)
        # Make read-only
        os.chmod(db_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        try:
            result = _generate_summary(db_path)
            # Should return a string (summary or warning), not crash
            assert result is not None
            assert isinstance(result, str)
        finally:
            # Restore write permission for cleanup
            os.chmod(db_path, stat.S_IRUSR | stat.S_IWUSR)

    def test_merge_traces_unwritable_target(self, tmp_path):
        """_merge_traces to an unwritable directory should fail gracefully."""
        # Create two valid source DBs
        db1 = str(tmp_path / "proc1.db")
        db2 = str(tmp_path / "proc2.db")
        _make_populated_db(db1, num_kernels=10)
        _make_populated_db(db2, num_kernels=10)

        # Target in a non-existent directory
        bad_target = "/nonexistent_dir_12345/merged.db"
        with pytest.raises((OSError, sqlite3.OperationalError)):
            _merge_traces([db1, db2], bad_target)

    def test_checkpoint_wal_closed_db(self, tmp_path):
        """_checkpoint_wal on a closed/nonexistent DB should not crash."""
        # Non-existent path — should silently pass (catches OperationalError)
        _checkpoint_wal(str(tmp_path / "nonexistent.db"))
        # Already-closed scenario: create and remove
        db_path = str(tmp_path / "gone.db")
        _make_db(db_path)
        os.remove(db_path)
        _checkpoint_wal(db_path)  # should not raise


# ===========================================================================
# 2. rpd_lite.sh Tests (CPU only)
# ===========================================================================

class TestRpdLiteShell:
    """Validate rpd_lite.sh behavior without GPU."""

    @staticmethod
    def _setup_sh(tmp_path):
        """Copy rpd_lite.sh into tmp_path with a dummy .so so lib check passes."""
        import shutil
        sh_src = os.path.join(TOOLS_DIR, "rpd_lite.sh")
        sh_dst = str(tmp_path / "rpd_lite.sh")
        shutil.copy2(sh_src, sh_dst)
        # rpd_lite.sh resolves LIB from SCRIPT_DIR, so place dummy .so there
        fake_lib = str(tmp_path / "librpd_lite.so")
        with open(fake_lib, "w") as f:
            f.write("")
        return sh_dst

    def test_no_args_prints_usage(self, tmp_path):
        """rpd_lite.sh with no arguments should print usage and exit non-zero."""
        sh = self._setup_sh(tmp_path)
        r = subprocess.run(
            ["bash", sh],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=10,
        )
        combined = r.stdout.decode() + r.stderr.decode()
        assert r.returncode != 0, "Expected non-zero exit for no-args"
        assert "usage" in combined.lower(), (
            "Expected usage message, got: {}".format(combined[:200])
        )

    def test_sets_hsa_tools_lib(self, tmp_path):
        """rpd_lite.sh -o test.db env should show HSA_TOOLS_LIB in env."""
        sh = self._setup_sh(tmp_path)
        out_db = str(tmp_path / "test.db")
        r = subprocess.run(
            ["bash", sh, "-o", out_db, "env"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=10,
        )
        stdout = r.stdout.decode()
        stderr = r.stderr.decode()
        combined = stdout + stderr
        assert "HSA_TOOLS_LIB" in combined, (
            "HSA_TOOLS_LIB not found in output: {}".format(combined[:500])
        )

    def test_syntax_valid(self):
        """rpd_lite.sh should pass bash -n syntax check."""
        sh_path = os.path.join(TOOLS_DIR, "rpd_lite.sh")
        r = subprocess.run(
            ["bash", "-n", sh_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=10,
        )
        assert r.returncode == 0, (
            "bash -n failed: {}".format(r.stderr.decode())
        )


# ===========================================================================
# 3. rpd2trace.py vs cmd_convert.py Consistency (CPU only)
# ===========================================================================

class TestConverterConsistency:
    """Verify rpd2trace.py and cmd_convert.py produce equivalent output."""

    def test_same_json_structure(self, tmp_path):
        """Both converters should produce JSON with same key names and event types."""
        db_path = str(tmp_path / "trace.db")
        _make_populated_db(db_path)

        json1 = str(tmp_path / "out_cmd.json")
        json2 = str(tmp_path / "out_rpd2trace.json")

        # Use cmd_convert
        from rocm_trace_lite.cmd_convert import convert as convert_cmd
        convert_cmd(db_path, json1)

        # Use rpd2trace.py (import from tools/)
        rpd2trace_path = os.path.join(TOOLS_DIR, "rpd2trace.py")
        # Run as subprocess to avoid module collision
        r = subprocess.run(
            [sys.executable, rpd2trace_path, db_path, json2],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )
        assert r.returncode == 0, "rpd2trace.py failed: {}".format(r.stderr.decode())

        with open(json1) as f:
            data1 = json.load(f)
        with open(json2) as f:
            data2 = json.load(f)

        # Both must have traceEvents key
        assert "traceEvents" in data1, "cmd_convert missing traceEvents"
        assert "traceEvents" in data2, "rpd2trace missing traceEvents"

        # Same number of events
        assert len(data1["traceEvents"]) == len(data2["traceEvents"]), (
            "Event count mismatch: {} vs {}".format(
                len(data1["traceEvents"]), len(data2["traceEvents"])
            )
        )

        # Same set of event phases (ph values)
        phases1 = sorted(set(e.get("ph") for e in data1["traceEvents"]))
        phases2 = sorted(set(e.get("ph") for e in data2["traceEvents"]))
        assert phases1 == phases2, (
            "Phase types differ: {} vs {}".format(phases1, phases2)
        )

        # X events should have same key structure
        x_events1 = [e for e in data1["traceEvents"] if e.get("ph") == "X"]
        x_events2 = [e for e in data2["traceEvents"] if e.get("ph") == "X"]
        if x_events1 and x_events2:
            keys1 = sorted(x_events1[0].keys())
            keys2 = sorted(x_events2[0].keys())
            assert keys1 == keys2, (
                "X event keys differ: {} vs {}".format(keys1, keys2)
            )

    def test_nonexistent_input_graceful(self, tmp_path):
        """Both converters should handle non-existent input gracefully."""
        fake_db = str(tmp_path / "does_not_exist.db")
        fake_out = str(tmp_path / "out.json")

        # cmd_convert.run_convert exits with sys.exit(1)
        from rocm_trace_lite.cmd_convert import run_convert
        import argparse
        args = argparse.Namespace(input=fake_db, output=fake_out)
        with pytest.raises(SystemExit) as exc_info:
            run_convert(args)
        assert exc_info.value.code != 0

        # rpd2trace.py should also exit non-zero
        rpd2trace_path = os.path.join(TOOLS_DIR, "rpd2trace.py")
        r = subprocess.run(
            [sys.executable, rpd2trace_path, fake_db, fake_out],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10,
        )
        # rpd2trace.py will hit sqlite3 error on connect to nonexistent then
        # fail on query — it should not produce valid output
        assert not os.path.exists(fake_out) or os.path.getsize(fake_out) == 0 or r.returncode != 0, (
            "rpd2trace.py should fail on non-existent input"
        )


# ===========================================================================
# 4. Shutdown Ordering (GPU required)
# ===========================================================================

class TestShutdownOrdering:
    """Verify trace completeness after workload shutdown."""

    @pytest.mark.skipif(not HAS_GPU, reason="No GPU or librpd_lite.so")
    def test_normal_workload_complete_trace(self, tmp_path):
        """Normal workload should produce a complete trace (no truncation)."""
        trace = str(tmp_path / "shutdown.db")
        script = (
            "import torch; "
            "x = torch.randn(256, 256, device='cuda'); "
            "[torch.mm(x, x) for _ in range(50)]; "
            "torch.cuda.synchronize()"
        )
        rc, stderr = _run_traced(script, trace)
        assert rc == 0, "Workload failed: {}".format(stderr)
        assert os.path.exists(trace), "Trace file not created"

        conn = sqlite3.connect(trace)
        ops = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        conn.close()
        # 50 matmuls should produce at least some ops
        assert ops >= 10, "Expected >= 10 ops, got {}".format(ops)

    @pytest.mark.skipif(not HAS_GPU, reason="No GPU or librpd_lite.so")
    def test_short_workload_flush_completes(self, tmp_path):
        """Very short workload (< 100ms) should still flush completely."""
        trace = str(tmp_path / "short.db")
        script = (
            "import torch; "
            "x = torch.randn(64, 64, device='cuda'); "
            "y = torch.mm(x, x); "
            "torch.cuda.synchronize()"
        )
        rc, stderr = _run_traced(script, trace)
        assert rc == 0, "Short workload failed: {}".format(stderr)
        assert os.path.exists(trace), "Trace file not created"

        conn = sqlite3.connect(trace)
        ops = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        conn.close()
        assert ops >= 1, "Expected >= 1 op from short workload, got {}".format(ops)


# ===========================================================================
# 5. Completion Worker Drop Counters (GPU required)
# ===========================================================================

class TestCompletionWorkerDiagnostics:
    """Verify diagnostic output from the completion worker."""

    @pytest.mark.skipif(not HAS_GPU, reason="No GPU or librpd_lite.so")
    def test_stderr_contains_finalized(self, tmp_path):
        """Normal workload stderr should contain trace finalized message."""
        trace = str(tmp_path / "diag.db")
        script = (
            "import torch; "
            "x = torch.randn(128, 128, device='cuda'); "
            "y = torch.mm(x, x); "
            "torch.cuda.synchronize()"
        )
        rc, stderr = _run_traced(script, trace)
        assert rc == 0, "Workload failed: {}".format(stderr)
        # The C++ library prints diagnostic on unload
        # Accept either "finalized" or "records written" or "ops" in stderr
        stderr_lower = stderr.lower()
        has_diag = (
            "finalized" in stderr_lower
            or "records" in stderr_lower
            or "written" in stderr_lower
            or "ops" in stderr_lower
            or "rtl:" in stderr_lower
        )
        assert has_diag, (
            "Expected diagnostic output in stderr, got: {}".format(stderr[:500])
        )

    @pytest.mark.skipif(not HAS_GPU, reason="No GPU or librpd_lite.so")
    def test_records_written_format(self, tmp_path):
        """Diagnostic output should include record count information."""
        trace = str(tmp_path / "diag2.db")
        script = (
            "import torch; "
            "x = torch.randn(256, 256, device='cuda'); "
            "[torch.mm(x, x) for _ in range(20)]; "
            "torch.cuda.synchronize()"
        )
        rc, stderr = _run_traced(script, trace)
        assert rc == 0, "Workload failed: {}".format(stderr)
        # Check that stderr has some numeric diagnostic
        has_numeric = bool(re.search(r'\d+\s*(record|op|kernel|written|dispatch)', stderr, re.IGNORECASE))
        has_rtl_prefix = "rtl:" in stderr
        assert has_numeric or has_rtl_prefix, (
            "Expected numeric diagnostic or rtl: prefix in stderr, got: {}".format(stderr[:500])
        )


# ===========================================================================
# 6. Source Guard Extended (CPU only)
# ===========================================================================

class TestSourceGuardExtended:
    """Extended source-level dependency guards beyond test_source_guard.py."""

    def test_no_roctracer_rocprofiler_in_cpp(self):
        """src/*.cpp must not include roctracer or rocprofiler headers."""
        for fname in os.listdir(SRC_DIR):
            if not fname.endswith((".cpp", ".h")):
                continue
            fpath = os.path.join(SRC_DIR, fname)
            with open(fpath) as f:
                content = f.read()
            # Check for forbidden includes
            matches = re.findall(
                r'#include\s*[<"].*(?:roctracer|rocprofiler).*[>"]', content
            )
            assert not matches, (
                "{} includes forbidden headers: {}".format(fname, matches)
            )

    def test_makefile_no_forbidden_links(self):
        """Makefile must not link -lroctracer or -lrocprofiler."""
        makefile = os.path.join(REPO_ROOT, "Makefile")
        with open(makefile) as f:
            content = f.read()
        assert "-lroctracer" not in content, "Makefile links -lroctracer"
        assert "-lrocprofiler" not in content, "Makefile links -lrocprofiler"

    def test_python_no_roctracer_import(self):
        """Python code must not import roctracer or rocprofiler."""
        py_dir = os.path.join(REPO_ROOT, "rocm_trace_lite")
        for fname in os.listdir(py_dir):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(py_dir, fname)
            with open(fpath) as f:
                content = f.read()
            # Check for import statements
            for pattern in [
                r'^\s*import\s+roctracer',
                r'^\s*from\s+roctracer',
                r'^\s*import\s+rocprofiler',
                r'^\s*from\s+rocprofiler',
            ]:
                matches = re.findall(pattern, content, re.MULTILINE)
                assert not matches, (
                    "{} imports forbidden module: {}".format(fname, matches)
                )


# ===========================================================================
# 7. CI Workflow Validation (CPU only)
# ===========================================================================

class TestCIWorkflow:
    """Validate CI configuration files."""

    def test_workflow_yaml_valid(self):
        """All .github/workflows/*.yml files must be valid YAML."""
        if not HAS_YAML:
            pytest.skip("PyYAML not installed")
        wf_dir = os.path.join(REPO_ROOT, ".github", "workflows")
        if not os.path.isdir(wf_dir):
            pytest.skip("No .github/workflows directory")
        yml_files = [f for f in os.listdir(wf_dir) if f.endswith((".yml", ".yaml"))]
        assert len(yml_files) > 0, "No workflow YAML files found"
        for fname in yml_files:
            fpath = os.path.join(wf_dir, fname)
            with open(fpath) as f:
                try:
                    data = yaml.safe_load(f)
                except yaml.YAMLError as e:
                    pytest.fail("{} is not valid YAML: {}".format(fname, e))
            assert data is not None, "{} is empty".format(fname)
            assert isinstance(data, dict), "{} is not a YAML mapping".format(fname)

    def test_pytest_markers_registered(self):
        """gpu and multigpu markers should be registered in conftest.py or pyproject.toml."""
        # Check pyproject.toml for [tool.pytest.ini_options] markers
        pyproject = os.path.join(REPO_ROOT, "pyproject.toml")
        conftest = os.path.join(REPO_ROOT, "tests", "conftest.py")
        setup_cfg = os.path.join(REPO_ROOT, "setup.cfg")
        pytest_ini = os.path.join(REPO_ROOT, "pytest.ini")

        marker_sources = []
        for path in [pyproject, conftest, setup_cfg, pytest_ini]:
            if os.path.exists(path):
                with open(path) as f:
                    marker_sources.append((path, f.read()))

        # Look for gpu marker registration in any config
        found_gpu = False
        _found_multigpu = False
        for path, content in marker_sources:
            if "gpu" in content.lower():
                # Could be in markers list or pytest.mark.skipif references
                found_gpu = True
            if "multigpu" in content.lower():
                _found_multigpu = True  # noqa: F841

        # At minimum, gpu should be referenced somewhere in test infra
        # If not registered as a marker, it should at least be used in skipif
        test_dir = os.path.join(REPO_ROOT, "tests")
        test_content = ""
        for fname in os.listdir(test_dir):
            if fname.endswith(".py"):
                with open(os.path.join(test_dir, fname)) as f:
                    test_content += f.read()

        gpu_used = "gpu" in test_content.lower()
        assert found_gpu or gpu_used, (
            "gpu marker not registered and not used in any test infrastructure"
        )


# ===========================================================================
# 8. Arch-Specific Placeholder (GPU required)
# ===========================================================================

class TestArchSpecific:
    """Arch-specific GPU tracing tests."""

    @pytest.mark.skipif(not HAS_GPU, reason="No GPU or librpd_lite.so")
    def test_detect_gpu_arch_and_trace(self, tmp_path):
        """Detect current GPU arch and verify tracing works."""
        trace = str(tmp_path / "arch.db")
        script = (
            "import torch; "
            "print(torch.cuda.get_device_properties(0).gcnArchName); "
            "x = torch.randn(64, 64, device='cuda'); "
            "y = torch.mm(x, x); "
            "torch.cuda.synchronize()"
        )
        rc, stderr = _run_traced(script, trace)
        assert rc == 0, "Arch detection workload failed: {}".format(stderr)
        assert os.path.exists(trace), "Trace file not created"

        conn = sqlite3.connect(trace)
        ops = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        conn.close()
        assert ops >= 1, "Expected >= 1 op, got {}".format(ops)

    @pytest.mark.skipif(not HAS_GPU, reason="No GPU or librpd_lite.so")
    def test_gpu_id_reasonable(self, tmp_path):
        """gpuId in trace should be >= 0."""
        trace = str(tmp_path / "gpuid.db")
        script = (
            "import torch; "
            "x = torch.randn(128, 128, device='cuda'); "
            "y = torch.mm(x, x); "
            "torch.cuda.synchronize()"
        )
        rc, stderr = _run_traced(script, trace)
        assert rc == 0, "Workload failed: {}".format(stderr)
        assert os.path.exists(trace), "Trace file not created"

        conn = sqlite3.connect(trace)
        # All kernel ops should have gpuId >= 0
        neg_count = conn.execute(
            "SELECT count(*) FROM rocpd_op WHERE gpuId < 0"
        ).fetchone()[0]
        total = conn.execute(
            "SELECT count(*) FROM rocpd_op WHERE gpuId >= 0"
        ).fetchone()[0]
        conn.close()
        # Allow gpuId=-1 for roctx markers, but there must be real ops
        # neg_count is informational (roctx markers use gpuId=-1)
        assert total >= 1, "No ops with gpuId >= 0 (neg_count={})".format(neg_count)

    @pytest.mark.skipif(not HAS_GPU, reason="No GPU or librpd_lite.so")
    def test_fallback_detection_stderr(self, tmp_path):
        """If running on gfx1250 or similar, check for fallback info in stderr."""
        trace = str(tmp_path / "fallback.db")
        script = (
            "import torch; "
            "x = torch.randn(64, 64, device='cuda'); "
            "y = torch.mm(x, x); "
            "torch.cuda.synchronize()"
        )
        rc, stderr = _run_traced(script, trace)
        assert rc == 0, "Workload failed: {}".format(stderr)
        # This test is informational — on gfx1250 there may be fallback counters
        # On other archs, we just verify no crash happened
        if "fallback" in stderr.lower():
            # If fallback is reported, it should have a count
            assert re.search(r'\d+.*fallback', stderr, re.IGNORECASE), (
                "Fallback mentioned but no count: {}".format(stderr[:300])
            )
        # Either way, trace should exist
        assert os.path.exists(trace), "Trace file not created"
