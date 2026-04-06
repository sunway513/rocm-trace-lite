"""Tests for the rpd-lite CLI and Python package structure.

Validates CLI subcommands, package imports, and wheel contents.
No GPU needed — uses synthetic trace data.
"""
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)  # noqa: E402
sys.path.insert(0, os.path.join(REPO_ROOT, "tests"))  # noqa: E402
from conftest import populate_synthetic_trace  # noqa: E402


def _run_cli(*args):
    """Run rpd-lite CLI via python -m."""
    r = subprocess.run(
        [sys.executable, "-m", "rocm_trace_lite.cli"] + list(args),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=REPO_ROOT,
    )
    return r.returncode, r.stdout.decode(), r.stderr.decode()


class TestPackageImport:
    """Package imports and structure."""

    def test_import_version(self):
        from rocm_trace_lite import __version__
        assert __version__ == "0.1.1"

    def test_import_get_lib_path(self):
        from rocm_trace_lite import get_lib_path
        assert callable(get_lib_path)

    def test_import_cli(self):
        from rocm_trace_lite.cli import main
        assert callable(main)

    def test_import_all_subcommands(self):
        from rocm_trace_lite.cmd_trace import run_trace
        from rocm_trace_lite.cmd_convert import run_convert
        from rocm_trace_lite.cmd_summary import run_summary
        from rocm_trace_lite.cmd_info import run_info
        assert all(callable(f) for f in [run_trace, run_convert, run_summary, run_info])


class TestCLIHelp:
    """CLI help and version output."""

    def test_help(self):
        rc, out, _ = _run_cli("--help")
        assert rc == 0
        assert "trace" in out
        assert "convert" in out
        assert "summary" in out
        assert "info" in out

    def test_version(self):
        rc, out, _ = _run_cli("--version")
        assert rc == 0
        assert "0.1.1" in out

    def test_no_args_shows_help(self):
        rc, out, _ = _run_cli()
        assert rc == 1  # exits with error
        assert "trace" in out or "usage" in out.lower()

    def test_trace_help(self):
        rc, out, _ = _run_cli("trace", "--help")
        assert rc == 0
        assert "--output" in out or "-o" in out

    def test_convert_help(self):
        rc, out, _ = _run_cli("convert", "--help")
        assert rc == 0

    def test_summary_help(self):
        rc, out, _ = _run_cli("summary", "--help")
        assert rc == 0
        assert "--limit" in out or "-n" in out


class TestSummaryCommand:
    """summary subcommand with synthetic data."""

    def test_summary_basic(self, tmp_path):
        rpd = str(tmp_path / "test.db")
        populate_synthetic_trace(rpd, num_kernels=50, num_gpus=2)
        rc, out, _ = _run_cli("summary", rpd)
        assert rc == 0
        assert "GPU ops:" in out
        assert "50" in out
        assert "Cijk_GEMM" in out

    def test_summary_limit(self, tmp_path):
        rpd = str(tmp_path / "test.db")
        populate_synthetic_trace(rpd, num_kernels=100)
        rc, out, _ = _run_cli("summary", rpd, "-n", "3")
        assert rc == 0
        # Should have header + separator + 3 data rows
        lines = [line for line in out.strip().split("\n") if line.strip() and "=" not in line and "GPU" not in line and "API" not in line and "Trace" not in line and "Kernel" not in line]
        assert len(lines) <= 5  # max 3 kernel rows + possible utilization

    def test_summary_missing_file(self):
        rc, _, err = _run_cli("summary", "/tmp/nonexistent.db")
        assert rc != 0

    def test_summary_gpu_utilization(self, tmp_path):
        rpd = str(tmp_path / "test.db")
        populate_synthetic_trace(rpd, num_kernels=50, num_gpus=2)
        rc, out, _ = _run_cli("summary", rpd)
        assert rc == 0
        assert "GPU 0" in out
        assert "GPU 1" in out


class TestInfoCommand:
    """info subcommand."""

    def test_info_basic(self, tmp_path):
        rpd = str(tmp_path / "test.db")
        populate_synthetic_trace(rpd, num_kernels=20)
        rc, out, _ = _run_cli("info", rpd)
        assert rc == 0
        assert "rocpd_op: 20 rows" in out
        assert "Unique kernels: 5" in out
        assert "Tables:" in out

    def test_info_missing_file(self):
        rc, _, err = _run_cli("info", "/tmp/nonexistent.db")
        assert rc != 0


class TestConvertCommand:
    """convert subcommand."""

    def test_convert_basic(self, tmp_path):
        rpd = str(tmp_path / "test.db")
        json_out = str(tmp_path / "test.json")
        populate_synthetic_trace(rpd, num_kernels=10)
        rc, out, _ = _run_cli("convert", rpd, "-o", json_out)
        assert rc == 0
        assert os.path.exists(json_out)
        assert os.path.getsize(json_out) > 0

    def test_convert_default_output(self, tmp_path):
        rpd = str(tmp_path / "test.db")
        populate_synthetic_trace(rpd, num_kernels=5)
        rc, _, _ = _run_cli("convert", rpd)
        assert rc == 0
        json_path = str(tmp_path / "test.json")
        assert os.path.exists(json_path)

    def test_convert_missing_file(self):
        rc, _, _ = _run_cli("convert", "/tmp/nonexistent.db")
        assert rc != 0


class TestTraceCommand:
    """trace subcommand (no GPU needed for error handling tests)."""

    def test_trace_no_command(self):
        rc, _, err = _run_cli("trace")
        assert rc != 0

    def test_trace_help(self):
        rc, out, _ = _run_cli("trace", "--help")
        assert rc == 0
        assert "--output" in out or "-o" in out


class TestPackageStructure:
    """Verify wheel-related files exist."""

    def test_pyproject_toml_exists(self):
        assert os.path.exists(os.path.join(REPO_ROOT, "pyproject.toml"))

    def test_manifest_in_exists(self):
        assert os.path.exists(os.path.join(REPO_ROOT, "MANIFEST.in"))

    def test_release_workflow_exists(self):
        assert os.path.exists(os.path.join(REPO_ROOT, ".github", "workflows", "release.yml"))

    def test_build_wheel_script_exists(self):
        path = os.path.join(REPO_ROOT, "build_wheel.sh")
        assert os.path.exists(path)
        assert os.access(path, os.X_OK)

    def test_rtl_alias_in_pyproject(self):
        """rtl should be registered as CLI alias."""
        with open(os.path.join(REPO_ROOT, "pyproject.toml")) as f:
            content = f.read()
        assert "rtl" in content, "rtl alias not in pyproject.toml"

    def test_lib_directory_exists(self):
        """rocm_trace_lite/lib/ directory must exist for .so staging."""
        assert os.path.isdir(os.path.join(REPO_ROOT, "rocm_trace_lite", "lib"))
