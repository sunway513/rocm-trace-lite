"""Tests for the preflight diagnostic checks in cmd_trace."""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock


class TestPreflightCheck:
    """Tests for _preflight_check()."""

    def _get_preflight(self):
        from rocm_trace_lite.cmd_trace import _preflight_check
        return _preflight_check

    def test_missing_lib_returns_false(self, capsys):
        check = self._get_preflight()
        result = check("/nonexistent/path/librpd_lite.so")
        assert result is False
        err = capsys.readouterr().err
        assert "not found" in err

    def test_existing_lib_dlopen_failure(self, tmp_path, capsys):
        """A file that exists but isn't a valid .so should fail dlopen."""
        fake_so = tmp_path / "librpd_lite.so"
        fake_so.write_text("not a real shared object")
        check = self._get_preflight()
        result = check(str(fake_so))
        assert result is False
        err = capsys.readouterr().err
        assert "cannot load" in err

    def test_conflicting_hsa_tools_lib_warns(self, tmp_path, capsys):
        """Warn when HSA_TOOLS_LIB is already set to a different path."""
        fake_so = tmp_path / "librpd_lite.so"
        fake_so.write_text("not real")
        check = self._get_preflight()
        with patch.dict(os.environ, {"HSA_TOOLS_LIB": "/some/other/lib.so"}):
            check(str(fake_so))
        err = capsys.readouterr().err
        assert "already set" in err
        assert "/some/other/lib.so" in err

    def test_matching_hsa_tools_lib_no_warn(self, capsys):
        """No conflict warning when HSA_TOOLS_LIB matches our lib path."""
        check = self._get_preflight()
        lib = "/our/librpd_lite.so"
        with patch.dict(os.environ, {"HSA_TOOLS_LIB": lib}):
            check(lib)
        err = capsys.readouterr().err
        assert "already set" not in err

    @patch("ctypes.CDLL")
    def test_successful_dlopen(self, mock_cdll, tmp_path, capsys):
        """When dlopen succeeds, print OK and return True."""
        fake_so = tmp_path / "librpd_lite.so"
        fake_so.write_bytes(b"\x7fELF")
        mock_cdll.return_value = MagicMock()
        check = self._get_preflight()
        with patch("os.path.exists", return_value=True):
            result = check(str(fake_so))
        assert result is True
        err = capsys.readouterr().err
        assert "librpd_lite.so OK" in err

    @patch("ctypes.CDLL")
    def test_dlopen_hsa_missing_message(self, mock_cdll, tmp_path, capsys):
        """When dlopen fails due to libhsa, show HSA-specific guidance."""
        fake_so = tmp_path / "librpd_lite.so"
        fake_so.write_bytes(b"\x7fELF")
        mock_cdll.side_effect = OSError(
            "libhsa-runtime64.so: cannot open shared object file"
        )
        check = self._get_preflight()
        result = check(str(fake_so))
        assert result is False
        err = capsys.readouterr().err
        assert "libhsa-runtime64.so not found" in err
        assert "ROCm HSA runtime is missing" in err

    @patch("ctypes.CDLL")
    def test_dlopen_sqlite_missing_message(self, mock_cdll, tmp_path, capsys):
        """When dlopen fails due to libsqlite3, show sqlite-specific guidance."""
        fake_so = tmp_path / "librpd_lite.so"
        fake_so.write_bytes(b"\x7fELF")
        mock_cdll.side_effect = OSError(
            "libsqlite3.so: cannot open shared object file"
        )
        check = self._get_preflight()
        result = check(str(fake_so))
        assert result is False
        err = capsys.readouterr().err
        assert "libsqlite3.so not found" in err
        assert "apt install" in err

    @patch("ctypes.CDLL")
    def test_rocm_path_env_searched(self, mock_cdll, tmp_path, capsys):
        """ROCM_PATH is used to find libhsa-runtime64.so."""
        fake_so = tmp_path / "librpd_lite.so"
        fake_so.write_bytes(b"\x7fELF")
        mock_cdll.return_value = MagicMock()
        rocm_lib = tmp_path / "rocm" / "lib"
        rocm_lib.mkdir(parents=True)
        (rocm_lib / "libhsa-runtime64.so").write_bytes(b"\x7fELF")
        check = self._get_preflight()
        with patch.dict(os.environ, {"ROCM_PATH": str(tmp_path / "rocm")}):
            result = check(str(fake_so))
        assert result is True
        err = capsys.readouterr().err
        assert "libhsa-runtime64.so OK" in err


class TestZeroOpsMessage:
    """Test that 0-ops-captured gives actionable diagnostics."""

    def test_zero_ops_message_content(self):
        """Verify the 0-ops warning includes key diagnostic info."""
        from rocm_trace_lite import cmd_trace
        import inspect
        source = inspect.getsource(cmd_trace.run_trace)
        assert "HSA_TOOLS_LIB was not inherited" in source
        assert "didn't run any GPU kernels" in source
        assert "export HSA_TOOLS_LIB=" in source


class TestStderrPrefix:
    """Verify C++ source uses 'rtl:' prefix, not 'rpd_lite:'."""

    def _read_src(self, filename):
        src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
        path = os.path.join(src_dir, filename)
        with open(path) as f:
            return f.read()

    def test_hsa_intercept_uses_rtl_prefix(self):
        src = self._read_src("hsa_intercept.cpp")
        assert '"rpd_lite:' not in src, "hsa_intercept.cpp still has rpd_lite: prefix"
        assert '"rtl:' in src

    def test_rpd_lite_cpp_uses_rtl_prefix(self):
        src = self._read_src("rpd_lite.cpp")
        assert '"rpd_lite:' not in src, "rpd_lite.cpp still has rpd_lite: prefix"
        assert '"rtl:' in src

    def test_roctx_shim_uses_rtl_prefix(self):
        src = self._read_src("roctx_shim.cpp")
        assert '"rpd_lite:' not in src, "roctx_shim.cpp still has rpd_lite: prefix"
        assert '"rtl:' in src
