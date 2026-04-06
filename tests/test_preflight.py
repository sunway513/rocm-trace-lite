"""Tests for the preflight diagnostic checks in cmd_trace."""
import os
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

    def test_existing_lib_prints_ok(self, tmp_path, capsys):
        """An existing .so prints OK for the file check."""
        fake_so = tmp_path / "librpd_lite.so"
        fake_so.write_text("not a real shared object")
        check = self._get_preflight()
        ldd_mock = MagicMock()
        ldd_mock.returncode = 0
        ldd_mock.stdout = ""
        with patch("subprocess.run", return_value=ldd_mock):
            with patch("os.path.exists", return_value=True):
                check(str(fake_so))
        err = capsys.readouterr().err
        assert "librpd_lite.so OK" in err

    def test_ldd_missing_dep_warns(self, tmp_path, capsys):
        """When ldd reports a missing dep, warn with specifics."""
        fake_so = tmp_path / "librpd_lite.so"
        fake_so.write_text("fake")
        ldd_mock = MagicMock()
        ldd_mock.returncode = 0
        ldd_mock.stdout = "\tlibhsa-runtime64.so => not found\n"
        check = self._get_preflight()
        with patch("subprocess.run", return_value=ldd_mock):
            result = check(str(fake_so))
        assert result is False
        err = capsys.readouterr().err
        assert "missing dependency" in err
        assert "ROCm HSA runtime is missing" in err

    def test_ldd_sqlite_missing_warns(self, tmp_path, capsys):
        """When ldd reports missing libsqlite3, show install hint."""
        fake_so = tmp_path / "librpd_lite.so"
        fake_so.write_text("fake")
        ldd_mock = MagicMock()
        ldd_mock.returncode = 0
        ldd_mock.stdout = "\tlibsqlite3.so => not found\n"
        check = self._get_preflight()
        with patch("subprocess.run", return_value=ldd_mock):
            result = check(str(fake_so))
        assert result is False
        err = capsys.readouterr().err
        assert "apt install" in err

    def test_ldd_all_found_ok(self, tmp_path, capsys):
        """When ldd finds everything, no warnings from dep check."""
        fake_so = tmp_path / "librpd_lite.so"
        fake_so.write_text("fake")
        ldd_mock = MagicMock()
        ldd_mock.returncode = 0
        ldd_mock.stdout = "\tlibhsa-runtime64.so => /opt/rocm/lib/libhsa-runtime64.so\n"
        check = self._get_preflight()
        with patch("subprocess.run", return_value=ldd_mock):
            with patch("os.path.exists", return_value=True):
                result = check(str(fake_so))
        assert result is True
        err = capsys.readouterr().err
        assert "missing dependency" not in err

    def test_conflicting_hsa_tools_lib_warns(self, tmp_path, capsys):
        """Warn when HSA_TOOLS_LIB is already set to a different path."""
        fake_so = tmp_path / "librpd_lite.so"
        fake_so.write_text("not real")
        check = self._get_preflight()
        ldd_mock = MagicMock()
        ldd_mock.returncode = 0
        ldd_mock.stdout = ""
        with patch("subprocess.run", return_value=ldd_mock):
            with patch.dict(os.environ, {"HSA_TOOLS_LIB": "/some/other/lib.so"}):
                with patch("os.path.exists", return_value=True):
                    check(str(fake_so))
        err = capsys.readouterr().err
        assert "already set" in err
        assert "/some/other/lib.so" in err

    def test_matching_hsa_tools_lib_no_warn(self, tmp_path, capsys):
        """No conflict warning when HSA_TOOLS_LIB matches our lib path."""
        fake_so = tmp_path / "librpd_lite.so"
        fake_so.write_text("not real")
        check = self._get_preflight()
        ldd_mock = MagicMock()
        ldd_mock.returncode = 0
        ldd_mock.stdout = ""
        with patch("subprocess.run", return_value=ldd_mock):
            with patch.dict(os.environ, {"HSA_TOOLS_LIB": str(fake_so)}):
                with patch("os.path.exists", return_value=True):
                    check(str(fake_so))
        err = capsys.readouterr().err
        assert "already set" not in err

    def test_rocm_path_env_searched(self, tmp_path, capsys):
        """ROCM_PATH is used to find libhsa-runtime64.so."""
        fake_so = tmp_path / "librpd_lite.so"
        fake_so.write_bytes(b"\x7fELF")
        rocm_lib = tmp_path / "rocm" / "lib"
        rocm_lib.mkdir(parents=True)
        (rocm_lib / "libhsa-runtime64.so").write_bytes(b"\x7fELF")
        check = self._get_preflight()
        ldd_mock = MagicMock()
        ldd_mock.returncode = 0
        ldd_mock.stdout = ""
        with patch("subprocess.run", return_value=ldd_mock):
            with patch.dict(os.environ, {"ROCM_PATH": str(tmp_path / "rocm")}):
                check(str(fake_so))
        err = capsys.readouterr().err
        assert "libhsa-runtime64.so OK" in err

    def test_ld_library_path_suggested_only_when_ldd_fails(self, tmp_path, capsys):
        """Only suggest LD_LIBRARY_PATH fix when ldd reports HSA not found."""
        fake_so = tmp_path / "librpd_lite.so"
        fake_so.write_text("fake")
        rocm_lib = tmp_path / "rocm" / "lib"
        rocm_lib.mkdir(parents=True)
        (rocm_lib / "libhsa-runtime64.so").write_bytes(b"\x7fELF")
        check = self._get_preflight()
        # ldd shows HSA as not found
        ldd_mock = MagicMock()
        ldd_mock.returncode = 0
        ldd_mock.stdout = "\tlibhsa-runtime64.so => not found\n"
        with patch("subprocess.run", return_value=ldd_mock):
            with patch.dict(os.environ, {
                "ROCM_PATH": str(tmp_path / "rocm"),
                "LD_LIBRARY_PATH": "/some/other/path",
            }):
                check(str(fake_so))
        err = capsys.readouterr().err
        assert "LD_LIBRARY_PATH" in err

    def test_no_ld_warning_when_ldd_resolves(self, tmp_path, capsys):
        """No LD_LIBRARY_PATH warning when ldd resolves HSA fine."""
        fake_so = tmp_path / "librpd_lite.so"
        fake_so.write_text("fake")
        rocm_lib = tmp_path / "rocm" / "lib"
        rocm_lib.mkdir(parents=True)
        (rocm_lib / "libhsa-runtime64.so").write_bytes(b"\x7fELF")
        check = self._get_preflight()
        # ldd resolves everything fine
        ldd_mock = MagicMock()
        ldd_mock.returncode = 0
        ldd_mock.stdout = "\tlibhsa-runtime64.so => /opt/rocm/lib/libhsa-runtime64.so\n"
        with patch("subprocess.run", return_value=ldd_mock):
            with patch.dict(os.environ, {
                "ROCM_PATH": str(tmp_path / "rocm"),
                "LD_LIBRARY_PATH": "/some/other/path",
            }):
                result = check(str(fake_so))
        err = capsys.readouterr().err
        assert "LD_LIBRARY_PATH" not in err
        assert result is True

    def test_no_dlopen_side_effects(self):
        """Preflight must NOT use ctypes.CDLL (avoids constructor side effects)."""
        from rocm_trace_lite import cmd_trace
        import inspect
        source = inspect.getsource(cmd_trace._preflight_check)
        assert "ctypes.CDLL" not in source
        assert "CDLL" not in source


class TestZeroOpsMessage:
    """Test that 0-ops-captured gives actionable diagnostics."""

    def test_zero_ops_message_content(self):
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
        assert '"rpd_lite:' not in src
        assert '"rtl:' in src

    def test_rpd_lite_cpp_uses_rtl_prefix(self):
        src = self._read_src("rpd_lite.cpp")
        assert '"rpd_lite:' not in src
        assert '"rtl:' in src

    def test_roctx_shim_uses_rtl_prefix(self):
        src = self._read_src("roctx_shim.cpp")
        assert '"rpd_lite:' not in src
        assert '"rtl:' in src
