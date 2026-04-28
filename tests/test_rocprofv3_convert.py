"""Tests for RTL → rocprofv3 JSON conversion (TraceLens compatibility)."""
import json
import os
import sqlite3
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")


def _create_test_db(tmp_path, n_kernels=5, n_copies=2, n_api=10):
    """Create a minimal RTL trace DB with test data."""
    db_path = str(tmp_path / "test_trace.db")
    conn = sqlite3.connect(db_path)

    # Read schema from source
    import re
    schema_src = open(os.path.join(SRC_DIR, "trace_db.cpp")).read()
    m = re.search(r'R"SQL\((.*?)\)SQL"', schema_src, re.DOTALL)
    assert m, "Cannot find schema in trace_db.cpp"
    conn.executescript(m.group(1))
    # Add correlation_id columns
    conn.execute("ALTER TABLE rocpd_op ADD COLUMN correlation_id INTEGER DEFAULT 0")
    conn.execute("ALTER TABLE rocpd_api ADD COLUMN correlation_id INTEGER DEFAULT 0")

    # Insert kernel names
    conn.execute("INSERT INTO rocpd_string VALUES (1, 'Cijk_Ailk_Bljk_GEMM_kernel')")
    conn.execute("INSERT INTO rocpd_string VALUES (2, 'KernelExecution')")
    conn.execute("INSERT INTO rocpd_string VALUES (3, 'CopyDeviceToHost')")
    conn.execute("INSERT INTO rocpd_string VALUES (4, 'CopyHostToDevice')")
    conn.execute("INSERT INTO rocpd_string VALUES (5, 'hipLaunchKernel')")
    conn.execute("INSERT INTO rocpd_string VALUES (6, 'hipMemcpy')")
    conn.execute("INSERT INTO rocpd_string VALUES (7, 'elementwise_add_kernel')")
    conn.execute("INSERT INTO rocpd_string VALUES (8, '')")  # empty args

    base_ns = 1000000000000  # 1 trillion ns

    # Insert kernel ops
    for i in range(n_kernels):
        start = base_ns + i * 1000000
        end = start + 500000  # 500us
        kid = 1 if i % 2 == 0 else 7
        conn.execute(
            "INSERT INTO rocpd_op (gpuId, queueId, sequenceId, start, end, "
            "description_id, opType_id, correlation_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (0, i % 4, i, start, end, kid, 2, i + 1)
        )

    # Insert copy ops
    for i in range(n_copies):
        start = base_ns + (n_kernels + i) * 1000000
        end = start + 200000
        copy_type = 3 if i == 0 else 4  # D2H, H2D
        conn.execute(
            "INSERT INTO rocpd_op (gpuId, queueId, sequenceId, start, end, "
            "description_id, opType_id, correlation_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (0, 0, n_kernels + i, start, end, copy_type, copy_type, n_kernels + i + 1)
        )

    # Insert API calls
    for i in range(n_api):
        start = base_ns + i * 900000
        end = start + 50000  # 50us
        api_name = 5 if i % 2 == 0 else 6
        conn.execute(
            "INSERT INTO rocpd_api (pid, tid, start, end, apiName_id, args_id, correlation_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (1234, 5678, start, end, api_name, 8, i + 1)
        )

    # Insert metadata
    conn.execute("INSERT INTO rocpd_metadata (tag, value) VALUES ('pid', '1234')")
    conn.commit()
    conn.close()
    return db_path


class TestRocprofv3Convert:
    """Test rocprofv3 JSON output format."""

    def test_basic_conversion(self, tmp_path):
        db_path = _create_test_db(tmp_path)
        out_path = str(tmp_path / "output_results.json")

        from rocm_trace_lite.cmd_convert_rocprofv3 import convert_to_rocprofv3
        result = convert_to_rocprofv3(db_path, out_path)
        assert result is True
        assert os.path.exists(out_path)

    def test_top_level_structure(self, tmp_path):
        db_path = _create_test_db(tmp_path)
        out_path = str(tmp_path / "output.json")

        from rocm_trace_lite.cmd_convert_rocprofv3 import convert_to_rocprofv3
        convert_to_rocprofv3(db_path, out_path)

        data = json.load(open(out_path))
        assert "rocprofiler-sdk-tool" in data
        assert isinstance(data["rocprofiler-sdk-tool"], list)
        assert len(data["rocprofiler-sdk-tool"]) == 1

        tool = data["rocprofiler-sdk-tool"][0]
        assert "metadata" in tool
        assert "agents" in tool
        assert "kernel_symbols" in tool
        assert "buffer_records" in tool

    def test_kernel_dispatch_records(self, tmp_path):
        db_path = _create_test_db(tmp_path, n_kernels=5)
        out_path = str(tmp_path / "output.json")

        from rocm_trace_lite.cmd_convert_rocprofv3 import convert_to_rocprofv3
        convert_to_rocprofv3(db_path, out_path)

        data = json.load(open(out_path))
        tool = data["rocprofiler-sdk-tool"][0]
        dispatches = tool["buffer_records"]["kernel_dispatch"]
        assert len(dispatches) == 5

        d = dispatches[0]
        assert "start_timestamp" in d
        assert "end_timestamp" in d
        assert d["end_timestamp"] > d["start_timestamp"]
        assert "dispatch_info" in d
        assert "kernel_id" in d["dispatch_info"]
        assert "agent_id" in d["dispatch_info"]
        assert "grid_size" in d["dispatch_info"]
        assert "workgroup_size" in d["dispatch_info"]
        assert "correlation_id" in d

    def test_kernel_symbols(self, tmp_path):
        db_path = _create_test_db(tmp_path, n_kernels=5)
        out_path = str(tmp_path / "output.json")

        from rocm_trace_lite.cmd_convert_rocprofv3 import convert_to_rocprofv3
        convert_to_rocprofv3(db_path, out_path)

        data = json.load(open(out_path))
        tool = data["rocprofiler-sdk-tool"][0]
        symbols = tool["kernel_symbols"]

        # 5 kernels using 2 unique names
        assert len(symbols) == 2

        for sym in symbols:
            assert "kernel_id" in sym
            assert "kernel_name" in sym
            assert "formatted_kernel_name" in sym
            assert "truncated_kernel_name" in sym
            assert sym["kernel_name"] != ""

    def test_kernel_id_linkage(self, tmp_path):
        """kernel_dispatch.dispatch_info.kernel_id must reference a kernel_symbols entry."""
        db_path = _create_test_db(tmp_path, n_kernels=3)
        out_path = str(tmp_path / "output.json")

        from rocm_trace_lite.cmd_convert_rocprofv3 import convert_to_rocprofv3
        convert_to_rocprofv3(db_path, out_path)

        data = json.load(open(out_path))
        tool = data["rocprofiler-sdk-tool"][0]
        symbol_ids = {s["kernel_id"] for s in tool["kernel_symbols"]}
        for d in tool["buffer_records"]["kernel_dispatch"]:
            kid = d["dispatch_info"]["kernel_id"]
            assert kid in symbol_ids, f"kernel_id {kid} not in kernel_symbols"

    def test_memory_copy_records(self, tmp_path):
        db_path = _create_test_db(tmp_path, n_copies=3)
        out_path = str(tmp_path / "output.json")

        from rocm_trace_lite.cmd_convert_rocprofv3 import convert_to_rocprofv3
        convert_to_rocprofv3(db_path, out_path)

        data = json.load(open(out_path))
        copies = data["rocprofiler-sdk-tool"][0]["buffer_records"]["memory_copy"]
        assert len(copies) == 3
        for c in copies:
            assert "start_timestamp" in c
            assert "end_timestamp" in c

    def test_hip_api_records(self, tmp_path):
        db_path = _create_test_db(tmp_path, n_api=8)
        out_path = str(tmp_path / "output.json")

        from rocm_trace_lite.cmd_convert_rocprofv3 import convert_to_rocprofv3
        convert_to_rocprofv3(db_path, out_path)

        data = json.load(open(out_path))
        api = data["rocprofiler-sdk-tool"][0]["buffer_records"]["hip_api"]
        assert len(api) == 8
        for a in api:
            assert "operation" in a
            assert "thread_id" in a
            assert "start_timestamp" in a

    def test_metadata(self, tmp_path):
        db_path = _create_test_db(tmp_path)
        out_path = str(tmp_path / "output.json")

        from rocm_trace_lite.cmd_convert_rocprofv3 import convert_to_rocprofv3
        convert_to_rocprofv3(db_path, out_path)

        data = json.load(open(out_path))
        meta = data["rocprofiler-sdk-tool"][0]["metadata"]
        assert meta["pid"] == 1234
        assert meta["init_time"] is not None
        assert meta["fini_time"] is not None
        assert meta["fini_time"] > meta["init_time"]

    def test_gzip_output(self, tmp_path):
        db_path = _create_test_db(tmp_path)
        out_path = str(tmp_path / "output.json.gz")

        from rocm_trace_lite.cmd_convert_rocprofv3 import convert_to_rocprofv3
        convert_to_rocprofv3(db_path, out_path)

        import gzip
        with gzip.open(out_path, 'rt') as f:
            data = json.load(f)
        assert "rocprofiler-sdk-tool" in data

    def test_empty_trace(self, tmp_path):
        """Empty trace should return False."""
        db_path = str(tmp_path / "empty.db")
        conn = sqlite3.connect(db_path)
        import re
        schema_src = open(os.path.join(SRC_DIR, "trace_db.cpp")).read()
        m = re.search(r'R"SQL\((.*?)\)SQL"', schema_src, re.DOTALL)
        conn.executescript(m.group(1))
        conn.close()

        from rocm_trace_lite.cmd_convert_rocprofv3 import convert_to_rocprofv3
        result = convert_to_rocprofv3(db_path, str(tmp_path / "out.json"))
        assert result is False

    def test_cli_format_flag(self, tmp_path):
        """Test CLI routing via --format rocprofv3."""
        import argparse
        args = argparse.Namespace(
            input=str(tmp_path / "nonexistent.db"),
            output=None,
            format="rocprofv3"
        )
        # Just verify the import works
        from rocm_trace_lite.cmd_convert_rocprofv3 import run_convert_rocprofv3
        assert callable(run_convert_rocprofv3)
