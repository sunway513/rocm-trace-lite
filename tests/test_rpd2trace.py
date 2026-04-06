"""T5 — rpd2trace.py converter tests.

Tests the RPD-to-Chrome-Trace/Perfetto JSON converter.
No GPU needed — uses synthetic trace data.
"""
import json
import os
import sqlite3
import subprocess
import sys
from conftest import SCHEMA_SQL, populate_synthetic_trace

RPD2TRACE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "rpd2trace.py"
)


class TestConverterBasic:
    """T5.1–T5.3 — Basic converter functionality."""

    def test_produces_valid_json(self, tmp_rpd, tmp_path):
        """T5.1 — Output is valid JSON with traceEvents key."""
        populate_synthetic_trace(tmp_rpd, num_kernels=10)
        out_json = str(tmp_path / "out.json")
        subprocess.run(
            [sys.executable, RPD2TRACE, tmp_rpd, out_json],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        with open(out_json) as f:
            data = json.load(f)
        assert "traceEvents" in data

    def test_all_ops_present(self, tmp_rpd, tmp_path):
        """T5.2 — All ops in SQLite appear in JSON."""
        n = 50
        populate_synthetic_trace(tmp_rpd, num_kernels=n)
        out_json = str(tmp_path / "out.json")
        subprocess.run(
            [sys.executable, RPD2TRACE, tmp_rpd, out_json],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        with open(out_json) as f:
            data = json.load(f)
        complete_events = [e for e in data["traceEvents"] if e.get("ph") == "X"]
        assert len(complete_events) == n, (
            f"Expected {n} complete events, got {len(complete_events)}"
        )

    def test_event_required_fields(self, tmp_rpd, tmp_path):
        """T5.3 — Every event has required Perfetto fields."""
        populate_synthetic_trace(tmp_rpd, num_kernels=10)
        out_json = str(tmp_path / "out.json")
        subprocess.run(
            [sys.executable, RPD2TRACE, tmp_rpd, out_json],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        with open(out_json) as f:
            data = json.load(f)
        for e in data["traceEvents"]:
            if e.get("ph") == "X":
                assert "name" in e, f"Missing 'name': {e}"
                assert "pid" in e, f"Missing 'pid': {e}"
                assert "tid" in e, f"Missing 'tid': {e}"
                assert "ts" in e, f"Missing 'ts': {e}"
                assert "dur" in e, f"Missing 'dur': {e}"
                assert isinstance(e["ts"], (int, float)), f"ts not numeric: {e['ts']}"
                assert isinstance(e["dur"], (int, float)), f"dur not numeric: {e['dur']}"

    def test_ts_and_dur_non_negative(self, tmp_rpd, tmp_path):
        """T5.3b — ts >= 0 and dur >= 0 for all events."""
        populate_synthetic_trace(tmp_rpd, num_kernels=20)
        out_json = str(tmp_path / "out.json")
        subprocess.run(
            [sys.executable, RPD2TRACE, tmp_rpd, out_json],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        with open(out_json) as f:
            data = json.load(f)
        for e in data["traceEvents"]:
            if e.get("ph") == "X":
                assert e["ts"] >= 0, f"Negative ts: {e['ts']}"
                assert e["dur"] >= 0, f"Negative dur: {e['dur']}"


class TestConverterEdgeCases:
    """T5.4–T5.5 — Edge cases."""

    def test_empty_trace_no_crash(self, tmp_rpd, tmp_path):
        """T5.4 — Empty trace prints error, doesn't crash."""
        conn = sqlite3.connect(tmp_rpd)
        conn.executescript(SCHEMA_SQL)
        conn.close()
        out_json = str(tmp_path / "out.json")
        result = subprocess.run(
            [sys.executable, RPD2TRACE, tmp_rpd, out_json],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        # Should not crash (exit 0) but print error
        stdout = result.stdout.decode() if isinstance(result.stdout, bytes) else (result.stdout or "")
        assert result.returncode == 0 or "empty" in stdout.lower()

    def test_large_queue_id_collapse(self, tmp_rpd, tmp_path):
        """T5.5 — Traces with many unique queueIds get collapsed."""
        conn = sqlite3.connect(tmp_rpd)
        conn.executescript(SCHEMA_SQL)
        conn.execute(
            "INSERT OR IGNORE INTO rocpd_string(string) VALUES('test_kernel.kd')"
        )
        conn.execute(
            "INSERT OR IGNORE INTO rocpd_string(string) VALUES('KernelExecution')"
        )
        base_ns = 1000000000
        # Insert 200 ops with unique queueIds
        for i in range(200):
            conn.execute(
                "INSERT INTO rocpd_op(gpuId, queueId, sequenceId, start, end, "
                "description_id, opType_id) VALUES(0, ?, 0, ?, ?, "
                "(SELECT id FROM rocpd_string WHERE string='test_kernel.kd'), "
                "(SELECT id FROM rocpd_string WHERE string='KernelExecution'))",
                (i, base_ns + i * 10000, base_ns + i * 10000 + 5000),
            )
        conn.commit()
        conn.close()

        out_json = str(tmp_path / "out.json")
        subprocess.run(
            [sys.executable, RPD2TRACE, tmp_rpd, out_json],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        with open(out_json) as f:
            data = json.load(f)
        # All events should be collapsed to tid=0 (not 200 different tids)
        tids = set(
            e["tid"] for e in data["traceEvents"] if e.get("ph") == "X"
        )
        assert len(tids) <= 2, f"Expected collapsed tids, got {len(tids)}: {tids}"

    def test_multi_gpu_separate_tracks(self, tmp_rpd, tmp_path):
        """Multi-GPU ops should appear on separate pid tracks."""
        populate_synthetic_trace(tmp_rpd, num_kernels=40, num_gpus=4)
        out_json = str(tmp_path / "out.json")
        subprocess.run(
            [sys.executable, RPD2TRACE, tmp_rpd, out_json],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        with open(out_json) as f:
            data = json.load(f)
        pids = set(
            e["pid"] for e in data["traceEvents"] if e.get("ph") == "X"
        )
        assert len(pids) == 4, f"Expected 4 GPU pids, got {len(pids)}: {pids}"

    def test_kernel_name_shortening(self, tmp_rpd, tmp_path):
        """Long Tensile kernel names should be shortened in 'name' field."""
        conn = sqlite3.connect(tmp_rpd)
        conn.executescript(SCHEMA_SQL)
        long_name = (
            "Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_"
            "MT16x16x512_MI16x16x1_SN_LDSB1_AFC1_LOTS_MORE_PARAMS.kd"
        )
        conn.execute("INSERT OR IGNORE INTO rocpd_string(string) VALUES(?)", (long_name,))
        conn.execute("INSERT OR IGNORE INTO rocpd_string(string) VALUES('KernelExecution')")
        conn.execute(
            "INSERT INTO rocpd_op(gpuId, queueId, sequenceId, start, end, "
            "description_id, opType_id) VALUES(0, 0, 0, 1000000, 1005000, "
            "(SELECT id FROM rocpd_string WHERE string=?), "
            "(SELECT id FROM rocpd_string WHERE string='KernelExecution'))",
            (long_name,),
        )
        conn.commit()
        conn.close()

        out_json = str(tmp_path / "out.json")
        subprocess.run(
            [sys.executable, RPD2TRACE, tmp_rpd, out_json],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        with open(out_json) as f:
            data = json.load(f)
        event = [e for e in data["traceEvents"] if e.get("ph") == "X"][0]
        # Short name should not contain _UserArgs_
        assert "_UserArgs_" not in event["name"]
        # But full name should be in args
        assert event["args"]["full_name"] == long_name


class TestConverterOutput:
    """Output format validation."""

    def test_output_file_created(self, tmp_rpd, tmp_path):
        """Output JSON file is created."""
        populate_synthetic_trace(tmp_rpd, num_kernels=5)
        out_json = str(tmp_path / "out.json")
        subprocess.run(
            [sys.executable, RPD2TRACE, tmp_rpd, out_json],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        assert os.path.exists(out_json)
        assert os.path.getsize(out_json) > 0

    def test_default_output_name(self, tmp_path):
        """Default output should be input.db -> input.json."""
        rpd_path = str(tmp_path / "mytrace.db")
        populate_synthetic_trace(rpd_path, num_kernels=5)
        subprocess.run(
            [sys.executable, RPD2TRACE, rpd_path],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        json_path = str(tmp_path / "mytrace.json")
        assert os.path.exists(json_path)

    def test_metadata_events_present(self, tmp_rpd, tmp_path):
        """Process name metadata events should be present."""
        populate_synthetic_trace(tmp_rpd, num_kernels=10)
        out_json = str(tmp_path / "out.json")
        subprocess.run(
            [sys.executable, RPD2TRACE, tmp_rpd, out_json],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        with open(out_json) as f:
            data = json.load(f)
        meta = [e for e in data["traceEvents"] if e.get("ph") == "M"]
        assert len(meta) > 0, "No metadata events"
        names = [e for e in meta if e.get("name") == "process_name"]
        assert len(names) > 0, "No process_name metadata"
