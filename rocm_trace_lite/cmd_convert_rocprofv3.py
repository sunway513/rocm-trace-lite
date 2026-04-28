"""
cmd_convert_rocprofv3 — Convert RTL trace DB to rocprofv3 JSON.

Emits rocprofiler-sdk-tool format that TraceLens can consume directly:
  TraceLens_generate_perf_report_rocprof --profile_json_path output.json

Format reference: TraceLens/util.py:RocprofParser.extract_kernel_events()
"""

import json
import gzip
import os
import sqlite3
import sys


def convert_to_rocprofv3(input_db, output_json):
    """Convert RTL trace DB to rocprofiler-sdk-tool JSON format."""
    conn = sqlite3.connect(input_db)

    # ---- Metadata ----
    pid = None
    init_time = None
    fini_time = None

    try:
        for tag, value in conn.execute("SELECT tag, value FROM rocpd_metadata"):
            if tag == "pid":
                pid = int(value)
    except sqlite3.OperationalError:
        pass

    # Get time range
    row = conn.execute("SELECT MIN(start), MAX(end) FROM rocpd_op WHERE end > start").fetchone()
    if row and row[0] is not None:
        init_time = row[0]
        fini_time = row[1]
    else:
        row = conn.execute("SELECT MIN(start), MAX(end) FROM rocpd_api").fetchone()
        if row and row[0] is not None:
            init_time = row[0]
            fini_time = row[1]

    if init_time is None:
        print("Error: trace is empty", file=sys.stderr)
        return False

    if pid is None:
        try:
            row = conn.execute("SELECT pid FROM rocpd_api LIMIT 1").fetchone()
            if row:
                pid = row[0]
        except sqlite3.OperationalError:
            pass
    pid = pid or 0

    # ---- Kernel symbols (deduplicate kernel names → synthetic IDs) ----
    kernel_name_to_id = {}
    kernel_symbols = []
    next_kid = 1

    # ---- GPU ops → kernel_dispatch + memory_copy ----
    kernel_dispatches = []
    memory_copies = []
    dispatch_id = 0

    ops = conn.execute("""
        SELECT o.id, o.gpuId, o.queueId, o.start, o.end,
               s.string, ot.string, o.completionSignal, o.correlation_id
        FROM rocpd_op o
        JOIN rocpd_string s ON o.description_id = s.id
        LEFT JOIN rocpd_string ot ON o.opType_id = ot.id
        WHERE o.end > o.start
        ORDER BY o.start
    """).fetchall()

    for op_id, gpu_id, queue_id, start_ns, end_ns, name, op_type, dispatch_info, corr_id in ops:
        gpu_id = gpu_id if gpu_id is not None and gpu_id >= 0 else 0
        queue_id = queue_id or 0
        corr_id = corr_id or dispatch_id

        # Parse grid/workgroup from dispatch_info string
        grid_x, grid_y, grid_z = 1, 1, 1
        wg_x, wg_y, wg_z = 1, 1, 1
        if dispatch_info:
            for part in str(dispatch_info).split():
                if part.startswith("grid="):
                    dims = part[5:].split(",")
                    if len(dims) >= 3:
                        grid_x, grid_y, grid_z = int(dims[0]), int(dims[1]), int(dims[2])
                elif part.startswith("wg="):
                    dims = part[3:].split(",")
                    if len(dims) >= 3:
                        wg_x, wg_y, wg_z = int(dims[0]), int(dims[1]), int(dims[2])

        # Also try rocpd_kernelapi for grid/workgroup dims
        if grid_x == 1 and grid_y == 1 and grid_z == 1:
            try:
                api_link = conn.execute(
                    "SELECT api_id FROM rocpd_api_ops WHERE op_id = ?", (op_id,)
                ).fetchone()
                if api_link:
                    ka = conn.execute(
                        "SELECT gridX, gridY, gridZ, workgroupX, workgroupY, workgroupZ "
                        "FROM rocpd_kernelapi WHERE api_id = ?", (api_link[0],)
                    ).fetchone()
                    if ka and ka[0]:
                        grid_x, grid_y, grid_z = ka[0], ka[1], ka[2]
                        wg_x, wg_y, wg_z = ka[3], ka[4], ka[5]
            except sqlite3.OperationalError:
                pass

        is_copy = op_type and ("Copy" in op_type or "copy" in op_type)

        if is_copy:
            memory_copies.append({
                "size": 48,
                "kind": 1,
                "operation": _map_copy_operation(op_type),
                "thread_id": pid,
                "correlation_id": {"internal": corr_id, "external": 0},
                "start_timestamp": start_ns,
                "end_timestamp": end_ns,
                "stream_id": {"handle": queue_id},
            })
        else:
            # Kernel dispatch
            if name not in kernel_name_to_id:
                kid = next_kid
                next_kid += 1
                kernel_name_to_id[name] = kid
                kernel_symbols.append({
                    "size": 0,
                    "kernel_id": kid,
                    "code_object_id": 0,
                    "kernel_name": name,
                    "formatted_kernel_name": name,
                    "truncated_kernel_name": name[:256] if len(name) > 256 else name,
                    "kernel_object": 0,
                    "kernarg_segment_size": 0,
                    "kernarg_segment_alignment": 0,
                    "group_segment_size": 0,
                    "private_segment_size": 0,
                    "sgpr_count": 0,
                    "arch_vgpr_count": 0,
                    "accum_vgpr_count": 0,
                    "kernel_code_entry_byte_offset": 0,
                    "kernel_address": {"handle": 0},
                })
            kid = kernel_name_to_id[name]

            kernel_dispatches.append({
                "size": 184,
                "kind": 11,
                "operation": 2,
                "thread_id": pid,
                "correlation_id": {"internal": corr_id, "external": 0},
                "start_timestamp": start_ns,
                "end_timestamp": end_ns,
                "dispatch_info": {
                    "size": 72,
                    "agent_id": {"handle": gpu_id},
                    "queue_id": {"handle": queue_id},
                    "kernel_id": kid,
                    "dispatch_id": dispatch_id,
                    "private_segment_size": 0,
                    "group_segment_size": 0,
                    "workgroup_size": {"x": wg_x, "y": wg_y, "z": wg_z},
                    "grid_size": {"x": grid_x, "y": grid_y, "z": grid_z},
                },
                "stream_id": {"handle": queue_id},
            })
            dispatch_id += 1

    # ---- HIP API events ----
    hip_api_events = []
    try:
        for row in conn.execute("""
            SELECT a.pid, a.tid, a.start, a.end, s.string, a.correlation_id
            FROM rocpd_api a
            JOIN rocpd_string s ON a.apiName_id = s.id
            WHERE a.end > a.start
            ORDER BY a.start
        """):
            api_pid, tid, start_ns, end_ns, api_name, corr_id = row
            hip_api_events.append({
                "size": 48,
                "kind": 3,
                "operation": api_name,
                "thread_id": tid or 0,
                "correlation_id": {"internal": corr_id or 0, "external": 0},
                "start_timestamp": start_ns,
                "end_timestamp": end_ns,
            })
    except sqlite3.OperationalError:
        pass

    conn.close()

    # ---- Assemble rocprofiler-sdk-tool JSON ----
    # Collect unique GPU IDs for agents list
    agent_ids = sorted(set(
        d["dispatch_info"]["agent_id"]["handle"] for d in kernel_dispatches
    ))
    agents = []
    for aid in agent_ids:
        agents.append({
            "size": 0,
            "id": {"handle": aid},
            "type": 2,  # GPU
            "name": f"GPU-Agent-{aid}",
        })

    output = {
        "rocprofiler-sdk-tool": [{
            "metadata": {
                "pid": pid,
                "init_time": init_time,
                "fini_time": fini_time,
            },
            "agents": agents,
            "counters": [],
            "strings": [],
            "code_objects": [],
            "kernel_symbols": kernel_symbols,
            "host_functions": [],
            "buffer_records": {
                "kernel_dispatch": kernel_dispatches,
                "memory_copy": memory_copies,
                "hip_api": hip_api_events,
            },
        }]
    }

    # Write JSON
    if output_json.endswith('.gz'):
        with gzip.open(output_json, 'wt') as f:
            json.dump(output, f)
    else:
        with open(output_json, 'w') as f:
            json.dump(output, f, indent=2)

    size_mb = os.path.getsize(output_json) / 1024 / 1024
    print(f"Written {output_json} ({size_mb:.1f} MB)")
    print(f"  Kernel dispatches: {len(kernel_dispatches)}")
    print(f"  Unique kernels:    {len(kernel_symbols)}")
    print(f"  Memory copies:     {len(memory_copies)}")
    print(f"  HIP API calls:     {len(hip_api_events)}")
    print("  Format: rocprofiler-sdk-tool (TraceLens compatible)")
    return True


def _map_copy_operation(op_type):
    """Map RTL copy type string to rocprofv3 operation enum."""
    if not op_type:
        return 0
    op_lower = op_type.lower()
    if "h2d" in op_lower or "hosttodevice" in op_lower:
        return 1  # MEMORY_COPY_HOST_TO_DEVICE
    elif "d2h" in op_lower or "devicetohost" in op_lower:
        return 2  # MEMORY_COPY_DEVICE_TO_HOST
    elif "d2d" in op_lower or "devicetodevice" in op_lower:
        return 3  # MEMORY_COPY_DEVICE_TO_DEVICE
    return 0


def run_convert_rocprofv3(args):
    """Entry point for 'rtl convert --format rocprofv3'."""
    input_db = args.input
    output_json = args.output or input_db.replace(".db", "_results.json")

    if not os.path.exists(input_db):
        print(f"Error: {input_db} not found", file=sys.stderr)
        sys.exit(1)

    if not convert_to_rocprofv3(input_db, output_json):
        sys.exit(1)
