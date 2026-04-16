"""
cmd_convert — Convert RPD trace to Chrome Trace / Perfetto JSON.

Inlines the converter logic so it works after pip install (tools/ not in package).
"""

import sys
import os
import json
import gzip
import sqlite3


def convert(input_rpd, output_json):
    """Convert an RPD trace file to Chrome Trace JSON format."""
    conn = sqlite3.connect(input_rpd)
    events = []

    # Get time range from ops
    row = conn.execute("SELECT MIN(start), MAX(end) FROM rocpd_op WHERE end > start").fetchone()
    if not row or row[0] is None:
        row = conn.execute("SELECT MIN(start), MAX(end) FROM rocpd_api").fetchone()
    if not row or row[0] is None:
        print("Error: trace is empty")
        return

    base_ns = row[0]
    duration_s = (row[1] - row[0]) / 1e9

    print(f"Trace duration: {duration_s:.3f}s")
    print(f"Base timestamp: {base_ns} ns")

    # Collect all GPU ops
    ops = []
    for r in conn.execute("""
        SELECT o.gpuId, o.queueId, o.start, o.end, s.string, ot.string,
               o.completionSignal
        FROM rocpd_op o
        JOIN rocpd_string s ON o.description_id = s.id
        LEFT JOIN rocpd_string ot ON o.opType_id = ot.id
        WHERE o.end > o.start
        ORDER BY o.start
    """):
        ops.append(r)

    # Pre-parse dispatch_info to extract hwq for each op
    parsed_ops = []
    for gpu_id, queue_id, start_ns, end_ns, name, op_type, dispatch_info in ops:
        hwq = None
        wg = None
        grid = None
        if dispatch_info:
            for part in dispatch_info.split():
                if part.startswith("hwq="):
                    hwq = part[4:]
                elif part.startswith("wg="):
                    wg = part[3:]
                elif part.startswith("grid="):
                    grid = part[5:]
        parsed_ops.append((gpu_id, queue_id, start_ns, end_ns, name, op_type, hwq, wg, grid))

    # Decide track layout: hwq-based (if dispatch_info present) or queue-based (fallback)
    gpu_ids = sorted(set(r[0] for r in parsed_ops if r[0] is not None and r[0] >= 0))
    if not gpu_ids:
        gpu_ids = [0]

    # Collect unique hwq addresses per GPU
    hwq_by_gpu = {}
    for gpu_id, _q, _s, _e, _n, _t, hwq, _wg, _grid in parsed_ops:
        gid = gpu_id if gpu_id is not None and gpu_id >= 0 else 0
        if hwq:
            hwq_by_gpu.setdefault(gid, set()).add(hwq)

    use_hwq_tracks = len(hwq_by_gpu) > 0

    if use_hwq_tracks:
        # HWQ-based tracks: one track per hardware queue address
        hwq_track = {}
        for gid in sorted(hwq_by_gpu):
            for i, hwq in enumerate(sorted(hwq_by_gpu[gid])):
                hwq_track[(gid, hwq)] = i
        fallback_tid = max(hwq_track.values(), default=-1) + 1

        print(f"  {len(hwq_track)} HW queues across {len(gpu_ids)} GPU(s)")

        for gid in gpu_ids:
            events.append({
                "name": "process_name", "ph": "M",
                "pid": int(gid), "tid": 0,
                "args": {"name": f"GPU {gid}"}
            })
        for (gid, hwq), tid in hwq_track.items():
            events.append({
                "name": "thread_name", "ph": "M",
                "pid": int(gid), "tid": tid,
                "args": {"name": f"HWQ {hwq}"}
            })
    else:
        # Queue-based fallback (no dispatch_info in trace)
        hwq_track = None
        all_same_gpu = len(gpu_ids) <= 1

        if all_same_gpu:
            queue_ids = sorted(set(r[1] for r in parsed_ops if r[1] is not None))
            if len(queue_ids) > 100:
                queue_to_track = {q: 0 for q in queue_ids}
                print(f"  Single GPU detected, {len(queue_ids)} unique queue IDs (per-dispatch) -> collapsing to 1 track")
            else:
                queue_to_track = {q: i for i, q in enumerate(queue_ids)}
                print(f"  Single GPU detected, {len(queue_ids)} queues -> using queue-based tracks")

            events.append({
                "name": "process_name", "ph": "M",
                "pid": 0, "tid": 0,
                "args": {"name": "GPU 0"}
            })
        else:
            queue_to_track = None
            for gid in gpu_ids:
                events.append({
                    "name": "process_name", "ph": "M",
                    "pid": int(gid), "tid": 0,
                    "args": {"name": f"GPU {gid}"}
                })

    # GPU ops -> complete events
    op_count = 0
    for gpu_id, queue_id, start_ns, end_ns, name, op_type, hwq, wg, grid in parsed_ops:
        if gpu_id is None or gpu_id < 0:
            gpu_id = 0

        # Shorten long kernel names for display
        short_name = name
        if '.kd' in name:
            parts = name.split('_UserArgs_')
            if len(parts) > 1:
                short_name = parts[0]
            elif len(name) > 120:
                short_name = name[:60] + "..." + name[-40:]

        if hwq_track is not None:
            pid = int(gpu_id)
            tid = hwq_track.get((gpu_id, hwq), fallback_tid)
        elif queue_to_track is not None:
            pid = 0
            tid = queue_to_track.get(queue_id, 0)
        else:
            pid = int(gpu_id)
            tid = int(queue_id) if queue_id else 0

        args = {
            "full_name": name,
            "gpu": gpu_id,
            "queue": queue_id,
        }
        if hwq:
            args["hwq"] = hwq
        if wg:
            args["workgroup"] = wg
        if grid:
            args["grid"] = grid

        events.append({
            "name": short_name,
            "cat": op_type or "gpu",
            "ph": "X",
            "pid": pid,
            "tid": tid,
            "ts": (start_ns - base_ns) / 1000.0,
            "dur": (end_ns - start_ns) / 1000.0,
            "args": args,
        })
        op_count += 1

    # Add thread names for queues (fallback mode only)
    if hwq_track is None and queue_to_track is not None:
        for qid, track in queue_to_track.items():
            events.append({
                "name": "thread_name", "ph": "M",
                "pid": 0, "tid": track,
                "args": {"name": f"Queue {qid}"}
            })

    # HIP API -> complete events (if present)
    api_count = 0
    api_pids = set()
    try:
        for r in conn.execute("""
            SELECT a.pid, a.tid, a.start, a.end, s.string, sa.string
            FROM rocpd_api a
            JOIN rocpd_string s ON a.apiName_id = s.id
            LEFT JOIN rocpd_string sa ON a.args_id = sa.id
            ORDER BY a.start
        """):
            pid, tid, start_ns, end_ns, name, args_str = r
            if start_ns is None or end_ns is None:
                continue

            host_pid = 1000000 + (pid or 0)
            api_pids.add((host_pid, pid))

            events.append({
                "name": name,
                "cat": "hip_api",
                "ph": "X",
                "pid": host_pid,
                "tid": tid or 0,
                "ts": (start_ns - base_ns) / 1000.0,
                "dur": max(0, (end_ns - start_ns) / 1000.0),
                "args": {"api_args": args_str or ""}
            })
            api_count += 1

        for host_pid, real_pid in api_pids:
            events.append({
                "name": "process_name", "ph": "M",
                "pid": host_pid, "tid": 0,
                "args": {"name": f"Host (PID {real_pid})"}
            })
    except sqlite3.OperationalError:
        pass

    conn.close()

    # Write JSON (gzip if output ends with .gz)
    trace = {"traceEvents": events}
    if output_json.endswith('.gz'):
        with gzip.open(output_json, 'wt') as f:
            json.dump(trace, f)
    else:
        with open(output_json, 'w') as f:
            json.dump(trace, f)

    size_mb = os.path.getsize(output_json) / 1024 / 1024
    print(f"Written {output_json} ({size_mb:.1f} MB)")
    print(f"  GPU ops:   {op_count}")
    print(f"  API calls: {api_count}")
    print(f"  Total events: {len(events)}")


def run_convert(args):
    """Entry point for the 'convert' subcommand."""
    input_rpd = args.input
    output_json = args.output or input_rpd.replace(".db", ".json.gz")

    if not os.path.exists(input_rpd):
        print(f"Error: {input_rpd} not found", file=sys.stderr)
        sys.exit(1)

    try:
        convert(input_rpd, output_json)
    except Exception as e:
        print(f"Error: conversion failed: {e}", file=sys.stderr)
        sys.exit(1)
