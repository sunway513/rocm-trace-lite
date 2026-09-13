"""
cmd_convert — Convert RPD trace to Chrome Trace / Perfetto JSON.

Inlines the converter logic so it works after pip install (tools/ not in package).
"""

import sys
import os
import json
import gzip
import sqlite3
import tempfile
from pathlib import Path


# Level 3 keeps compression CPU modest for large traces; decoded events are unchanged.
GZIP_LEVEL = 3


class _EventWriter:
    def __init__(self, stream):
        self.stream = stream
        self.count = 0

    def append(self, event):
        if self.count:
            self.stream.write(", ")
        self.stream.write(json.dumps(event))
        self.count += 1


def _parse_dispatch(info):
    fields = {}
    if info:
        for part in info.split():
            for key in ("hwq", "wg", "grid"):
                if part.startswith(key + "="):
                    fields[key] = part[len(key) + 1:]
    return fields.get("hwq"), fields.get("wg"), fields.get("grid")


def _iter_ops(conn):
    # Cursor iteration keeps only the current joined row in Python memory.
    for gpu, queue, start, end, name, kind, info in conn.execute("""
        SELECT o.gpuId, o.queueId, o.start, o.end, s.string, ot.string,
               o.completionSignal
        FROM rocpd_op o
        JOIN rocpd_string s ON o.description_id = s.id
        LEFT JOIN rocpd_string ot ON o.opType_id = ot.id
        WHERE o.end > o.start
        ORDER BY o.start
    """):
        yield gpu, queue, start, end, name, kind, *_parse_dispatch(info)


def convert(input_rpd, output_json):
    """Stream events into an atomic plain/gzip output, outside the source DB."""
    output = Path(output_json)
    if Path(input_rpd).resolve() == output.resolve():
        raise ValueError("Output must not replace the input database")
    conn = sqlite3.connect(input_rpd)
    temp = None
    try:
        # Keep SQLite's ORDER BY work off the Python heap and bounded in cache.
        conn.execute("PRAGMA temp_store=FILE")
        conn.execute("PRAGMA cache_size=-8192")
        conn.execute("BEGIN")
        fd, temp = tempfile.mkstemp(prefix="." + output.name + ".", suffix=".tmp", dir=output.parent)
        os.close(fd)
        opener = gzip.open if str(output).endswith('.gz') else open
        kwargs = {"compresslevel": GZIP_LEVEL} if opener is gzip.open else {}
        with opener(temp, "wt", **kwargs) as stream:
            stream.write('{"traceEvents": [')
            events = _EventWriter(stream)
            counts = _convert(conn, events)
            if counts is None:
                return
            stream.write(']}')
        os.replace(temp, output)
        temp = None
        size_mb = output.stat().st_size / 1024 / 1024
        print(f"Written {output_json} ({size_mb:.1f} MB)")
        print(f"  GPU ops:   {counts[0]}")
        print(f"  API calls: {counts[1]}")
        print(f"  Total events: {events.count}")
    finally:
        conn.close()
        if temp is not None:
            os.unlink(temp)


def _convert(conn, events):
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

    # Discover only track metadata; never retain one Python object per op.
    gpu_ids = set()
    hwq_by_gpu = {}
    for gpu_id, info in conn.execute("""
        SELECT o.gpuId, o.completionSignal FROM rocpd_op o
        JOIN rocpd_string s ON o.description_id = s.id
        WHERE o.end > o.start
    """):
        if gpu_id is not None and gpu_id >= 0:
            gpu_ids.add(gpu_id)
        hwq, _wg, _grid = _parse_dispatch(info)
        gid = gpu_id if gpu_id is not None and gpu_id >= 0 else 0
        if hwq:
            hwq_by_gpu.setdefault(gid, set()).add(hwq)
    gpu_ids = sorted(gpu_ids) or [0]

    queue_sql = """
        SELECT DISTINCT o.queueId FROM rocpd_op o
        JOIN rocpd_string s ON o.description_id = s.id
        WHERE o.end > o.start AND o.queueId IS NOT NULL
        ORDER BY o.queueId
    """
    collapsed_queues = False
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
            # A per-dispatch queue ID must not create an event-sized dictionary.
            queue_count = conn.execute("SELECT COUNT(*) FROM (" + queue_sql + ")").fetchone()[0]
            collapsed_queues = queue_count > 100
            if collapsed_queues:
                queue_to_track = {}
                print(f"  Single GPU detected, {queue_count} unique queue IDs (per-dispatch) -> collapsing to 1 track")
            else:
                queue_to_track = {q: i for i, (q,) in enumerate(conn.execute(queue_sql))}
                print(f"  Single GPU detected, {queue_count} queues -> using queue-based tracks")

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
    for gpu_id, queue_id, start_ns, end_ns, name, op_type, hwq, wg, grid in _iter_ops(conn):
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
        queue_tracks = ((q, 0) for (q,) in conn.execute(queue_sql)) if collapsed_queues else queue_to_track.items()
        for qid, track in queue_tracks:
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

    return op_count, api_count


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
