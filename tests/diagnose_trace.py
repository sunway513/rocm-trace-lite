#!/usr/bin/env python3
"""Diagnose per-process trace files for TP>1 profiling issues.

Usage:
    python tests/diagnose_trace.py trace_*.db
    python tests/diagnose_trace.py /path/to/trace_12345.db /path/to/trace_12346.db

Reports per-file:
  - Total kernel count (rocpd_op with gpuId >= 0)
  - GPU IDs seen
  - Unique kernel names (top 10)
  - File size
  - HIP API call count
  - Whether completion worker data looks complete

Flags asymmetry between processes (one rank missing kernels = Issue #31).
"""
import os
import re
import sqlite3
import sys
import glob


def diagnose_file(db_path):
    """Analyze a single trace .db file."""
    info = {
        "path": db_path,
        "size_kb": os.path.getsize(db_path) / 1024,
        "kernel_count": 0,
        "api_count": 0,
        "gpu_ids": [],
        "top_kernels": [],
        "pid": None,
        "error": None,
    }

    try:
        conn = sqlite3.connect(db_path)

        # Kernel ops (gpuId >= 0, excluding markers)
        try:
            info["kernel_count"] = conn.execute(
                "SELECT count(*) FROM rocpd_op WHERE gpuId >= 0"
            ).fetchone()[0]
        except sqlite3.OperationalError as e:
            info["error"] = f"rocpd_op query failed: {e}"
            conn.close()
            return info

        # GPU IDs
        rows = conn.execute(
            "SELECT DISTINCT gpuId FROM rocpd_op WHERE gpuId >= 0 ORDER BY gpuId"
        ).fetchall()
        info["gpu_ids"] = [r[0] for r in rows]

        # Top kernel names
        try:
            rows = conn.execute(
                "SELECT s.string, count(*) as cnt "
                "FROM rocpd_op o "
                "JOIN rocpd_string s ON o.description_id = s.id "
                "WHERE o.gpuId >= 0 "
                "GROUP BY s.string ORDER BY cnt DESC LIMIT 10"
            ).fetchall()
            info["top_kernels"] = [(name, cnt) for name, cnt in rows]
        except sqlite3.OperationalError:
            pass

        # HIP API count
        try:
            info["api_count"] = conn.execute(
                "SELECT count(*) FROM rocpd_api"
            ).fetchone()[0]
        except sqlite3.OperationalError:
            info["api_count"] = 0

        # PID from metadata
        try:
            row = conn.execute(
                "SELECT value FROM rocpd_metadata WHERE tag='pid'"
            ).fetchone()
            if row:
                info["pid"] = int(row[0])
        except (sqlite3.OperationalError, TypeError, ValueError):
            pass

        conn.close()
    except Exception as e:
        info["error"] = str(e)

    return info


def print_report(infos):
    """Print diagnostic report and flag issues."""
    print("=" * 80)
    print("rocm-trace-lite TP>1 Diagnostic Report")
    print("=" * 80)

    for i, info in enumerate(infos):
        print(f"\n--- File {i+1}: {info['path']} ---")
        print(f"  Size:         {info['size_kb']:.1f} KB")
        print(f"  PID:          {info['pid'] or 'unknown'}")
        print(f"  Kernel ops:   {info['kernel_count']}")
        print(f"  HIP API ops:  {info['api_count']}")
        print(f"  GPU IDs:      {info['gpu_ids']}")

        if info["error"]:
            print(f"  ERROR:        {info['error']}")

        if info["top_kernels"]:
            print("  Top kernels:")
            for name, cnt in info["top_kernels"]:
                short = name[:70] + "..." if len(name) > 70 else name
                print(f"    {cnt:>6}x  {short}")

    # Asymmetry detection
    print("\n" + "=" * 80)
    print("ASYMMETRY CHECK")
    print("=" * 80)

    kernel_counts = [info["kernel_count"] for info in infos]
    if not kernel_counts:
        print("  No files to compare.")
        return

    max_k = max(kernel_counts)
    min_k = min(kernel_counts)

    if max_k == 0:
        print("  WARNING: ALL files have 0 kernels — profiler not capturing anything!")
        return

    for info in infos:
        status = "OK"
        if info["kernel_count"] == 0:
            status = "MISSING KERNELS (Issue #31 candidate)"
        elif max_k > 0 and info["kernel_count"] < max_k * 0.5:
            status = f"LOW ({info['kernel_count']}/{max_k} = {100*info['kernel_count']/max_k:.0f}%)"
        print(f"  {os.path.basename(info['path'])}: "
              f"{info['kernel_count']} kernels -> {status}")

    if min_k == 0 and max_k > 0:
        print("\n  DIAGNOSIS: At least one process has 0 kernels while others have data.")
        print("  This confirms Issue #31 — TP>1 missing compute kernels.")
        print("  Likely causes:")
        print("    1. HSA_TOOLS_LIB not inherited by worker process")
        print("    2. OnLoad not called in worker (HSA runtime not re-initialized)")
        print("    3. fork() without exec — completion worker thread lost")
        print("    4. Signal race — profiling_signal recycled before timestamp read")
    elif min_k > 0 and min_k < max_k * 0.8:
        print(f"\n  NOTE: Kernel counts vary ({min_k}-{max_k}). "
              "Minor asymmetry may be normal for different ranks.")
    elif min_k > 0:
        print("\n  All processes captured kernels. TP>1 profiling appears functional.")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} trace_*.db [trace_*.db ...]")
        print(f"       {sys.argv[0]} /path/to/trace_dir/  (auto-finds trace_*.db)")
        sys.exit(1)

    files = []
    for arg in sys.argv[1:]:
        if os.path.isdir(arg):
            # Find trace_*.db in directory
            found = sorted(glob.glob(os.path.join(arg, "trace_*.db")))
            found = [f for f in found
                     if re.match(r"^trace_\d+\.db$", os.path.basename(f))]
            files.extend(found)
        elif os.path.isfile(arg):
            files.append(arg)
        else:
            # Try glob expansion
            expanded = sorted(glob.glob(arg))
            files.extend(expanded)

    if not files:
        print("No trace files found.")
        sys.exit(1)

    infos = [diagnose_file(f) for f in files]
    print_report(infos)


if __name__ == "__main__":
    main()
