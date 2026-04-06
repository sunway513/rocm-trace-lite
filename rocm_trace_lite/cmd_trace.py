import glob
import os
import sys
import subprocess
import sqlite3


def run_trace(args):
    cmd = [c for c in args.cmd if c != "--"]
    if not cmd:
        print("Error: no command specified. Usage: rpd-lite trace python3 script.py", file=sys.stderr)
        sys.exit(1)

    from rocm_trace_lite import get_lib_path
    lib = get_lib_path()
    output = args.output

    # Multi-process safety: use %p pattern so each process writes its own file.
    # After the workload, we merge all per-process files into the final output.
    trace_dir = os.path.dirname(os.path.abspath(output)) or "."
    trace_base = os.path.splitext(os.path.basename(output))[0]
    per_process_pattern = os.path.join(trace_dir, f"{trace_base}_%p.db")

    env = os.environ.copy()
    env["HSA_TOOLS_LIB"] = lib
    env["RPD_LITE_OUTPUT"] = per_process_pattern

    # Clean stale per-process files (only those matching our PID pattern)
    import re
    for f in glob.glob(os.path.join(trace_dir, f"{trace_base}_*.db")):
        # Only remove files matching trace_base_DIGITS.db (PID pattern)
        basename = os.path.basename(f)
        if re.match(rf"^{re.escape(trace_base)}_\d+\.db$", basename):
            os.remove(f)
    if os.path.exists(output) and os.path.isfile(output):
        os.remove(output)

    result = subprocess.run(cmd, env=env)

    # Collect per-process trace files
    # Collect per-process files (strict PID pattern: trace_DIGITS.db)
    per_process_files = sorted([
        f for f in glob.glob(os.path.join(trace_dir, f"{trace_base}_*.db"))
        if re.match(rf"^{re.escape(trace_base)}_\d+\.db$", os.path.basename(f))
    ])

    if not per_process_files:
        print("Warning: no trace files produced", file=sys.stderr)
        sys.exit(result.returncode)

    if len(per_process_files) == 1:
        # Single process — just rename
        os.rename(per_process_files[0], output)
    else:
        # Multi-process — merge all into one
        _merge_traces(per_process_files, output)
        # Clean up per-process files
        for f in per_process_files:
            os.remove(f)

    # Print summary
    _print_summary(output)
    sys.exit(result.returncode)


def _checkpoint_wal(db_path):
    """Checkpoint WAL journal to ensure all data is in the main DB file."""
    try:
        c = sqlite3.connect(db_path)
        c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        c.close()
    except sqlite3.OperationalError:
        pass


def _merge_traces(input_files, output_path):
    """Merge multiple per-process RPD trace files into one."""
    # Use the largest file as the base (likely the main ModelRunner)
    input_files.sort(key=lambda f: os.path.getsize(f), reverse=True)

    # Copy the largest as the base
    import shutil
    _checkpoint_wal(input_files[0])
    shutil.copy2(input_files[0], output_path)

    conn = sqlite3.connect(output_path)
    merged_ops = 0

    for idx, src_file in enumerate(input_files[1:]):
        alias = f"src{idx}"
        try:
            _checkpoint_wal(src_file)
            # Path is internally generated (not user input), but sanitize for safety
            safe_path = src_file.replace("'", "''")
            conn.execute(f"ATTACH DATABASE '{safe_path}' AS {alias}")
            src_ops = conn.execute(f"SELECT count(*) FROM {alias}.rocpd_op").fetchone()[0]
            if src_ops == 0:
                conn.execute(f"DETACH DATABASE {alias}")
                continue
            conn.execute(f"INSERT OR IGNORE INTO rocpd_string(string) SELECT string FROM {alias}.rocpd_string")
            count = conn.execute(f"""
                INSERT INTO rocpd_op(gpuId, queueId, sequenceId, start, end, description_id, opType_id)
                SELECT o.gpuId, o.queueId, o.sequenceId, o.start, o.end,
                    (SELECT id FROM rocpd_string WHERE string = (SELECT string FROM {alias}.rocpd_string WHERE id = o.description_id)),
                    (SELECT id FROM rocpd_string WHERE string = (SELECT string FROM {alias}.rocpd_string WHERE id = o.opType_id))
                FROM {alias}.rocpd_op o
            """).rowcount
            merged_ops += count
            try:
                conn.execute(f"DETACH DATABASE {alias}")
            except sqlite3.OperationalError:
                pass  # DETACH may fail on some SQLite versions, data already merged
        except sqlite3.OperationalError as e:
            print(f"Warning: could not merge {src_file}: {e}", file=sys.stderr)

    conn.commit()
    conn.close()

    if merged_ops > 0:
        print(f"Merged {len(input_files)} process traces ({merged_ops} additional ops)", file=sys.stderr)


def _print_summary(output):
    """Print trace summary after workload completes."""
    if not os.path.exists(output):
        return
    try:
        conn = sqlite3.connect(output)
        ops = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        print(f"\nTrace: {output} ({ops} GPU ops)")
        try:
            rows = conn.execute("SELECT * FROM top LIMIT 5").fetchall()
        except sqlite3.OperationalError:
            rows = []
        if rows:
            print(f"{'Kernel':<60} {'Calls':>6} {'Total(us)':>10} {'Avg(us)':>8} {'%':>6}")
            print("-" * 96)
            for r in rows:
                name = r[0][:57] + "..." if len(r[0]) > 60 else r[0]
                print(f"{name:<60} {r[1]:>6} {r[2]/1000:>10.1f} {r[3]/1000:>8.1f} {r[6]:>6.1f}")
        conn.close()
    except Exception as e:
        print(f"Warning: could not read trace: {e}", file=sys.stderr)
