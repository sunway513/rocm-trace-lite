import glob
import os
import sys
import subprocess
import sqlite3


def _preflight_check(lib_path):
    """Advisory check: verify the profiler library and its dependencies.

    Checks (all advisory — prints warnings but never blocks tracing):
      1. librtl.so exists
      2. librtl.so dependencies are resolvable (advisory ldd check)
      3. libhsa-runtime64.so is findable on disk
      4. HSA_TOOLS_LIB is not already set to a conflicting value

    Returns True if all checks pass, False if any warning was emitted.
    """
    ok = True

    def warn(msg):
        print("rtl: WARNING: {}".format(msg), file=sys.stderr)

    def info(msg):
        print("rtl: {}".format(msg), file=sys.stderr)

    # 1. Check librtl.so exists
    if not os.path.isfile(lib_path):
        warn("librtl.so not found at {}".format(lib_path))
        return False

    info("librtl.so OK ({})".format(lib_path))

    # 2. Advisory dependency check via ldd (avoids loading the library into
    #    this Python process, which would run .so constructors / HSA OnLoad)
    hsa_missing_in_ldd = False
    try:
        ldd_out = subprocess.run(
            ["ldd", lib_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=5
        )
        if ldd_out.returncode == 0:
            for line in ldd_out.stdout.splitlines():
                if "not found" in line:
                    lib_name = line.split("=>")[0].strip() if "=>" in line else line.strip()
                    warn("missing dependency: {}".format(lib_name))
                    if "libhsa-runtime64" in line:
                        hsa_missing_in_ldd = True
                        warn("ROCm HSA runtime is missing")
                        _suggest_rocm_paths()
                    elif "libsqlite3" in line:
                        warn("install with: apt install libsqlite3-dev")
                    ok = False
    except (OSError, subprocess.TimeoutExpired):
        pass  # ldd not available, skip

    # 3. Check ROCm / HSA runtime on disk (only suggest LD_LIBRARY_PATH
    #    fix if ldd actually failed to resolve it — the loader may find it
    #    via RPATH, ld.so.cache, or default system paths)
    rocm_path = os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH")
    search_dirs = ["/opt/rocm/lib", "/opt/rocm/lib64"]
    if rocm_path:
        search_dirs.insert(0, os.path.join(rocm_path, "lib"))
        search_dirs.insert(1, os.path.join(rocm_path, "lib64"))

    hsa_found = False
    for d in search_dirs:
        hsa_lib = os.path.join(d, "libhsa-runtime64.so")
        if os.path.exists(hsa_lib):
            hsa_found = True
            info("libhsa-runtime64.so OK ({})".format(hsa_lib))
            if hsa_missing_in_ldd:
                warn("  fix: export LD_LIBRARY_PATH={}:$LD_LIBRARY_PATH".format(d))
            break

    if not hsa_found:
        warn("libhsa-runtime64.so not found in any standard ROCm path")
        _suggest_rocm_paths()
        ok = False

    # 4. Check for conflicting HSA_TOOLS_LIB
    existing = os.environ.get("HSA_TOOLS_LIB")
    if existing and existing != lib_path:
        warn("HSA_TOOLS_LIB already set to: {}".format(existing))
        warn("  rtl will override it with: {}".format(lib_path))

    return ok


def _suggest_rocm_paths():
    """Print suggestions for finding ROCm."""

    def warn(msg):
        print("rtl: WARNING: {}".format(msg), file=sys.stderr)
    rocm_candidates = []
    for d in ["/opt/rocm/lib", "/usr/lib/x86_64-linux-gnu"]:
        if os.path.isdir(d):
            hsa = os.path.join(d, "libhsa-runtime64.so")
            if os.path.exists(hsa):
                rocm_candidates.append(d)
    if rocm_candidates:
        warn("  found ROCm libs at: {}".format(", ".join(rocm_candidates)))
        warn("  fix: export LD_LIBRARY_PATH={}:$LD_LIBRARY_PATH".format(rocm_candidates[0]))
    else:
        warn("  ROCm does not appear to be installed")
        warn("  install ROCm or set ROCM_PATH/LD_LIBRARY_PATH to your ROCm installation")


def run_trace(args):
    cmd = [c for c in args.cmd if c != "--"]
    if not cmd:
        print("Error: no command specified. Usage: rtl trace python3 script.py", file=sys.stderr)
        sys.exit(1)

    from rocm_trace_lite import get_lib_path
    lib = get_lib_path()

    # Advisory preflight: warn about missing deps, never block
    _preflight_check(lib)

    output = args.output

    # Multi-process safety: use %p pattern so each process writes its own file.
    # After the workload, we merge all per-process files into the final output.
    trace_dir = os.path.dirname(os.path.abspath(output)) or "."
    trace_base = os.path.splitext(os.path.basename(output))[0]
    per_process_pattern = os.path.join(trace_dir, f"{trace_base}_%p.db")

    env = os.environ.copy()
    env["HSA_TOOLS_LIB"] = lib
    env["RTL_OUTPUT"] = per_process_pattern
    # LD_PRELOAD librtl.so so roctx shim symbols are globally visible.
    # Without this, ctypes.CDLL(None).roctxRangePushA won't find our symbols.
    existing_preload = env.get("LD_PRELOAD", "")
    env["LD_PRELOAD"] = "{}:{}".format(lib, existing_preload) if existing_preload else lib
    if hasattr(args, 'mode') and args.mode:
        env["RTL_MODE"] = args.mode

    # Ensure the subprocess can dlopen librtl.so and its dependencies.
    # Add the .so's directory and common ROCm paths to LD_LIBRARY_PATH.
    lib_dir = os.path.dirname(os.path.abspath(lib))
    extra_dirs = [lib_dir]
    for rocm_lib in ["/opt/rocm/lib", "/opt/rocm/lib64"]:
        if os.path.isdir(rocm_lib):
            extra_dirs.append(rocm_lib)
    rocm_path = os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH")
    if rocm_path:
        extra_dirs.append(os.path.join(rocm_path, "lib"))
    existing = env.get("LD_LIBRARY_PATH", "")
    combined = ":".join(extra_dirs)
    env["LD_LIBRARY_PATH"] = "{}:{}".format(combined, existing) if existing else combined

    # Clean stale per-process files (only those matching our PID pattern)
    import re
    for f in glob.glob(os.path.join(trace_dir, f"{trace_base}_*.db")):
        basename = os.path.basename(f)
        if re.match(rf"^{re.escape(trace_base)}_\d+\.db$", basename):
            try:
                os.remove(f)
            except OSError:
                pass
    if os.path.exists(output) and os.path.isfile(output):
        try:
            os.remove(output)
        except OSError:
            pass

    result = subprocess.run(cmd, env=env)

    # Collect per-process trace files
    # Collect per-process files (strict PID pattern: trace_DIGITS.db)
    per_process_files = sorted([
        f for f in glob.glob(os.path.join(trace_dir, f"{trace_base}_*.db"))
        if re.match(rf"^{re.escape(trace_base)}_\d+\.db$", os.path.basename(f))
    ])

    if not per_process_files:
        print("rtl: WARNING: no trace files produced — 0 GPU ops captured", file=sys.stderr)
        print("rtl: Possible causes:", file=sys.stderr)
        print("rtl:   - HSA_TOOLS_LIB was not inherited by GPU worker subprocess", file=sys.stderr)
        print("rtl:   - The workload didn't run any GPU kernels", file=sys.stderr)
        print("rtl:   - librtl.so failed to load (check warnings above)", file=sys.stderr)
        print("rtl: Try: export HSA_TOOLS_LIB=$(python3 -c 'from rocm_trace_lite import get_lib_path; print(get_lib_path())')", file=sys.stderr)
        print("rtl:      export RTL_OUTPUT=trace_%p.db", file=sys.stderr)
        print("rtl:      <your command>", file=sys.stderr)
        sys.exit(result.returncode)

    import shutil
    if len(per_process_files) == 1:
        # Single process — move to final output
        shutil.move(per_process_files[0], output)
    else:
        # Multi-process — merge all into one
        _merge_traces(per_process_files, output)
        # Clean up per-process files (best-effort)
        for f in per_process_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except OSError:
                pass

    # Generate all output artifacts from one command
    base = os.path.splitext(output)[0]
    summary_file = base + "_summary.txt"
    json_file = base + ".json.gz"

    # 1. Summary → terminal + .txt file
    summary_text = _generate_summary(output)
    if summary_text:
        print(summary_text)
        with open(summary_file, "w") as f:
            f.write(summary_text)

    # 2. Perfetto JSON → compressed .json.gz
    _generate_perfetto(output, json_file)

    # 3. Print output locations
    print("\nOutput files:")
    print(f"  {output}")
    if os.path.exists(summary_file):
        print(f"  {summary_file}")
    if os.path.exists(json_file):
        size_mb = os.path.getsize(json_file) / 1024 / 1024
        print(f"  {json_file} ({size_mb:.1f} MB → open in https://ui.perfetto.dev)")

    sys.exit(result.returncode)


def _checkpoint_wal(db_path):
    """Checkpoint WAL journal and convert to DELETE mode for clean ATTACH."""
    try:
        c = sqlite3.connect(db_path)
        c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        c.execute("PRAGMA journal_mode=DELETE")
        c.commit()
        c.close()
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        pass


def _merge_traces(input_files, output_path):
    """Atomically merge complete process traces, preserving foreign keys.

    SQLite backup includes committed WAL pages. Inputs are retained until the
    caller has a complete output; any failure leaves all inputs recoverable.
    """
    import tempfile
    from contextlib import closing

    def connect_source(path):
        from pathlib import Path
        return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)

    def rows(db, table):
        exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                            (table,)).fetchone()
        if not exists:
            return []
        cur = db.execute('SELECT * FROM ' + table)
        names = [col[0] for col in cur.description]
        return [dict(zip(names, row)) for row in cur]

    def insert(db, table, row):
        columns = ','.join(row)
        marks = ','.join('?' for _ in row)
        return db.execute(f'INSERT INTO {table}({columns}) VALUES({marks})',
                          tuple(row.values())).lastrowid

    if not input_files:
        raise ValueError('No input traces')
    inputs = sorted(input_files, key=os.path.getsize, reverse=True)
    fd, temp = tempfile.mkstemp(prefix='.rtl-merge-', suffix='.db',
                                dir=os.path.dirname(os.path.abspath(output_path)))
    os.close(fd)
    try:
        with closing(sqlite3.connect(temp)) as dst:
            with closing(connect_source(inputs[0])) as src:
                src.backup(dst)
            # Keep the output self-contained before the atomic rename.
            dst.execute('PRAGMA journal_mode=DELETE')
            cols = {r[1] for r in dst.execute('PRAGMA table_info(rocpd_op)')}
            if 'roctxId' not in cols:
                dst.execute('ALTER TABLE rocpd_op ADD COLUMN roctxId INTEGER DEFAULT 0')
            next_range = dst.execute('SELECT COALESCE(MAX(roctxId),0) FROM rocpd_op').fetchone()[0]
            for path in inputs[1:]:
                with closing(connect_source(path)) as src, dst:
                    strings = {}
                    for row in rows(src, 'rocpd_string'):
                        dst.execute('INSERT OR IGNORE INTO rocpd_string(string) VALUES(?)',
                                    (row['string'],))
                        strings[row['id']] = dst.execute(
                            'SELECT id FROM rocpd_string WHERE string IS ?', (row['string'],)).fetchone()[0]
                    maps = {'rocpd_api': {}, 'rocpd_op': {}}
                    ranges = {}
                    for table, string_cols in (
                            ('rocpd_api', ('apiName_id', 'args_id')),
                            ('rocpd_op', ('description_id', 'opType_id'))):
                        for row in rows(src, table):
                            old_id = row.pop('id')
                            for col in string_cols:
                                if row.get(col) is not None:
                                    row[col] = strings[row[col]]
                            if table == 'rocpd_op':
                                old_range = row.get('roctxId') or 0
                                if old_range and old_range not in ranges:
                                    next_range += 1
                                    ranges[old_range] = next_range
                                row['roctxId'] = ranges.get(old_range, 0)
                            maps[table][old_id] = insert(dst, table, row)
                    for table in ('rocpd_api_ops', 'rocpd_kernelapi', 'rocpd_copyapi',
                                  'rocpd_metadata', 'rocpd_monitor'):
                        for row in rows(src, table):
                            row.pop('id', None)
                            for col, parent in (('api_id', 'rocpd_api'), ('op_id', 'rocpd_op')):
                                if row.get(col) is not None:
                                    row[col] = maps[parent][row[col]]
                            if row.get('kernelName_id') is not None:
                                row['kernelName_id'] = strings[row['kernelName_id']]
                            insert(dst, table, row)
            dst.commit()
            if dst.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise sqlite3.DatabaseError('Merged trace failed integrity_check')
            if dst.execute('PRAGMA foreign_key_check').fetchone():
                raise sqlite3.DatabaseError('Merged trace has invalid references')
        os.replace(temp, output_path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    print(f'Merged {len(inputs)} process traces', file=sys.stderr)


def _generate_summary(db_path):
    """Generate summary text from trace database."""
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        ops = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
        lines = [f"Trace: {db_path} ({ops} GPU ops)", ""]

        # Top kernels
        try:
            rows = conn.execute("SELECT * FROM top LIMIT 20").fetchall()
        except sqlite3.OperationalError:
            rows = []
        if rows:
            lines.append(f"{'Kernel':<60} {'Calls':>6} {'Total(us)':>10} {'Avg(us)':>8} {'%':>6}")
            lines.append("=" * 96)
            for r in rows:
                name = r[0][:57] + "..." if len(r[0]) > 60 else r[0]
                total_us = r[2] / 1000 if r[2] else 0
                avg_us = r[3] / 1000 if r[3] else 0
                pct = r[6] if r[6] else 0
                lines.append(f"{name:<60} {r[1]:>6} {total_us:>10.1f} {avg_us:>8.1f} {pct:>6.1f}")

        # GPU utilization
        try:
            busy = conn.execute("SELECT * FROM busy").fetchall()
        except sqlite3.OperationalError:
            busy = []
        if busy:
            lines.append("")
            lines.append("GPU Utilization:")
            for row in busy:
                lines.append(f"  GPU {row[0]}: {row[1]}% ({row[2]} ops, {row[3]/1e6:.1f}ms busy, {row[4]/1e6:.1f}ms wall)")

        conn.close()
        return "\n".join(lines)
    except Exception as e:
        return f"Warning: could not read trace: {e}"


def _generate_perfetto(db_path, json_gz_path):
    """Convert trace to compressed Perfetto JSON."""
    if not os.path.exists(db_path):
        return
    try:
        from rocm_trace_lite.cmd_convert import convert
        import gzip
        import tempfile

        # Convert to temp JSON first
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_json = tmp.name

        convert(db_path, tmp_json)

        # Compress to .json.gz
        with open(tmp_json, "rb") as f_in:
            with gzip.open(json_gz_path, "wb") as f_out:
                f_out.write(f_in.read())

        os.unlink(tmp_json)
    except Exception as e:
        print(f"Warning: could not generate Perfetto trace: {e}", file=sys.stderr)
