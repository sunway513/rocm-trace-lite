"""
cmd_info — Print structural info about an RPD trace file.

Shows file size, tables, record counts, time range, and metadata.
"""

import sqlite3
import sys
import os


def run_info(args):
    """Entry point for the 'info' subcommand."""
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    with sqlite3.connect(args.input) as conn:
        print(f"Trace: {args.input}")
        print(f"  Size: {os.path.getsize(args.input) / 1024 / 1024:.1f} MB")

        # Tables
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()]
        print(f"  Tables: {', '.join(tables)}")

        # Record counts (guard on table existence)
        for t in ["rocpd_op", "rocpd_api", "rocpd_string", "rocpd_metadata"]:
            if t in tables:
                count = conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                print(f"  {t}: {count} rows")

        # Time range (guard on rocpd_op existence)
        if "rocpd_op" in tables:
            row = conn.execute(
                "SELECT MIN(start), MAX(end) FROM rocpd_op WHERE end > start"
            ).fetchone()
            if row and row[0]:
                duration_s = (row[1] - row[0]) / 1e9
                print(f"  Duration: {duration_s:.3f}s")

        # Metadata
        if "rocpd_metadata" in tables:
            try:
                meta = conn.execute("SELECT tag, value FROM rocpd_metadata").fetchall()
                if meta:
                    print("  Metadata:")
                    for tag, val in meta:
                        print(f"    {tag}: {val}")
            except sqlite3.OperationalError:
                pass

        # Unique kernels
        if "rocpd_op" in tables:
            kernels = conn.execute(
                "SELECT count(DISTINCT description_id) FROM rocpd_op"
            ).fetchone()[0]
            print(f"  Unique kernels: {kernels}")
