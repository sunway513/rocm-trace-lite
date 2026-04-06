"""
cmd_summary — Print a summary of an RPD trace file.

Shows total stats, top kernels by time, and GPU utilization.
"""

import sqlite3
import sys
import os


def run_summary(args):
    """Entry point for the 'summary' subcommand."""
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(args.input)

    # Total stats
    ops = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
    apis = conn.execute("SELECT count(*) FROM rocpd_api").fetchone()[0]
    print(f"Trace: {args.input}")
    print(f"  GPU ops:   {ops}")
    print(f"  API calls: {apis}")
    print()

    # Top kernels
    print(f"{'Kernel':<60} {'Calls':>6} {'Total(us)':>10} {'Avg(us)':>8} {'%':>6}")
    print("=" * 96)
    rows = conn.execute("SELECT * FROM top LIMIT ?", (args.limit,)).fetchall()
    for r in rows:
        name = r[0][:57] + "..." if len(r[0]) > 60 else r[0]
        total_us = r[2] / 1000 if r[2] else 0
        avg_us = r[3] / 1000 if r[3] else 0
        pct = r[6] if r[6] else 0
        print(f"{name:<60} {r[1]:>6} {total_us:>10.1f} {avg_us:>8.1f} {pct:>6.1f}")

    # GPU utilization
    print()
    busy = conn.execute("SELECT * FROM busy").fetchall()
    if busy:
        print("GPU Utilization:")
        for row in busy:
            print(f"  GPU {row[0]}: {row[1]}% ({row[2]} ops, {row[3]/1e6:.1f}ms busy, {row[4]/1e6:.1f}ms wall)")

    conn.close()
