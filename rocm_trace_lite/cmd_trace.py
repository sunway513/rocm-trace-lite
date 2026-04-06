import os
import sys
import subprocess
import sqlite3


def run_trace(args):
    if not args.cmd:
        print("Error: no command specified. Usage: rpd-lite trace python3 script.py", file=sys.stderr)
        sys.exit(1)

    from rocm_trace_lite import get_lib_path
    lib = get_lib_path()
    output = args.output

    env = os.environ.copy()
    env["HSA_TOOLS_LIB"] = lib
    env["RPD_LITE_OUTPUT"] = output

    # Remove stale trace
    if os.path.exists(output):
        os.remove(output)

    result = subprocess.run(args.cmd, env=env)

    # Print summary
    if os.path.exists(output):
        try:
            conn = sqlite3.connect(output)
            ops = conn.execute("SELECT count(*) FROM rocpd_op").fetchone()[0]
            print(f"\nTrace: {output} ({ops} GPU ops)")
            rows = conn.execute("SELECT * FROM top LIMIT 5").fetchall()
            if rows:
                print(f"{'Kernel':<60} {'Calls':>6} {'Total(us)':>10} {'Avg(us)':>8} {'%':>6}")
                print("-" * 96)
                for r in rows:
                    name = r[0][:57] + "..." if len(r[0]) > 60 else r[0]
                    print(f"{name:<60} {r[1]:>6} {r[2]/1000:>10.1f} {r[3]/1000:>8.1f} {r[6]:>6.1f}")
            conn.close()
        except Exception as e:
            print(f"Warning: could not read trace: {e}", file=sys.stderr)

    sys.exit(result.returncode)
