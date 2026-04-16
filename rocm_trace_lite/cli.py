#!/usr/bin/env python3
"""rtl: Self-contained GPU kernel profiler for ROCm."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="rtl",
        description="Self-contained GPU kernel profiler for ROCm. Zero roctracer dependency.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {get_version()}")
    sub = parser.add_subparsers(dest="command")

    # trace
    trace_p = sub.add_parser("trace", help="Trace a workload")
    trace_p.add_argument("-o", "--output", default="trace.db", help="Output trace file (default: trace.db)")
    trace_p.add_argument("-m", "--mode", choices=["lite", "standard", "default", "full"], default=None,
                         help="Profiling mode: "
                              "lite (~0%% overhead, skip has-signal packets, safe for all ROCm, default), "
                              "standard (GPU timing for all count==1 dispatches, skip graph replay), "
                              "full (profile everything including graph replay, requires ROCm 7.13+). "
                              "'default' is accepted as alias for 'standard'.")
    trace_p.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to trace")

    # convert
    conv_p = sub.add_parser("convert", help="Convert RPD trace to Perfetto JSON")
    conv_p.add_argument("input", help="Input .db file")
    conv_p.add_argument("-o", "--output", default=None, help="Output .json file (default: input with .json extension)")

    # summary
    sum_p = sub.add_parser("summary", help="Show top kernels from trace")
    sum_p.add_argument("input", help="Input .db file")
    sum_p.add_argument("-n", "--limit", type=int, default=20, help="Number of rows (default: 20)")

    # info
    info_p = sub.add_parser("info", help="Show trace metadata")
    info_p.add_argument("input", help="Input .db file")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # dispatch
    if args.command == "trace":
        from rocm_trace_lite.cmd_trace import run_trace
        run_trace(args)
    elif args.command == "convert":
        from rocm_trace_lite.cmd_convert import run_convert
        run_convert(args)
    elif args.command == "summary":
        from rocm_trace_lite.cmd_summary import run_summary
        run_summary(args)
    elif args.command == "info":
        from rocm_trace_lite.cmd_info import run_info
        run_info(args)


def get_version():
    try:
        from rocm_trace_lite import __version__
        return __version__
    except ImportError:
        return "unknown"


if __name__ == "__main__":
    main()
