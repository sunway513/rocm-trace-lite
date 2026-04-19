"""rocm-trace-lite: Self-contained GPU kernel profiler for ROCm."""

__version__ = "0.3.6"


def get_lib_path() -> str:
    """Return the path to librtl.so.

    Search order:
      1. Package-local lib/ directory
      2. /usr/local/lib/
      3. JIT compile from source if ROCm headers available
    Raises FileNotFoundError if the library cannot be found or built.
    """
    import os

    # 1. Check package directory lib/
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_lib = os.path.join(pkg_dir, "lib", "librtl.so")
    if os.path.isfile(pkg_lib):
        return pkg_lib

    # 2. Check system path
    sys_lib = "/usr/local/lib/librtl.so"
    if os.path.isfile(sys_lib):
        return sys_lib

    # 3. JIT compile from source
    try:
        from rocm_trace_lite._build import compile_librtl
        lib_dir = os.path.join(pkg_dir, "lib")
        ok, msg = compile_librtl(lib_dir)
        if ok and os.path.isfile(pkg_lib):
            import sys
            print("rtl: JIT compiled librtl.so -> %s" % msg, file=sys.stderr)
            return pkg_lib
    except Exception:
        pass

    raise FileNotFoundError(
        "librtl.so not found. Searched:\n"
        f"  1. {pkg_lib}\n"
        f"  2. {sys_lib}\n"
        "  3. JIT compilation (ROCm headers not found or compilation failed)\n"
        "Please build the library first (make) or ensure ROCm is installed."
    )
