"""rocm-trace-lite: Self-contained GPU kernel profiler for ROCm."""

__version__ = "0.1.1"


def get_lib_path() -> str:
    """Return the path to librtl.so.

    Search order:
      1. Package-local lib/ directory
      2. /usr/local/lib/
    Raises FileNotFoundError if the library cannot be found.
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

    raise FileNotFoundError(
        "librtl.so not found. Searched:\n"
        f"  1. {pkg_lib}\n"
        f"  2. {sys_lib}\n"
        "Please build the library first (make) or install the wheel."
    )
