"""Custom setup: JIT compile librtl.so + platform-specific wheel tag.

librtl.so must be compiled against the target system's ROCm HSA headers.
This setup.py attempts compilation during `pip install` from source.
If ROCm is not available, the install succeeds but tracing is disabled.
"""
import os
import time
import shutil
import sys

from setuptools import setup
from setuptools.command.build_py import build_py

try:
    from wheel.bdist_wheel import bdist_wheel

    class PlatformWheel(bdist_wheel):
        """Force platform-specific wheel tag."""
        def finalize_options(self):
            super().finalize_options()
            self.root_is_pure = False

        def get_tag(self):
            import sysconfig
            plat = sysconfig.get_platform().replace("-", "_").replace(".", "_")
            return "py3", "none", plat

    wheel_cmdclass = {"bdist_wheel": PlatformWheel}
except ImportError:
    wheel_cmdclass = {}


class BuildWithLibrtl(build_py):
    """Standard build_py + attempt to compile librtl.so at install time."""

    def run(self):
        super().run()

        lib_dest = os.path.join(self.build_lib, "rocm_trace_lite", "lib")
        so_path = os.path.join(lib_dest, "librtl.so")

        # Skip if .so already exists (pre-built via `make`)
        if os.path.isfile(so_path):
            return

        # Check if user already ran `make` in source tree
        # Prefer repo-root librtl.so (make output) over package-dir copy
        # to avoid packaging a stale .so
        for src_so in ["librtl.so", os.path.join("rocm_trace_lite", "lib", "librtl.so")]:
            if os.path.isfile(src_so):
                age_s = time.time() - os.path.getmtime(src_so)
                if age_s > 3600:
                    print("rocm-trace-lite: WARNING: packaging %s (%.0f min old) "
                          "— run `make` to rebuild" % (src_so, age_s / 60),
                          file=sys.stderr)
                os.makedirs(lib_dest, exist_ok=True)
                shutil.copy2(src_so, so_path)
                print("rocm-trace-lite: packaged %s (%.1f KB)" %
                      (src_so, os.path.getsize(so_path) / 1024))
                return

        # Attempt JIT compilation
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rocm_trace_lite"))
            from _build import compile_librtl
            ok, msg = compile_librtl(lib_dest)
            if ok:
                print("rocm-trace-lite: compiled librtl.so -> %s" % msg)
            else:
                print("rocm-trace-lite: WARNING: skipping librtl.so build (%s)" % msg,
                      file=sys.stderr)
                print("rocm-trace-lite: CLI tools (convert, summary, info) will still work.",
                      file=sys.stderr)
        except Exception as e:
            print("rocm-trace-lite: WARNING: librtl.so build failed: %s" % e,
                  file=sys.stderr)


setup(cmdclass={"build_py": BuildWithLibrtl, **wheel_cmdclass})
