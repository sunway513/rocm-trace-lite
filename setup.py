"""Build a platform wheel containing a freshly compiled native profiler."""
import os
import sys

from setuptools import setup
from setuptools.command.build_py import build_py

try:
    try:
        from setuptools.command.bdist_wheel import bdist_wheel
    except ImportError:  # setuptools before 70.1
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
    """Build the native library from the same source as the Python package."""

    def run(self):
        super().run()
        lib_dest = os.path.join(self.build_lib, "rocm_trace_lite", "lib")
        # A copied package-local .so may belong to another checkout/runtime.
        # Always rebuild; a source install must not silently omit the profiler.
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rocm_trace_lite"))
        try:
            from _build import compile_librtl
            ok, message = compile_librtl(lib_dest, force=True)
        finally:
            sys.path.pop(0)
        if not ok:
            raise RuntimeError(
                "Cannot build the native profiler: %s. Install ROCm HSA headers, "
                "g++, make and libsqlite3-dev, or install a validated release wheel."
                % message
            )
        print("rocm-trace-lite: compiled native library -> %s" % message)


setup(cmdclass={"build_py": BuildWithLibrtl, **wheel_cmdclass})
