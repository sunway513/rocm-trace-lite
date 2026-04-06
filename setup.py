"""Custom setup to force platform-specific wheel tag.

librtl.so is Linux x86_64 only. Without this, setuptools
produces py3-none-any which is incorrect.
"""
from setuptools import setup
from wheel.bdist_wheel import bdist_wheel


class PlatformWheel(bdist_wheel):
    """Force platform tag to linux_x86_64 (not 'any')."""
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        return "py3", "none", "manylinux_2_28_x86_64"


setup(cmdclass={"bdist_wheel": PlatformWheel})
