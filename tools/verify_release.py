#!/usr/bin/env python3
"""Validate distribution contents and install the wheel outside the checkout."""
import argparse
import hashlib
import json
import os
import platform
import re
from pathlib import Path
import subprocess
import tarfile
import tempfile
import venv
import zipfile


def verify(dist):
    wheels = list(dist.glob("*.whl"))
    sources = list(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise RuntimeError("Expected exactly one wheel and one sdist in dist/")
    wheel, source = wheels[0], sources[0]
    with tarfile.open(source) as archive:
        names = archive.getnames()
        if any(name.endswith((".so", ".o", ".d")) for name in names):
            raise RuntimeError("sdist must contain sources, not native build products")
        for required in ("Makefile", "src/hsa_intercept.cpp", "src/trace_db.h"):
            if not any(name.endswith("/" + required) for name in names):
                raise RuntimeError("Missing sdist source: " + required)
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        libraries = [name for name in names if name.endswith("/lib/librtl.so")]
        if len(libraries) != 1:
            raise RuntimeError("Wheel must contain exactly one native profiler")
        binary = archive.read(libraries[0])
        if binary[:4] != b"\x7fELF":
            raise RuntimeError("Packaged profiler is not an ELF binary")
        expected_hash = hashlib.sha256(binary).hexdigest()
        metadata = archive.read(next(n for n in names if n.endswith(".dist-info/WHEEL"))).decode()
        if "Root-Is-Purelib: false" not in metadata or "Tag: py3-none-linux_x86_64" not in metadata:
            raise RuntimeError("Expected an explicitly platform-specific Linux x86_64 wheel")
    with tempfile.TemporaryDirectory(prefix="rtl-wheel-check-") as directory:
        root = Path(directory)
        venv.create(root / "venv", with_pip=True)
        python = root / "venv/bin/python"
        clean_env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "HSA_TOOLS_LIB", "LD_PRELOAD")}
        subprocess.run([str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)], cwd=root, env=clean_env, check=True)
        probe = '''import ctypes,hashlib,json,sys
from pathlib import Path
import rocm_trace_lite as rtl
p=Path(rtl.get_lib_path()).resolve()
assert Path(sys.prefix).resolve() in p.parents, str(p)
lib=ctypes.CDLL(str(p))
for name in ("OnLoad", "OnUnload", "roctxRangePushA", "roctxRangePop", "roctxMarkA", "roctxRangeStartA", "roctxRangeStop"):
 assert getattr(lib,name)
print(json.dumps(dict(version=rtl.__version__,library=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
'''
        result = subprocess.run([str(python), "-I", "-c", probe], cwd=root, env=clean_env, check=True, text=True, capture_output=True)
        identity = json.loads(result.stdout)
        if identity["sha256"] != expected_hash:
            raise RuntimeError("Installed library differs from the wheel")
        tag = os.environ.get("GITHUB_REF", "")
        if tag.startswith("refs/tags/") and tag != "refs/tags/v" + identity["version"]:
            raise RuntimeError("Release tag and package version differ")
        deps = subprocess.run(["ldd", identity["library"]], env=clean_env, check=True, text=True, capture_output=True).stdout
        for forbidden in ("not found", "roctracer", "rocprofiler-sdk", "libamdhip64", "libroctx64"):
            if forbidden in deps:
                raise RuntimeError("Missing or forbidden dependency: " + forbidden)
        subprocess.run([str(root / "venv/bin/rtl"), "--version"], cwd=root, env=clean_env, check=True)
        identity["dependencies"] = deps
        # A linux_x86_64 tag does not express a glibc compatibility floor.
        versions = subprocess.run(
            ["readelf", "--version-info", identity["library"]],
            check=True, text=True, capture_output=True,
        ).stdout
        glibc_versions = sorted(set(re.findall(r"\bGLIBC_(\d+(?:\.\d+)+)\b", versions)),
                                key=lambda value: tuple(map(int, value.split("."))))
        if not glibc_versions:
            raise RuntimeError("No versioned glibc requirements found in native library")
        identity["native_abi"] = {
            "build_libc": dict(zip(("name", "version"), platform.libc_ver())),
            "direct_glibc_requirements": glibc_versions,
            "minimum_direct_glibc": glibc_versions[-1],
            "os_release": Path("/etc/os-release").read_text(),
            "wheel_tag_encodes_glibc_floor": False,
            "scope": "Direct ELF requirements only; transitive dependencies also need validation",
        }
        # The temporary install is removed; its path records validation provenance.
        (dist / "validation.json").write_text(json.dumps(identity, indent=2) + "\n")
    files = [wheel, source, dist / "validation.json"]
    (dist / "SHA256SUMS").write_text("".join(hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.name + "\n" for p in files))
    print("Validated sdist, platform wheel, isolated installation, CLI and native library")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", type=Path)
    args = parser.parse_args()
    verify(args.dist.resolve())
