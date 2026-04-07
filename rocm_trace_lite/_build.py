"""JIT compilation of librtl.so against the local ROCm installation."""
import os
import shutil
import subprocess
import sys


def find_rocm_path():
    """Detect ROCm installation. Returns path or None."""
    for env in ("ROCM_PATH", "HIP_PATH"):
        p = os.environ.get(env)
        if p and os.path.isdir(os.path.join(p, "include", "hsa")):
            return p
    for candidate in ("/opt/rocm",):
        if os.path.isdir(os.path.join(candidate, "include", "hsa")):
            return candidate
    return None


def can_compile(rocm_path):
    """Check if g++ and required headers/libs exist."""
    try:
        subprocess.run(["g++", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False, "g++ not found"
    hsa_header = os.path.join(rocm_path, "include", "hsa", "hsa.h")
    if not os.path.isfile(hsa_header):
        return False, "HSA headers not found at %s" % hsa_header
    return True, ""


def find_source_dir():
    """Find the C++ source directory (src/ relative to package root)."""
    # When installed from source or editable: ../src/ relative to this file
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(pkg_dir)
    src = os.path.join(root, "src")
    if os.path.isfile(os.path.join(src, "hsa_intercept.cpp")):
        return root, src
    return None, None


def compile_librtl(output_dir, rocm_path=None):
    """Compile librtl.so into output_dir. Returns (success, message)."""
    if rocm_path is None:
        rocm_path = find_rocm_path()
    if rocm_path is None:
        return False, "ROCm not found (set ROCM_PATH or install ROCm)"

    ok, reason = can_compile(rocm_path)
    if not ok:
        return False, reason

    project_root, src_dir = find_source_dir()
    if src_dir is None:
        return False, "C++ source files not found (need src/hsa_intercept.cpp)"

    os.makedirs(output_dir, exist_ok=True)

    env = os.environ.copy()
    env["HIP_PATH"] = rocm_path

    try:
        subprocess.check_call(
            ["make", "-j%d" % (os.cpu_count() or 1), "librtl.so"],
            cwd=project_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        built = os.path.join(project_root, "librtl.so")
        if not os.path.isfile(built):
            return False, "make succeeded but librtl.so not found"
        dest = os.path.join(output_dir, "librtl.so")
        shutil.copy2(built, dest)
        return True, dest
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace")[:500] if e.stderr else ""
        return False, "Compilation failed (exit %d): %s" % (e.returncode, stderr)
