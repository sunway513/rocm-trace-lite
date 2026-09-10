# Installation and compatibility

The 0.4.0rc1 ROCm 10 candidate is under validation and has not been published. Obtain released assets from [GitHub Releases](https://github.com/sunway513/rocm-trace-lite/releases), then follow the requirements for that exact version. The existing v0.3.7 release does not contain this candidate's changes. Do not assume a plain `pip install rocm-trace-lite` retrieves a supported release from PyPI.

## Candidate wheel

The CI wheel is built and validated on **Ubuntu 24.04 with glibc 2.39**.
Its native library requires `GLIBC_2.38`; it cannot load on Ubuntu 22.04
(glibc 2.35), including the pinned vLLM serving image. The `linux_x86_64`
filename does not encode this requirement, so pip can accept an incompatible
wheel. On Ubuntu 22.04, build the same-version sdist inside the target
ROCm environment; that route still requires its own validation.

Download the candidate wheel, sdist, validation record and `SHA256SUMS` from the same workflow artifact or release. In that directory:

```bash
sha256sum --check SHA256SUMS
# Check inside the target environment/container, not on its host.
python3 -c 'import platform; n,v=platform.libc_ver(); assert n=="glibc" and tuple(map(int,v.split("."))) >= (2,39), "Use the matching sdist: this CI wheel targets Ubuntu 24.04 / glibc 2.39"'
python3 -m venv .venv
. .venv/bin/activate
python -m pip install ./rocm_trace_lite-0.4.0rc1-py3-none-linux_x86_64.whl
rtl --version
python -c 'import ctypes; from rocm_trace_lite import get_lib_path; p=get_lib_path(); ctypes.CDLL(p); print(p)'
```

Checksums detect mismatched or damaged files; obtain the checksum file from the same trusted GitHub release. The wheel includes `librtl.so`; it does not include ROCm, SQLite or PyTorch. Test installation from a directory outside a source checkout so local Python files cannot hide a broken package.

## Target-environment source build

For an Ubuntu 22.04 serving container, use the **same candidate sdist** and
compile inside that container with its ROCm/HSA headers and compiler. Do not
reuse the Ubuntu 24.04 wheel or replace the container libc. For example, after
installing build prerequisites in the target environment:

```bash
python -m pip wheel --no-deps ./rocm_trace_lite-0.4.0rc1.tar.gz -w target-wheels
python -m pip install target-wheels/*.whl
python -c 'import ctypes; from rocm_trace_lite import get_lib_path; ctypes.CDLL(get_lib_path())'
```

In the pinned TheRock-based vLLM image only, the development SDK is at
`/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel`; set `ROCM_PATH` to
that directory before building. Other environments must use their actual ROCm
header/library root. SDK 10 metadata does not identify the overlaid HIP runtime:
the pinned serving image actually loads HIP 7.16.26361, as described below.

This avoids importing a newer host libc requirement. It is a build route, not a
claim that Ubuntu 22.04 GPU capture has passed release validation. Keep the
source hash and target-built native library hash with the validation results.

## Source development

Use a checkout of the candidate branch inside the validated ROCm 10 development environment:

```bash
sudo apt-get install g++ make libsqlite3-dev python3-venv
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
```

Set `ROCM_PATH` if HSA headers are outside `/opt/rocm`. A source build always compiles the profiler from source; it fails if compilation or native dependency validation fails. It does not reuse an old packaged `.so`. For maintainers, `python -m build` builds an sdist and then a wheel from that sdist; `python tools/verify_release.py dist` verifies the contents and an isolated installation.

## Validated environment

| Component | Candidate validation |
|---|---|
| OS / wheel | Ubuntu 24.04 / glibc 2.39 build and validation; current native ELF requires GLIBC_2.38. Linux x86_64 tag does not enforce libc compatibility; not a manylinux wheel |
| Native CI / build image | `rocm/dev-ubuntu-24.04@sha256:a90cf047f615abe70fbef83c64def0a2d549ef37a39c8ea545430aba4981b374` |
| Runtime | ROCm 10; native CI HIP 7.15.26333 |
| GPU | MI355X, `gfx950`, single GPU and eight GPUs |
| Python | Installation verified on Python 3.12; declared minimum 3.8 has not yet been revalidated for this candidate |
| Shared libraries | HSA runtime, SQLite, C/C++ runtime; no roctracer, rocprofiler-sdk, HIP or libroctx link dependency |

Other OS/runtime/GPU combinations require validation. The ROCm 10 implementation removes the old graph batch-skip workaround and relies on ROCR staging-buffer fix `559d48b1`. Older unpatched runtimes are unsupported; selecting lite mode does not repair the runtime.

The separate vLLM E2E image uses Ubuntu 22.04 / glibc 2.35 and requires a target-built profiler; the CI wheel above cannot load there. It has ROCm 10 SDK metadata but an overlaid HIP **7.16.26361** library. It is not identical to the native CI image; see [performance](performance.md) for comparison limits.

## Troubleshooting

- **`GLIBC_2.38 not found`:** the Ubuntu 24.04 CI wheel was installed into an older libc environment. Build the matching sdist inside the target environment, or use the validated Ubuntu 24.04 environment. Pip installation success is not a native compatibility check.
- **Native library cannot load:** inspect `ldd` on the exact path printed by `get_lib_path()`. Resolve every `not found` entry. For the pinned dev image, set `LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-10.0/lib` before loading. Do not copy a `.so` from an unrelated runtime to bypass an error.
- **Zero GPU records:** confirm the workload actually launches kernels; use `--mode standard`, check ROCm compatibility and the `rtl:` diagnostics. Lite intentionally omits some dispatches. Verify GPU workers inherit the profiler environment before their first HIP/HSA call. A successful application exit alone does not prove successful tracing.
- **Missing workers or final records:** ensure every GPU worker synchronizes and finalizes its trace before merge/export. Keep per-process databases for investigation. Abrupt termination is not a supported guarantee of complete traces.
- **No ROCm/compiler on the analysis machine:** install the matching wheel to use SQLite conversion/summary commands. GPU capture requires the validated ROCm runtime on the machine running the workload.

Continue with the [first trace](quickstart.md).
