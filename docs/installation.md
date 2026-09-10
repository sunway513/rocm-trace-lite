# Installation and compatibility

The 0.4.0rc1 ROCm 10 candidate is under validation and has not been published. Obtain released assets from [GitHub Releases](https://github.com/sunway513/rocm-trace-lite/releases), then follow the requirements for that exact version. The existing v0.3.7 release does not contain this candidate's changes. Do not assume a plain `pip install rocm-trace-lite` retrieves a supported release from PyPI.

## Candidate wheel

Download the candidate wheel, sdist, validation record and `SHA256SUMS` from the same workflow artifact or release. In that directory:

```bash
sha256sum --check SHA256SUMS
python3 -m venv .venv
. .venv/bin/activate
python -m pip install ./rocm_trace_lite-0.4.0rc1-py3-none-linux_x86_64.whl
rtl --version
python -c 'from rocm_trace_lite import get_lib_path; print(get_lib_path())'
```

Checksums detect mismatched or damaged files; obtain the checksum file from the same trusted GitHub release. The wheel includes `librtl.so`; it does not include ROCm, SQLite or PyTorch. Test installation from a directory outside a source checkout so local Python files cannot hide a broken package.

## Source build

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
| OS / wheel | Ubuntu 24.04, Linux x86_64; this is not a manylinux wheel |
| Native CI / build image | `rocm/dev-ubuntu-24.04@sha256:a90cf047f615abe70fbef83c64def0a2d549ef37a39c8ea545430aba4981b374` |
| Runtime | ROCm 10; native CI HIP 7.15.26333 |
| GPU | MI355X, `gfx950`, single GPU and eight GPUs |
| Python | Installation verified on Python 3.12; declared minimum 3.8 has not yet been revalidated for this candidate |
| Shared libraries | HSA runtime, SQLite, C/C++ runtime; no roctracer, rocprofiler-sdk, HIP or libroctx link dependency |

Other OS/runtime/GPU combinations require validation. The ROCm 10 implementation removes the old graph batch-skip workaround and relies on ROCR staging-buffer fix `559d48b1`. Older unpatched runtimes are unsupported; selecting lite mode does not repair the runtime.

The separate vLLM E2E image has ROCm 10 SDK metadata but an overlaid HIP **7.16.26361** library. It is not identical to the native CI image; see [performance](performance.md) for comparison limits.

## Troubleshooting

- **Native library cannot load:** inspect `ldd` on the exact path printed by `get_lib_path()`. Resolve every `not found` entry. For the pinned dev image, set `LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-10.0/lib` before loading. Do not copy a `.so` from an unrelated runtime to bypass an error.
- **Zero GPU records:** confirm the workload actually launches kernels; use `--mode standard`, check ROCm compatibility and the `rtl:` diagnostics. Lite intentionally omits some dispatches. Verify GPU workers inherit the profiler environment before their first HIP/HSA call. A successful application exit alone does not prove successful tracing.
- **Missing workers or final records:** ensure every GPU worker synchronizes and finalizes its trace before merge/export. Keep per-process databases for investigation. Abrupt termination is not a supported guarantee of complete traces.
- **No ROCm/compiler on the analysis machine:** install the matching wheel to use SQLite conversion/summary commands. GPU capture requires the validated ROCm runtime on the machine running the workload.

Continue with the [first trace](quickstart.md).
