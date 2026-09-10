# ROCm 10 SDK + matching PyTorch/AITER/vLLM, published 2026-09-10.
# Pin the manifest: mutable nightly tags are not regression provenance.
ARG BASE_IMAGE=vllm/vllm-openai-rocm@sha256:72e90adf360ccffb231a098a15eadfb808719c029c6cd43f6386095064552b7e
FROM ${BASE_IMAGE}
WORKDIR /opt/rtl
COPY . .
# This image uses the Python-packaged ROCm SDK; HIP_PATH is supplied by it.
RUN make clean && make -j8 \
    && mkdir -p rocm_trace_lite/lib \
    && cp librtl.so rocm_trace_lite/lib/librtl.so \
    && sha256sum librtl.so rocm_trace_lite/lib/librtl.so
ENV PYTHONPATH=/opt/rtl:/opt/rtl/benchmarks/e2e
ENTRYPOINT []
CMD ["python3", "-m", "rocm_trace_lite.cli", "--help"]
