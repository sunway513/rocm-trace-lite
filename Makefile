# rocm-trace-lite — Self-contained GPU kernel profiler
#
# Backends:
#   BACKEND=hsa             (default) HSA_TOOLS_LIB/OnLoad interception
#   BACKEND=rocprofiler_sdk           rocprofiler-sdk callback tracing

HIP_PATH ?= /opt/rocm
PREFIX ?= /usr/local
BACKEND ?= hsa

CXX ?= g++
CXXFLAGS = -O3 -g -fPIC -std=c++17 \
           -I$(HIP_PATH)/include \
           -I$(HIP_PATH)/include/hsa \
           -D__HIP_PLATFORM_AMD__ -DAMD_INTERNAL_BUILD

LDFLAGS_COMMON = -shared -rdynamic \
          -L$(HIP_PATH)/lib \
          -Wl,-rpath,$(HIP_PATH)/lib \
          -lsqlite3 \
          -ldl -lpthread

SRCDIR = src

# Backend-specific source and link flags
ifeq ($(BACKEND),rocprofiler_sdk)
  BACKEND_SRC = rocprofiler_backend.cpp
  # Don't link librocprofiler-sdk — symbols are resolved at runtime by
  # rocprofiler-register (which already has them loaded). Avoids pulling
  # libamd_comgr → libLLVM into our dep chain (LLVM conflict on ROCm 10).
  LDFLAGS = $(LDFLAGS_COMMON)
  CXXFLAGS += -DRTL_BACKEND_ROCPROFILER_SDK
else
  BACKEND_SRC = hsa_intercept.cpp
  LDFLAGS = $(LDFLAGS_COMMON) -lhsa-runtime64
  CXXFLAGS += -DRTL_BACKEND_HSA
endif

COMMON_SRCS = trace_db.cpp hip_api_intercept.cpp roctx_shim.cpp
SRCS = $(COMMON_SRCS) $(BACKEND_SRC)
OBJS = $(addprefix $(SRCDIR)/,$(SRCS:.cpp=.o))
TARGET = librtl.so

.PHONY: all clean install test test-cpu test-gpu rocp

all: $(TARGET)

# Convenience target: make rocp
rocp:
	$(MAKE) BACKEND=rocprofiler_sdk

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)
	@echo "Built $(TARGET) [backend=$(BACKEND)]"
ifeq ($(BACKEND),rocprofiler_sdk)
	@echo "  Dependencies: librocprofiler-sdk, libsqlite3"
	@ldd $@ | grep -E "roctracer|libamdhip64" && echo "ERROR: unwanted dependency!" && exit 1 || echo "  Verified: clean dependency chain"
else
	@echo "  Dependencies: libhsa-runtime64, libsqlite3"
	@ldd $@ | grep -E "roctracer|rocprofiler-sdk|libamdhip64" && echo "ERROR: unwanted dependency!" && exit 1 || echo "  Verified: clean dependency chain (no roctracer, no rocprofiler-sdk, no libamdhip64)"
endif

$(SRCDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

install: $(TARGET)
	install -d $(PREFIX)/lib $(PREFIX)/bin
	install -m 755 $(TARGET) $(PREFIX)/lib/
	install -m 755 tools/rtl.sh $(PREFIX)/bin/
	install -m 755 tools/rpd2trace.py $(PREFIX)/bin/
	ldconfig
	@echo "Installed to $(PREFIX)"

GPU_WORKLOAD = tests/gpu_workload

$(GPU_WORKLOAD): tests/gpu_workload.hip
	hipcc -O2 -o $@ $< -lpthread

clean:
	rm -f $(addprefix $(SRCDIR)/,$(COMMON_SRCS:.cpp=.o)) \
	      $(SRCDIR)/hsa_intercept.o \
	      $(SRCDIR)/rocprofiler_backend.o \
	      $(TARGET) $(GPU_WORKLOAD)

# Non-GPU tests (runs in CI)
test-cpu:
	cd tests && python3 -m pytest -v --tb=short

test: test-cpu

# GPU smoke test (requires ROCm GPU, no PyTorch needed)
test-gpu: $(TARGET) $(GPU_WORKLOAD)
	@echo "=== GPU smoke test ==="
ifeq ($(BACKEND),rocprofiler_sdk)
	LD_PRELOAD=$(CURDIR)/$(TARGET) RTL_OUTPUT=trace_%p.db $(CURDIR)/$(GPU_WORKLOAD) gemm 256 10
else
	HSA_TOOLS_LIB=$(CURDIR)/$(TARGET) $(CURDIR)/$(GPU_WORKLOAD) gemm 256 10
endif
	@echo "=== Trace results ==="
	sqlite3 trace.db "SELECT * FROM top LIMIT 10;" 2>/dev/null || true
	sqlite3 trace.db "SELECT count(*) || ' kernel ops' FROM rocpd_op;" 2>/dev/null || true
