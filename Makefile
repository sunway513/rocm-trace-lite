# rocm-trace-lite — Self-contained GPU kernel profiler
# Dependencies: libhsa-runtime64, libsqlite3
# NO roctracer, NO rocprofiler-sdk

HIP_PATH ?= /opt/rocm
PREFIX ?= /usr/local

CXX ?= g++
CXXFLAGS = -O3 -g -fPIC -std=c++17 \
           -I$(HIP_PATH)/include \
           -I$(HIP_PATH)/include/hsa \
           -D__HIP_PLATFORM_AMD__ -DAMD_INTERNAL_BUILD

LDFLAGS = -shared -rdynamic \
          -L$(HIP_PATH)/lib \
          -Wl,-rpath,$(HIP_PATH)/lib \
          -lhsa-runtime64 -lsqlite3 \
          -ldl -lpthread

SRCDIR = src
SRCS = trace_db.cpp hip_api_intercept.cpp roctx_shim.cpp hsa_intercept.cpp
OBJS = $(addprefix $(SRCDIR)/,$(SRCS:.cpp=.o))
DEPS = $(OBJS:.o=.d)

.DEFAULT_GOAL := all
-include $(DEPS)
TARGET = librtl.so

.PHONY: all clean install test test-cpu test-gpu

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)
	@echo "Built $(TARGET)"
	@echo "  Dependencies: libhsa-runtime64, libsqlite3"
	@echo "  NO roctracer, NO rocprofiler-sdk"
	@deps="$$(ldd $@)" || exit 1; \
	if printf '%s\n' "$$deps" | grep -E "not found|roctracer|rocprofiler-sdk|libamdhip64|libroctx64"; then \
	  echo "ERROR: missing or forbidden native dependency" >&2; exit 1; \
	fi

$(SRCDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

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

tests/trace_regions: tests/trace_regions.hip
	$(HIP_PATH)/bin/hipcc -O2 -o $@ $< -ldl -lpthread

clean:
	rm -f $(OBJS) $(DEPS) $(TARGET) $(GPU_WORKLOAD)

# Non-GPU tests (runs in CI)
test-cpu:
	cd tests && python3 -m pytest -v --tb=short

test: test-cpu

# GPU smoke test (requires ROCm GPU, no PyTorch needed)
test-gpu: $(TARGET) $(GPU_WORKLOAD)
	@echo "=== GPU smoke test ==="
	HSA_TOOLS_LIB=$(CURDIR)/$(TARGET) $(CURDIR)/$(GPU_WORKLOAD) gemm 256 10
	@echo "=== Trace results ==="
	sqlite3 trace.db "SELECT * FROM top LIMIT 10;" 2>/dev/null || true
	sqlite3 trace.db "SELECT count(*) || ' kernel ops' FROM rocpd_op;" 2>/dev/null || true
