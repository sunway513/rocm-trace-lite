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
SRCS = rpd_lite.cpp hip_intercept.cpp roctx_shim.cpp hsa_intercept.cpp
OBJS = $(addprefix $(SRCDIR)/,$(SRCS:.cpp=.o))
TARGET = librtl.so

.PHONY: all clean install test test-cpu test-gpu

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)
	@echo "Built $(TARGET)"
	@echo "  Dependencies: libhsa-runtime64, libsqlite3"
	@echo "  NO roctracer, NO rocprofiler-sdk"
	@ldd $@ | grep -E "roctracer|rocprofiler-sdk" && echo "ERROR: unwanted dependency!" && exit 1 || echo "  Verified: clean dependency chain"

$(SRCDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

install: $(TARGET)
	install -d $(PREFIX)/lib $(PREFIX)/bin
	install -m 755 $(TARGET) $(PREFIX)/lib/
	install -m 755 tools/rtl.sh $(PREFIX)/bin/
	install -m 755 tools/rpd2trace.py $(PREFIX)/bin/
	ldconfig
	@echo "Installed to $(PREFIX)"

clean:
	rm -f $(OBJS) $(TARGET)

# Non-GPU tests (runs in CI)
test-cpu:
	cd tests && python3 -m pytest -v --tb=short

test: test-cpu

# GPU smoke test (requires ROCm GPU)
test-gpu: $(TARGET)
	@echo "=== GPU smoke test ==="
	HSA_TOOLS_LIB=$(CURDIR)/$(TARGET) \
		python3 -c "import torch; x=torch.randn(512,512,device='cuda'); y=x@x; torch.cuda.synchronize(); print('OK')"
	@echo "=== Trace results ==="
	sqlite3 trace.db "SELECT * FROM top LIMIT 10;" 2>/dev/null || true
	sqlite3 trace.db "SELECT count(*) || ' kernel ops' FROM rocpd_op;" 2>/dev/null || true
