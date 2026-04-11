ROCM_PATH ?= /opt/rocm
HIPCC      ?= $(ROCM_PATH)/bin/hipcc
CXX        ?= g++

HSA_INC    = $(ROCM_PATH)/include
HSA_LIB    = $(ROCM_PATH)/lib

# Number of kernels in the graph (256 is threshold for the bug)
NUM_KERNELS ?= 256

.PHONY: all clean test test-plain test-intercept

all: librepro_passthrough.so repro_hipgraph_stress

librepro_passthrough.so: repro_passthrough_lib.cpp
	$(CXX) -shared -fPIC -O2 -DAMD_INTERNAL_BUILD -o $@ $< \
		-I$(HSA_INC) -L$(HSA_LIB) -lhsa-runtime64

repro_hipgraph_stress: repro_hipgraph_stress.hip
	$(HIPCC) -O2 -DNUM_KERNELS=$(NUM_KERNELS) -o $@ $<

# Run without interception (should PASS)
test-plain: repro_hipgraph_stress
	@echo "========================================"
	@echo "Running WITHOUT interceptible queues..."
	@echo "========================================"
	./repro_hipgraph_stress

# Run with interception (triggers SEGFAULT on ROCm <= 7.2)
test-intercept: all
	@echo "========================================"
	@echo "Running WITH interceptible queues..."
	@echo "========================================"
	HSA_TOOLS_LIB=./librepro_passthrough.so ./repro_hipgraph_stress

# Run both
test: test-plain test-intercept

clean:
	rm -f librepro_passthrough.so repro_hipgraph_stress
