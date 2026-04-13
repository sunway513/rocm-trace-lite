Run E2E serving benchmarks on remote GPU nodes. Handles pre-flight, server launch, benchmark execution, monitoring, and post-flight cleanup.

MANDATORY: Use this skill for ANY work involving remote GPU servers. No exception.

## Arguments
$ARGUMENTS — model name / benchmark description / specific configs

## Pre-flight Checklist (MANDATORY)

### 1. Verify GPU Cleanliness
Check KFD processes (actual GPU users), not just container status:
```bash
ssh $NODE 'for d in /sys/class/kfd/kfd/proc/*/; do pid=$(basename $d); echo "PID=$pid CMD=$(cat /proc/$pid/cmdline 2>/dev/null | tr "\0" " " | head -c 200)"; done'
```

### 2. Docker Image — Pull Fresh
NEVER trust existing images. Always pull before benchmark:
```bash
ssh $NODE 'docker pull $IMAGE && docker inspect $IMAGE --format "{{.Created}}"'
```

### 3. Check ATOM Server Args
Different ATOM versions support different args. ALWAYS check before scripting:
```bash
# Try module path first, fall back to script path
ssh $NODE 'docker run --rm $IMAGE python3 -m atom.entrypoints.openai_server --help 2>&1 | head -30'
# If that fails, try: python3 /app/ATOM/atom/entrypoints/openai_server.py --help
```

### 4. Verify Benchmark Module Path
```bash
ssh $NODE 'docker run --rm $IMAGE python3 -m atom.benchmarks.benchmark_serving --help 2>&1 | head -5'
```

### 5. Pre-launch Port Check
```bash
if ss -tlnp | grep -q ":${PORT} "; then echo "ERROR: port in use"; exit 1; fi
```

## RTL Wheel Verification (MANDATORY after install)

Stale .so in wheel caused phantom -32.7% overhead. Always verify:
```bash
INSTALLED=$(python3 -c "from rocm_trace_lite import get_lib_path; print(get_lib_path())")
md5sum $INSTALLED
nm -D $INSTALLED | grep -c hip_profiler  # 0=main, 2=hip branch
ls -la $INSTALLED  # check timestamp
```

## RTL Diagnostic Validation (MANDATORY after profiled run)

Capture intercept/inject/skip stats. Catches broken lite mode immediately:
```bash
# Expected for lite mode with CUDAGraph:
#   drop (not kernel): M  ← should be >50% (lite skip)
#   drop (batch skip): K  ← CUDAGraph replay skips
# RED FLAG: if inject ≈ intercept (near 100%), lite is NOT skipping
```

## Warmup Before Benchmark (MANDATORY)

Send warmup prompts BEFORE timed run. torch.compile/JIT compiles on first invocation:
```bash
python3 -m atom.benchmarks.benchmark_serving \
    --backend openai --endpoint /v1/completions --model "$MODEL" \
    --dataset-name random --random-input-len $ISL --random-output-len $OSL \
    --num-prompts 5 --max-concurrency $CONC --ignore-eos > /dev/null 2>&1
```

## Server Launch Pattern

Logs MUST go to docker stdout (use tee, NOT redirect to file only):
```bash
python3 -m atom.entrypoints.openai_server \
    --model "$MODEL" -tp $TP --gpu-memory-utilization 0.9 --port $PORT \
    2>&1 | tee "$OUTDIR/${label}_server.log" &
```

## RTL Profiler Env Vars (set on SERVER process)
```bash
# lite mode (zero overhead with CUDAGraph)
HSA_TOOLS_LIB="$LIBRTL" RTL_OUTPUT="$OUTDIR/trace_%p.db" RTL_MODE=lite

# hip mode (requires CLR profiler-enabled libamdhip64.so)
HSA_TOOLS_LIB="$LIBRTL" RTL_MODE=hip RTL_OUTPUT="$OUTDIR/trace_%p.db" GPU_CLR_PROFILE_OUTPUT=/dev/null

```

## Process Cleanup (CRITICAL)

kill $PID only kills main process. ATOM spawns 8+ workers.
WARNING: `pkill -f openai_server` inside docker exec can kill the rtl trace wrapper too,
causing container restart and trace data loss. Use targeted PID kill instead:
```bash
# Kill server by PID, not pattern — preserves rtl trace wrapper
kill $(pgrep -f "openai_server.py" | head -1) 2>/dev/null
# Wait for RTL to finalize trace DB
wait $RTL_PID 2>/dev/null
sleep 5
# Then clean up remaining workers
pkill -9 -f "multiprocessing.spawn"
pkill -9 -f "compile_worker"
# Verify GPU release
for i in $(seq 1 60); do
    pids=$(cat /sys/class/kfd/kfd/proc/*/pasid 2>/dev/null | wc -l)
    [ "$pids" -eq 0 ] && break; sleep 1
done
```

## Trust Boundaries (HARD RULES)

1. NEVER trust binaries on remote systems — always build/pull fresh
2. NEVER trust Docker images already on nodes — always pull
3. NEVER reuse custom-tagged images from previous sessions
4. ALWAYS verify after pull — check creation date, key binaries, versions

## Known Issues

- RTL lite: ~0% overhead on MoE with CUDAGraph, ~5-6% on enforce-eager
- RTL hip mode: GPU_CLR_PROFILE_OUTPUT + CUDAGraph = crash (CLR dispatch wrapper incompatibility)
- setup.py packaging: always `make` before `bdist_wheel` — repo-root librtl.so takes priority
