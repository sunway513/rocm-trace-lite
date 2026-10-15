/*
 * mock_hsa_table.cpp — GPU-free conformance harness for RTL's HSA table hooks.
 *
 * Loads librtl.so's OnLoad() with a synthetic HsaApiTable whose function
 * pointers are recording stubs, then drives the installed hooks directly.
 * This exercises interception logic (descriptor parsing, fallback paths, ABI
 * version guards) without a GPU, a real HSA runtime, or any kernel dispatch.
 *
 * Output is one line per check:
 *     CHECK <name> <PASS|FAIL|SKIP> <detail>
 * The process always exits 0 unless it crashes; test_queue_create_intercept.py
 * turns each line into an individual pytest result.
 *
 * Build: see the `test-mock-hsa` target in the top-level Makefile.
 */
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstddef>
#include <unistd.h>
#include <cstdarg>
#include <sys/mman.h>
#include <sys/wait.h>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_api_trace.h>

extern "C" bool OnLoad(void*, uint64_t, uint64_t, const char* const*);

// ---------------------------------------------------------------- reporting
static int g_fails = 0;

// Checks whose expected behaviour is understood but not implemented yet. They
// report XFAIL instead of FAIL so a known gap does not read the same as a
// regression. Remove the entry once the gap is closed; an unexpected PASS is
// reported normally and will then be held to.
static const struct { const char* name; const char* why; } kKnownOpen[] = {
    {"nonzero_flags_not_dropped",
     "ring-buffer placement cannot be expressed through hsa_amd_queue_intercept_create; "
     "needs gfx94x/gfx125x hardware to decide, see issue #124 item 3"},
};

static const char* known_open(const char* name) {
    for (const auto& k : kKnownOpen)
        if (strcmp(k.name, name) == 0) return k.why;
    return nullptr;
}

static void check(const char* name, bool ok, const char* fmt, ...) {
    char detail[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(detail, sizeof(detail), fmt, ap);
    va_end(ap);
    const char* open_reason = ok ? nullptr : known_open(name);
    if (open_reason) {
        printf("CHECK %s XFAIL %s -- %s\n", name, detail, open_reason);
        return;
    }
    printf("CHECK %s %s %s\n", name, ok ? "PASS" : "FAIL", detail);
    if (!ok) g_fails++;
}

static void skip(const char* name, const char* why) {
    printf("CHECK %s SKIP %s\n", name, why);
}

// ------------------------------------------------------------ recording stubs
struct Rec {
    int   intercept_create = 0;
    uint32_t size = 0, type = 0, priv = 0, grp = 0;
    int   cu_set_mask = 0;
    uint32_t cu_mask_count = 0;
    int   set_priority = 0;
    int   last_priority = -1;
    int   prof_enable = 0;
    int   intercept_register = 0;
    int   passthrough = 0;
    bool  fail_intercept_create = false;
};
static Rec R;
static void reset() { bool f = R.fail_intercept_create; R = Rec(); R.fail_intercept_create = f; }

static hsa_queue_t g_queues[32];
static int         g_nq = 0;
static uint64_t    g_ring[1024];

static hsa_queue_t* fresh_queue(uint32_t size) {
    hsa_queue_t* q = &g_queues[g_nq++ % 32];
    memset(q, 0, sizeof(*q));
    q->base_address = g_ring;
    q->size = size;
    return q;
}

static hsa_status_t s_intercept_create(hsa_agent_t, uint32_t size, hsa_queue_type32_t type,
                                       void (*)(hsa_status_t, hsa_queue_t*, void*), void*,
                                       uint32_t priv, uint32_t grp, hsa_queue_t** q) {
    R.intercept_create++;
    R.size = size; R.type = (uint32_t)type; R.priv = priv; R.grp = grp;
    if (R.fail_intercept_create) { *q = nullptr; return HSA_STATUS_ERROR_OUT_OF_RESOURCES; }
    *q = fresh_queue(size);
    return HSA_STATUS_SUCCESS;
}
static hsa_status_t s_intercept_register(hsa_queue_t*, hsa_amd_queue_intercept_handler, void*) {
    R.intercept_register++; return HSA_STATUS_SUCCESS;
}
static hsa_status_t s_prof_enable(hsa_queue_t*, int) { R.prof_enable++; return HSA_STATUS_SUCCESS; }
static hsa_status_t s_async_copy_enable(bool) { return HSA_STATUS_SUCCESS; }
static hsa_status_t s_set_priority(hsa_queue_t*, hsa_amd_queue_priority_t p) {
    R.set_priority++; R.last_priority = (int)p; return HSA_STATUS_SUCCESS;
}
static hsa_status_t s_cu_set_mask(const hsa_queue_t*, uint32_t n, const uint32_t*) {
    R.cu_set_mask++; R.cu_mask_count = n; return HSA_STATUS_SUCCESS;
}
static hsa_status_t s_prof_dispatch_time(hsa_agent_t, hsa_signal_t,
                                         hsa_amd_profiling_dispatch_time_t* t) {
    if (t) { t->start = 0; t->end = 0; }
    return HSA_STATUS_SUCCESS;
}
static hsa_status_t s_core_queue_create(hsa_agent_t, uint32_t size, hsa_queue_type32_t,
                                        void (*)(hsa_status_t, hsa_queue_t*, void*), void*,
                                        uint32_t, uint32_t, hsa_queue_t** q) {
    *q = fresh_queue(size); return HSA_STATUS_SUCCESS;
}
static uint64_t g_sig = 1;
static hsa_status_t s_signal_create(hsa_signal_value_t, uint32_t, const hsa_agent_t*, hsa_signal_t* s) {
    s->handle = g_sig++; return HSA_STATUS_SUCCESS;
}
static hsa_status_t s_signal_destroy(hsa_signal_t) { return HSA_STATUS_SUCCESS; }
static void s_signal_store(hsa_signal_t, hsa_signal_value_t) {}
static void s_signal_sub(hsa_signal_t, hsa_signal_value_t) {}
static hsa_signal_value_t s_signal_wait(hsa_signal_t, hsa_signal_condition_t, hsa_signal_value_t,
                                        uint64_t, hsa_wait_state_t) { return 0; }
static hsa_status_t s_exec_freeze(hsa_executable_t, const char*) { return HSA_STATUS_SUCCESS; }

#ifdef HSA_AMD_QUEUE_CREATE_DESC_VERSION
static hsa_status_t s_orig_amd_queue_create(hsa_agent_t, hsa_amd_queue_create_desc_t* d, uint32_t n) {
    R.passthrough++;
    for (uint32_t i = 0; i < n; i++) d[i].queue = fresh_queue(d[i].queue_size_bytes / 64);
    return HSA_STATUS_SUCCESS;
}
#endif

// ------------------------------------- interposed direct libhsa-runtime64 syms
static hsa_agent_t g_agent = { 0xA6E47 };
static bool        g_publish_agent = true;

extern "C" hsa_status_t hsa_iterate_agents(hsa_status_t (*cb)(hsa_agent_t, void*), void* data) {
    return g_publish_agent ? cb(g_agent, data) : HSA_STATUS_SUCCESS;
}
extern "C" hsa_status_t hsa_agent_get_info(hsa_agent_t, hsa_agent_info_t attr, void* v) {
    if (attr == HSA_AGENT_INFO_DEVICE) *(hsa_device_type_t*)v = HSA_DEVICE_TYPE_GPU;
    return HSA_STATUS_SUCCESS;
}
extern "C" hsa_status_t hsa_queue_destroy(hsa_queue_t*) { return HSA_STATUS_SUCCESS; }
extern "C" hsa_status_t hsa_executable_iterate_symbols(
    hsa_executable_t, hsa_status_t (*)(hsa_executable_t, hsa_executable_symbol_t, void*), void*) {
    return HSA_STATUS_SUCCESS;
}
extern "C" hsa_status_t hsa_executable_symbol_get_info(
    hsa_executable_symbol_t, hsa_executable_symbol_info_t, void*) { return HSA_STATUS_SUCCESS; }

// ------------------------------------------------------------- table building
static CoreApiTable g_core;
static HsaApiTable  g_table;

static void fill_core() {
    memset(&g_core, 0, sizeof(g_core));
    g_core.version.minor_id = sizeof(CoreApiTable);
    g_core.hsa_queue_create_fn              = s_core_queue_create;
    g_core.hsa_executable_freeze_fn         = s_exec_freeze;
    g_core.hsa_signal_create_fn             = s_signal_create;
    g_core.hsa_signal_destroy_fn            = s_signal_destroy;
    g_core.hsa_signal_store_relaxed_fn      = s_signal_store;
    g_core.hsa_signal_subtract_screlease_fn = s_signal_sub;
    g_core.hsa_signal_wait_scacquire_fn     = s_signal_wait;
}

static void fill_ext(AmdExtTable* ext, size_t published_size) {
    memset(ext, 0, published_size);
    ext->version.minor_id = (uint32_t)published_size;
    ext->hsa_amd_queue_intercept_create_fn          = s_intercept_create;
    ext->hsa_amd_queue_intercept_register_fn        = s_intercept_register;
    ext->hsa_amd_profiling_set_profiler_enabled_fn  = s_prof_enable;
    ext->hsa_amd_profiling_async_copy_enable_fn     = s_async_copy_enable;
    ext->hsa_amd_profiling_get_dispatch_time_fn     = s_prof_dispatch_time;
    ext->hsa_amd_queue_set_priority_fn              = s_set_priority;
    ext->hsa_amd_queue_cu_set_mask_fn               = s_cu_set_mask;
#ifdef HSA_AMD_QUEUE_CREATE_DESC_VERSION
    // Only write the slot if the simulated runtime actually published it —
    // otherwise the harness itself would be the one going out of bounds.
    if (published_size >= offsetof(AmdExtTable, hsa_amd_queue_create_fn) + sizeof(void*))
        ext->hsa_amd_queue_create_fn = s_orig_amd_queue_create;
#endif
}

#ifdef HSA_AMD_QUEUE_CREATE_DESC_VERSION
static uint32_t g_cu_mask[2] = { 0xFFFFFFFFu, 0x0000FFFFu };

static hsa_amd_queue_create_desc_t desc(uint16_t version, uint16_t flags, uint8_t engine,
                                        uint32_t bytes, uint32_t priv, bool cu_mask,
                                        hsa_amd_queue_priority_t prio) {
    hsa_amd_queue_create_desc_t d;
    memset(&d, 0, sizeof(d));
    d.version          = version;
    d.flags            = flags;
    d.engine_type      = engine;
    d.queue_size_bytes = bytes;
    d.priority         = prio;
    d.queue            = (hsa_queue_t*)0xBADC0FFEE0DDF00Dull;   // must be cleared by the callee
    if (engine == HSA_AMD_QUEUE_ENGINE_COMPUTE) {
        d.engine.compute.type                 = HSA_QUEUE_TYPE_MULTI;
        d.engine.compute.private_segment_size = priv;
        if (cu_mask) {
            d.engine.compute.cu_mask       = g_cu_mask;
            d.engine.compute.cu_mask_count = 64;
        }
    }
    return d;
}
#endif

// --------------------------------------------------------------------- main
//
// OnLoad() may only be called once per process: it assigns to g_worker, and a
// second assignment onto a joinable std::thread terminates. Each phase
// therefore runs in its own forked child; the parent only collects verdicts.

#ifdef HSA_AMD_QUEUE_CREATE_DESC_VERSION
static const uintptr_t kSentinel = 0x5E5E5E5E5E5E5E5Eull;

// Phase 1a: runtime published a table that stops short of the
// hsa_amd_queue_create_fn slot. The slot is backed by memory and poisoned, so
// a stray access is observable instead of fatal.
static void phase_abi_slot() {
    const size_t published = offsetof(AmdExtTable, hsa_amd_queue_create_fn);
    static AmdExtTable ext;
    fill_core();
    fill_ext(&ext, sizeof(AmdExtTable));
    ext.version.minor_id = (uint32_t)published;            // older runtime
    ext.hsa_amd_queue_create_fn = (decltype(ext.hsa_amd_queue_create_fn))kSentinel;
    g_publish_agent = false;                               // load path only

    memset(&g_table, 0, sizeof(g_table));
    g_table.version.minor_id = sizeof(HsaApiTable);
    g_table.core_    = &g_core;
    g_table.amd_ext_ = &ext;

    OnLoad(&g_table, 70200, 0, nullptr);

    bool untouched = (uintptr_t)ext.hsa_amd_queue_create_fn == kSentinel;
    check("abi_hook_install_guarded", untouched,
          "amd_ext minor_id=%zu, offsetof(hsa_amd_queue_create_fn)=%zu, "
          "sizeof(AmdExtTable)=%zu -> slot %s",
          published, offsetof(AmdExtTable, hsa_amd_queue_create_fn),
          sizeof(AmdExtTable), untouched ? "untouched" : "OVERWRITTEN");
}

// Phase 1b: same situation, but the bytes past minor_id genuinely are not
// ours. A guard page turns any overread into a signal the parent can see.
static void phase_abi_guard_page() {
    const size_t published = offsetof(AmdExtTable, hsa_amd_queue_create_fn);
    long   pg   = sysconf(_SC_PAGESIZE);
    size_t body = ((published + pg - 1) / pg) * pg;
    char*  base = (char*)mmap(nullptr, body + pg, PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) _exit(120);
    mprotect(base + body, pg, PROT_NONE);
    AmdExtTable* ext = (AmdExtTable*)(base + body - published);
    fill_core();
    fill_ext(ext, published);
    g_publish_agent = false;

    memset(&g_table, 0, sizeof(g_table));
    g_table.version.minor_id = sizeof(HsaApiTable);
    g_table.core_    = &g_core;
    g_table.amd_ext_ = ext;

    OnLoad(&g_table, 70200, 0, nullptr);
}

// Phase 2: a full, current table. The hook must install and behave.
static void phase_descriptors() {
    static AmdExtTable ext;
    fill_core();
    fill_ext(&ext, sizeof(AmdExtTable));
    g_publish_agent = true;

    memset(&g_table, 0, sizeof(g_table));
    g_table.version.minor_id = sizeof(HsaApiTable);
    g_table.core_    = &g_core;
    g_table.amd_ext_ = &ext;

    OnLoad(&g_table, 100100, 0, nullptr);

    auto hook = ext.hsa_amd_queue_create_fn;
    if (hook == s_orig_amd_queue_create) {
        skip("hook_installed", "RTL build does not intercept hsa_amd_queue_create");
        return;
    }
    check("hook_installed", true, "amd_ext_->hsa_amd_queue_create_fn replaced");

    // queue_size_bytes is bytes; hsa_amd_queue_intercept_create takes packets
    reset();
    { auto d = desc(1, 0, HSA_AMD_QUEUE_ENGINE_COMPUTE, 4096, 0, false, HSA_AMD_QUEUE_PRIORITY_NORMAL);
      hook(g_agent, &d, 1);
      check("queue_size_in_packets", R.size == 4096 / 64,
            "queue_size_bytes=4096 -> intercept_create size=%u (want 64)", R.size); }

    // CU mask must reach the queue; the runtime applies it at creation
    reset();
    { auto d = desc(1, 0, HSA_AMD_QUEUE_ENGINE_COMPUTE, 4096, 0, true, HSA_AMD_QUEUE_PRIORITY_NORMAL);
      hook(g_agent, &d, 1);
      check("cu_mask_forwarded", R.cu_set_mask == 1 && R.cu_mask_count == 64,
            "cu_mask_count=64 requested -> cu_set_mask calls=%d (count=%u)",
            R.cu_set_mask, R.cu_mask_count); }

    // ring-buffer placement flags cannot be honoured by intercept_create
    reset();
    { auto d = desc(1, HSA_AMD_QUEUE_CREATE_DEVICE_MEM_RING_BUF, HSA_AMD_QUEUE_ENGINE_COMPUTE,
                    4096, 0, false, HSA_AMD_QUEUE_PRIORITY_NORMAL);
      hook(g_agent, &d, 1);
      check("nonzero_flags_not_dropped", R.passthrough == 1 && R.intercept_create == 0,
            "flags=DEVICE_MEM_RING_BUF -> passthrough=%d intercept_create=%d "
            "(want passthrough; intercept cannot place the ring in device memory)",
            R.passthrough, R.intercept_create); }

    // an unknown descriptor version must not be reinterpreted as v1
    reset();
    { auto d = desc(HSA_AMD_QUEUE_CREATE_DESC_VERSION + 1, 0, HSA_AMD_QUEUE_ENGINE_COMPUTE,
                    4096, 0, false, HSA_AMD_QUEUE_PRIORITY_NORMAL);
      auto st = hook(g_agent, &d, 1);
      check("unknown_desc_version_rejected",
            R.intercept_create == 0 && (R.passthrough == 1 || st != HSA_STATUS_SUCCESS),
            "version=%d -> intercept_create=%d passthrough=%d ret=0x%x (want passthrough or error)",
            HSA_AMD_QUEUE_CREATE_DESC_VERSION + 1, R.intercept_create, R.passthrough,
            (unsigned)st); }

    // the runtime rejects queue_size_bytes == 0; the hook must not invent one
    reset();
    { auto d = desc(1, 0, HSA_AMD_QUEUE_ENGINE_COMPUTE, 0, 0, false, HSA_AMD_QUEUE_PRIORITY_NORMAL);
      auto st = hook(g_agent, &d, 1);
      check("zero_queue_size_rejected", st != HSA_STATUS_SUCCESS || R.passthrough == 1,
            "queue_size_bytes=0 -> ret=0x%x intercept_create=%d size=%u passthrough=%d "
            "(runtime returns INVALID_ARGUMENT)",
            (unsigned)st, R.intercept_create, R.size, R.passthrough); }

    // DEFAULT means "runtime default scratch", not "no scratch"
    reset();
    { auto d = desc(1, 0, HSA_AMD_QUEUE_ENGINE_COMPUTE, 4096,
                    HSA_AMD_PRIVATE_SEGMENT_SIZE_DEFAULT, false, HSA_AMD_QUEUE_PRIORITY_NORMAL);
      hook(g_agent, &d, 1);
      check("default_private_segment_preserved", R.priv == HSA_AMD_PRIVATE_SEGMENT_SIZE_DEFAULT,
            "priv=DEFAULT(0x%x) -> intercept_create priv=0x%x (0 means no scratch)",
            (unsigned)HSA_AMD_PRIVATE_SEGMENT_SIZE_DEFAULT, R.priv); }

    // SDMA is not an AQL queue; it must pass straight through
    reset();
    { auto d = desc(1, 0, HSA_AMD_QUEUE_ENGINE_SDMA, 4096, 0, false, HSA_AMD_QUEUE_PRIORITY_NORMAL);
      hook(g_agent, &d, 1);
      check("sdma_passthrough", R.passthrough == 1 && R.intercept_create == 0,
            "engine=SDMA -> passthrough=%d intercept_create=%d", R.passthrough, R.intercept_create); }

    reset();
    { auto d = desc(1, 0, HSA_AMD_QUEUE_ENGINE_COMPUTE, 4096, 0, false, HSA_AMD_QUEUE_PRIORITY_HIGH);
      hook(g_agent, &d, 1);
      check("priority_applied",
            R.set_priority == 1 && R.last_priority == (int)HSA_AMD_QUEUE_PRIORITY_HIGH,
            "priority=HIGH -> set_priority calls=%d value=%d", R.set_priority, R.last_priority); }

    reset();
    { auto d = desc(1, 0, HSA_AMD_QUEUE_ENGINE_COMPUTE, 4096, 0, false, HSA_AMD_QUEUE_PRIORITY_NORMAL);
      hook(g_agent, &d, 1);
      check("profiling_enabled", R.prof_enable == 1 && R.intercept_register == 1,
            "prof_enable=%d intercept_register=%d", R.prof_enable, R.intercept_register); }

    reset();
    { hsa_amd_queue_create_desc_t d[3] = {
          desc(1, 0, HSA_AMD_QUEUE_ENGINE_COMPUTE, 4096, 0, false, HSA_AMD_QUEUE_PRIORITY_NORMAL),
          desc(1, 0, HSA_AMD_QUEUE_ENGINE_COMPUTE, 8192, 0, false, HSA_AMD_QUEUE_PRIORITY_NORMAL),
          desc(1, 0, HSA_AMD_QUEUE_ENGINE_SDMA,    4096, 0, false, HSA_AMD_QUEUE_PRIORITY_NORMAL) };
      auto st = hook(g_agent, d, 3);
      bool ok = st == HSA_STATUS_SUCCESS && R.intercept_create == 2 && R.passthrough == 1 &&
                d[0].queue && d[1].queue && d[2].queue;
      check("batch_mixed_engines", ok,
            "[compute,compute,sdma] -> ret=0x%x intercept_create=%d passthrough=%d queues=%p/%p/%p",
            (unsigned)st, R.intercept_create, R.passthrough,
            (void*)d[0].queue, (void*)d[1].queue, (void*)d[2].queue); }

    reset();
    { auto st1 = hook(g_agent, nullptr, 1);
      auto d = desc(1, 0, HSA_AMD_QUEUE_ENGINE_COMPUTE, 4096, 0, false, HSA_AMD_QUEUE_PRIORITY_NORMAL);
      auto st2 = hook(g_agent, &d, 0);
      check("null_and_empty_args",
            st1 == HSA_STATUS_ERROR_INVALID_ARGUMENT && st2 == HSA_STATUS_ERROR_INVALID_ARGUMENT,
            "descs=NULL ret=0x%x, num_descs=0 ret=0x%x (want 0x%x both)",
            (unsigned)st1, (unsigned)st2, (unsigned)HSA_STATUS_ERROR_INVALID_ARGUMENT); }

    reset();
    { auto d = desc(1, 0, HSA_AMD_QUEUE_ENGINE_SDMA, 4096, 0, false, HSA_AMD_QUEUE_PRIORITY_NORMAL);
      hook(g_agent, &d, 1);
      check("queue_out_field_owned", d.queue != (hsa_queue_t*)0xBADC0FFEE0DDF00Dull,
            "poisoned desc.queue was %s",
            d.queue == (hsa_queue_t*)0xBADC0FFEE0DDF00Dull ? "left untouched"
                                                           : "replaced by the callee"); }

    reset();
    R.fail_intercept_create = true;
    { auto d = desc(1, 0, HSA_AMD_QUEUE_ENGINE_COMPUTE, 4096, 0, false, HSA_AMD_QUEUE_PRIORITY_NORMAL);
      auto st = hook(g_agent, &d, 1);
      check("intercept_failure_fallback", st == HSA_STATUS_SUCCESS && d.queue != nullptr,
            "intercept_create forced to fail -> ret=0x%x queue=%p passthrough=%d",
            (unsigned)st, (void*)d.queue, R.passthrough); }
    R.fail_intercept_create = false;
}

// Run one phase in a child. Returns the wait status.
static int g_child_fails = 0;

static int run_child(void (*fn)()) {
    fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) { g_fails = 0; fn(); fflush(stdout); _exit(g_fails > 99 ? 99 : g_fails); }
    int st = 0;
    waitpid(pid, &st, 0);
    if (WIFEXITED(st) && WEXITSTATUS(st) < 100) g_child_fails += WEXITSTATUS(st);
    return st;
}
#endif

int main() {
#ifndef HSA_AMD_QUEUE_CREATE_DESC_VERSION
    skip("rocm10_headers", "HSA_AMD_QUEUE_CREATE_DESC_VERSION undefined (ROCm < 10 headers)");
    return 0;
#else
    setenv("RTL_MODE", "lite", 1);

    int st = run_child(phase_abi_slot);
    if (WIFSIGNALED(st))
        check("abi_hook_install_guarded", false,
              "OnLoad died with signal %d on a short amd_ext table", WTERMSIG(st));

    st = run_child(phase_abi_guard_page);
    if (WIFEXITED(st) && WEXITSTATUS(st) == 120) {
        skip("abi_table_copy_bounded", "mmap failed, could not build the guard page");
    } else {
        bool crashed = WIFSIGNALED(st);
        check("abi_table_copy_bounded", !crashed,
              "OnLoad against a %zu-byte amd_ext table backed by a guard page: %s "
              "(RTL copies sizeof(AmdExtTable)=%zu bytes)",
              offsetof(AmdExtTable, hsa_amd_queue_create_fn),
              crashed ? "FAULTED" : "clean", sizeof(AmdExtTable));
    }

    st = run_child(phase_descriptors);
    if (WIFSIGNALED(st))
        check("hook_installed", false,
              "descriptor phase died with signal %d", WTERMSIG(st));

    int total = g_fails + g_child_fails;
    printf("SUMMARY fails=%d (parent=%d children=%d)\n", total, g_fails, g_child_fails);
    fflush(stdout);
    return total == 0 ? 0 : 1;
#endif
}
