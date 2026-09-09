// Deterministic execution of the real completion worker with fake HSA signals.
#include "hsa_intercept.cpp"
#include <cassert>
#include <chrono>
static bool b_forwarded = false;
static int forwards = 0;
static hsa_signal_value_t wait_signal(hsa_signal_t signal, hsa_signal_condition_t,
    hsa_signal_value_t, uint64_t, hsa_wait_state_t) {
    return signal.handle == 1 && !b_forwarded ? 1 : 0;
}
static void forward(hsa_signal_t signal, hsa_signal_value_t n) {
    assert(n == 1);
    if (signal.handle == 11) assert(b_forwarded);  // A depends on B
    if (signal.handle == 12) b_forwarded = true;
    ++forwards;
}
int main() {
    using namespace hsa_intercept;
    for (size_t i=0; i<CENTRAL_CAPACITY; ++i) g_central_slots[i].sequence.store(i);
    g_orig_core.hsa_signal_wait_scacquire_fn = wait_signal;
    g_orig_core.hsa_signal_subtract_screlease_fn = forward;
    auto* a = new DispatchData{};
    a->profiling_signal.handle = 1; a->original_signal.handle = 11; a->device_id = -1;
    auto* b = new DispatchData{};
    b->profiling_signal.handle = 2; b->original_signal.handle = 12; b->device_id = -1;
    g_work_queue.push_back(a); g_work_queue.push_back(b);
    g_shutdown.store(true);
    completion_worker();
    assert(forwards == 2 && g_work_queue.empty());
    // Shutdown must not let the worker exit before an admitted producer publishes.
    g_shutdown.store(false);
    auto* admission = new InterceptAdmission;
    assert(admission->admitted);
    g_shutdown.store(true);
    std::atomic<bool> exited{false};
    std::thread worker([&] { completion_worker(); exited.store(true); });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    assert(!exited.load());
    delete admission;
    worker.join();
    assert(exited.load());
}
