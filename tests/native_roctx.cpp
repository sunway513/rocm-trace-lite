#include "trace_db.h"
#include <thread>
#include <vector>
#include <cassert>
extern "C" {
uint64_t roctxRangeStartA(const char*);
void roctxRangeStop(uint64_t);
int roctxRangePushA(const char*);
int roctxRangePop();
}
int main() {
    std::vector<uint64_t> ids;
    for (int i=0; i<100; ++i) ids.push_back(roctxRangeStartA("cross-thread"));
    std::vector<std::thread> workers;
    for (auto id : ids) workers.emplace_back([id] { roctxRangeStop(id); });
    for (auto& worker : workers) worker.join();
    auto id = roctxRangeStartA("exactly-once");
    std::thread a([id] { roctxRangeStop(id); });
    std::thread b([id] { roctxRangeStop(id); });
    a.join(); b.join();
    assert(roctxRangePushA("outer") == 0);
    assert(roctxRangePushA("inner") == 1);
    assert(roctxRangePop() == 1);
    assert(roctxRangePop() == 0);
    assert(roctxRangePop() == -1);
    trace_db::get_trace_db().close();
}
