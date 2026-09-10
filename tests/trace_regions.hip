#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <unistd.h>
#define CHECK(x) do { auto e=(x); if(e!=hipSuccess) {fprintf(stderr,"%s\n",hipGetErrorString(e)); return 2;} } while(0)
__global__ void tagged_kernel(int* p) { *p += 1; }
int main() {
  using Push=int(*)(const char*); using Pop=int(*)();
  using Start=uint64_t(*)(const char*); using Stop=void(*)(uint64_t);
  auto push=(Push)dlsym(RTLD_DEFAULT,"roctxRangePushA");
  auto pop=(Pop)dlsym(RTLD_DEFAULT,"roctxRangePop");
  auto start=(Start)dlsym(RTLD_DEFAULT,"roctxRangeStartA");
  auto stop=(Stop)dlsym(RTLD_DEFAULT,"roctxRangeStop");
  if(!push||!pop||!start||!stop) return 3;
  int* p; CHECK(hipMalloc(&p,sizeof(int))); CHECK(hipMemset(p,0,sizeof(int)));
  push("parent-region");
  for(int i=0;i<10;++i) hipLaunchKernelGGL(tagged_kernel,dim3(1),dim3(1),0,0,p);
  CHECK(hipDeviceSynchronize()); pop();
  auto id=start("cross-thread-region"); std::thread t([=]{stop(id);}); t.join();
  int result=0; CHECK(hipMemcpy(&result,p,sizeof(int),hipMemcpyDeviceToHost)); CHECK(hipFree(p));
  printf("pid=%d result=%d\n",getpid(),result);
  return result==10 ? 0 : 4;
}
