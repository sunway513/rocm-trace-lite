"""Test RTL profiling with spawn workers + CUDA graph replay."""
import multiprocessing as mp
import os
import sys

def worker(gpu_id):
    import torch
    torch.cuda.set_device(gpu_id)
    a = torch.randn(256, 256, device='cuda')
    b = torch.randn(256, 256, device='cuda')
    for _ in range(5):
        c = torch.mm(a, b)
    torch.cuda.synchronize()
    print(f'[worker {gpu_id}, pid={os.getpid()}] 5 direct matmuls done', file=sys.stderr)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        g = torch.cuda.CUDAGraph()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(g):
            c = torch.mm(a, b)
        for _ in range(10):
            g.replay()
        s.synchronize()
    print(f'[worker {gpu_id}, pid={os.getpid()}] 10 graph replays done', file=sys.stderr)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    procs = []
    for i in range(2):
        p = mp.Process(target=worker, args=(i,))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    print('all done', file=sys.stderr)
