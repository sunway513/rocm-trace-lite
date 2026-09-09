"""Execute a Python workload under torch.profiler in its own process."""
import os
from pathlib import Path
import runpy
import sys


def main():
    out, *args = sys.argv[1:]
    if not args:
        raise ValueError('Missing Python workload entrypoint')
    import torch
    from torch.profiler import profile, ProfilerActivity
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    Path(out).mkdir(parents=True, exist_ok=True)
    trace = str(Path(out) / f'{os.getpid()}.json')
    # export explicitly so even an exception leaves a trace for diagnosis;
    # the workload's nonzero exit still fails the benchmark sanity gate.
    prof = profile(activities=activities, record_shapes=False, with_stack=False)
    try:
        with prof:
            if args[0] == '-c':
                sys.argv = ['-c', *args[2:]]
                sys.path.insert(0, os.getcwd())
                exec(compile(args[1], '<string>', 'exec'), {'__name__': '__main__'})
            elif args[0] == '-m':
                sys.argv = [args[1], *args[2:]]
                sys.path.insert(0, os.getcwd())
                runpy.run_module(args[1], run_name='__main__', alter_sys=True)
            elif not args[0].startswith('-'):
                sys.argv = args
                sys.path.insert(0, str(Path(args[0]).resolve().parent))
                runpy.run_path(args[0], run_name='__main__')
            else:
                raise ValueError('Unsupported Python interpreter option: ' + args[0])
    finally:
        prof.export_chrome_trace(trace)



if __name__ == '__main__':
    main()
