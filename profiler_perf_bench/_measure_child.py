"""Fresh, small supervisor: child RSS must not inherit a torch-loaded runner."""
import json
import resource
import subprocess
import sys

def main():
    result = subprocess.run(sys.argv[2:])
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    with open(sys.argv[1], 'w') as output:
        json.dump({'peak_rss_MB': rss / (1024.0**2 if sys.platform == 'darwin' else 1024.0),
                   'returncode': result.returncode}, output)


if __name__ == '__main__':
    main()
