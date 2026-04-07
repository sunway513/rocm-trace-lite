.PHONY: bench bench-quick report check clean

# Full benchmark: all workloads x all profilers, 5 runs each
bench:
	python3 overhead_bench.py --workload all --profiler all --runs 5 --output results.json
	python3 analyze.py --input results.json

# Quick: gemm only, all profilers, 3 runs
bench-quick:
	python3 overhead_bench.py --workload gemm --profiler all --runs 3 --output results_quick.json
	python3 analyze.py --input results_quick.json

# Generate report from existing results
report:
	python3 analyze.py --input results.json --format markdown

# CI regression check (rtl overhead must stay < 5%)
check:
	python3 analyze.py --input results.json --check-regression --threshold 5

clean:
	rm -f results*.json
	rm -rf __pycache__ workloads/__pycache__
