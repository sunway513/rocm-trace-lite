#!/usr/bin/env python3
"""Matched HTTP latency samples; server ownership and GPU isolation are external gates."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import statistics
import time
import urllib.request
from pathlib import Path

PROMPT_REPETITIONS = 24
METRIC = 'Client HTTP end-to-end latency; no TTFT/TPOT claim'

def prompt(index):
    return ('You are explaining GPU computing to an engineering student. '
            'Discuss memory bandwidth, matrix multiplication, synchronization, '
            'and the difference between prefill and token decoding. ' * PROMPT_REPETITIONS
            + f'\nExample {index}: Give a detailed step-by-step explanation.\n')


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def worker_setup_barrier(url, timeout, expected_workers=8):
    """Require an acknowledgement from every TP worker before timing requests."""
    request = urllib.request.Request(
        url.rstrip('/') + '/collective_rpc',
        json.dumps({'method': 'rtl_latency_setup_barrier', 'args': [],
                    'timeout': timeout}).encode(),
        {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read())
    workers = result['results']
    if (len(workers) != expected_workers
            or {w['rank'] for w in workers} != set(range(expected_workers))
            or len({w['pid'] for w in workers}) != expected_workers
            or any(w.get('synchronized') is not True for w in workers)):
        raise ValueError(f'Incomplete worker setup barrier: {result}')
    return {'workers': workers, 'completed_ns': time.monotonic_ns()}


def atom_worker_setup_barrier(url, directory, timeout, expected_workers=8):
    """Request the baseline overlay barrier and verify fresh, per-rank acknowledgements."""
    import uuid
    directory.mkdir(parents=True, exist_ok=False)
    nonce = uuid.uuid4().hex
    request_file = directory / 'request.json'
    request_file.write_text(json.dumps({'nonce': nonce}))
    request = urllib.request.Request(url.rstrip('/') + '/start_profile', b'{}',
                                     {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()
    deadline = time.monotonic() + timeout
    while True:
        paths = [directory / f'rank-{rank}.json' for rank in range(expected_workers)]
        if all(p.exists() for p in paths):
            workers = [json.loads(p.read_text()) for p in paths]
            if ({w['rank'] for w in workers} != set(range(expected_workers))
                    or len({w['pid'] for w in workers}) != expected_workers
                    or any(w.get('nonce') != nonce or w.get('synchronized') is not True for w in workers)):
                raise ValueError('Invalid/stale ATOM worker setup acknowledgements')
            return {'workers': workers, 'completed_ns': time.monotonic_ns(), 'adapter': 'atom-setup-overlay'}
        if time.monotonic() >= deadline:
            raise TimeoutError('Missing ATOM worker setup acknowledgements')
        time.sleep(0.05)


def sample(url, model, index, tokens, expected_prompt_tokens, allow_eos, timeout, length_finish_reason="length"):
    body = dict(model=model, prompt=prompt(index), max_tokens=tokens,
                temperature=0, ignore_eos=not allow_eos, seed=42, stream=False)
    payload = json.dumps(body).encode()
    request = urllib.request.Request(url.rstrip('/') + '/v1/completions', payload,
                                    {'Content-Type': 'application/json'})
    record = {'index': index, 'request': body, 'valid': False,
              'start_ns': time.monotonic_ns()}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            record['http_status'] = response.status
        # Exclude JSON parsing, validation, and artifact writes from HTTP latency.
        record['end_ns'] = time.monotonic_ns()
        result = json.loads(raw)
        record['response'] = result
        choice = result['choices'][0]
        usage = result['usage']
        if len(result['choices']) != 1 or not choice['text'].strip():
            raise ValueError('Expected one nonempty completion')
        if usage['prompt_tokens'] != expected_prompt_tokens:
            raise ValueError(f'Prompt length mismatch: {usage}')
        if allow_eos:
            if not 1 <= usage['completion_tokens'] <= tokens:
                raise ValueError(f'Unexpected output length: {usage}')
            if choice['finish_reason'] not in ('stop', 'length'):
                raise ValueError(f'Unexpected finish reason: {choice}')
        elif usage['completion_tokens'] != tokens or choice['finish_reason'] != length_finish_reason:
            raise ValueError(f'Fixed-output requirement failed: {usage}, {choice}')
        record['valid'] = True
    except Exception as error:
        record.setdefault('end_ns', time.monotonic_ns())
        record['error'] = f'{type(error).__name__}: {error}'
    record['latency_ms'] = (record['end_ns'] - record['start_ns']) / 1e6
    return record


def phase(args, out, name, concurrency, count):
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        started = time.monotonic_ns()
        futures = []
        for index in range(count):
            if args.request_rate:
                due = started / 1e9 + index / args.request_rate
                time.sleep(max(0, due - time.monotonic()))
            futures.append(pool.submit(sample, args.url, args.model, index,
                                       args.output_tokens, args.prompt_tokens,
                                       args.allow_eos, args.timeout, args.length_finish_reason))
        records = [future.result() for future in futures]
    ended = max(r['end_ns'] for r in records)
    # Save responses only after the entire phase, avoiding trace/storage I/O overlap.
    (out / f'{name}-requests.json').write_text(json.dumps(records, indent=2) + '\n')
    valid = all(r['valid'] for r in records)
    result = dict(name=name, concurrency=concurrency, count=count,
                  start_ns=started, end_ns=ended, valid=valid)
    if not valid:
        result['errors'] = [r.get('error') for r in records if not r['valid']]
        return result
    latencies = [r['latency_ms'] for r in records]
    output_tokens = sum(r['response']['usage']['completion_tokens'] for r in records)
    result.update(median_ms=statistics.median(latencies), p95_ms=percentile(latencies, .95),
                  mean_ms=statistics.mean(latencies), output_tokens=output_tokens,
                  output_tokens_per_second=output_tokens / ((ended - started) / 1e9))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:18861')
    parser.add_argument('--model', required=True)
    parser.add_argument('--condition', required=True)
    parser.add_argument('--manifest', type=Path, required=True,
                        help='Recorded server command, pinned image, model and RTL hashes')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--concurrency', type=int, nargs='+', default=[1, 8])
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--tensor-parallel-size', type=int, default=4)
    parser.add_argument('--requests', type=int, default=32,
                        help='Requests per concurrency and round; use the same count for every condition')
    parser.add_argument('--warmup-requests', type=int, default=8)
    parser.add_argument('--prompt-tokens', type=int, default=709)
    parser.add_argument('--prompt-repetitions', type=int, default=24)
    parser.add_argument('--output-tokens', type=int, default=128)
    parser.add_argument('--request-rate', type=float, default=0,
                        help='0 means closed-loop load at the concurrency cap; positive means paced arrivals')
    parser.add_argument('--length-finish-reason', choices=['length', 'max_tokens'], default='length', help='Pinned server schema spelling; output count remains exact')
    parser.add_argument('--allow-eos', action='store_true',
                        help='Companion natural-stop workload; never pool with fixed-output results')
    parser.add_argument('--timeout', type=float, default=600)
    barrier = parser.add_mutually_exclusive_group()
    barrier.add_argument('--atom-worker-setup-dir', type=Path,
                         help='Fresh shared directory matching the ATOM overlay RTL_BENCH_SETUP_DIR')
    barrier.add_argument('--worker-setup-barrier', action='store_true',
                        help='After warmup, synchronize all TP workers through collective_rpc')
    args = parser.parse_args()
    global PROMPT_REPETITIONS
    PROMPT_REPETITIONS = args.prompt_repetitions
    if min(args.concurrency + [args.rounds, args.requests, args.warmup_requests,
                               args.prompt_tokens, args.output_tokens]) < 1 or args.request_rate < 0:
        parser.error('Counts must be positive; request rate must be nonnegative')
    if args.timeout <= 0:
        parser.error('Timeout must be positive')
    manifest = json.loads(args.manifest.read_text())
    required = {'image_digest', 'server_command', 'model_config_sha256', 'rtl_mode',
                'rtl_library_sha256', 'tensor_parallel_size', 'graph_enabled',
                'prefix_cache_enabled'}
    if required - manifest.keys():
        parser.error(f'Manifest missing {sorted(required - manifest.keys())}')
    if (manifest['tensor_parallel_size'] != args.tensor_parallel_size or manifest['graph_enabled'] is not True
            or manifest['prefix_cache_enabled'] is not False):
        parser.error('Protocol requires matching TP size, graphs enabled, prefix cache disabled')
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {'complete': False, 'condition': args.condition, 'manifest': manifest,
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               'metric': METRIC,
               'inference_limits': ['Within-server rounds are not independent server restarts.',
                                    'GPU isolation, trace coverage and graph replay require separate audits.',
                                    'A different serving image changes the entire stack, not only ROCm.',
                                    'P95 is a descriptive sample quantile; no confidence interval implied.'],
               'warmups': [], 'phases': []}

    def save():
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    save()
    for concurrency in args.concurrency:
        result = phase(args, args.output, f'warmup-c{concurrency}', concurrency,
                       max(args.warmup_requests, concurrency))
        summary['warmups'].append(result)
        save()
        if not result['valid']:
            raise SystemExit('Warmup validation failed; timings rejected')
    if args.worker_setup_barrier or args.atom_worker_setup_dir:
        try:
            summary['setup_barrier'] = (
                worker_setup_barrier(args.url, args.timeout, args.tensor_parallel_size) if args.worker_setup_barrier
                else atom_worker_setup_barrier(args.url, args.atom_worker_setup_dir, args.timeout, args.tensor_parallel_size))
        except Exception as error:
            summary['setup_barrier_error'] = f'{type(error).__name__}: {error}'
            save()
            raise SystemExit('Worker setup barrier failed; no measured requests launched')
        save()
    for round_index in range(args.rounds):
        for concurrency in args.concurrency:
            result = phase(args, args.output, f'r{round_index}-c{concurrency}',
                           concurrency, args.requests)
            summary['phases'].append(result)
            save()
            print(json.dumps(result), flush=True)
            if not result['valid']:
                raise SystemExit('Measured request validation failed; run rejected')
    summary['complete'] = True
    save()


if __name__ == '__main__':
    main()
