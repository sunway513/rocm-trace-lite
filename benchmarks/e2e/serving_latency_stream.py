#!/usr/bin/env python3
"""Diagnostic streaming transport timings; separate from primary HTTP baselines."""
import json
import statistics
import time
import urllib.request

import serving_latency as base


def read_stream(response, record):
    events, texts, content_times, finishes = [], [], [], []
    usage = None
    done = False
    for raw in response:
        if not raw.startswith(b'data:'):
            continue
        data = raw[5:].strip()
        received = time.monotonic_ns()
        if data == b'[DONE]':
            done = True
            break
        event = json.loads(data)
        events.append({'received_ns': received, 'event': event})
        if event.get('usage'):
            usage = event['usage']
        for choice in event.get('choices', []):
            if choice.get('index', 0) != 0:
                raise ValueError('Unexpected additional completion')
            if choice.get('text'):
                texts.append(choice['text'])
                content_times.append(received)
            if choice.get('finish_reason'):
                finishes.append(choice['finish_reason'])
    record.update(events=events, done=done, finish_reasons=finishes)
    if not done or not usage or not content_times or not finishes:
        raise ValueError('Incomplete SSE stream, usage, content or finish metadata')
    record['response'] = {'usage': usage, 'choices': [{'text': ''.join(texts)}]}
    record['first_content_ns'] = content_times[0]
    record['last_content_ns'] = content_times[-1]
    record['content_chunks'] = len(content_times)


def sample(url, model, index, tokens, expected_prompt_tokens, allow_eos,
           timeout, length_finish_reason='length'):
    assert not allow_eos
    body = dict(model=model, prompt=base.prompt(index), max_tokens=tokens,
                temperature=0, ignore_eos=True, seed=42, stream=True,
                stream_options={'include_usage': True})
    request = urllib.request.Request(url.rstrip('/') + '/v1/completions',
                                    json.dumps(body).encode(),
                                    {'Content-Type': 'application/json'})
    record = dict(index=index, request=body, valid=False, start_ns=time.monotonic_ns())
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            record['http_status'] = response.status
            read_stream(response, record)
        record['end_ns'] = time.monotonic_ns()
        usage = record['response']['usage']
        if usage['prompt_tokens'] != expected_prompt_tokens or usage['completion_tokens'] != tokens:
            raise ValueError(f'Fixed length requirement failed: {usage}')
        # ATOM emits an extra terminal "stop" even after reaching max_tokens.
        if (length_finish_reason not in record['finish_reasons']
                or any(reason not in (length_finish_reason, 'stop') for reason in record['finish_reasons'])):
            raise ValueError('Unexpected streaming finish reason')
        record['first_content_ms'] = (record['first_content_ns'] - record['start_ns']) / 1e6
        record['post_first_content_ms'] = (record['last_content_ns'] - record['first_content_ns']) / 1e6
        record['estimated_tpot_ms'] = record['post_first_content_ms'] / (tokens - 1)
        record['valid'] = True
    except Exception as error:
        record.setdefault('end_ns', time.monotonic_ns())
        record['error'] = f'{type(error).__name__}: {error}'
    record['latency_ms'] = (record['end_ns'] - record['start_ns']) / 1e6
    return record


_phase = base.phase


def phase(args, out, name, concurrency, count):
    result = _phase(args, out, name, concurrency, count)
    if result['valid']:
        records = json.loads((out / f'{name}-requests.json').read_text())
        for key in ('first_content_ms', 'post_first_content_ms', 'estimated_tpot_ms', 'content_chunks'):
            result['median_' + key] = statistics.median(r[key] for r in records)
        result['timing_scope'] = ('Client first nonempty SSE content and subsequent content span; '
                                  'includes scheduling/network/serialization, not pure GPU prefill/decode. '
                                  'TPOT estimated from span/127, chunks need not map one-to-one to tokens.')
    return result


if __name__ == '__main__':
    base.METRIC = 'Client SSE end-to-end latency, first nonempty content, and estimated TPOT; includes transport and scheduling'
    base.sample = sample
    base.phase = phase
    base.__file__ = __file__
    base.main()
