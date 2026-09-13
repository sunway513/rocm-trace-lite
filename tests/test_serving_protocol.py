"""Fail closed on incomplete worker barriers and malformed streaming completions."""
import io
import json
import unittest
from unittest.mock import patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/e2e"))

import serving_latency as base
import serving_latency_stream as stream


class Response(io.BytesIO):
    status = 200


def response_stream(*, done=True, tokens=128, finish='length'):
    events = [dict(choices=[dict(index=0, text='answer')]),
              dict(choices=[dict(index=0, text='', finish_reason=finish)],
                   usage=dict(prompt_tokens=709, completion_tokens=tokens))]
    data = b''.join(b'data: ' + json.dumps(e).encode() + b'\n\n' for e in events)
    return Response(data + (b'data: [DONE]\n\n' if done else b''))


class ProtocolTest(unittest.TestCase):
    def test_accepts_complete_stream(self):
        with patch('urllib.request.urlopen', return_value=response_stream()):
            result = stream.sample('http://localhost', 'test', 0, 128, 709, False, 3)
        self.assertTrue(result['valid'], result)

    def test_rejects_truncated_wrong_length_and_wrong_finish(self):
        for options in [dict(done=False), dict(tokens=127), dict(finish='stop')]:
            with self.subTest(options=options), patch('urllib.request.urlopen', return_value=response_stream(**options)):
                self.assertFalse(stream.sample('http://localhost', 'test', 0, 128, 709, False, 3)['valid'])

    def test_rejects_missing_duplicate_or_unsynchronized_worker(self):
        valid = [dict(rank=i, pid=100+i, synchronized=True) for i in range(4)]
        for workers in [valid[:-1], [valid[0]] * 4, [dict(w, synchronized=False) for w in valid]]:
            with self.subTest(workers=workers), patch('urllib.request.urlopen', return_value=Response(json.dumps(dict(results=workers)).encode())):
                with self.assertRaises(ValueError):
                    base.worker_setup_barrier('http://localhost', 3, 4)
        with patch('urllib.request.urlopen', return_value=Response(json.dumps(dict(results=valid)).encode())):
            self.assertEqual(len(base.worker_setup_barrier('http://localhost', 3, 4)['workers']), 4)


if __name__ == '__main__':
    unittest.main()
