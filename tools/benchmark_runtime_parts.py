#!/usr/bin/env python3
"""Split or reconstruct a runtime archive with publisher hashes for every part."""
import argparse
import hashlib
import json
from pathlib import Path


def split(archive, output, chunk_size=128 * 1024 * 1024):
    output.mkdir(parents=True, exist_ok=True)
    total, parts = hashlib.sha256(), []
    with archive.open('rb') as source:
        for index, content in enumerate(iter(lambda: source.read(chunk_size), b'')):
            name = f'runtime.part{index:04d}'
            (output / name).write_bytes(content)
            total.update(content)
            parts.append({'name': name, 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)})
    (output / 'parts.json').write_text(json.dumps({'sha256': total.hexdigest(), 'parts': parts}, indent=2) + '\n')


def join(source, archive):
    manifest = json.loads((source / 'parts.json').read_text())
    total = hashlib.sha256()
    temporary = archive.with_name(archive.name + '.partial')
    try:
        with temporary.open('wb') as output:
            for index, part in enumerate(manifest['parts']):
                assert part['name'] == f'runtime.part{index:04d}', 'Invalid part name/order'
                checksum, size = hashlib.sha256(), 0
                with (source / part['name']).open('rb') as stream:
                    for content in iter(lambda: stream.read(1024 * 1024), b''):
                        checksum.update(content)
                        total.update(content)
                        size += len(content)
                        output.write(content)
                assert checksum.hexdigest() == part['sha256'] and size == part['bytes'], part['name']
                print(f"Verified {part['name']}: {size} bytes", flush=True)
        assert total.hexdigest() == manifest['sha256'], 'Combined archive hash mismatch'
        temporary.replace(archive)
        archive.with_name(archive.name + '.sha256').write_text(f"{total.hexdigest()}  {archive.name}\n")
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['split', 'join'])
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    (split if args.operation == 'split' else join)(args.source, args.output)
