#!/usr/bin/env python3
"""Bundle the pinned image's eager Torch/profiler dependencies, without rebuilding them."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile

from packaging.requirements import Requirement


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def build(output, source_image):
    if '@sha256:' not in source_image:
        raise ValueError('An immutable source image digest is required')
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / 'benchmark-runtime'
        root.mkdir()
        records, distributions, seen = {}, {}, set()

        def copy(source, relative):
            source = Path(source).resolve()
            target = root / relative
            if target.exists():
                if digest(target) != digest(source):
                    raise ValueError(f'Conflicting dependency: {relative}')
                return
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            records[str(relative)] = {'source': str(source), 'sha256': digest(target),
                                      'bytes': target.stat().st_size}

        pending = ['torch', 'pytest', 'numpy', 'pyyaml', 'packaging']
        while pending:
            name = pending.pop()
            normalized = re.sub(r'[-_.]+', '-', name).lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            distribution = metadata.distribution(name)
            distributions[distribution.metadata['Name']] = distribution.version
            for requirement_text in distribution.requires or []:
                requirement = Requirement(requirement_text)
                if requirement.marker is None or requirement.marker.evaluate({'extra': ''}):
                    pending.append(requirement.name)
            for item in distribution.files or []:
                # Entry-point scripts outside site-packages are unnecessary:
                # every check uses python -m. Preserve all package runtime data.
                if '..' in item.parts or '__pycache__' in item.parts:
                    continue
                source = Path(distribution.locate_file(item))
                if source.is_file():
                    copy(source, Path('python') / item)

        # ldd resolves transitive DT_NEEDED dependencies. Dynamic ROCtracer loading
        # additionally requires explicit profiler libraries from the same image.
        seeds = [Path(record['source']) for path, record in records.items()
                 if '.so' in Path(path).name]
        seeds += list(Path('/opt/rocm/lib').glob('libroctracer64.so*'))
        seeds += list(Path('/opt/rocm/lib').glob('libroctx64.so*'))
        package_sources = {record['source'] for record in records.values()}
        system_glibc = re.compile(r'^(?:ld-linux.*|lib(?:c|m|pthread|dl|rt|util|resolv|anl)\.so(?:\..*)?)$')
        for seed in seeds:
            result = subprocess.run(['ldd', str(seed)], text=True, capture_output=True)
            if 'not found' in result.stdout:
                raise RuntimeError(result.stdout)
            dependencies = re.findall(r'=> (/\S+)', result.stdout)
            if str(seed).startswith('/opt/rocm/'):
                dependencies.append(str(seed))
            for dependency in dependencies:
                path = Path(dependency)
                if str(path.resolve()) in package_sources or system_glibc.match(path.name):
                    continue
                copy(path, Path('lib') / path.name)

        manifest = {'source_image': source_image, 'distributions': distributions,
                    'files': records, 'uncompressed_bytes': sum(x['bytes'] for x in records.values()),
                    'scope': 'Original pinned Torch packages and ELF dependencies; system Python/glibc supplied by Ubuntu 24.04'}
        (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        checksums = {path: record['sha256'] for path, record in records.items()}
        checksums['manifest.json'] = digest(root / 'manifest.json')
        (root / 'SHA256SUMS').write_text(''.join(f'{sha}  {path}\n' for path, sha in sorted(checksums.items())))
        with tarfile.open(output, 'w:gz', compresslevel=1) as archive:
            archive.add(root, arcname=root.name)
        output.with_name(output.name + '.sha256').write_text(f'{digest(output)}  {output.name}\n')
        print(json.dumps({'archive_bytes': output.stat().st_size,
                          'uncompressed_bytes': manifest['uncompressed_bytes'],
                          'source_image': source_image}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-image', required=True)
    args = parser.parse_args()
    build(args.output, args.source_image)
