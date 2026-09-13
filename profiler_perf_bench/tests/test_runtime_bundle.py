import hashlib
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest
from tools import build_benchmark_runtime as builder
from tools import benchmark_runtime_parts as parts


def test_bundle_preserves_requested_device_and_library_extras(tmp_path, monkeypatch):
    requirements = {
        'torch': ['rocm[libraries]', 'device; extra == "device-gfx950"',
                  'unwanted; extra == "device-gfx942"'],
        'rocm': ['libraries; extra == "libraries"'],
    }

    def distribution(name):
        source = tmp_path / (name + '.py')
        source.write_text('# Original package bytes\n')
        return SimpleNamespace(metadata={'Name': name}, version='1.0',
                               requires=requirements.get(name, []), files=[Path(source.name)],
                               locate_file=lambda _: source)

    monkeypatch.setattr(builder.metadata, 'requires', lambda _: requirements['torch'])
    monkeypatch.setattr(builder.metadata, 'distribution', distribution)
    original_glob = Path.glob
    monkeypatch.setattr(Path, 'glob', lambda self, pattern: [] if str(self) == '/opt/rocm/lib'
                        else original_glob(self, pattern))
    archive = tmp_path / 'runtime.tar.gz'
    builder.build(archive, 'example@sha256:' + 'a' * 64)
    with tarfile.open(archive) as stream:
        manifest = json.load(stream.extractfile('benchmark-runtime/manifest.json'))
        assert {'device', 'libraries'} <= manifest['distributions'].keys()
        assert 'unwanted' not in manifest['distributions']
        for name, record in manifest['files'].items():
            content = stream.extractfile('benchmark-runtime/' + name).read()
            assert hashlib.sha256(content).hexdigest() == record['sha256']
    assert archive.with_name(archive.name + '.sha256').read_text().startswith(
        hashlib.sha256(archive.read_bytes()).hexdigest())


def test_parts_reconstruct_and_reject_corruption_without_replacing_output(tmp_path):
    source = tmp_path / 'original'
    source.write_bytes(b'abcdefghijk')
    directory = tmp_path / 'parts'
    parts.split(source, directory, chunk_size=4)
    output = tmp_path / 'combined'
    parts.join(directory, output)
    assert output.read_bytes() == source.read_bytes()
    (directory / 'runtime.part0001').write_bytes(b'bad!')
    with pytest.raises(AssertionError):
        parts.join(directory, output)
    assert output.read_bytes() == source.read_bytes()
    assert not output.with_name(output.name + '.partial').exists()
