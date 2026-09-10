#!/usr/bin/env python3
"""Build a small CI-only HIP graph runtime from the pinned development image."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output, source_image):
    repo = Path(__file__).resolve().parents[1]
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    if '@sha256:' not in source_image:
        raise ValueError('Record the pinned development image digest')
    with tempfile.TemporaryDirectory(prefix='rtl-gpu-runtime-') as directory:
        root = Path(directory) / 'gpu-test-runtime'
        (root/'lib').mkdir(parents=True)
        workload = root/'graph_stress'
        source = repo/'repro/repro_hipgraph_stress.hip'
        subprocess.run(['hipcc','-O2','--offload-arch=gfx950','-o',str(workload),str(source)],check=True)
        deps = subprocess.run(['ldd',str(workload)],check=True,text=True,capture_output=True).stdout
        if 'not found' in deps:
            raise RuntimeError('Unresolved workload dependency:\n'+deps)
        manifest = {'source_image':source_image,'workload_source_sha256':digest(source),
                    'workload_sha256':digest(workload),'target':'gfx950',
                    'rocm_libraries':[], 'system_dependencies':[]}
        paths = re.findall(r'=> (/\S+)',deps)
        for value in sorted(set(paths)):
            src = Path(value)
            if value.startswith('/opt/rocm/') or value.startswith('/opt/rocm-'):
                dst = root/'lib'/src.name
                if dst.exists() and digest(dst) != digest(src):
                    raise RuntimeError('Library basename collision: '+src.name)
                shutil.copy2(src,dst,follow_symlinks=True)
                manifest['rocm_libraries'].append({'name':src.name,'source_path':value,
                    'resolved_source_path':str(src.resolve()),'sha256':digest(dst),'bytes':dst.stat().st_size})
            else:
                manifest['system_dependencies'].append(value)
        names = {x['name'] for x in manifest['rocm_libraries']}
        if not {'libamdhip64.so.7','libhsa-runtime64.so.1'} <= names:
            raise RuntimeError('Missing pinned HIP/HSA runtime in dependency closure')
        relocated = subprocess.run(['ldd',str(workload)],
            env=dict(os.environ,LD_LIBRARY_PATH=str(root/'lib')),
            check=True,text=True,capture_output=True).stdout
        if 'not found' in relocated:
            raise RuntimeError('Relocated dependency is missing:\n'+relocated)
        for name in names:
            if f'{name} => {root}/lib/{name} ' not in relocated:
                raise RuntimeError('Dependency escaped the bundled runtime: '+name)
        (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        files = [workload,root/'manifest.json',*sorted((root/'lib').iterdir())]
        (root/'SHA256SUMS').write_text(''.join(digest(p)+'  '+str(p.relative_to(root))+'\n' for p in files))
        with tarfile.open(output,'w:gz',compresslevel=6) as archive:
            archive.add(root,arcname='gpu-test-runtime')
    print(json.dumps({'archive':str(output),'bytes':output.stat().st_size,'sha256':digest(output)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-image',required=True)
    args=parser.parse_args()
    build(args.output,args.source_image)
