#!/usr/bin/env python3
"""Create a GPU-only documentation excerpt without copying requests or host metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def make_example(source, output, duration_ns):
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source.as_uri()+'?mode=ro', uri=True) as src:
        marker = src.execute("SELECT max(o.start) FROM rocpd_op o JOIN rocpd_string s ON s.id=o.description_id WHERE s.string LIKE 'e2e_requests_begin%'").fetchone()[0]
        if marker is None:
            raise ValueError('Expected explicit E2E begin markers')
        start = src.execute("SELECT min(o.start) FROM rocpd_op o JOIN rocpd_string s ON s.id=o.opType_id WHERE s.string='KernelExecution' AND o.start>=?", (marker,)).fetchone()[0]
        rows = src.execute("SELECT o.gpuId,o.queueId,o.start,o.end,s.string,o.completionSignal FROM rocpd_op o JOIN rocpd_string s ON s.id=o.description_id JOIN rocpd_string t ON t.id=o.opType_id WHERE t.string='KernelExecution' AND o.start>=? AND o.start<? AND o.end>o.start ORDER BY o.start,o.id", (start,start+duration_ns)).fetchall()
        if {row[0] for row in rows} != set(range(8)):
            raise ValueError('Example must contain every TP8 GPU')
        metadata = {'example_scope':'GPU kernel excerpt, not a full trace or performance benchmark',
                    'source_sha256':sha256(source),'source_window_start_ns':start,
                    'start_time_selection_width_ns':duration_ns,'kernels':len(rows),
                    'timestamps':'relative to first selected kernel; original durations preserved',
                    'queues':'hardware queue addresses replaced by per-GPU integers',
                    'omitted':'requests, responses, host metadata, process IDs, APIs and ROCTX'}
        strings, queues, counts = {}, {}, {}
        with sqlite3.connect(output) as dst:
            dst.execute('PRAGMA foreign_keys=ON')
            for name in ('rocpd_string','rocpd_op','rocpd_api','rocpd_metadata','top','busy'):
                dst.execute(src.execute('SELECT sql FROM sqlite_master WHERE name=?',(name,)).fetchone()[0])
            def intern(value):
                if value not in strings:
                    strings[value]=len(strings)+1
                    dst.execute('INSERT INTO rocpd_string VALUES (?,?)',(strings[value],value))
                return strings[value]
            typ = intern('KernelExecution')
            for index,(gpu,queue,begin,end,name,info) in enumerate(rows,1):
                parts=dict(p.split('=',1) for p in (info or '').split() if '=' in p)
                key=(gpu,parts.get('hwq',str(queue)))
                if key not in queues:
                    queues[key]=sum(1 for g,_ in queues if g==gpu)
                dimensions=' '.join(k+'='+parts[k] for k in ('wg','grid') if k in parts)
                dst.execute('INSERT INTO rocpd_op VALUES (?,?,?,?,?,?,?,?,?,?)',
                    (index,gpu,queues[key],index,dimensions,begin-start,end-start,intern(name),typ,0))
                counts[gpu]=counts.get(gpu,0)+1
            metadata['kernels_per_gpu']=counts
            for key,value in metadata.items():
                dst.execute('INSERT INTO rocpd_metadata(tag,value) VALUES (?,?)',(key,json.dumps(value)))
            assert dst.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
            assert dst.execute('PRAGMA foreign_key_check').fetchall()==[]
    output.with_suffix('.provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--duration-ns',type=int,default=10_000_000)
    args=parser.parse_args()
    if args.duration_ns<=0: parser.error('duration must be positive')
    make_example(args.source,args.output,args.duration_ns)
