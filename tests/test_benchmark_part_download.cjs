const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const crypto = require('node:crypto');
const {downloadParts} = require('../tools/download_benchmark_parts.cjs');

async function fixture(t, count = 12) {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'rtl-download-test-'));
  t.after(async () => {
    const target = path.resolve(directory);
    assert.equal(path.dirname(target), path.resolve(os.tmpdir()));
    assert.ok(path.basename(target).startsWith('rtl-download-test-'));
    await fs.rm(target, {recursive: true});
  });
  const data = Array.from({length: count}, (_, i) => Buffer.from(`part ${i} contents`));
  const manifest = {parts: data.map((content, i) => ({
    name: `runtime.part${String(i).padStart(4, '0')}`, bytes: content.length,
    sha256: crypto.createHash('sha256').update(content).digest('hex')}))};
  const artifacts = manifest.parts.map((part, id) => ({id, name: `benchmark-${part.name}`}));
  const write = async (artifact, stage, content = data[artifact.id]) => {
    await fs.writeFile(path.join(stage, manifest.parts[artifact.id].name), content);
  };
  return {directory, data, manifest, artifacts, write, delay: async () => {}};
}

test('starts at most four transfers, rather than eagerly creating all download promises', async t => {
  const f = await fixture(t, 24);
  let active = 0, peak = 0, calls = 0;
  const results = await downloadParts({...f, downloadArtifact: async (artifact, stage) => {
    active++; calls++; peak = Math.max(peak, active);
    try {
      await new Promise(resolve => setTimeout(resolve, 8));
      await f.write(artifact, stage);
    } finally { active--; }
  }});
  assert.equal(calls, 24);
  assert.ok(peak > 1 && peak <= 4, `observed concurrency ${peak}`);
  assert.equal(results.length, 24);
  assert.equal(active, 0);
  for (let i = 0; i < 24; i++) assert.deepEqual(await fs.readFile(path.join(f.directory, f.manifest.parts[i].name)), f.data[i]);
});

test('retries transport failures and corrupt bytes while reusing only hash-verified parts', async t => {
  const f = await fixture(t, 3);
  await fs.writeFile(path.join(f.directory, f.manifest.parts[2].name), f.data[2]);
  const calls = [0, 0, 0];
  const results = await downloadParts({...f, downloadArtifact: async (artifact, stage) => {
    const attempt = ++calls[artifact.id];
    if (artifact.id === 0 && attempt === 1) throw new Error('transient ListArtifacts timeout');
    await f.write(artifact, stage, artifact.id === 1 && attempt === 1 ? Buffer.alloc(f.data[artifact.id].length) : f.data[artifact.id]);
  }});
  assert.deepEqual(calls, [2, 2, 0]);
  assert.equal(results[2].reused, true);
  assert.ok((await fs.readdir(f.directory)).some(name => name.startsWith('.rtl-download-')));
});

test('exhausted integrity retries fail and cannot replace an existing output', async t => {
  const f = await fixture(t, 1);
  const output = path.join(f.directory, f.manifest.parts[0].name);
  await fs.writeFile(output, 'previous evidence');
  let calls = 0;
  await assert.rejects(downloadParts({...f, downloadArtifact: async (artifact, stage) => {
    calls++; await f.write(artifact, stage, Buffer.from('corrupt'));
  }}), /failed after 3 attempts/);
  assert.equal(calls, 3);
  assert.equal(await fs.readFile(output, 'utf8'), 'previous evidence');
});

test('rejects missing, duplicate and unsafe part metadata before downloading', async t => {
  const f = await fixture(t, 1);
  const downloadArtifact = async () => assert.fail('must not start download');
  await assert.rejects(downloadParts({...f, artifacts: [], downloadArtifact}), /exactly one artifact/);
  await assert.rejects(downloadParts({...f, artifacts: [...f.artifacts, ...f.artifacts], downloadArtifact}), /exactly one artifact/);
  const manifest = {parts: [{...f.manifest.parts[0], name: '../escape'}]};
  await assert.rejects(downloadParts({...f, manifest, downloadArtifact}), /Invalid manifest part/);
});

test('waits for active transfers when another worker fails', async t => {
  const f = await fixture(t, 8);
  let active = 0;
  await assert.rejects(downloadParts({...f, attempts: 1, downloadArtifact: async (artifact, stage) => {
    active++;
    try {
      if (artifact.id === 0) throw new Error('permanent failure');
      await new Promise(resolve => setTimeout(resolve, 25));
      await f.write(artifact, stage);
    } finally { active--; }
  }}), /permanent failure/);
  assert.equal(active, 0);
});
