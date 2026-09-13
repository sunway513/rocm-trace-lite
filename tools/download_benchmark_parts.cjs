// Bounded, verified artifact downloads. Creating every Promise before chunking
// eagerly starts every transfer, so workers must claim parts only when idle.
const fs = require('node:fs');
const fsp = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');

async function matches(filename, part) {
  let stat;
  try { stat = await fsp.lstat(filename); }
  catch (error) { if (error.code === 'ENOENT') return false; throw error; }
  if (!stat.isFile() || stat.size !== part.bytes) return false;
  const hash = crypto.createHash('sha256');
  for await (const chunk of fs.createReadStream(filename)) hash.update(chunk);
  return hash.digest('hex') === part.sha256;
}

async function downloadParts({manifest, artifacts, directory, downloadArtifact,
  concurrency = 4, attempts = 3, log = () => {},
  delay = ms => new Promise(resolve => setTimeout(resolve, ms))}) {
  if (!Number.isInteger(concurrency) || concurrency < 1 || concurrency > 4 ||
      !Number.isInteger(attempts) || attempts < 1 || attempts > 5) {
    throw new Error('Invalid bounded download settings');
  }
  if (!Array.isArray(manifest.parts) || manifest.parts.length === 0) {
    throw new Error('Empty part manifest');
  }
  const entries = manifest.parts.map((part, index) => {
    if (part.name !== `runtime.part${String(index).padStart(4, '0')}` ||
        !/^[a-f0-9]{64}$/.test(part.sha256) ||
        !Number.isSafeInteger(part.bytes) || part.bytes <= 0) {
      throw new Error(`Invalid manifest part at index ${index}`);
    }
    const found = artifacts.filter(artifact => artifact.name === `benchmark-${part.name}`);
    if (found.length !== 1) throw new Error(`Expected exactly one artifact for ${part.name}`);
    return {part, artifact: found[0]};
  });
  await fsp.mkdir(directory, {recursive: true});
  let cursor = 0;
  let failure;
  const results = new Array(entries.length);

  async function transfer({part, artifact}) {
    const output = path.join(directory, part.name);
    if (await matches(output, part)) {
      log(`Verified existing ${part.name}; transfer skipped`);
      return {name: part.name, reused: true, attempts: 0};
    }
    for (let attempt = 1; attempt <= attempts; attempt++) {
      const stage = await fsp.mkdtemp(path.join(directory, '.rtl-download-'));
      try {
        await downloadArtifact(artifact, stage);
        const members = await fsp.readdir(stage);
        if (members.length !== 1 || members[0] !== part.name ||
            !await matches(path.join(stage, part.name), part)) {
          throw new Error(`Size, SHA256 or archive contents mismatch for ${part.name}`);
        }
        await fsp.rename(path.join(stage, part.name), output);
        await fsp.rmdir(stage); // Only the now-empty, task-created directory.
        log(`Verified ${part.name}: ${part.bytes} bytes, attempt ${attempt}`);
        return {name: part.name, reused: false, attempts: attempt};
      } catch (error) {
        // Retain failed attempts separately; never expose a partial final part.
        log(`${part.name} attempt ${attempt}/${attempts} failed: ${error.message}`);
        if (attempt === attempts) throw new Error(`${part.name} failed after ${attempts} attempts: ${error.message}`);
        await delay(2000 * attempt);
      }
    }
  }

  const workers = Array.from({length: Math.min(concurrency, entries.length)}, async () => {
    while (!failure) {
      const index = cursor++;
      if (index >= entries.length) return;
      try { results[index] = await transfer(entries[index]); }
      catch (error) { failure = failure || error; throw error; }
    }
  });
  // Wait for already-started work even if one worker fails; do not leave
  // background transfers writing after this step has reported a failure.
  await Promise.allSettled(workers);
  if (failure) throw failure;
  if (results.filter(Boolean).length !== entries.length) throw new Error('Missing verified artifact parts');
  return results;
}

module.exports = {downloadParts};
