"""Download the pinned official test classifier and verify upstream LFS hashes."""
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REPO = 'cais/HarmBench-Llama-2-13b-cls'
REVISION = 'bda705349d1144fa618770bea64d99ce54e3835b'


def main():
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    destination = ROOT/'models/HarmBench-Llama-2-13b-cls'
    snapshot_download(REPO, revision=REVISION, local_dir=destination, max_workers=6)
    files = []
    for entry in info.siblings:
        p = destination/entry.rfilename
        if not p.is_file() or p.stat().st_size != entry.size:
            raise ValueError('Missing/incomplete classifier file: '+entry.rfilename)
        h = hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):
                h.update(b)
        checksum = h.hexdigest()
        if entry.lfs and checksum != entry.lfs.sha256:
            raise ValueError('Classifier LFS hash mismatch: '+entry.rfilename)
        files.append({'file':entry.rfilename,'bytes':entry.size,'sha256':checksum})
    report = {'repo':REPO,'revision':REVISION,'files':files,'total_bytes':sum(f['bytes'] for f in files)}
    (ROOT/'reports/harmbench_classifier_manifest.json').write_text(json.dumps(report,indent=2))
    print('Official classifier downloaded and verified',flush=True)


if __name__=='__main__':
    main()
