"""Fetch and verify the fixed HF FinEval reference; never add it to training."""
from pathlib import Path
import hashlib,json,urllib.request,zipfile,io

ROOT=Path(__file__).resolve().parents[1]

def main():
    lock=json.loads((ROOT/'configs/fineval_reference.lock.json').read_text())
    url=f"https://huggingface.co/datasets/{lock['dataset']}/resolve/{lock['revision']}/FinEval.zip"
    data=urllib.request.urlopen(url,timeout=120).read()
    if hashlib.sha256(data).hexdigest()!=lock['archive_sha256']:raise ValueError('Pinned FinEval archive differs')
    destination=ROOT/'evaluation_data/FinEval/academic'
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for record in lock['csv_files']:
            relative=record['file'];matches=[]
            for name in z.namelist():
                if name==relative or name.endswith('/'+relative):
                    b=z.read(name)
                    if hashlib.sha256(b).hexdigest()==record['sha256']:matches.append(b)
            if len(matches)!=1:raise ValueError('Cannot uniquely verify '+relative)
            target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and target.read_bytes()!=matches[0]:raise ValueError('Existing reference differs: '+relative)
            target.write_bytes(matches[0])
    print('FinEval reference verified:',destination)

if __name__=='__main__':main()
