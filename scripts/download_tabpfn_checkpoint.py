"""Fetch public original TabPFN-v2 files; sends no project tables or labels.

Built with PriorLabs-TabPFN. Weights retain their original Prior Labs License 1.1.
No sign-in, hosted inference or automatic agreement acceptance is implemented.
"""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/tabpfn_experiment'
REPOSITORY = 'Prior-Labs/TabPFN-v2-clf'
REVISION = '275f19d2add9c2c0de38f9ebaded3856599fa2e7'
FILENAME = 'tabpfn-v2-classifier.ckpt'
SHA256 = 'f65a35685aeef42e31b796d9bfa34e68d6fc780bc98e7bff7763802964cf435f'


def download():
    model_dir = OUT / 'models'
    model_dir.mkdir(parents=True,exist_ok=True)
    for name in [FILENAME,'LICENSE.txt','README.md']:
        destination = model_dir / name
        if destination.exists():
            if name == FILENAME:
                assert hashlib.sha256(destination.read_bytes()).hexdigest() == SHA256, 'Existing checkpoint changed.'
            continue
        url = f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}'
        temporary = destination.with_suffix(destination.suffix+'.part')
        with urllib.request.urlopen(url,timeout=120) as response,temporary.open('wb') as stream:
            while chunk := response.read(1024*1024):
                stream.write(chunk)
        if name == FILENAME:
            assert hashlib.sha256(temporary.read_bytes()).hexdigest() == SHA256, 'Downloaded checkpoint hash differs.'
        temporary.replace(destination)
    record = dict(repository=REPOSITORY,revision=REVISION,filename=FILENAME,gated=False,sha256=SHA256,
        source_url=f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{FILENAME}',license_files=['LICENSE.txt'])
    (OUT/'checkpoint.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Public original TabPFN-v2 checkpoint verified:',model_dir/FILENAME)


if __name__=='__main__':
    download()
