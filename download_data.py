"""Obtain the unmodified public Data82.mat file and verify both checksums."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parent
URL = 'https://zenodo.org/records/1254206/files/Data82.mat?download=1'
MD5 = '5698942708300bc34b87931c6d91f6b6'
SHA256 = '03a35c57ae52026f980ffcf540a4243a02f198b3fcdd8dff72cb251f70b067cc'


def verify(path):
    value = path.read_bytes()
    if hashlib.md5(value).hexdigest() != MD5 or hashlib.sha256(value).hexdigest() != SHA256:
        raise ValueError(f'Checksum mismatch: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-file', type=Path, help='copy an existing verified Data82.mat')
    args = parser.parse_args()
    target = ROOT/'realdata_ct/data/Data82.mat'
    if target.exists():
        verify(target)
        print('Data82.mat is present and both checksums match.')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.mat.part')
    if args.local_file:
        verify(args.local_file)
        shutil.copyfile(args.local_file, temporary)
    else:
        request = urllib.request.Request(URL, headers={'User-Agent': 'Muon-DFO-reproduction/1.0'})
        with urllib.request.urlopen(request, timeout=120) as source, temporary.open('wb') as output:
            shutil.copyfileobj(source, output)
    verify(temporary)
    temporary.replace(target)
    print('Data82.mat installed; published MD5 and recorded SHA-256 match.')


if __name__ == '__main__':
    main()
