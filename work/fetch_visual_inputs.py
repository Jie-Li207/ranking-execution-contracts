from pathlib import Path
import ftplib
import hashlib
import json
from urllib.parse import urlparse
import h5py
import numpy as np
import requests
ROOT = Path(__file__).resolve().parents[1]
SOURCES = [{'file': 'oxford5k_siamac.mat', 'url': 'ftp://ftp.irisa.fr/local/texmex/corpus/diffusion/data/oxford5k_siamac.mat', 'bytes': 234276021, 'sha256': 'dfb368cb7fa6b14cfed6b265c0b2a542cc38a05e97ebdf103d98ab3fa462c468'}, {'file': 'gnd_oxford5k.pkl', 'url': 'https://cmp.felk.cvut.cz/cnnimageretrieval/data/test/oxford5k/gnd_oxford5k.pkl', 'sha256': 'c84e5eb68524fd46aab5c0e80b90f970d7bf677507aeb8cddd6b12990d4bc19e'}, {'file': 'paris6k_siamac.mat', 'url': 'ftp://ftp.irisa.fr/local/texmex/corpus/diffusion/data/paris6k_siamac.mat', 'bytes': 295481172, 'sha256': 'b0cf8d7d0168abd8b362f2c2a41e33b50a681ef41f9570116dce7d81c035923e'}, {'file': 'gnd_paris6k.pkl', 'url': 'https://cmp.felk.cvut.cz/cnnimageretrieval/data/test/paris6k/gnd_paris6k.pkl', 'sha256': 'def916cf4f197a182d9a7840cdb6294489fa57c2121f9742a1bcc6f2209bb345'}]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4194304), b''):
            h.update(block)
    return h.hexdigest()

def main():
    directory = ROOT / 'data/rev3_visual'
    directory.mkdir(parents=True, exist_ok=True)
    for row in SOURCES:
        target = directory / row['file']
        if not target.exists():
            part = target.with_suffix(target.suffix + '.partial')
            url = urlparse(row['url'])
            if url.scheme == 'ftp':
                with ftplib.FTP(url.hostname, timeout=120) as ftp:
                    ftp.login()
                    with part.open('wb') as output:
                        ftp.retrbinary('RETR ' + url.path, output.write, blocksize=1048576)
            else:
                with requests.get(row['url'], stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    with part.open('wb') as output:
                        for block in response.iter_content(1048576):
                            output.write(block)
            assert digest(part) == row['sha256'], row['file']
            part.replace(target)
        assert digest(target) == row['sha256'], row['file']
        print('Verified', row['file'])
    for dataset in ['oxford5k', 'paris6k']:
        with h5py.File(directory / (dataset + '_siamac.mat')) as mat:
            for field, name in [('Q', 'query'), ('V', 'gallery')]:
                vectors = np.squeeze(np.array([mat[item[0]][:] for item in mat['/glob/' + field]]), axis=1).astype(np.float32)
                np.save(directory / (dataset + '_' + name + '.npy'), vectors)
if __name__ == '__main__':
    main()
