from pathlib import Path
import argparse, json, hashlib, requests
from bs4 import BeautifulSoup
p = argparse.ArgumentParser()
p.add_argument('item', choices=['clip', 'WikiMEL', 'RichpediaMEL'])
a = p.parse_args()
r = Path(__file__).resolve().parents[1]
if a.item == 'clip':
    sha = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
    url = f'https://openaipublic.azureedge.net/clip/models/{sha}/ViT-B-32.pt'
    dest = r / 'models/ViT-B-32.pt'
else:
    receipt = json.load(open(r / 'sources' / f'openmel_{a.item.lower()}_inventory.json'))
    sha = receipt['sha256']
    fid = receipt['source'].split('/d/')[1].split('/')[0]
    url = 'https://drive.google.com/uc?export=download&id=' + fid
    dest = r / 'archives' / (a.item + '.zip')

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
if dest.exists():
    assert digest(dest) == sha, 'Existing file hash mismatch; preserve and investigate.'
    print('Already verified', dest)
    raise SystemExit(0)
dest.parent.mkdir(parents=True, exist_ok=True)
session = requests.Session()
resp = session.get(url, stream=True, timeout=(30, 120))
resp.raise_for_status()
if 'text/html' in resp.headers.get('Content-Type', ''):
    form = BeautifulSoup(resp.text, 'html.parser').find('form', id='download-form')
    assert form is not None, 'Public download form unavailable'
    fields = {x['name']: x.get('value', '') for x in form.find_all('input') if x.get('name')}
    resp = session.get(form['action'], params=fields, stream=True, timeout=(30, 180))
    resp.raise_for_status()
assert 'text/html' not in resp.headers.get('Content-Type', '')
part = dest.with_suffix(dest.suffix + '.part')
h = hashlib.sha256()
with open(part, 'wb') as f:
    for block in resp.iter_content(4 * 1024 * 1024):
        f.write(block)
        h.update(block)
assert h.hexdigest() == sha, 'Downloaded hash mismatch; .part preserved for diagnosis'
part.replace(dest)
print('Verified', dest, sha)
