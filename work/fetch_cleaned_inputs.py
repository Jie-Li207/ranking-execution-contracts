from pathlib import Path
import argparse
import hashlib
import json
import re
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

INPUTS = {
    'WikiDiverse': ('https://mailustceducn-my.sharepoint.com/:u:/g/personal/pfluo_mail_ustc_edu_cn/EQgQKn4VeghChY_lhUoyBIMBKz6aTS00DFKOL1dqxP_bEg?e=yRpKkU', 1937927168, 'e8ccae2b02e44347489d05bb57ed200603f15b99cf64df33d760cc544ff51e97'),
    'WikiMEL': ('https://mailustceducn-my.sharepoint.com/:u:/g/personal/pfluo_mail_ustc_edu_cn/ETtT1zwqdDdAmE-uxHMX5EAB7bCGb1Eh2AuafB0tijDdyg?e=IT9E8a', 3181946368, 'b151e248cd9d068636cf2700ddd951801d6f8d56122965c9ee2f6d3c5bd23c58')
}

def resolve(session, url):
    response = session.get(url, timeout=(30, 90))
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    if soup.find('input', {'name': 'txtPassword'}):
        form = soup.find('form')
        values = {x['name']: x.get('value', '') for x in form.find_all('input') if x.get('name') and x.get('type') == 'hidden'}
        values.update({'txtPassword': 'kdd2023', '__EVENTTARGET': 'btnSubmitPassword', '__EVENTARGUMENT': ''})
        response = session.post(urljoin(response.url, form['action']), data=values, timeout=(30, 90))
        response.raise_for_status()
    match = re.search(r'"\.downloadUrl"\s*:\s*("(?:[^"\\]|\\.)*")', response.text)
    if not match:
        raise RuntimeError('The public share did not expose a download; consult the MIMIC data release.')
    return json.loads(match.group(1))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=list(INPUTS))
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    url, size, expected = INPUTS[args.dataset]
    session = requests.Session()
    download = resolve(session, url)
    if args.probe:
        with session.get(download, headers={'Range': 'bytes=0-1023'}, stream=True, timeout=(30, 90)) as response:
            response.raise_for_status()
            data = next(response.iter_content(1024))
            assert response.status_code == 206 and len(data) == 1024
            assert response.headers['Content-Range'].endswith('/' + str(size))
            print(json.dumps({'dataset': args.dataset, 'public_source': url, 'range_status': response.status_code,
                              'bytes_checked': len(data), 'archive_bytes': size, 'expected_full_sha256': expected,
                              'full_hash_verified': False}))
        return
    target = args.output or Path(__file__).resolve().parents[1] / 'archives' / (args.dataset + '.tar')
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + '.part')
    h = hashlib.sha256()
    with session.get(download, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        with part.open('wb') as f:
            for block in response.iter_content(4194304):
                f.write(block)
                h.update(block)
    assert part.stat().st_size == size and h.hexdigest() == expected
    part.replace(target)
    print(json.dumps({'dataset': args.dataset, 'bytes': size, 'sha256': expected, 'output': str(target)}))

if __name__ == '__main__':
    main()
