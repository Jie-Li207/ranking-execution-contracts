from pathlib import Path
import argparse
import json
import hashlib
import numpy as np

def verify(root):
    checks = []
    for mode in ['text', 'multimodal']:
        for split in ['dev', 'test']:
            a = np.load(root / f'reranking_{mode}_{split}.npz')
            record = json.loads((root / f'reranking_{mode}_{split}.json').read_text())
            for i, row in enumerate(record['rows']):
                r = float(np.mean(np.any(a['ranks'][i, :, :5] == a['gold'][:, None], axis=1)))
                assert abs(r - row['R5']) < 1e-12
                assert int(np.sum(a['ranks'][i, :, 0] != a['ranks'][0, :, 0])) == row['top1_changes']
                checks.append(f'{mode}/{split}/{i}')
    for ds in ['WikiMEL', 'RichpediaMEL']:
        for mode in ['text', 'multimodal']:
            a = np.load(root / f'transfer_{ds}_{mode}.npz')
            record = json.loads((root / f'transfer_{ds}_{mode}.json').read_text())
            for i, row in enumerate(record['rows']):
                assert abs(float(np.mean(np.any(a['ranks'][i, :, :5] == a['gold'][:, None], axis=1))) - row['R5']) < 1e-12
                checks.append(f'{ds}/{mode}/{i}')
    record = json.loads((root / 'cohort_controls_summary.json').read_text())
    for row in record['rows']:
        lam = row['lambda']
        size = row['batch']
        rev = int(row['reverse'])
        a = np.load(root / row['file'])
        ref = np.load(root / f'cohort64_lambda{lam}_batch1024_reverse0.npz')['ranks']
        assert int(np.sum(a['ranks'][:, 0] != ref[:, 0])) == row['top1_changes']
        assert abs(float(np.mean(np.any(a['ranks'][:, :5] == a['gold'][:, None], axis=1))) - row['R5']) < 1e-12
        checks.append(f'cohort/{lam}/{size}/{rev}')
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir()) if p.suffix in ['.json', '.npz'] and p.name != 'verification.json'}
    result = {'checks': len(checks), 'all_passed': True, 'sha256': manifest}
    (root / 'verification.json').write_text(json.dumps(result, indent=2))
    return result
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.results), indent=2))
