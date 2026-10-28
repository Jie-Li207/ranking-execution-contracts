from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('evidence', type=Path)
p.add_argument('--solve', action='store_true')
args = p.parse_args()
root = args.evidence
checks = []

def check(name, condition):
    assert condition, name
    checks.append(name)

manifest = json.loads((root / 'SHA256.json').read_text())
for name, record in manifest.items():
    check('sha256:' + name, hashlib.sha256((root / name).read_bytes()).hexdigest() == record['sha256'])
rows = json.loads((root / 'corrections/paired_records.json').read_text())
check('correction_records_present', len(rows) > 0)
for row in rows:
    old, new = row['original'], row['corrected']
    check('correction_exact:' + str((row['condition'], new['qi'], new['B'])), new['exact'] and new['status'] == 0)
    if old['opt_value'] is not None:
        check('corrected_value_not_lower', new['opt_value'] >= old['opt_value'] - 1e-7)

factorial = json.loads((root / 'rev10/factorial_summary.json').read_text())
metadata = json.loads((root / 'rev10/query_metadata.json').read_text())
groups = {}
for i, row in enumerate(metadata): groups.setdefault(row['imgPath'] or f'missing-{i}', []).append(i)
cluster_counts = np.array([len(v) for v in groups.values()])
draws = np.random.default_rng(20261028).integers(len(cluster_counts), size=(2000, len(cluster_counts)))
for row in factorial:
    name = row['representation']
    a = np.load(root / f'rev10/factorial_{name}.npz')
    for i, record in enumerate(row['reranking']):
        r5 = np.any(a['ranks'][i, :, :5] == a['gold'][:, None], axis=1).mean()
        check(f'{name}:{record["operator"]}:R5', abs(r5 - record['R5']) < 1e-12)
        delta = np.any(a['ranks'][i, :, :5] == a['gold'][:, None], axis=1).astype(float) - np.any(a['ranks'][0, :, :5] == a['gold'][:, None], axis=1)
        sums = np.array([delta[v].sum() for v in groups.values()])
        ci = 100 * np.quantile(sums[draws].sum(1) / cluster_counts[draws].sum(1), [.025, .975])
        check(f'{name}:{record["operator"]}:CI95', np.max(np.abs(ci - record['CI95_delta_pp'])) < 1e-10)
        change = np.count_nonzero(a['ranks'][i, :, 0] != a['ranks'][0, :, 0])
        check(f'{name}:{record["operator"]}:first', change == record['top1_changes'])
    full = np.load(root / f'rev10/factorial_{name}_cohort1024_reverse0.npz')
    for record in row['cohort']:
        b = np.load(root / f'rev10/factorial_{name}_cohort{record["batch"]}_reverse{int(record["reverse"])}.npz')
        changed = b['ranks'][:, 0] != full['ranks'][:, 0]
        counts = {'top1_changes': int(changed.sum()), 'ordered10_changes': int(np.any(b['ranks'][:, :10] != full['ranks'][:, :10], axis=1).sum()), 'set10_changes': int(np.any(np.sort(b['ranks'][:, :10]) != np.sort(full['ranks'][:, :10]), axis=1).sum()), 'strict_first_changes': int((changed & (b['margin'] > 1e-8) & (full['margin'] > 1e-8)).sum())}
        for key, value in counts.items(): check(f'{name}:{record["batch"]}:{record["reverse"]}:{key}', value == record[key])

service = json.loads((root / 'rev10/sqlite_service/summary.json').read_text())
if isinstance(service, dict): service = service['rows']
types = json.loads((root / 'rev10/gallery_types.json').read_text())
for row in service:
    stem = f'seed{row["seed"]}_stable{row["stable_cohort_fraction"]}'
    ref = np.load(root / f'rev10/sqlite_service/expected_{stem}.npz')['ranks'].reshape(-1, 10)
    a = np.load(root / f'rev10/sqlite_service/{stem}_{row["policy"]}.npz')
    actual = a['ranks'].reshape(-1, 10)
    for key, value in [('incorrect_first', np.count_nonzero(actual[:, 0] != ref[:, 0])), ('incorrect_ordered10', np.any(actual != ref, axis=1).sum()), ('incorrect_set10', np.any(np.sort(actual) != np.sort(ref), axis=1).sum()), ('incorrect_joined_type', sum(types[str(i)] != types[str(j)] for i, j in zip(actual[:, 0], ref[:, 0])))]:
        check(stem + ':' + row['policy'] + ':' + key, value == row[key])
    for q in [50, 95]: check(stem + ':latency' + str(q), abs(np.percentile(a['latency_ms'], q) - row[f'latency_p{q}_ms']) < 1e-8)
    if row['policy'] in ['none', 'cohort']: check(stem + ':valid_cache', np.array_equal(actual, ref))
check('27_sqlite_runs', len(service) == 27)

solves = 0
if args.solve:
    from enumeration_oracle import enumerate_roots
    for ds in ['WikiMEL', 'RichpediaMEL']:
        a = np.load(root / f'oracle_inputs/{ds}/inputs.npz')
        threshold = json.loads((root / f'oracle_inputs/{ds}/protocol.json').read_text())['threshold']
        selected = {int(q): i for i, q in enumerate(a['selected'])}
        lookup = {int(v): i for i, v in enumerate(a['unique'])}
        with (root / f'enumerated/{ds}_author3_t0.6_nall.csv').open(newline='') as f:
            records = {(int(x['qi']), int(x['B'])): x for x in csv.DictReader(f)}
        take = sorted(selected, key=lambda q: hashlib.sha256(f'{ds}:{q}'.encode()).digest())[:128]
        for qi in take:
            i = selected[qi]
            mask = a['scores'][i] > threshold
            roots = a['roots'][i][mask].astype(int).tolist()
            edges = [(u, int(v), float(w)) for u in roots for v, w in zip(a['nn'][lookup[u]], a['dd'][lookup[u]]) if int(v) not in roots and w > threshold]
            for actual in enumerate_roots(roots, a['scores'][i][mask], edges, [1, 5, 10, 20], gold=int(a['gold'][i])):
                saved = records[(qi, actual['B'])]
                check(f'fresh_solve:{ds}:{qi}:{actual["B"]}', abs(actual['value'] - float(saved['opt_value'])) < 1e-10 and actual['strict_first_exclusion'] == (saved['strict_first_exclusion'] == 'True'))
                solves += 1
report = {'passed_checks': len(checks), 'fresh_graphs': solves // 4, 'fresh_budget_solves': solves, 'scope': 'Hash verification and stored-output reaggregation; fresh exact solving only when --solve is used. No fresh encoder inference.'}
print(json.dumps(report, indent=2))
