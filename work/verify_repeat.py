from pathlib import Path
import argparse
import json
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('evidence', type=Path)
args = p.parse_args()
root = args.evidence
repeat = root / 'repeat_20261029'
checks = []

def check(name, value):
    if not value:
        raise AssertionError(name)
    checks.append(name)

for file in sorted(repeat.glob('factorial_*.npz')):
    old = np.load(root / 'rev10' / file.name)
    new = np.load(file)
    check(file.name + ':arrays', set(old.files) == set(new.files))
    for key in old.files:
        check(file.name + ':' + key, np.array_equal(old[key], new[key]))
check('all_factorial_files', len(list(repeat.glob('factorial_*.npz'))) == 16)
rows = json.loads((repeat / 'sqlite_service/summary.json').read_text())['rows']
types = json.loads((root / 'rev10/gallery_types.json').read_text())
check('27_runs', len(rows) == 27)
check('unique_conditions', len({(r['seed'], r['stable_cohort_fraction'], r['policy']) for r in rows}) == 27)
for row in rows:
    stem = f'seed{row["seed"]}_stable{row["stable_cohort_fraction"]}'
    reference = np.load(repeat / f'sqlite_service/expected_{stem}.npz')
    ref = reference['ranks'].reshape(-1, 10)
    old = np.load(root / f'rev10/sqlite_service/expected_{stem}.npz')
    check(stem + ':same_reference', np.array_equal(reference['ranks'], old['ranks']))
    check(stem + ':same_cohorts', np.array_equal(reference['cohorts'], old['cohorts']))
    array = np.load(repeat / f'sqlite_service/{stem}_{row["policy"]}.npz')
    actual = array['ranks'].reshape(-1, 10)
    old_actual = np.load(root / f'rev10/sqlite_service/{stem}_{row["policy"]}.npz')['ranks']
    check(stem + ':' + row['policy'] + ':repeat', np.array_equal(array['ranks'], old_actual))
    measures = {
        'incorrect_first': np.count_nonzero(actual[:, 0] != ref[:, 0]),
        'incorrect_ordered10': np.any(actual != ref, axis=1).sum(),
        'incorrect_set10': np.any(np.sort(actual) != np.sort(ref), axis=1).sum(),
        'incorrect_joined_type': sum(types[str(i)] != types[str(j)] for i, j in zip(actual[:, 0], ref[:, 0])),
    }
    for key, value in measures.items():
        check(stem + ':' + row['policy'] + ':' + key, value == row[key])
    for q in [50, 95]:
        check(stem + ':latency' + str(q), abs(np.percentile(array['latency_ms'], q) - row[f'latency_p{q}_ms']) < 1e-8)
    expected_hits = 0 if row['policy'] == 'none' else 1024 if row['policy'] == 'query' else 1024 * row['stable_cohort_fraction']
    check(stem + ':hits', row['hits'] == expected_hits)
    check(stem + ':misses', row['misses'] == 2048 - expected_hits)
    check(stem + ':calls', row['graph_calls'] == (2048 - expected_hits) // 64)
    payload = 0 if row['policy'] == 'none' else (2048 - expected_hits) * 72
    check(stem + ':payload', row['payload_bytes'] == payload)
    if row['policy'] in ['none', 'cohort']:
        check(stem + ':valid_output', np.array_equal(actual, ref))
intervals = json.loads((repeat / 'native_strict_intervals.json').read_text())
for ds in ['WikiMEL', 'RichpediaMEL']:
    fresh = json.loads((repeat / f'native_strict/{ds}/timings.json').read_text())
    old = json.loads((root / f'timing/native_{ds}.json').read_text())
    signature = lambda row: (row['qi'], row['rep'], row['plan'], row['primary'])
    check(ds + ':same_timed_queries', sorted(map(signature, fresh)) == sorted(map(signature, old)))
    check(ds + ':exact_order', all(not x['prior_order_mismatch'] for x in fresh))
    ids = sorted({x['qi'] for x in fresh if x['primary']})
    summary = json.loads((repeat / f'native_strict/{ds}/summary.json').read_text())
    matrices = {}
    for plan in ['native_eager', 'compiled_prefix']:
        matrix = np.array([[x['ms'] for x in fresh if x['qi'] == qi and x['plan'] == plan and x['primary']] for qi in ids])
        check(ds + ':' + plan + ':shape', matrix.shape == (128, 3))
        row = next(x for x in summary if x['plan'] == plan and x['subset'] == 'primary')
        check(ds + ':' + plan + ':median', abs(np.median(matrix) - row['p50_ms']) < 1e-8)
        check(ds + ':' + plan + ':p95', abs(np.percentile(matrix, 95) - row['p95_ms']) < 1e-8)
        check(ds + ':' + plan + ':serial_rate', abs(1000 / matrix.mean() - row['serial_queries_per_s']) < 1e-8)
        matrices[plan] = matrix
    a, b = matrices['native_eager'], matrices['compiled_prefix']
    draws = np.random.default_rng(20261025).integers(128, size=(2000, 128))
    ci = 100 * np.quantile(b[draws].mean((1, 2)) / a[draws].mean((1, 2)) - 1, [.025, .975])
    reference = next(x for x in intervals if x['phase'] == 'strict' and x['dataset'] == ds)
    check(ds + ':paired_interval', np.max(np.abs(ci - reference['CI95'])) < 1e-8)
print(json.dumps({'passed_checks': len(checks), 'factorial_files': 16, 'sqlite_runs': len(rows), 'scope': 'Fresh repeated operator outputs versus retained historical outputs; complete workload reaggregation. Timings are independently recorded, not asserted equal.'}, indent=2))
