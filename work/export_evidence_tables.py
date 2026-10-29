from pathlib import Path
from collections import defaultdict
import argparse
import csv
import json
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('evidence', type=Path)
args = p.parse_args()
root = args.evidence
out = root / 'generated'
out.mkdir(exist_ok=True)

def save(name, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (out / (name + '.csv')).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

native = []
for ds in ['WikiMEL', 'RichpediaMEL']:
    rows = json.loads((root / f'native/{ds}_query_results.json').read_text())
    for B in [1, 5, 10, 20]:
        native.append({'dataset': ds, 'B': B, 'n': len(rows), 'fallbacks': sum(not x['name_filter_matched'] for x in rows), 'native_hits': sum(x['gold'] in x['native_top20'][:B] for x in rows), 'graph_off_hits': sum(x['gold'] in x['no_graph_top20'][:B] for x in rows), 'ordered_changes': sum(x['native_top20'][:B] != x['no_graph_top20'][:B] for x in rows)})
save('native', native)
objective = []
for file in sorted((root / 'enumerated').glob('*.csv')):
    with file.open(newline='') as f: rows = list(csv.DictReader(f))
    for B in [1, 5, 10, 20]:
        block = [x for x in rows if int(x['B']) == B]
        if not block: continue
        row = {'condition': file.stem, 'B': B, 'n': len(block)}
        for key in ['root_hit', 'greedy_hit', 'opt_hit', 'opt_hit_min', 'opt_hit_max', 'strict_first_exclusion']:
            row[key + '_n'] = sum(x[key] == 'True' for x in block)
        objective.append(row)
save('objective', objective)
control = []
for file in sorted((root / 'controls').glob('*.csv')):
    groups = defaultdict(list)
    with file.open(newline='') as f:
        for row in csv.DictReader(f): groups[tuple(row.get(k, '') for k in ['partition', 'method', 'param', 'B'])].append(row)
    for key, rows in groups.items():
        outrow = dict(zip(['partition', 'method', 'param', 'B'], key))
        outrow.update(source=file.stem, n=len(rows))
        for field in ['hit', 'root_hit', 'strict', 'gold_can', 'gold_must', 'dominance']:
            if field in rows[0]: outrow[field + '_n'] = sum(x[field] == 'True' for x in rows)
        control.append(outrow)
save('controls', control)
timing = []
allscale = {n: json.loads((root / f'timing/scale_{n}_timings.json').read_text()) for n in [8192, 32768, 132460]}
common = set.intersection(*[{x['qi'] for x in rows} for rows in allscale.values()])
for n, rows in allscale.items():
    for plan in ['eager_batch_native', 'prefix_batch_flat']:
        x = [v['ms'] for v in rows if v['qi'] in common and v['backend'] == 'blas' and v['B'] == 5 and v['plan'] == plan]
        timing.append({'n': n, 'queries': len(common), 'plan': plan, 'median_ms': float(np.median(x))})
save('scale', timing)
native_latency = []
for ds in ['WikiMEL', 'RichpediaMEL']:
    path = root / f'repeat_20261029/native_strict/{ds}/timings.json'
    if not path.exists():
        continue
    rows = [x for x in json.loads(path.read_text()) if x['primary']]
    ids = sorted({x['qi'] for x in rows})
    matrices = {}
    for plan in ['native_eager', 'compiled_prefix']:
        mat = np.array([[x['ms'] for x in rows if x['qi'] == qi and x['plan'] == plan] for qi in ids])
        assert mat.shape == (128, 3)
        matrices[plan] = mat
        native_latency.append({'dataset': ds, 'plan': plan, 'queries': len(ids), 'repeats': 3, 'p50_ms': float(np.median(mat)), 'p95_ms': float(np.percentile(mat, 95)), 'serial_QPS': float(1000 / mat.mean())})
    draw = np.random.default_rng(20261025).integers(len(ids), size=(2000, len(ids)))
    a, b = matrices['native_eager'], matrices['compiled_prefix']
    ci = 100 * np.quantile(b[draw].mean((1, 2)) / a[draw].mean((1, 2)) - 1, [.025, .975])
    native_latency[-1].update(relative_mean_change_pct=float(100 * (b.mean() / a.mean() - 1)), CI_low=float(ci[0]), CI_high=float(ci[1]))
if native_latency:
    save('native_strict_latency', native_latency)

fig3 = []
for backend in ['direct', 'blas']:
    for plan in ['eager_batch_native', 'prefix_batch_flat']:
        for B in [1, 5, 10, 20]:
            rows = [x for x in allscale[132460] if x['backend'] == backend and x['plan'] == plan and x['B'] == B]
            ids = sorted({x['qi'] for x in rows})
            mat = np.array([[x['ms'] for x in rows if x['qi'] == qi] for qi in ids])
            draw = np.random.default_rng(20261025).integers(len(ids), size=(2000, len(ids)))
            lo, hi = np.quantile(np.median(mat[draw].reshape(2000, -1), axis=1), [.025, .975])
            fig3.append({'backend': backend, 'plan': plan, 'B': B, 'median_ms': float(np.median(mat)), 'CI_low': lo, 'CI_high': hi})
save('figure3', fig3)
fig4 = []
for mode in ['text', 'multimodal']:
    rows = json.loads((root / f'results/rev3/superglobal_{mode}_test_queries.json').read_text())
    counts = np.zeros((5, 5), int)
    for row in rows:
        bins = []
        for stage in ['initial', 'full']:
            rank = row[stage].index(row['gold']) + 1 if row['gold'] in row[stage] else 21
            bins.append(next(i for i, cut in enumerate([1, 5, 10, 20, 21]) if rank <= cut))
        counts[tuple(bins)] += 1
    for i in range(5):
        for j in range(5): fig4.append({'mode': mode, 'initial_bin': i, 'full_bin': j, 'count': int(counts[i, j]), 'row_percent': float(100 * counts[i, j] / counts[i].sum())})
save('figure4_rank_transition', fig4)
fig6 = []
diffusion = []
for ds in ['oxford5k', 'paris6k']:
    a = np.load(root / f'visual/{ds}.npz')
    for method in ['joint_full', 'joint_partition', 'gallery_only']:
        diffusion.append({'dataset': ds, 'method': method, 'mAP1000': float(a[method + '_ap1000'].mean())})
    for i, (x, y) in enumerate(zip(a['full'], a['partition'])):
        fig6.append({'dataset': ds, 'qi': i, 'replaced_top10_members': 10 - len(set(x[:10]) & set(y[:10])), 'AP_delta_pp': 100 * (a['partition_ap'][i] - a['full_ap'][i])})
save('figure6', fig6)
save('diffusion_common_depth', diffusion)
factorial = json.loads((root / 'rev10/factorial_summary.json').read_text())
save('encoder_reranking', [{'representation': row['representation'], **v, 'CI95_delta_pp': json.dumps(v['CI95_delta_pp'])} for row in factorial for v in row['reranking']])
save('encoder_cohorts', [{'representation': row['representation'], **v} for row in factorial for v in row['cohort']])
service_path = root / 'repeat_20261029/sqlite_service/summary.json'
if not service_path.exists():
    service_path = root / 'rev10/sqlite_service/summary.json'
service = json.loads(service_path.read_text())['rows']
save('sqlite_all_runs', [{k: v for k, v in row.items() if k != 'stage_times'} for row in service])
groups = defaultdict(list)
for row in service: groups[(row['stable_cohort_fraction'], row['policy'])].append(row)
summary = []
for (stable, policy), rows in groups.items():
    row = {'stable_fraction': stable, 'policy': policy, 'repeats': len(rows)}
    for key in ['wall_seconds', 'latency_p50_ms', 'latency_p95_ms', 'incorrect_ordered10', 'incorrect_joined_type', 'hits', 'payload_bytes', 'cache_page_bytes']:
        values = [x[key] for x in rows]
        row.update({key + '_median': float(np.median(values)), key + '_min': min(values), key + '_max': max(values)})
    summary.append(row)
save('sqlite_summary', summary)
print(json.dumps({'output': str(out), 'csv_files': len(list(out.glob('*.csv'))), 'sqlite_source': str(service_path.relative_to(root))}))
