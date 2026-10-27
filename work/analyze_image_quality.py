from pathlib import Path
import json, numpy as np
r = Path(__file__).resolve().parents[1]
p = r / 'results/rev4/image_quality_audit.json'
audit = json.loads(p.read_text(encoding='utf-8'))
for m in audit['clean_metrics']:
    rows = json.loads((r / f'results/rev3/superglobal_{m['mode']}_test_queries.json').read_text())
    chosen = [rows[i] for i in audit['clean_indices']]
    groups = {}
    for x in chosen:
        groups.setdefault(x['cluster'], []).append(int(x['full_5']) - int(x['initial_5']))
    sums = np.array([sum(v) for v in groups.values()])
    sizes = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(20261026)
    stats = []
    for _ in range(2000):
        idx = rng.integers(len(groups), size=len(groups))
        stats.append(100 * sums[idx].sum() / sizes[idx].sum())
    m.update(full_minus_initial_pp=100 * sum(sums) / sum(sizes), full_minus_initial_CI95_pp=np.quantile(stats, [0.025, 0.975]).tolist(), bootstrap_clusters=len(groups), bootstrap_replicates=2000, bootstrap_seed=20261026)
audit['inference'] = 'Descriptive posthoc sensitivity; no model selection or replacement of full-test primary outcomes.'
p.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(audit['clean_metrics'], indent=2))
