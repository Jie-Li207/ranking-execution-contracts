from pathlib import Path
import argparse
import ast
import heapq
import json
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('traces', type=Path)
p.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1] / 'vendor/OpenMEL-main/maximum_spanning.py')
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
source = ast.parse(args.source.read_text(encoding='utf8'))
ns = {'heapq': heapq}
keep = ['MaxHeap', 'find', 'union', 'maximum_spanning_tree']
exec(compile(ast.Module(body=[x for x in source.body if isinstance(x, (ast.FunctionDef, ast.ClassDef)) and x.name in keep], type_ignores=[]), str(args.source), 'exec'), ns)
native = ns['maximum_spanning_tree']
runner = Path(__file__).with_name('author_cache_reproduction.py')
module = ast.parse(runner.read_text(encoding='utf8'))
flat_node = next(x for x in module.body if isinstance(x, ast.FunctionDef) and x.name == 'flat')
exec(compile(ast.Module(body=[flat_node], type_ignores=[]), str(runner), 'exec'), ns)
flat = ns['flat']
node = next(x for x in ast.walk(module) if isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'preconditions' for t in x.targets))
guard_code = compile(ast.Expression(node.value), str(runner), 'eval')

def guard(rt, ss, adj, ww):
    return eval(guard_code, {'ss': ss, 'rt': rt, 'ww': ww, 'adj': adj})

counterexamples = []
for delta in [1e-8, 5e-8, 9e-8]:
    rt, ss, adj, ww = [0], [.5], {0: [1]}, {0: [.5 + delta]}
    actual = native(-1, rt, ss, adj, ww, 3)[0][1:]
    proposed = flat(rt, ss, adj, ww, 2)
    assert all(0 <= w <= ss[j] + 1e-7 for j, u in enumerate(rt) for w in ww[u])
    assert proposed != actual and not guard(rt, ss, adj, ww)
    counterexamples.append({'excess_weight': delta, 'native': actual, 'unconditionally_compiled': proposed, 'strict_admission': False})
summary, details = [], []
for ds in ['WikiMEL', 'RichpediaMEL']:
    rows = json.loads((args.traces / f'{ds}_native_execution_traces.json').read_text())
    admitted = rejected = mismatch = 0
    for row in rows:
        rt, ss = row['roots'], row['root_scores']
        adj = {int(k): v for k, v in row['adjacency'].items()}
        ww = {int(k): v for k, v in row['weights'].items()}
        allowed = guard(rt, ss, adj, ww)
        before = all(s >= 0 for s in ss) and all(0 <= w <= ss[j] + 1e-7 for j, u in enumerate(rt) for w in ww.get(u, []))
        expected = native(-1, rt, ss, adj, ww, 21)[0][1:]
        actual = flat(rt, ss, adj, ww, 20) if allowed else expected
        assert expected == row['native_top20'] and actual == expected
        admitted += allowed
        rejected += not allowed
        reasons = []
        if any(s < 0 for s in ss): reasons.append('negative_root')
        for j, u in enumerate(rt):
            for v, w in zip(adj.get(u, []), ww.get(u, [])):
                if not 0 <= w <= ss[j]: reasons.append('self_edge' if v == u else 'nonself_edge')
        details.append({'dataset': ds, 'qi': row['qi'], 'old_guard': before, 'strict_guard': allowed, 'reasons': sorted(set(reasons))})
    summary.append({'dataset': ds, 'traces': len(rows), 'strict_admitted': admitted, 'strict_rejected': rejected, 'ordered_mismatches': mismatch, 'fallback_traces': sum(not x['name_match'] for x in rows)})
rng = np.random.default_rng(20261029)
synthetic = 0
for i in range(1000):
    rt = list(range(int(rng.integers(1, 9))))
    ss = np.round(rng.random(len(rt)), 1).tolist()
    adj = {u: rng.choice(18, size=int(rng.integers(1, 9)), replace=False).tolist() for u in rt}
    ww = {u: [ss[u] if i % 3 == 0 else float(ss[u] * rng.random()) for v in adj[u]] for u in rt}
    assert guard(rt, ss, adj, ww)
    for B in [1, 5, 10, 20]:
        assert native(-1, rt, ss, adj, ww, B + 1)[0][1:] == flat(rt, ss, adj, ww, B)
        synthetic += 1
result = {'summary': summary, 'trace_records': len(details), 'synthetic_prefix_checks': synthetic, 'tolerance_counterexamples': counterexamples, 'records': details, 'scope': 'Source-extracted strict guard and flat function; fresh traversal of saved graphs. This does not regenerate embeddings or measure latency.'}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2))
