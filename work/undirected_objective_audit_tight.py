import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '2'
import json, time, argparse, datetime, hashlib
from pathlib import Path
import numpy as np
from undirected_tree_oracle_tight import solve_tree, prim_prefix, verify_small
p = argparse.ArgumentParser()
p.add_argument('--input', required=True)
p.add_argument('--limit', type=int, default=128)
args = p.parse_args()
r = Path(__import__('os').environ.get('ICDE_STUDY_ROOT', str(Path(__file__).resolve().parents[1])))
src = Path(args.input)
a = dict(np.load(src / 'inputs.npz'))
out = r / 'results/undirected_tight' / src.name
out.mkdir(parents=True, exist_ok=True)
reg = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'input': str(src), 'query_selection': 'first128 by SHA256(query_id) from same fixed development input, independent of labels', 'limit': args.limit, 'budgets': [1, 5, 10, 20], 'threshold': json.load(open(src / 'protocol.json'))['threshold'], 'oracle': 'undirected bounded rooted tree via single-commodity flow and edge_count=node_count; standard MILP', 'primary': ['certified utility gap', 'gold inclusion', 'strict exclusion of initial top1'], 'timeouts': '5s per solve; retain gap/bound, never treat timeout as optimum'}
(out / 'protocol.json').write_text(json.dumps(reg, indent=2))
(out / 'oracle_validation.json').write_text(json.dumps(verify_small(cases=50), indent=2))
orig = json.load(open(src / 'query_results.json'))
idmap = {z['qi']: z['query_id'] for z in orig}
ix = sorted(range(len(a['selected'])), key=lambda i: hashlib.sha256(idmap[int(a['selected'][i])].encode()).hexdigest())[:args.limit]
lookup = {int(u): j for j, u in enumerate(a['unique'])}
records = []
for count, i in enumerate(ix):
    rt = a['roots'][i].tolist()
    ss = a['scores'][i]
    rootset = set(rt)
    edges = [(-1, u, float(s)) for u, s in zip(rt, ss)]
    edgedict = {}
    for u in rt:
        j = lookup[u]
        for v, w in zip(a['nn'][j], a['dd'][j]):
            v = int(v)
            if v in rootset or w <= reg['threshold']:
                continue
            key = tuple(sorted([u, v]))
            edgedict[key] = max(edgedict.get(key, 0), float(w))
    edges.extend(((u, v, w) for (u, v), w in edgedict.items()))
    nodes = sorted({x for u, v, w in edges for x in [u, v]} - {-1})
    gold = int(a['gold'][i])
    for B in reg['budgets']:
        t = time.perf_counter()
        res, sel = solve_tree(nodes, edges, B)
        elapsed = time.perf_counter() - t
        greedy, gv = prim_prefix(edges, B)
        row = {'qi': int(a['selected'][i]), 'B': B, 'status': int(res.status), 'gap': float(getattr(res, 'mip_gap', np.nan)), 'value': -res.fun if res.fun is not None else None, 'dual_upper': -float(res.mip_dual_bound) if getattr(res, 'mip_dual_bound', None) is not None else None, 'set': sel, 'gold_hit': gold in sel, 'greedy_hit': gold in greedy, 'greedy_value': gv, 'root_hit': gold in rt[:B], 'solve_ms': elapsed * 1000}
        if res.status == 0:
            assert -res.fun >= gv - 1e-07
            if rt[0] in sel:
                row['strict_first_exclusion'] = False
            else:
                forced, _ = solve_tree(nodes, edges, B, required=[rt[0]])
                row['forced_status'] = int(forced.status)
                row['strict_first_exclusion'] = forced.fun - res.fun > 1e-07 if forced.status == 0 else None
        records.append(row)
    print(src.name, count + 1, len(ix), flush=True)
    if count % 8 == 0:
        (out / 'query_results.partial.json').write_text(json.dumps(records))
(out / 'query_results.json').write_text(json.dumps(records))
summary = []
for B in reg['budgets']:
    rows = [z for z in records if z['B'] == B]
    ex = [z for z in rows if z['status'] == 0]
    summary.append({'B': B, 'n': len(rows), 'exact_n': len(ex), 'greedy_suboptimal': sum((z['value'] - z['greedy_value'] > 1e-07 for z in ex)), 'exact_recall': float(np.mean([z['gold_hit'] for z in ex])), 'greedy_recall_exact_subset': float(np.mean([z['greedy_hit'] for z in ex])), 'root_recall_exact_subset': float(np.mean([z['root_hit'] for z in ex])), 'strict_first_exclusion_n': sum((z.get('strict_first_exclusion') is True for z in ex)), 'unresolved_first_n': sum((z.get('strict_first_exclusion') is None for z in ex))})
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
