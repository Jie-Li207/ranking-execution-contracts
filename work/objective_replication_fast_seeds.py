import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '2'
import sys, json, time, hashlib, datetime, importlib.util, warnings, argparse
from pathlib import Path
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix
from threadpoolctl import threadpool_limits
threadpool_limits(2)
p = argparse.ArgumentParser()
p.add_argument('--dataset', choices=['wikidiverse', 'wikimel'], required=True)
p.add_argument('--roots', choices=['native3', 'paper4', 'fast128'], required=True)
p.add_argument('--threshold', type=float, default=0.6)
p.add_argument('--limit', type=int, default=0)
p.add_argument('--fast-seed', type=int, default=101)
args = p.parse_args()
r = Path(__import__('os').environ.get('ICDE_STUDY_ROOT', str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(r / 'work/deps'))
import faiss
faiss.omp_set_num_threads(2)
sp = importlib.util.spec_from_file_location('native', r / 'vendor/OpenMEL-main/maximum_spanning.py')
native = importlib.util.module_from_spec(sp)
sp.loader.exec_module(native)
tag = f'{args.dataset}_{args.roots}_s{args.fast_seed}_t{args.threshold:g}_n{args.limit or 'all'}'
out = r / 'results/objective_v2' / tag
out.mkdir(parents=True, exist_ok=True)
reg = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'dataset': args.dataset, 'root_mode': args.roots, 'threshold': args.threshold, 'limit': args.limit, 'stage': 'development only; fixed operator, not full OpenMEL', 'B': [1, 5, 10, 20], 'root_budget': 10, 'neighbor_budget': 10, 'primary': ['exact objective vs greedy utility', 'candidate inclusion (not a ranking from an optimal set)', 'strict utility loss when canonical first root is required', 'best/worst gold inclusion across exactly optimal sets'], 'MILP': 'depth-2 directed arborescence, root-to-root edges removed; roots and selected child edges each cost one entity; at most one parent per child', 'strict_tolerance': 1e-07, 'timeout_s': 3, 'gold_use': 'only post-hoc optimal-set tie diagnostic, never retrieval/optimization selection', 'frozen_features': 'CLIP ViT-B32 unexpanded name+description, or already fixed FAST-MEL128 seed101 adapter embeddings', 'scope': 'paper4 uses raw query image to entity-text as proxy for caption-text, explicitly NOT paper caption expansion; only aggregation channel count matches paper', 'success': 'replicate sign of objective/relevance mismatch and certify unavoidable root-top1 exclusion; negative and unresolved cases retained'}
(out / 'protocol.json').write_text(json.dumps(reg, indent=2))
f = r / 'data' / args.dataset / 'features'
meta = json.load(open(f / 'dev_metadata.json', encoding='utf-8'))
gold = np.array([x['answer_id'] for x in meta])
selected = np.arange(len(meta))
if args.limit:
    selected = np.array(sorted(selected, key=lambda i: hashlib.sha256(str(meta[i]['id']).encode()).hexdigest())[:args.limit])

def unit(path):
    a = np.load(path).astype(np.float32)
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-08)
    return a
D = unit(f / 'entity_name_desc.npy')
if args.roots == 'native3':
    source = r / 'results' / ('coherence_audit' if args.dataset == 'wikidiverse' else 'coherence_wikimel') / 'real_operator_inputs.npz'
    a = np.load(source)
    roots = a['roots']
    rs = (1 + a['root_scores'].astype(float)) / 2
else:
    roots = []
    rs = []
    if args.roots == 'fast128':
        assert args.dataset == 'wikidiverse'
        ff = r / 'results' / ('fastmel_lowshot_resumable' if args.fast_seed == 102 else 'fastmel_lowshot') / f'b128_s{args.fast_seed}'
        E = unit(ff / 'entity_embeddings.npy')
        T = unit(ff / 'dev_embeddings.npy')
    else:
        E = D
        I = unit(f / 'entity_image.npy')
        T = unit(f / 'dev_text.npy')
        Q = unit(f / 'dev_image.npy')
        im = np.load(f / 'entity_image_mask.npy')
        qm = np.load(f / 'dev_image_mask.npy')
    for start in range(0, len(T), 24):
        S = T[start:start + 24] @ E.T
        if args.roots == 'paper4':
            it = Q[start:start + 24] @ D.T
            it[~qm[start:start + 24]] = -np.inf
            ti = T[start:start + 24] @ I.T
            ti[:, ~im] = -np.inf
            ii = Q[start:start + 24] @ I.T
            ii[:, ~im] = -np.inf
            ii[~qm[start:start + 24]] = -np.inf
            S = np.maximum.reduce([S, it, ti, ii])
        for s in S:
            ix = np.argpartition(-s, 9)[:10]
            ix = ix[np.lexsort((ix, -s[ix]))]
            roots.append(ix)
            rs.append((1 + s[ix].astype(float)) / 2)
    roots = np.asarray(roots)
    rs = np.asarray(rs)
idx = faiss.IndexFlatIP(512)
idx.add(D)
unique = np.unique(roots[selected])
dd, nn = idx.search(D[unique], 10)
dd = np.clip((1 + dd.astype(float)) / 2, 0, 1)
lookup = {int(u): i for i, u in enumerate(unique)}
np.savez_compressed(out / 'inputs.npz', roots=roots[selected], scores=rs[selected], gold=gold[selected], selected=selected, unique=unique, nn=nn, dd=dd)
records = []
tol = reg['strict_tolerance']
start_time = time.perf_counter()
for count, qi in enumerate(selected):
    rt = roots[qi].tolist()
    s = rs[qi]
    rootset = set(rt)
    adj = {u: [int(v) for v, w in zip(nn[lookup[u]], dd[lookup[u]]) if int(v) not in rootset and w > args.threshold] for u in rt}
    ww = {u: [float(w) for v, w in zip(nn[lookup[u]], dd[lookup[u]]) if int(v) not in rootset and w > args.threshold] for u in rt}
    edges = [(j, v, w) for j, u in enumerate(rt) for v, w in zip(adj[u], ww[u])]
    children = sorted(set((v for _, v, _ in edges)))
    childrow = {v: i for i, v in enumerate(children)}
    n = len(rt) + len(edges)
    A = lil_matrix((1 + len(edges) + len(children), n), dtype=float)
    A[0, :] = 1
    ub = np.ones(A.shape[0])
    lb = np.full(A.shape[0], -np.inf)
    for j, (parent, v, w) in enumerate(edges):
        A[1 + j, len(rt) + j] = 1
        A[1 + j, parent] = -1
        ub[1 + j] = 0
        A[1 + len(edges) + childrow[v], len(rt) + j] = 1
    values = np.concatenate([s, np.array([w for _, _, w in edges])])
    A = A.tocsr()
    goldvars = [j for j, u in enumerate(rt) if u == int(gold[qi])] + [len(rt) + j for j, (_, v, _) in enumerate(edges) if v == int(gold[qi])]

    def solve(B, forcefirst=False, forcegold=None):
        upper = ub.copy()
        upper[0] = B
        lo = np.zeros(n)
        hi = np.ones(n)
        constraints = [LinearConstraint(A, lb, upper)]
        if forcefirst:
            lo[0] = 1
        if forcegold is not None:
            indicator = np.zeros((1, n))
            indicator[0, goldvars] = 1
            constraints.append(LinearConstraint(indicator, int(forcegold), int(forcegold)))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            res = milp(-values, integrality=np.ones(n), bounds=Bounds(lo, hi), constraints=constraints, options={'time_limit': 3, 'mip_rel_gap': 0, 'threads': 1})
        return res
    for B in reg['B']:
        t = time.perf_counter()
        res = solve(B)
        elapsed = time.perf_counter() - t
        exact = res.status == 0 and getattr(res, 'mip_gap', 0) == 0
        sel = []
        if res.x is not None:
            sel = [rt[j] for j in range(len(rt)) if res.x[j] > 0.5] + [v for j, (_, v, _) in enumerate(edges) if res.x[len(rt) + j] > 0.5]
        g, ge = native.maximum_spanning_tree(-1, rt, s.tolist(), adj, ww, B + 1)
        gs = g[1:]
        gv = sum((z[2] for z in ge))
        ov = -res.fun if res.fun is not None else None
        row = {'qi': int(qi), 'query_id': str(meta[qi]['id']), 'B': B, 'status': int(res.status), 'exact': bool(exact), 'opt_value': ov, 'greedy_value': gv, 'solve_ms': elapsed * 1000, 'optimal_set': sel, 'greedy_set': gs, 'gold': int(gold[qi]), 'opt_hit': int(gold[qi]) in sel, 'greedy_hit': int(gold[qi]) in gs, 'root_hit': int(gold[qi]) in rt[:B], 'root_top1': rt[0], 'gold_in_graph': bool(goldvars), 'nodes': len(rt) + len(children), 'edges': len(edges)}
        if exact:
            assert ov >= gv - tol
            if rt[0] in sel:
                forced_value = ov
                forced_status = 0
            else:
                forced = solve(B, forcefirst=True)
                forced_value = -forced.fun if forced.fun is not None else None
                forced_status = int(forced.status)
            row.update(forced_first_status=forced_status, forced_first_value=forced_value, strict_first_exclusion=ov - forced_value > tol if forced_status == 0 else None)
            if not goldvars:
                low = high = False
                tie_status = 0
            else:
                alt = solve(B, forcegold=not row['opt_hit'])
                tie_status = int(alt.status)
                same = alt.status == 0 and ov + alt.fun <= tol
                if alt.status not in [0, 2]:
                    low = high = None
                else:
                    low = row['opt_hit'] and (not same)
                    high = row['opt_hit'] or same
            row.update(opt_hit_min=low, opt_hit_max=high, gold_tie_status=tie_status)
        records.append(row)
    if count % 200 == 0:
        print(tag, count, len(selected), 'seconds', round(time.perf_counter() - start_time, 1), flush=True)
    if count % 200 == 0:
        (out / 'query_results.partial.json').write_text(json.dumps(records))
summary = []
for B in reg['B']:
    rows = [z for z in records if z['B'] == B]
    ex = [z for z in rows if z['exact']]
    tie = [z for z in ex if z.get('opt_hit_min') is not None]
    forced = [z for z in ex if z.get('strict_first_exclusion') is not None]
    summary.append({'B': B, 'n': len(rows), 'exact_n': len(ex), 'greedy_suboptimal_n': sum((z['opt_value'] - z['greedy_value'] > tol for z in ex)), 'mean_relative_gap': float(np.mean([(z['opt_value'] - z['greedy_value']) / z['opt_value'] for z in ex])), 'optimal_recall': float(np.mean([z['opt_hit'] for z in ex])), 'optimal_recall_min': float(np.mean([z['opt_hit_min'] for z in tie])), 'optimal_recall_max': float(np.mean([z['opt_hit_max'] for z in tie])), 'gold_tie_certified_n': len(tie), 'greedy_recall': float(np.mean([z['greedy_hit'] for z in ex])), 'root_recall_KminB10': float(np.mean([z['root_hit'] for z in ex])), 'strict_first_exclusion_n': sum((z['strict_first_exclusion'] for z in forced)), 'forced_first_certified_n': len(forced), 'solve_p50_ms': float(np.median([z['solve_ms'] for z in ex]))})
(out / 'query_results.json').write_text(json.dumps(records))
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
