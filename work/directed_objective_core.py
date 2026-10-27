import json, time, warnings
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix

def run_directed(roots, rs, gold, unique, dd, nn, meta, selected, threshold, reg, out, native, tag):
    lookup = {int(u): i for i, u in enumerate(unique)}
    records = []
    tol = reg['strict_tolerance']
    start_time = time.perf_counter()
    for count, qi in enumerate(selected):
        keep = np.flatnonzero(rs[qi] > threshold)
        rt = roots[qi][keep].tolist()
        s = rs[qi][keep]
        assert len(rt) > 0, 'No retained root: explicit empty-result handler required'
        rootset = set(rt)
        adj = {u: [int(v) for v, w in zip(nn[lookup[u]], dd[lookup[u]]) if int(v) not in rootset and w > threshold] for u in rt}
        ww = {u: [float(w) for v, w in zip(nn[lookup[u]], dd[lookup[u]]) if int(v) not in rootset and w > threshold] for u in rt}
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
