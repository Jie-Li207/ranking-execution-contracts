import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '2'
from pathlib import Path
import json, time, hashlib, datetime, ast, heapq
import numpy as np
from scipy.special import expit
from scipy.linalg import solve
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from enumeration_oracle import enumerate_roots
threadpool_limits(2)
r = Path(os.environ.get('ICDE_STUDY_ROOT', str(Path(__file__).resolve().parents[1])))
out = r / 'results/revision_review'
out.mkdir(parents=True, exist_ok=True)

def write(name, obj):
    (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def h(v):
    return hashlib.sha256(str(v).encode()).hexdigest()
protocol = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'status': 'post-review exploratory extension; original test results already known, not a new preregistered confirmation', 'fixed_before_new_outputs': True, 'B': [5, 10], 'neighbor_counts': [5, 10], 'thresholds': [0.55, 0.6, 0.65], 'blend_lambda': [0, 0.25, 0.5, 0.75, 1], 'pagerank_alpha': [0, 0.15, 0.5, 0.85], 'label_budget': 128, 'calibrator': 'single monotone logistic map from normalized three-channel max score to gold probability, 128 WikiDiverse train queries, up to100 hard candidates/query; no class reweighting', 'selection': 'lambda or alpha selected by B5 gold inclusion on even SHA256 parity WikiDiverse development; smallest parameter wins exact ties; remaining WD dev and all author-cache queries evaluate fixed choices', 'root_scores': 'unchanged graph root order; original raw, monotone calibrated, and parent attenuation variants named separately', 'sampling': 'K10 full sets; K5 fixed first512 hash-query IDs per dataset; threshold structure first256 fixed IDs for each original sensitivity run', 'statistics': 'paired image-cluster bootstrap 2000 reps, seed20261026; descriptive intervals, not fresh confirmatory p-values', 'scope': 'additional graph operator, not reproduction of a second published MEL system', 'threshold_case': 'explicitly measure edges/reachability/optimal set changes; report ineffective sweeps as such'}
write('protocol.json', protocol)
source = r / 'vendor/OpenMEL-main/maximum_spanning.py'
t = ast.parse(source.read_text(encoding='utf-8'))
ns = {'heapq': heapq}
exec(compile(ast.Module(body=[n for n in t.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in ['MaxHeap', 'find', 'union', 'maximum_spanning_tree']], type_ignores=[]), str(source), 'exec'), ns)
native = ns['maximum_spanning_tree']

def unit(p):
    x = np.load(p).astype(np.float32)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-08)
    return x

def arrays(ds, train=False):
    if ds == 'WikiDiverse':
        f = r / 'data/wikidiverse/features'
        split = 'train' if train else 'dev'
        names = ['entity_name_desc', 'entity_image', split + '_text', split + '_image']
        im = np.load(f / 'entity_image_mask.npy')
        qm = np.load(f / (split + '_image_mask.npy'))
        meta = json.load(open(f / (split + '_metadata.json'), encoding='utf-8'))
    else:
        f = r / 'data/author_cache' / ds
        names = ['entity_text', 'entity_image', 'query_text', 'query_image']
        im = np.load(f / 'entity_image_mask.npy')
        qm = np.load(f / 'query_image_mask.npy')
        meta = json.load(open(f / 'queries.json', encoding='utf-8'))
    return (*[unit(f / (n + '.npy')) for n in names], im, qm, meta)

def score(D, I, T, Q, im, qm, qi, entities):
    s = np.maximum(T[qi] @ D[entities].T, np.where(im[entities], T[qi] @ I[entities].T, -np.inf))
    if qm[qi]:
        s = np.maximum(s, Q[qi] @ D[entities].T)
    return np.clip((1 + s.astype(float)) / 2, 0, 1)
nat = []
changes = []
for ds in ['WikiMEL', 'RichpediaMEL']:
    records = json.load(open(r / 'results/author_reproduction' / ds / 'query_results.json'))
    qmeta = json.load(open(r / 'data/author_cache' / ds / 'queries.json', encoding='utf-8'))
    for subset in ['fallback', 'name_match']:
        zz = [z for z in records if (not z['name_filter_matched']) == (subset == 'fallback')]
        for B in [1, 5, 10, 20]:
            gained = [z for z in zz if z['native_rank'] <= B < z['no_graph_rank']]
            lost = [z for z in zz if z['no_graph_rank'] <= B < z['native_rank']]
            nat.append({'dataset': ds, 'subset': subset, 'B': B, 'n': len(zz), 'native_hits': sum((z['native_rank'] <= B for z in zz)), 'off_hits': sum((z['no_graph_rank'] <= B for z in zz)), 'gained': len(gained), 'lost': len(lost), 'ordered_prefix_changes': sum((z['native_top20'][:B] != z['no_graph_top20'][:B] for z in zz)), 'set_changes': sum((set(z['native_top20'][:B]) != set(z['no_graph_top20'][:B]) for z in zz))})
            for z in gained + lost:
                changes.append({'dataset': ds, 'B': B, 'qi': z['qi'], 'native_rank': z['native_rank'], 'off_rank': z['no_graph_rank'], 'mention': qmeta[z['qi']].get('mentions'), 'query_id': z['query_id'], 'has_image': bool(qmeta[z['qi']].get('imgPath'))})
write('native_conditional.json', nat)
write('native_changed_cases.json', changes)
print('Native conditional done', flush=True)
struct = []
for ds in ['WikiMEL', 'RichpediaMEL']:
    for mode in [3, 4]:
        folders = [f'{ds}_author{mode}_t{x}_n256' for x in ['0.55', '0.65']]
        aa = [dict(np.load(r / 'results/author_objective' / f / 'inputs.npz')) for f in folders]
        maps = [{int(q): i for i, q in enumerate(a['selected'])} for a in aa]
        look = [{int(q): i for i, q in enumerate(a['unique'])} for a in aa]
        rows = [{(z['qi'], z['B']): z for z in json.load(open(r / 'results/enumerated' / f / 'query_results.json'))} for f in folders]
        totaledges = [0, 0]
        totnodes = [0, 0]
        components = [0, 0]
        changed = 0
        rootchanged = 0
        optchanged = {5: 0, 10: 0}
        weightchanged = {5: 0, 10: 0}
        for qi in maps[0]:
            sets = []
            for k, th in enumerate([0.55, 0.65]):
                a = aa[k]
                i = maps[k][qi]
                rt = [int(u) for u, s in zip(a['roots'][i], a['scores'][i]) if s > th]
                ed = {(u, int(v)) for u in rt for v, w in zip(a['nn'][look[k][u]], a['dd'][look[k][u]]) if int(v) not in rt and w > th}
                nodes = set(rt) | {v for u, v in ed}
                totaledges[k] += len(ed)
                totnodes[k] += len(nodes)
                parent = {v: v for v in nodes}

                def find(v):
                    while parent[v] != v:
                        v = parent[v]
                    return v
                for u, v in ed:
                    parent[find(v)] = find(u)
                components[k] += len({find(v) for v in nodes})
                sets.append((set(rt), ed, nodes))
            changed += sets[0] != sets[1]
            rootchanged += sets[0][0] != sets[1][0]
            for B in [5, 10]:
                x, y = [m[qi, B] for m in rows]
                optchanged[B] += set(x['optimal_set']) != set(y['optimal_set'])
                weightchanged[B] += abs(x['opt_value'] - y['opt_value']) > 1e-07
        struct.append({'dataset': ds, 'mode': mode, 'n': len(maps[0]), 'thresholds': [0.55, 0.65], 'entity_edges_total': totaledges, 'reachable_entities_total': totnodes, 'components_without_query_total': components, 'graphs_changed': changed, 'root_sets_changed': rootchanged, 'optimal_sets_changed': optchanged, 'optimal_values_changed': weightchanged})
write('threshold_structure.json', struct)
print('Threshold structure done', flush=True)
D, I, T, Q, im, qm, meta = arrays('WikiDiverse', train=True)
ids = sorted(range(len(T)), key=lambda i: h(meta[i]['id']))[:128]
X = []
Y = []
trainrows = []
allids = np.arange(len(D))
for qi in ids:
    s = score(D, I, T, Q, im, qm, qi, allids)
    ix = np.argpartition(-s, 99)[:100]
    ix = ix[np.lexsort((ix, -s[ix]))]
    gold = int(meta[qi]['answer_id'])
    X.extend(s[ix])
    Y.extend(ix == gold)
    trainrows.append({'query_id': meta[qi]['id'], 'qi': qi, 'gold_in_candidates': bool(gold in ix)})
clf = LogisticRegression(C=1000, solver='lbfgs', max_iter=2000, tol=1e-10).fit(np.array(X)[:, None], np.array(Y))
coef = float(clf.coef_[0, 0])
intercept = float(clf.intercept_[0])
assert coef > 0
write('calibration.json', {'coefficient': coef, 'intercept': intercept, 'labeled_queries': 128, 'candidate_rows': len(X), 'positive_rows': int(sum(Y)), 'queries': trainrows, 'note': 'Single-score calibration, not an entity linker novelty. Evaluated across independently encoded and author-cache domains without test fitting.'})
cal = lambda s: expit(coef * np.asarray(s) + intercept)
del D, I, T, Q

def pagerank(rt, ss, ed, alpha):
    nodes = sorted(set(rt) | {v for u, v, w in ed})
    ix = {u: j for j, u in enumerate(nodes)}
    p = np.zeros(len(nodes))
    p[[ix[u] for u in rt]] = ss / np.sum(ss)
    W = np.zeros((len(nodes), len(nodes)))
    for u, v, w in ed:
        W[ix[u], ix[v]] = max(W[ix[u], ix[v]], w)
        W[ix[v], ix[u]] = max(W[ix[v], ix[u]], w)
    deg = W.sum(1)
    P = W / np.maximum(deg[:, None], 1e-30)
    P[deg == 0] = p
    z = solve(np.eye(len(nodes)) - alpha * P.T, (1 - alpha) * p, assume_a='gen')
    assert np.max(np.abs(z - ((1 - alpha) * p + alpha * z @ P))) < 1e-09
    order = np.lexsort((np.asarray(nodes), -z))
    return ([nodes[j] for j in order], P, p, z, nodes)
allsummary = []
choices = {}
networkxchecks = 0
start = time.perf_counter()
for ds, folder, parent in [('WikiDiverse', 'wikidiverse_native3_t0.6_nall', 'objective_v2'), ('WikiMEL', 'WikiMEL_author3_t0.6_nall', 'author_objective'), ('RichpediaMEL', 'RichpediaMEL_author3_t0.6_nall', 'author_objective')]:
    a = dict(np.load(r / 'results' / parent / folder / 'inputs.npz'))
    lookup = {int(u): j for j, u in enumerate(a['unique'])}
    D, I, T, Q, im, qm, meta = arrays(ds)
    records = []
    payload = []
    hashids = sorted(range(len(a['selected'])), key=lambda i: h(meta[int(a['selected'][i])].get('id', int(a['selected'][i]))))
    k5set = set(hashids[:512])
    maxscoreerr = 0
    for i, qi in enumerate(a['selected']):
        qi = int(qi)
        rt = a['roots'][i].astype(int).tolist()
        s = a['scores'][i].astype(float)
        gold = int(a['gold'][i])
        qid = str(meta[qi].get('id', qi))
        cluster = meta[qi].get('imgPath') or 'missing-' + qid
        partition = 'select' if ds == 'WikiDiverse' and int(h(qid), 16) % 2 == 0 else 'evaluate'
        ed = [(u, int(v), float(w), j) for u in rt for j, (v, w) in enumerate(zip(a['nn'][lookup[u]], a['dd'][lookup[u]])) if int(v) not in rt and w > 0.6]
        nodes = sorted(set(rt) | {v for u, v, w, j in ed})
        cs = score(D, I, T, Q, im, qm, qi, np.asarray(nodes))
        scoremap = dict(zip(nodes, cs))
        err = max((abs(scoremap[u] - x) for u, x in zip(rt, s)))
        maxscoreerr = max(maxscoreerr, err)
        assert err < 1e-05, (ds, qi, err)
        scoremap.update(zip(rt, s))
        payload.append({'qi': qi, 'nodes': nodes, 'query_scores': [float(scoremap[v]) for v in nodes]})
        for K in [10, 5] if i in k5set else [10]:
            edges = [(u, v, w) for u, v, w, j in ed if j < K]
            parents = dict(zip(rt, s))
            base = {'dataset': ds, 'qi': qi, 'query_id': qid, 'cluster': cluster, 'partition': partition, 'K': K, 'gold': gold}
            variants = [('attenuated', None, s, [(u, v, parents[u] * w) for u, v, w in edges]), ('clipped', None, s, [(u, v, min(parents[u], w)) for u, v, w in edges])]
            for typ, fn in [('blend_raw', lambda x: np.asarray(x)), ('blend_calibrated', cal)]:
                rs = fn(s)
                for lam in [0, 0.25, 0.5, 0.75, 1]:
                    variants.append((typ, lam, rs, [(u, v, float((1 - lam) * fn(scoremap[v]) + lam * w)) for u, v, w in edges]))
            for name, param, rs, es in variants:
                opts = enumerate_roots(rt, rs, es, [5, 10], gold=gold)
                for z in opts:
                    records.append(dict(base, method=name, param=param, B=z['B'], hit=gold in z['set'], root_hit=gold in rt[:z['B']], strict=z['strict_first_exclusion'], gold_can=z['gold_can_appear'], gold_must=z['gold_must_appear'], dominance=all((w <= rs[rt.index(u)] + 1e-07 for u, v, w in es)), selected=z['set'], objective=z['value']))
                if name in ['attenuated', 'clipped']:
                    adj = {u: [v for p, v, w in es if p == u] for u in rt}
                    ww = {u: [w for p, v, w in es if p == u] for u in rt}
                    got, e = native(-1, rt, rs.tolist(), adj, ww, 11)
                    for z in opts:
                        assert abs(sum((w for u, v, w in e[:z['B']])) - z['value']) < 1e-07
            for alpha in [0, 0.15, 0.5, 0.85]:
                rank, P, p, z, nd = pagerank(rt, cal(s), edges, alpha)
                if ds == 'WikiDiverse' and alpha == 0.85 and (K == 10) and (networkxchecks < 32):
                    import networkx as nx
                    G = nx.DiGraph()
                    G.add_nodes_from(nd)
                    for u in range(len(nd)):
                        for v in np.flatnonzero(P[u]):
                            G.add_edge(nd[u], nd[v], weight=float(P[u, v]))
                    chk = nx.pagerank(G, alpha=alpha, personalization=dict(zip(nd, p)), tol=1e-13, max_iter=2000)
                    assert max((abs(chk[v] - z[j]) for j, v in enumerate(nd))) < 1e-09
                    networkxchecks += 1
                for B in [5, 10]:
                    records.append(dict(base, method='pagerank', param=alpha, B=B, hit=gold in rank[:B], root_hit=gold in rt[:B], top1_changed=rank[0] != rt[0], selected=rank[:B]))
        if i % 500 == 0:
            print(ds, i, 'elapsed', round(time.perf_counter() - start, 1), flush=True)
    write(ds + '_candidate_scores.json', payload)
    write(ds + '_query_results.json', records)
    if ds == 'WikiDiverse':
        for meth in ['blend_raw', 'blend_calibrated', 'pagerank']:
            grid = protocol['pagerank_alpha'] if meth == 'pagerank' else protocol['blend_lambda']
            vals = []
            for par in grid:
                zz = [z for z in records if z['partition'] == 'select' and z['K'] == 10 and (z['B'] == 5) and (z['method'] == meth) and (z['param'] == par)]
                vals.append((sum((z['hit'] for z in zz)) / len(zz), par))
            choices[meth] = max(vals, key=lambda z: (z[0], -z[1]))[1]
        write('selected_parameters.json', choices)
    summary = []
    for K in [5, 10]:
        for meth in ['attenuated', 'clipped', 'blend_raw', 'blend_calibrated', 'pagerank']:
            pars = [None] if meth in ['attenuated', 'clipped'] else protocol['pagerank_alpha'] if meth == 'pagerank' else protocol['blend_lambda']
            for par in pars:
                for B in [5, 10]:
                    zz = [z for z in records if z['K'] == K and z['B'] == B and (z['method'] == meth) and (z['param'] == par) and (z['partition'] == 'evaluate')]
                    x = np.array([z['hit'] for z in zz], float)
                    b = np.array([z['root_hit'] for z in zz], float)
                    _, ix = np.unique([z['cluster'] for z in zz], return_inverse=True)
                    G = ix.max() + 1
                    cnt = np.bincount(ix)
                    diff = np.bincount(ix, weights=x - b)
                    rng = np.random.default_rng(20261026)
                    draw = rng.integers(G, size=(2000, G))
                    boot = diff[draw].sum(1) / cnt[draw].sum(1)
                    row = {'dataset': ds, 'K': K, 'method': meth, 'param': par, 'B': B, 'n': len(zz), 'root_recall': float(b.mean()), 'recall': float(x.mean()), 'delta_pp': float(100 * (x - b).mean()), 'delta_CI95_pp': (100 * np.quantile(boot, [0.025, 0.975])).tolist(), 'dev_selected': par == choices.get(meth), 'strict_rate': float(np.mean([z['strict'] for z in zz])) if 'strict' in zz[0] else None, 'dominance_rate': float(np.mean([z['dominance'] for z in zz])) if 'dominance' in zz[0] else None, 'top1_change_rate': float(np.mean([z['top1_changed'] for z in zz])) if 'top1_changed' in zz[0] else None}
                    summary.append(row)
    write(ds + '_summary.json', summary)
    allsummary.extend(summary)
    write('runtime_' + ds + '.json', {'max_root_score_error': maxscoreerr, 'elapsed_from_start_s': time.perf_counter() - start, 'n_queries': len(a['selected'])})
    del D, I, T, Q
write('summary.json', allsummary)
write('validation.json', {'pagerank_networkx_cases': networkxchecks, 'dominance_native_objective_checks': 'all attenuated/clipped instances at both budgets', 'complete': True, 'elapsed_s': time.perf_counter() - start})
print('COMPLETE', choices, round(time.perf_counter() - start, 1), flush=True)
