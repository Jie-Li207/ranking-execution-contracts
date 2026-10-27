import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '2'
import json, time, hashlib, datetime, sys
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from enumeration_oracle import enumerate_roots
threadpool_limits(2)
r = Path(os.environ.get('ICDE_STUDY_ROOT', str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(r / 'work/deps'))
import faiss
faiss.omp_set_num_threads(2)
out = r / 'results/revision_review'
grid = [0, 0.25, 0.5, 0.75, 1]

def write(name, obj):
    (out / name).write_text(json.dumps(obj, indent=2), encoding='utf-8')
write('joint_calibration_protocol.json', {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'reason': 'Query-only probability calibration leaves cosine coherence uncalibrated; add a shared gold-event model for both root and child edges.', 'features': ['destination query score', 'parent query score', 'entity cosine weight (0 for query-root)', 'is child edge'], 'model': 'StandardScaler plus LogisticRegression C=1, no class reweighting, per-entity incoming-edge weights sum to1', 'labels': 'same128 fixed hashed WikiDiverse training queries as scalar calibration', 'lambda_grid': grid, 'selection': 'same even-hash WD dev half, maximize B5, smallest lambda wins ties', 'evaluation': 'odd-hash WD dev and all author-cache queries, no test selection', 'status': 'post-review exploratory, not fresh independent confirmation', 'B': [5, 10], 'neighbors': 10})

def unit(p):
    a = np.load(p).astype(np.float32)
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-08)
    return a
f = r / 'data/wikidiverse/features'
D = unit(f / 'entity_name_desc.npy')
I = unit(f / 'entity_image.npy')
T = unit(f / 'train_text.npy')
Q = unit(f / 'train_image.npy')
im = np.load(f / 'entity_image_mask.npy')
qm = np.load(f / 'train_image_mask.npy')
meta = json.load(open(f / 'train_metadata.json', encoding='utf-8'))
ca = json.load(open(out / 'calibration.json'))
idx = faiss.IndexFlatIP(512)
idx.add(D)
xx = []
yy = []
sw = []
for entry in ca['queries']:
    qi = entry['qi']
    g = int(meta[qi]['answer_id'])
    s = np.maximum(T[qi] @ D.T, np.where(im, T[qi] @ I.T, -np.inf))
    if qm[qi]:
        s = np.maximum(s, Q[qi] @ D.T)
    s = np.clip((1 + s.astype(float)) / 2, 0, 1)
    rt = np.argpartition(-s, 9)[:10]
    rt = rt[np.lexsort((rt, -s[rt]))]
    dd, nn = idx.search(D[rt], 10)
    ed = [(int(u), int(v), float((1 + w) / 2)) for u, vs, ws in zip(rt, nn, dd) for v, w in zip(vs, ws) if v not in rt and (1 + w) / 2 > 0.6]
    freq = {v: sum((t == v for p, t, w in ed)) for u, v, c in ed}
    for u in rt:
        xx.append([s[u], s[u], 0, 0])
        yy.append(u == g)
        sw.append(1.0)
    for u, v, c in ed:
        xx.append([s[v], s[u], c, 1])
        yy.append(v == g)
        sw.append(1 / freq[v])
sc = StandardScaler().fit(xx, sample_weight=sw)
clf = LogisticRegression(C=1, solver='lbfgs', tol=1e-10, max_iter=2000).fit(sc.transform(xx), yy, sample_weight=sw)
write('joint_calibration_model.json', {'feature_mean': sc.mean_.tolist(), 'feature_scale': sc.scale_.tolist(), 'coefficient': clf.coef_[0].tolist(), 'intercept': clf.intercept_.tolist(), 'training_rows': len(xx), 'weighted_positive_rows': float(np.sum(np.asarray(sw) * yy)), 'query_label_count': 128, 'C': 1})
pred = lambda x: clf.predict_proba(sc.transform(x))[:, 1]
scalar = lambda s: expit(ca['coefficient'] * np.asarray(s) + ca['intercept'])
del D, I, T, Q, idx
start = time.perf_counter()
summaries = []
choice = None
for ds, folder, parent in [('WikiDiverse', 'wikidiverse_native3_t0.6_nall', 'objective_v2'), ('WikiMEL', 'WikiMEL_author3_t0.6_nall', 'author_objective'), ('RichpediaMEL', 'RichpediaMEL_author3_t0.6_nall', 'author_objective')]:
    a = dict(np.load(r / 'results' / parent / folder / 'inputs.npz'))
    lookup = {int(u): i for i, u in enumerate(a['unique'])}
    scores = {x['qi']: dict(zip(x['nodes'], x['query_scores'])) for x in json.load(open(out / (ds + '_candidate_scores.json')))}
    prev = json.load(open(out / (ds + '_query_results.json')))
    metadata = {z['qi']: {k: z[k] for k in ['query_id', 'cluster', 'partition', 'gold']} for z in prev}
    del prev
    rows = []
    for i, qi in enumerate(a['selected']):
        qi = int(qi)
        rt = a['roots'][i].astype(int).tolist()
        rs = a['scores'][i].astype(float)
        sq = scores[qi]
        ed = [(u, int(v), float(w)) for u in rt for v, w in zip(a['nn'][lookup[u]], a['dd'][lookup[u]]) if int(v) not in rt and w > 0.6]
        rootp = pred([[s, s, 0, 0] for s in rs])
        childp = pred([[sq[v], sq[u], w, 1] for u, v, w in ed]) if ed else []
        base = metadata[qi]
        for lam in grid:
            ss = (1 - lam) * scalar(rs) + lam * rootp
            es = [(u, v, float((1 - lam) * scalar(sq[v]) + lam * p)) for (u, v, w), p in zip(ed, childp)]
            sol = enumerate_roots(rt, ss, es, [5, 10], gold=base['gold'])
            for z in sol:
                rows.append(dict(base, dataset=ds, qi=qi, method='joint_calibrated', param=lam, B=z['B'], hit=base['gold'] in z['set'], root_hit=base['gold'] in rt[:z['B']], strict=z['strict_first_exclusion'], dominance=all((w <= ss[rt.index(u)] + 1e-07 for u, v, w in es)), selected=z['set'], gold_can=z['gold_can_appear'], gold_must=z['gold_must_appear']))
        if i % 1000 == 0:
            print(ds, i, round(time.perf_counter() - start, 1), flush=True)
    if ds == 'WikiDiverse':
        values = [(np.mean([x['hit'] for x in rows if x['partition'] == 'select' and x['B'] == 5 and (x['param'] == lam)]), lam) for lam in grid]
        choice = max(values, key=lambda z: (z[0], -z[1]))[1]
        write('joint_selected_parameter.json', {'lambda': choice, 'selection_values': values})
    for B in [5, 10]:
        for lam in grid:
            z = [x for x in rows if x['partition'] == 'evaluate' and x['B'] == B and (x['param'] == lam)]
            x = np.asarray([v['hit'] for v in z], float)
            b = np.asarray([v['root_hit'] for v in z], float)
            _, ix = np.unique([v['cluster'] for v in z], return_inverse=True)
            G = ix.max() + 1
            cnt = np.bincount(ix)
            dif = np.bincount(ix, weights=x - b)
            rng = np.random.default_rng(20261026)
            draw = rng.integers(G, size=(2000, G))
            boot = dif[draw].sum(1) / cnt[draw].sum(1)
            summaries.append({'dataset': ds, 'method': 'joint_calibrated', 'B': B, 'param': lam, 'n': len(z), 'recall': float(x.mean()), 'root_recall': float(b.mean()), 'delta_pp': float(100 * (x - b).mean()), 'delta_CI95_pp': (100 * np.quantile(boot, [0.025, 0.975])).tolist(), 'strict_rate': float(np.mean([v['strict'] for v in z])), 'dominance_rate': float(np.mean([v['dominance'] for v in z])), 'dev_selected': lam == choice})
    write(ds + '_joint_query_results.json', rows)
write('joint_summary.json', summaries)
write('joint_validation.json', {'complete': True, 'elapsed_s': time.perf_counter() - start, 'selected_lambda': choice})
print('JOINT COMPLETE', choice, flush=True)
