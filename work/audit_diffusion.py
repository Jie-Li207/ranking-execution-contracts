import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '2'
os.environ.pop('CUDA_VISIBLE_DEVICES', None)
from pathlib import Path
import sys, json, time, pickle, hashlib, types, argparse
import numpy as np
r = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(r / 'work/deps'), str(r / 'vendor/diffusion')]
import faiss, scipy.sparse as sparse, scipy.sparse.linalg as sla
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
threadpool_limits(2)
faiss.omp_set_num_threads(2)
arg = argparse.ArgumentParser()
arg.add_argument('--dataset', choices=['oxford5k', 'paris6k'], required=True)
arg.add_argument('--tight', action='store_true')
arg.add_argument('--certified', action='store_true')
args = arg.parse_args()
ds = args.dataset
out = r / 'results/rev3' / (ds + ('_certified' if args.certified else '_tight' if args.tight else ''))
out.mkdir(parents=True, exist_ok=True)
dat = r / 'data/rev3_visual'
source = r / 'vendor/diffusion/diffusion.py'
original = source.read_text()
code = original.replace('tol=1e-6', 'rtol=1e-6, atol=0').replace('n_jobs=-1', 'n_jobs=2')
if args.tight or args.certified:
    code = code.replace('rtol=1e-6', 'rtol=1e-8').replace('maxiter=20', 'maxiter=200')
module = types.ModuleType('native_diffusion')
exec(compile(code, str(source), 'exec'), module.__dict__)
ev = r / 'vendor/diffusion/evaluate.py'
e = types.ModuleType('official_evaluate')
exec(compile(ev.read_text().replace('np.in1d(', 'np.isin('), str(ev), 'exec'), e.__dict__)
record = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'patches': ['scipy tol -> rtol, atol=0', 'joblib n_jobs=-1 -> 2', 'numpy in1d -> isin evaluator'], 'dataset': ds, 'tight_numerical_control': args.tight, 'certified_direct_fallback': args.certified, 'runs': []}
if (dat / (ds + '_query.npy')).exists() and (dat / (ds + '_gallery.npy')).exists():
    Q = np.load(dat / (ds + '_query.npy'))
    G = np.load(dat / (ds + '_gallery.npy'))
else:
    import h5py
    with h5py.File(dat / f'{ds}_siamac.mat') as f:
        Q = np.squeeze(np.array([f[x[0]][:] for x in f['/glob/Q']]), axis=1).astype(np.float32)
        G = np.squeeze(np.array([f[x[0]][:] for x in f['/glob/V']]), axis=1).astype(np.float32)
    np.save(dat / (ds + '_query.npy'), Q)
    np.save(dat / (ds + '_gallery.npy'), G)
gnd = pickle.load(open(dat / f'gnd_{ds}.pkl', 'rb'))
n = len(Q)
N = len(G)
order = sorted(range(n), key=lambda i: hashlib.sha256(gnd['qimlist'][i].encode()).hexdigest())
groups = [order[j::4] for j in range(4)]
(out / 'workload.json').write_text(json.dumps({'queries': n, 'gallery': N, 'dimension': Q.shape[1], 'cohorts': groups, 'query_names': gnd['qimlist'], 'params': {'truncation': 1000, 'kd': 50, 'kq': 10, 'alpha': 0.99, 'gamma': 3}}, indent=2))
base_cg = sla.cg
cglog = []

def audited_cg(A, b, **kwargs):
    x, info = base_cg(A, b, **kwargs)
    res = np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-30)
    fallback = False
    if args.certified and (info != 0 or res > 1e-08):
        x = sla.spsolve(A.astype(np.float64).tocsc(), b)
        res = np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-30)
        assert np.isfinite(res) and res <= 1e-08
        info = 0
        fallback = True
    cglog.append((int(info), float(res), fallback))
    return (x, info)
sla.cg = audited_cg

def graph_run(ids, tag, cache=None):
    global cglog
    cglog = []
    F = np.vstack([Q[ids], G])
    cache = cache or out / ('cache_' + tag)
    cache.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    D = module.Diffusion(F, str(cache))
    O = D.get_offline_results(1000, 50)
    build = time.perf_counter() - t
    t = time.perf_counter()
    Z = normalize(O, norm='l2', axis=1)
    scores = (Z[:len(ids)] @ Z[len(ids):].T).toarray()
    ranks = np.argsort(-scores, axis=1)
    search = time.perf_counter() - t
    sim, nn = D.knn.search(F, 50)
    A = D.get_affinity(sim.copy(), nn)
    gal = A[len(ids):, len(ids):].tocsr()
    edgekeys = gal.tocoo().row.astype(np.int64) * N + gal.tocoo().col
    np.savez_compressed(out / (tag + '_results.npz'), ids=ids, ranks=ranks.astype(np.int32), top_scores=np.take_along_axis(scores, ranks[:, :20], axis=1), gallery_edges=np.sort(edgekeys), gallery_degrees=np.asarray(A.sum(1)).ravel()[len(ids):])
    m, ap, _, _ = e.compute_map(ranks.T, [gnd['gnd'][i] for i in ids])
    np.save(out / (tag + '_ap.npy'), ap)
    v = {'tag': tag, 'query_ids': ids, 'mAP': float(m), 'build_s': build, 'query_stage_s': search, 'graph_edges_directed': int(A.nnz), 'graph_memory_bytes': sum((x.nbytes for x in [A.data, A.indices, A.indptr])), 'offline_memory_bytes': sum((x.nbytes for x in [O.data, O.indices, O.indptr])), 'cg_solves': len(cglog), 'cg_nonconverged': sum((x[0] != 0 for x in cglog)), 'max_relative_residual': max((x[1] for x in cglog), default=None), 'self_not_first': int(np.sum(nn[:, 0] != np.arange(len(F))))}
    v['direct_solver_fallbacks'] = sum((x[2] for x in cglog))
    v['asymmetric_affinity_entries'] = int((A - A.T).nnz)
    record['runs'].append(v)
    (out / 'summary_partial.json').write_text(json.dumps(record, indent=2))
    print('DONE', json.dumps(v), flush=True)
    return (ranks, ap, cache)

def erase_own_cache(cache):
    p = (cache / 'offline.jbl').resolve()
    assert p.is_relative_to(out.resolve()) and p.name == 'offline.jbl'
    if p.exists():
        p.unlink()
print('BEGIN', ds, Q.shape, G.shape, flush=True)
t = time.perf_counter()
cos = Q @ G.T
direct = np.argsort(-cos, axis=1)
m, ap, _, _ = e.compute_map(direct.T, gnd['gnd'])
np.savez_compressed(out / 'direct.npz', ranks=direct.astype(np.int32), aps=ap)
record['direct'] = {'mAP': float(m), 'seconds': time.perf_counter() - t}
full, fullap, fullcache = graph_run(list(range(n)), 'full')
combined = np.empty_like(full)
batchap = np.empty(n)
for j, ids in enumerate(groups):
    rr, aa, ca = graph_run(ids, f'partition{j}')
    combined[ids] = rr
    batchap[ids] = aa
    erase_own_cache(ca)
reverse = list(reversed(range(n)))
rr, aa, ca = graph_run(reverse, 'reverse_fresh')
restored = rr[::-1]
erase_own_cache(ca)
stale, staleap, _ = graph_run(reverse, 'reverse_stale', fullcache)
erase_own_cache(fullcache)
cglog = []
cache = out / 'cache_gallery'
cache.mkdir(parents=True, exist_ok=True)
t = time.perf_counter()
D = module.Diffusion(G, str(cache))
O = D.get_offline_results(1000, 50)
prep = time.perf_counter() - t
t = time.perf_counter()
ss, ix = D.knn.search(Q, 10)
scores = np.stack([np.asarray(ss[i] ** 3 @ O[ix[i]]).ravel() for i in range(n)])
old = np.argsort(-scores, axis=1)
on = time.perf_counter() - t
per = np.empty_like(old)
for ids in groups:
    ss2, ix2 = D.knn.search(Q[ids], 10)
    for j, i in enumerate(ids):
        per[i] = np.argsort(-np.asarray(ss2[j] ** 3 @ O[ix2[j]]).ravel())
mm, oldap, _, _ = e.compute_map(old.T, gnd['gnd'])
np.savez_compressed(out / 'gallery_only.npz', ranks=old.astype(np.int32), aps=oldap)
erase_own_cache(cache)

def comparison(x, y):
    return {'top1_changes': int(np.sum(x[:, 0] != y[:, 0])), 'top10_set_changes': sum((set(xx[:10]) != set(yy[:10]) for xx, yy in zip(x, y))), 'mean_top10_overlap': float(np.mean([len(set(xx[:10]) & set(yy[:10])) / 10 for xx, yy in zip(x, y)])), 'full_order_changes': int(np.sum(np.any(x != y, axis=1)))}
record['gallery_only'] = {'mAP': float(mm), 'build_s': prep, 'query_stage_s': on, 'cohort_comparison': comparison(old, per), 'cg_nonconverged': sum((x[0] != 0 for x in cglog)), 'max_relative_residual': max((x[1] for x in cglog))}
record['full_vs_partition'] = comparison(full, combined)
record['permutation_fresh'] = comparison(full, restored)
record['permutation_stale'] = comparison(full, stale[::-1])
record['partition_mAP'] = float(batchap.mean())
record['stale_reversed_mAP'] = float(staleap.mean())
record['full_vs_direct'] = comparison(full, direct)
np.savez_compressed(out / 'comparisons.npz', full=full.astype(np.int32), partition=combined.astype(np.int32), fresh_reversed=restored.astype(np.int32), stale_reversed=stale[::-1].astype(np.int32), full_ap=fullap, partition_ap=batchap)
dif = batchap - fullap
rng = np.random.default_rng(20261026)
record['partition_minus_full_mAP_pp_CI'] = (100 * np.quantile([rng.choice(dif, size=n).mean() for _ in range(2000)], [0.025, 0.975])).tolist()
record['scope'] = 'Native transductive default is not claimed erroneous. Partition changes its graph. Stale-cache experiment deliberately violates namespace precondition. Gallery-only is the published original independent-query path, not a novel algorithm.'
(out / 'summary.json').write_text(json.dumps(record, indent=2))
print('FINAL', json.dumps({k: v for k, v in record.items() if k != 'runs'}), flush=True)
