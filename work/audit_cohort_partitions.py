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
out = r / 'results/rev6' / ('cohort_' + ds)
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
protocol = json.loads((r / 'results/rev6/cohort_protocol.json').read_text())
for name, digest in protocol['source_sha256'].items():
    assert hashlib.sha256((r / name).read_bytes()).hexdigest() == digest, name
assert args.certified
base = ds + '_certified'
z = np.load(r / f'results/rev3/{base}/comparisons.npz')
full = z['full']
fullap = z['full_ap']
assert len(full) == len(Q) == 55
newgroups = {str(seed): [np.random.default_rng(seed).permutation(n).tolist()[j::4] for j in range(4)] for seed in protocol['seeds']}
(out / 'workload.json').write_text(json.dumps({'queries': n, 'gallery': N, 'seeds_and_cohorts': newgroups, 'reference': f'results/rev3/{base}/comparisons.npz', 'protocol': 'results/rev6/cohort_protocol.json'}, indent=2))
record['repeated_partitions'] = []
for seed in protocol['seeds']:
    combined = np.empty_like(full)
    batchap = np.empty(n)
    for j, ids in enumerate(newgroups[str(seed)]):
        rr, aa, cache = graph_run(ids, f'seed{seed}_part{j}')
        combined[ids] = rr
        batchap[ids] = aa
        erase_own_cache(cache)
    result = {'seed': seed, 'top1_changes': int(np.sum(full[:, 0] != combined[:, 0])), 'top10_set_changes': sum((set(x[:10]) != set(y[:10]) for x, y in zip(full, combined))), 'mAP_full': float(fullap.mean()), 'mAP_partition': float(batchap.mean()), 'delta_mAP_pp': float(100 * (batchap.mean() - fullap.mean()))}
    np.savez_compressed(out / f'seed{seed}_comparisons.npz', full=full, partition=combined, full_ap=fullap, partition_ap=batchap)
    record['repeated_partitions'].append(result)
    print('PARTITION', json.dumps(result), flush=True)
    record['scope'] = protocol['scope']
    (out / 'summary.json').write_text(json.dumps(record, indent=2))
print('FINAL', json.dumps(record['repeated_partitions']), flush=True)
