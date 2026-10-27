import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '2'
import sys, time, json, hashlib, importlib.util, datetime, argparse, psutil
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(2)
p = argparse.ArgumentParser()
p.add_argument('--dataset', choices=['wikidiverse', 'wikimel'], default='wikidiverse')
p.add_argument('--queries', type=int, default=64)
p.add_argument('--repeats', type=int, default=3)
p.add_argument('--scale', type=int, default=0)
args = p.parse_args()
r = Path(__import__('os').environ.get('ICDE_STUDY_ROOT', str(Path(__file__).resolve().parents[1])))
sys.path[:0] = [str(r / 'work'), str(r / 'work/deps')]
import faiss
faiss.omp_set_num_threads(2)
sp = importlib.util.spec_from_file_location('native', r / 'vendor/OpenMEL-main/maximum_spanning.py')
native = importlib.util.module_from_spec(sp)
sp.loader.exec_module(native)
source = r / 'results' / ('coherence_audit' if args.dataset == 'wikidiverse' else 'coherence_wikimel')
a = np.load(source / 'real_operator_inputs.npz')
roots = a['roots']
ss = a['root_scores']
f = r / 'data' / args.dataset / 'features'
D = np.load(f / 'entity_name_desc.npy').astype(np.float32)
D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-08)
meta = json.load(open(f / 'dev_metadata.json', encoding='utf-8'))
selected = sorted(range(len(meta)), key=lambda i: hashlib.sha256(str(meta[i]['id']).encode()).hexdigest())[:args.queries]
N = args.scale or len(D)
E = D[:N]
index = faiss.IndexFlatIP(512)
t = time.perf_counter()
index.add(E)
build = time.perf_counter() - t
out = r / 'results/physical_v3' / f'{args.dataset}_n{N}'
out.mkdir(parents=True, exist_ok=True)
reg = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'queries': args.queries, 'repeats': args.repeats, 'database_size': N, 'seed_query_policy': 'fixed full-corpus root embeddings, search against nested real KB prefixes for scale experiment; this measures physical cost, not scale-dependent retrieval accuracy', 'stage': 'neighbors + graph selection only, excludes identical root retrieval and encoding', 'threads': 2, 'vector_bytes': int(E.nbytes), 'index_bytes': int(faiss.serialize_index(index).nbytes), 'index_build_s': build, 'backends': {'direct': 'FAISS1.15.1 default threshold128000 on nx*d', 'blas': 'FAISS forced BLAS threshold1'}, 'plans': ['eager_batch_native', 'prefix_batch_flat', 'warm_native', 'warm_flat'], 'scope': 'background other user tasks may run; randomize plan order and retain raw timings; report observed rather than isolated hardware ceiling', 'B': [1, 5, 10, 20]}
(out / 'protocol.json').write_text(json.dumps(reg, indent=2))

def flat(rt, s, adj, ww, B):
    best = {v: (-float(x), -1, v) for v, x in zip(rt, s)}
    for u in adj:
        for v, w in zip(adj[u], ww[u]):
            if v == u:
                continue
            key = (-float(w), u, v)
            if v not in best or key < best[v]:
                best[v] = key
    return [z[2] for z in sorted(best.values())[:B]]

def adjacency(rt, s, positions):
    if not positions:
        return ({}, {})
    dd, nn = index.search(D[[rt[j] for j in positions]], 5)
    adj = {rt[j]: nn[k].tolist() for k, j in enumerate(positions)}
    w = {rt[j]: (np.clip(dd[k], 0, 1) * s[j]).tolist() for k, j in enumerate(positions)}
    return (adj, w)

def eager(rt, s, B):
    adj, w = adjacency(rt, s, list(range(len(rt))))
    return native.maximum_spanning_tree(-1, rt, s, adj, w, B + 1)[0][1:]

def prefix(rt, s, B):
    if B == 1:
        return rt[:1]
    positions = list(range(len(rt))) if B > len(rt) else [j for j in range(len(rt)) if s[j] >= s[B - 1]]
    adj, w = adjacency(rt, s, positions)
    return flat(rt, s, adj, w, B)
records = []
rng = np.random.default_rng(20261025)
for backend, threshold in [('direct', 128000), ('blas', 1)]:
    faiss.cvar.distance_compute_blas_threshold = threshold
    cache = {i: adjacency(roots[i].tolist(), ss[i].astype(float).tolist(), list(range(10))) for i in selected}
    for B in reg['B']:
        for rep in range(args.repeats):
            for qi in rng.permutation(selected):
                qi = int(qi)
                rt = roots[qi].tolist()
                s = ss[qi].astype(float).tolist()
                adj, w = cache[qi]
                reference = native.maximum_spanning_tree(-1, rt, s, adj, w, B + 1)[0][1:]
                plans = {'eager_batch_native': lambda: eager(rt, s, B), 'prefix_batch_flat': lambda: prefix(rt, s, B), 'warm_native': lambda: native.maximum_spanning_tree(-1, rt, s, adj, w, B + 1)[0][1:], 'warm_flat': lambda: flat(rt, s, adj, w, B)}
                for name in rng.permutation(list(plans)):
                    t = time.perf_counter_ns()
                    got = plans[name]()
                    elapsed = (time.perf_counter_ns() - t) / 1000000.0
                    records.append({'qi': qi, 'rep': rep, 'B': B, 'backend': backend, 'plan': name, 'ms': elapsed, 'mismatch': got != reference})
        print(args.dataset, N, backend, 'B', B, flush=True)
        (out / 'timings.partial.json').write_text(json.dumps(records))
summary = []
for backend in reg['backends']:
    for B in reg['B']:
        for plan in reg['plans']:
            rows = [x for x in records if x['backend'] == backend and x['B'] == B and (x['plan'] == plan)]
            ts = [x['ms'] for x in rows]
            summary.append({'backend': backend, 'B': B, 'plan': plan, 'n': len(rows), 'p50_ms': float(np.median(ts)), 'p95_ms': float(np.quantile(ts, 0.95)), 'mean_ms': float(np.mean(ts)), 'ordered_mismatches': sum((x['mismatch'] for x in rows))})
(out / 'timings.json').write_text(json.dumps(records))
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
