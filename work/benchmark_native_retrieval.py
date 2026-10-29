import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '2'
import sys, json, time, hashlib, datetime, importlib.util, argparse
from pathlib import Path
from difflib import SequenceMatcher
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(2)
p = argparse.ArgumentParser()
p.add_argument('dataset', choices=['WikiMEL', 'RichpediaMEL'])
args = p.parse_args()
r = Path(__import__('os').environ.get('ICDE_STUDY_ROOT', str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(r / 'work/deps'))
import faiss
from rapidfuzz import process, fuzz
faiss.omp_set_num_threads(2)
faiss.cvar.distance_compute_blas_threshold = 1
sp = importlib.util.spec_from_file_location('native', r / 'vendor/OpenMEL-main/maximum_spanning.py')
native = importlib.util.module_from_spec(sp)
sp.loader.exec_module(native)
f = r / 'data/author_cache' / args.dataset
out = r / 'results/native_retrieval_latency' / args.dataset
out.mkdir(parents=True, exist_ok=True)

def load(n):
    a = np.load(f / (n + '.npy')).astype(np.float32)
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-08)
    return a
D = load('entity_text')
I = load('entity_image')
T = load('query_text')
Q = load('query_image')
im = np.load(f / 'entity_image_mask.npy')
qm = np.load(f / 'query_image_mask.npy')
meta = json.load(open(f / 'queries.json', encoding='utf-8'))
kb = json.load(open(f / 'kb_metadata.json', encoding='utf-8'))
names = [e['entity_name'] for e in kb]
prior = json.load(open(r / 'results/author_reproduction' / args.dataset / 'query_results.json'))
idx = faiss.IndexFlatIP(512)
idx.add(D)
selected = sorted(range(len(meta)), key=lambda i: hashlib.sha256(str(meta[i]['id']).encode()).hexdigest())[:128]
fallback = [z['qi'] for z in prior if not z['name_filter_matched']]
extra = sorted(set(fallback) - set(selected), key=lambda i: hashlib.sha256(str(meta[i]['id']).encode()).hexdigest())[:32]
order = selected + extra
reg = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'stage': 'complete retrieval stage given cached query/entity embeddings: full-corpus3-channel scan, exact name filter, candidate sorting and graph selection. Excludes LLM generation and embedding encoding, shared between plans.', 'B': 10, 'queries_primary': 128, 'fallback_extra': len(extra), 'repeats': 3, 'threads': 2, 'backend': 'Faiss1.15.1 BLAS threshold1', 'plans': ['native_eager', 'compiled_prefix'], 'filter': 'RapidFuzz upper bound then exact SequenceMatcher>=0.5 every timed invocation; no cross-query name cache', 'input': 'author cached original vectors, all entities', 'timing': 'random interleaving, other tasks may run; retain timings and reference-order differences'}
(out / 'protocol.json').write_text(json.dumps(reg, indent=2))

def run(qi, compiled):
    s = T[qi] @ D.T
    ti = T[qi] @ I.T
    ti[~im] = -np.inf
    s = np.maximum(s, ti)
    if qm[qi]:
        s = np.maximum(s, Q[qi] @ D.T)
    candidates = process.extract(meta[qi]['mentions'], names, scorer=fuzz.ratio, score_cutoff=49.99999, limit=None)
    c = np.array(sorted((i for name, score, i in candidates if SequenceMatcher(None, meta[qi]['mentions'], name).ratio() >= 0.5)), dtype=np.int64)
    if len(c):
        c = c[np.lexsort((c, -s[c]))]
        rt = c.tolist()
        ss = s[c].astype(float).tolist()
        return rt[:10] if compiled else native.maximum_spanning_tree(-1, rt, ss, {}, {}, 11)[0][1:]
    c = np.argpartition(-s, 9)[:10]
    c = c[np.lexsort((c, -s[c]))]
    rt = c.tolist()
    ss = s[c].astype(float).tolist()
    dd, nn = idx.search(D[c], 5)
    adj = {u: nn[j].tolist() for j, u in enumerate(rt)}
    w = {u: (dd[j] * ss[j]).tolist() for j, u in enumerate(rt)}
    if not compiled or any(x < 0 for x in ss) or any((not 0 <= x <= ss[j] for j, u in enumerate(rt) for x in w[u])):
        return native.maximum_spanning_tree(-1, rt, ss, adj, w, 11)[0][1:]
    best = {u: (-x, -1, u) for u, x in zip(rt, ss)}
    for u in rt:
        for v, x in zip(adj[u], w[u]):
            if v == u:
                continue
            key = (-x, u, v)
            if v not in best or key < best[v]:
                best[v] = key
    return [z[2] for z in sorted(best.values())[:10]]
for qi in order[:4]:
    run(qi, False)
    run(qi, True)
rng = np.random.default_rng(20261025)
records = []
for rep in range(3):
    for count, qi in enumerate(rng.permutation(order)):
        qi = int(qi)
        outputs = {}
        for name in rng.permutation(reg['plans']):
            t = time.perf_counter_ns()
            got = run(qi, name == 'compiled_prefix')
            elapsed = (time.perf_counter_ns() - t) / 1000000.0
            outputs[name] = got
            records.append({'qi': qi, 'rep': rep, 'plan': name, 'ms': elapsed, 'primary': qi in selected, 'name_match': prior[qi]['name_filter_matched'], 'prior_order_mismatch': got != prior[qi]['native_top20'][:10]})
        assert outputs['native_eager'] == outputs['compiled_prefix'], (qi, outputs)
    print(args.dataset, 'rep', rep, flush=True)
(out / 'timings.json').write_text(json.dumps(records))
summary = []
for subset in ['primary', 'fallback']:
    for name in reg['plans']:
        rr = [x for x in records if x['plan'] == name and (x['primary'] if subset == 'primary' else not x['name_match'])]
        ts = [x['ms'] for x in rr]
        if rr:
            summary.append({'subset': subset, 'plan': name, 'n': len(rr), 'p50_ms': float(np.median(ts)), 'p95_ms': float(np.quantile(ts, 0.95)), 'serial_queries_per_s': 1000 / float(np.mean(ts)), 'prior_order_mismatches': sum((x['prior_order_mismatch'] for x in rr))})
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
