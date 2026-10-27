import os, sys, json, time, hashlib, datetime, importlib.util
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(2)
r = Path(__import__('os').environ.get('ICDE_STUDY_ROOT', str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(r / 'work' / 'deps'))
import faiss
faiss.omp_set_num_threads(2)
source = r / 'vendor' / 'OpenMEL-main' / 'maximum_spanning.py'
spec = importlib.util.spec_from_file_location('openmel_original', source)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
out = r / 'results' / 'coherence_wikimel'
out.mkdir(parents=True, exist_ok=True)
reg = {'time': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'status': 'new candidate: operator-level measurement and exact query compilation, not yet manuscript contribution', 'hypotheses': ['H1: native greedy graph procedure cannot change the first returned entity once root edges are fixed', 'H2: for nonnegative root s and one-hop edge w(parent,child)<=s(parent), greedy node ranking equals a top-k of each node maximal incoming key; all budgets can share same exact compiled ranking', 'H3: graph expansion materially improves later candidate coverage beyond strong lexical+dense fusion; if false, no new relevance claim'], 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'protocol': 'unmodified native maximum_spanning_tree function; dev CLIP features without LLM expansion, explicitly an operator reproduction not full OpenMEL accuracy reproduction', 'synthetic': '10000 random nonnegative bounded-edge directed adjacency instances including ties; fail H2 on first mismatch and retain counterexample', 'real': 'all1664 development queries, complete132460entity KB, K10 roots,5 neighbors, B1/5/10/20; original test untouched', 'candidate_definition': 'native max(tt,query_image-to-entity_text,query_text-to-entity_image); image-image not used by released main.py. No name prefilter for WikiDiverse as its comment instructs.'}
(r / 'work' / 'preregistration_coherence_wikimel.json').write_text(json.dumps(reg, indent=2))

def compiled(roots, s, adj, w):
    best = {int(v): (float(t), -1, int(v)) for v, t in zip(roots, s)}
    for u in roots:
        for v, t in zip(adj.get(int(u), []), w.get(int(u), [])):
            v = int(v)
            u = int(u)
            if u == v:
                continue
            candidate = (float(t), u, v)
            old = best.get(v)
            if old is None or (-candidate[0], candidate[1], candidate[2]) < (-old[0], old[1], old[2]):
                best[v] = candidate
    return [x[2] for x in sorted(best.values(), key=lambda x: (-x[0], x[1], x[2]))]
print('synthetic checks reused from identical compiled function previously verified', flush=True)
f = r / 'data' / 'wikimel' / 'features'

def load(n):
    a = np.load(f / (n + '.npy')).astype(np.float32)
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-08)
    return a
D = load('entity_name_desc')
I = load('entity_image')
T = load('dev_text')
Q = load('dev_image')
imask = np.load(f / 'entity_image_mask.npy')
qm = np.load(f / 'dev_image_mask.npy')
rr = json.load(open(f / 'dev_metadata.json', encoding='utf-8'))
gold = np.array([x['answer_id'] for x in rr])
roots = []
root_s = []
t = time.perf_counter()
for start in range(0, len(T), 24):
    tt = T[start:start + 24] @ D.T
    it = Q[start:start + 24] @ D.T
    ti = T[start:start + 24] @ I.T
    ti[:, ~imask] = -np.inf
    it[~qm[start:start + 24]] = -np.inf
    ss = np.maximum(tt, np.maximum(it, ti))
    for S in ss:
        a = np.argpartition(-S, 9)[:10]
        a = a[np.lexsort((a, -S[a]))]
        roots.append(a)
        root_s.append(S[a])
roots = np.array(roots)
root_s = np.array(root_s)
assert root_s.min() >= 0
scan_seconds = time.perf_counter() - t
unique = np.unique(roots)
idx = faiss.IndexFlatIP(D.shape[1])
idx.add(D)
t = time.perf_counter()
distance, neighbors = idx.search(D[unique], 5)
neighbor_seconds = time.perf_counter() - t
lookup = {int(u): i for i, u in enumerate(unique)}
np.savez_compressed(out / 'real_operator_inputs.npz', roots=roots, root_scores=root_s, unique_seeds=unique, neighbors=neighbors, distances=distance, gold=gold)
rank_native = []
rank_comp = []
first_diff = 0
times_native = []
times_comp = []
mismatch = []
for i in range(len(T)):
    rt = roots[i].tolist()
    s = root_s[i].tolist()
    adj = {u: neighbors[lookup[u]].tolist() for u in rt}
    w = {u: (np.clip(distance[lookup[u]], 0, 1) * s[j]).tolist() for j, u in enumerate(rt)}
    t = time.perf_counter()
    a = native.maximum_spanning_tree(-1, rt, s, adj, w, 21)[0][1:]
    times_native.append(time.perf_counter() - t)
    t = time.perf_counter()
    b = compiled(rt, s, adj, w)[:20]
    times_comp.append(time.perf_counter() - t)
    if a != b:
        mismatch.append(i)
    first_diff += int(a[0] != rt[0])
    rank_native.append(a.index(int(gold[i])) + 1 if int(gold[i]) in a else 999999)
    rank_comp.append(b.index(int(gold[i])) + 1 if int(gold[i]) in b else 999999)
rank_native = np.array(rank_native)
rank_comp = np.array(rank_comp)
np.savez_compressed(out / 'real_operator_ranks.npz', native=rank_native, compiled=rank_comp, gold=gold, query_id=[x['id'] for x in rr], native_latency=times_native, compiled_latency=times_comp)
report = {'n': len(T), 'unique_seeds': len(unique), 'top1_changed_by_graph': first_diff, 'native_compiled_order_mismatches': mismatch, 'root_score_min': float(root_s.min()), 'all_scan_seconds': scan_seconds, 'neighbors_seconds': neighbor_seconds, 'native_operator_median_us': float(np.median(times_native) * 1000000.0), 'compiled_operator_median_us': float(np.median(times_comp) * 1000000.0), 'recall': {str(k): float((rank_native <= k).mean()) for k in [1, 5, 10, 20]}, 'roots_recall': {str(k): float((roots[:, :k] == gold[:, None]).any(1).mean()) for k in [1, 5, 10]}, 'scope': 'native ranking component on own unexpanded CLIP development features; not the full OpenMEL pipeline; finite precision clamped neighbor cosine to [0,1] for theoretical assumptions'}
(out / 'summary.json').write_text(json.dumps(report, indent=2))
print(report, flush=True)
