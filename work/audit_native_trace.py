import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '2'
from pathlib import Path
import sys, json, hashlib, time, ast, heapq
from difflib import SequenceMatcher
import numpy as np
r = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(r / 'work/deps'))
import faiss
from rapidfuzz import process, fuzz
from threadpoolctl import threadpool_limits
threadpool_limits(2)
faiss.omp_set_num_threads(2)
faiss.cvar.distance_compute_blas_threshold = 1
source = r / 'vendor/OpenMEL-main/maximum_spanning.py'
t = ast.parse(source.read_text(encoding='utf-8'))
ns = {'heapq': heapq}
exec(compile(ast.Module(body=[x for x in t.body if isinstance(x, (ast.ClassDef, ast.FunctionDef)) and x.name in ['MaxHeap', 'find', 'union', 'maximum_spanning_tree']], type_ignores=[]), str(source), 'exec'), ns)
native = ns['maximum_spanning_tree']
out = r / 'results/rev3'
reports = []
for ds in ['WikiMEL', 'RichpediaMEL']:
    f = r / 'data/author_cache' / ds

    def load(name):
        a = np.load(f / (name + '.npy')).astype(np.float32)
        a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-08)
        return a
    D, I, T, Q = [load(n) for n in ['entity_text', 'entity_image', 'query_text', 'query_image']]
    im = np.load(f / 'entity_image_mask.npy')
    qm = np.load(f / 'query_image_mask.npy')
    meta = json.load(open(f / 'queries.json', encoding='utf-8'))
    names = [z['entity_name'] for z in json.load(open(f / 'kb_metadata.json', encoding='utf-8'))]
    old = json.load(open(r / 'results/author_reproduction' / ds / 'query_results.json'))
    timings = json.load(open(r / 'results/native_retrieval_latency' / ds / 'timings.json'))
    selected = sorted({z['qi'] for z in timings} | {z['qi'] for z in old if not z['name_filter_matched']})
    index = faiss.IndexFlatIP(512)
    index.add(D)
    rows = []
    start = time.perf_counter()
    for qi in selected:
        score = T[qi] @ D.T
        ti = T[qi] @ I.T
        ti[~im] = -np.inf
        score = np.maximum(score, ti)
        if qm[qi]:
            score = np.maximum(score, Q[qi] @ D.T)
        cand = sorted((i for name, s, i in process.extract(meta[qi]['mentions'], names, scorer=fuzz.ratio, score_cutoff=49.99999, limit=None) if SequenceMatcher(None, meta[qi]['mentions'], name).ratio() >= 0.5))
        matched = bool(cand)
        adj = {}
        ww = {}
        if not cand:
            cand = np.argpartition(-score, 9)[:10].tolist()
        cand = sorted(cand, key=lambda i: (-score[i], i))
        ss = score[cand].astype(float).tolist()
        if not matched:
            dd, nn = index.search(D[cand], 5)
            for j, u in enumerate(cand):
                adj[u] = nn[j].tolist()
                ww[u] = (dd[j] * ss[j]).tolist()
        actual = native(-1, cand, ss, adj, ww, 21)[0][1:]
        assert actual == old[qi]['native_top20'], (ds, qi)
        rows.append({'qi': qi, 'roots': cand, 'root_scores': ss, 'adjacency': adj, 'weights': ww, 'native_top20': actual, 'name_match': matched, 'source_query_id': str(meta[qi]['id'])})
    (out / f'{ds}_native_execution_traces.json').write_text(json.dumps(rows))
    v = {'dataset': ds, 'traces': len(rows), 'all_fallback_included': sum((not x['name_match'] for x in rows)), 'original_output_mismatches': 0, 'elapsed_s': time.perf_counter() - start, 'timing_record_count': len(timings), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    reports.append(v)
    print(json.dumps(v), flush=True)
(out / 'native_execution_trace_summary.json').write_text(json.dumps({'status': 'PASS', 'datasets': reports, 'scope': 'Recomputed full entity scan and exact filtering, then checked full ordered prefixes against recorded original outputs. Compact traces allow graph/compiled equivalence replay without embeddings; latency itself still requires full inputs.'}, indent=2))
