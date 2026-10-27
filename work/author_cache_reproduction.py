import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '2'
import sys, json, time, datetime, hashlib, argparse, importlib.util
from pathlib import Path
from difflib import SequenceMatcher
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(2)
p = argparse.ArgumentParser()
p.add_argument('dataset', choices=['WikiMEL', 'RichpediaMEL'])
args = p.parse_args()
r = Path(__import__('os').environ.get('ICDE_STUDY_ROOT', str(Path(__file__).resolve().parents[1])))
sys.path[:0] = [str(r / 'work/deps')]
import faiss
from rapidfuzz import process, fuzz
faiss.omp_set_num_threads(2)
spec = importlib.util.spec_from_file_location('native', r / 'vendor/OpenMEL-main/maximum_spanning.py')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
assert (r / 'work/preregistration_confirmatory_native.json').exists()
f = r / 'data/author_cache' / args.dataset
out = r / 'results/author_reproduction' / args.dataset
out.mkdir(parents=True, exist_ok=True)
amend = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'timing': 'before retrieval outcomes', 'schema_repair': 'Richpedia has349 empty image paths. Preserve empty before jpg suffix conversion and use source intended no-image branch; original code converts empty to .jpg and attempts unavailable dictionary key. WikiMEL has none.', 'performance_only': 'RapidFuzz Indel similarity is an upper bound on SequenceMatcher matching-block ratio; use it to reject provable nonmatches then original SequenceMatcher for exact >=0.5 decision. Validate exhaustive comparisons for fixed first5 query strings.', 'arithmetic': 'author raw float16 numbers preserved exactly; normalize in float32 and vectorize cosine. Fixed-root operator equivalence uses identical scores; no bit-exact scalar vs BLAS claim.'}
(out / 'implementation_amendment.json').write_text(json.dumps(amend, indent=2))

def load(name):
    a = np.load(f / (name + '.npy')).astype(np.float32)
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-08)
    return a
D = load('entity_text')
I = load('entity_image')
T = load('query_text')
Q = load('query_image')
im = np.load(f / 'entity_image_mask.npy')
qm = np.load(f / 'query_image_mask.npy')
queries = json.load(open(f / 'queries.json', encoding='utf-8'))
kb = json.load(open(f / 'kb_metadata.json', encoding='utf-8'))
gold = np.load(f / 'gold.npy')
names = [e['entity_name'] for e in kb]
idx = faiss.IndexFlatIP(512)
idx.add(D)
filtercache = {}
checked = 0

def name_candidates(mention):
    global checked
    if mention in filtercache:
        return filtercache[mention]
    possible = process.extract(mention, names, scorer=fuzz.ratio, score_cutoff=49.99999, limit=None)
    c = sorted((i for name, score, i in possible if SequenceMatcher(None, mention, name).ratio() >= 0.5))
    if checked < 5:
        expected = [i for i, name in enumerate(names) if SequenceMatcher(None, mention, name).ratio() >= 0.5]
        assert c == expected
        checked += 1
    filtercache[mention] = np.array(c, dtype=np.int64)
    return filtercache[mention]

def flat(rt, ss, adj, ww, B):
    best = {v: (-float(s), -1, v) for v, s in zip(rt, ss)}
    for u in rt:
        for v, w in zip(adj.get(u, []), ww.get(u, [])):
            if v == u:
                continue
            key = (-float(w), u, v)
            if v not in best or key < best[v]:
                best[v] = key
    return [x[2] for x in sorted(best.values())[:B]]
records = []
rootsets = {3: [], 4: []}
scores = {3: [], 4: []}
start = time.perf_counter()
mismatch = []
mask_violations = []
for begin in range(0, len(T), 16):
    tt = T[begin:begin + 16] @ D.T
    it = Q[begin:begin + 16] @ D.T
    it[~qm[begin:begin + 16]] = -np.inf
    ti = T[begin:begin + 16] @ I.T
    ti[:, ~im] = -np.inf
    ii = Q[begin:begin + 16] @ I.T
    ii[:, ~im] = -np.inf
    ii[~qm[begin:begin + 16]] = -np.inf
    S3 = np.maximum(tt, np.maximum(it, ti))
    S4 = np.maximum(S3, ii)
    for j, s3 in enumerate(S3):
        qi = begin + j
        for mode, ss in [(3, s3), (4, S4[j])]:
            cand = np.argpartition(-ss, 9)[:10]
            cand = cand[np.lexsort((cand, -ss[cand]))]
            rootsets[mode].append(cand)
            scores[mode].append(ss[cand])
        c = name_candidates(queries[qi]['mentions'])
        match = bool(len(c))
        adj = {}
        ww = {}
        if match:
            c = c[np.lexsort((c, -s3[c]))]
            rt = c.tolist()
            ss = s3[c].astype(float).tolist()
        else:
            rt = rootsets[3][-1].tolist()
            ss = s3[rt].astype(float).tolist()
            dd, nn = idx.search(D[rt], 5)
            for u, vv, ds, s in zip(rt, nn, dd, ss):
                adj[u] = vv.tolist()
                ww[u] = (ds * s).tolist()
        got = native.maximum_spanning_tree(-1, rt, ss, adj, ww, 21)[0][1:]
        base = rt[:20]
        preconditions = all((s >= 0 for s in ss)) and all((0 <= w <= ss[j] + 1e-07 for j, u in enumerate(rt) for w in ww.get(u, [])))
        compiled = flat(rt, ss, adj, ww, 20) if preconditions else None
        if compiled is not None and got != compiled:
            mismatch.append(qi)
        if not preconditions:
            mask_violations.append(qi)
        g = int(gold[qi])
        rank = lambda ls: ls.index(g) + 1 if g in ls else 999999
        records.append({'qi': qi, 'query_id': str(queries[qi].get('id', qi)), 'gold': g, 'name_filter_matched': match, 'name_candidates': len(c), 'native_rank': rank(got), 'no_graph_rank': rank(base), 'native_top20': got, 'no_graph_top20': base, 'top1_changed': got[:1] != base[:1], 'compilation_preconditions': preconditions, 'image_cluster': queries[qi]['imgPath'] or 'missing-query-' + str(qi)})
    if begin % 256 == 0:
        print(args.dataset, begin, len(T), 'elapsed', round(time.perf_counter() - start, 1), flush=True)
    if begin % 512 == 0:
        (out / 'query_results.partial.json').write_text(json.dumps(records))
for mode in [3, 4]:
    roots = np.array(rootsets[mode])
    raw = np.array(scores[mode])
    unique = np.unique(roots)
    dd, nn = idx.search(D[unique], 10)
    np.savez_compressed(out / f'controlled_inputs_{mode}.npz', roots=roots, root_scores=raw, scores=(1 + raw.astype(float)) / 2, gold=gold, selected=np.arange(len(T)), unique=unique, nn=nn, dd=np.clip((1 + dd.astype(float)) / 2, 0, 1))
(out / 'query_results.json').write_text(json.dumps(records))
summary = {'dataset': args.dataset, 'queries': len(records), 'entities': len(D), 'name_matched': sum((z['name_filter_matched'] for z in records)), 'fallback_graph_queries': sum((not z['name_filter_matched'] for z in records)), 'top1_changed': sum((z['top1_changed'] for z in records)), 'compiled_mismatch_queries': mismatch, 'precondition_violation_queries': mask_violations, 'filter_exhaustive_checks': checked, 'filter_unique_mentions': len(filtercache), 'elapsed_s': time.perf_counter() - start, 'recall': {method: {str(B): float(np.mean([z[method + '_rank'] <= B for z in records])) for B in [1, 5, 10, 20]} for method in ['native', 'no_graph']}, 'scope': 'repaired author released retrieval using original cached inputs; does not regenerate LLM captions or claim equality to published table'}
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
