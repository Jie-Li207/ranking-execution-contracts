import os, json, time, hashlib, re, unicodedata
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
from pathlib import Path
from urllib.parse import unquote
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits
threadpool_limits(2)
r = Path(__import__('os').environ.get('ICDE_STUDY_ROOT', str(Path(__file__).resolve().parents[1])))
d = r / 'data' / 'wikidiverse' / 'cleaned' / 'WikiDiverse'
o = r / 'results' / 'B_full'
o.mkdir(exist_ok=True, parents=True)
e = json.load(open(d / 'kb_entity.json', encoding='utf-8'))
qid = json.load(open(d / 'qid2id.json', encoding='utf-8'))
assert all((x['id'] == i for i, x in enumerate(e)))
rows = {s: [x for x in json.load(open(d / f'wiki_diverse_{s}.json', encoding='utf-8')) if x['answer'] in qid] for s in ['train', 'dev']}
rows['train'] = sorted(rows['train'], key=lambda x: hashlib.sha256(str(x['id']).encode()).hexdigest())[:1024]

def norm(x):
    return ' '.join(re.findall('\\w+', unicodedata.normalize('NFKC', unquote(x)).casefold()))
reg = {'time': time.time(), 'stage': 'lexical baseline before any scores', 'models': ['char_wb3-5 TFIDF entity name vs mention', 'word1-2 TFIDF entity name+desc vs mention+sentence'], 'min_df': 1, 'max_features': 300000, 'dtype': 'float32', 'tie_break': 'ascending entity id', 'zero_matches': 'unretrieved, never padded with label-informed entries', 'budgets': [1, 10, 50, 100, 500]}
(r / 'work' / 'preregistration_lexical.json').write_text(json.dumps(reg, indent=2))
summary = []
for mode in ['char_name', 'word_desc']:
    vec = TfidfVectorizer(analyzer='char_wb' if mode == 'char_name' else 'word', ngram_range=(3, 5) if mode == 'char_name' else (1, 2), max_features=300000, dtype=np.float32)
    docs = [norm(x['entity_name']) if mode == 'char_name' else norm(x['entity_name'] + ' ' + x.get('desc', '')) for x in e]
    t = time.perf_counter()
    mat = vec.fit_transform(docs)
    build = time.perf_counter() - t
    print(mode, 'build', build, 'shape', mat.shape, 'nnz', mat.nnz, flush=True)
    for split, rr in rows.items():
        qq = [norm(x['mentions']) if mode == 'char_name' else norm(x['mentions'] + ' ' + x['sentence']) for x in rr]
        qm = vec.transform(qq)
        top = np.full((len(rr), 500), -1, dtype=np.int32)
        val = np.zeros((len(rr), 500), np.float32)
        ranks = np.full(len(rr), len(e) + 1, dtype=np.int32)
        times = []
        for i in range(len(rr)):
            t = time.perf_counter()
            v = (qm[i] @ mat.T).tocsr()
            ids = v.indices
            vals = v.data
            rank = np.lexsort((ids, -vals))
            sel = rank[:500]
            top[i, :len(sel)] = ids[sel]
            val[i, :len(sel)] = vals[sel]
            gold = qid[rr[i]['answer']]
            hit = np.where(ids[rank] == gold)[0]
            ranks[i] = int(hit[0] + 1) if len(hit) else len(e) + 1
            times.append(time.perf_counter() - t)
        np.savez_compressed(o / f'{split}_{mode}.npz', top=top, scores=val, gold_rank=ranks, query_id=[x['id'] for x in rr], gold=[qid[x['answer']] for x in rr], latency=times)
        row = dict(mode=mode, split=split, n=len(rr), build_seconds=build, index_bytes=int(mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes), p50_ms=float(np.median(times) * 1000), p95_ms=float(np.quantile(times, 0.95) * 1000), **{f'R@{k}': float((ranks <= k).mean()) for k in [1, 10, 50, 100, 500]})
        summary.append(row)
        print(row, flush=True)
(o / 'lexical_summary.json').write_text(json.dumps(summary, indent=2))
