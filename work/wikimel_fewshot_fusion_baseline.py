import os, time, json, hashlib, datetime
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
threadpool_limits(2)
r = Path(__import__('os').environ.get('ICDE_STUDY_ROOT', str(Path(__file__).resolve().parents[1])))
f = r / 'data' / 'wikimel' / 'features'
o = r / 'results' / 'B_wikimel'
out = o / 'fewshot'
out.mkdir(parents=True, exist_ok=True)
reg = {'time': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'status': 'strong simple baseline, not proposed method', 'budgets': [32, 128, 512], 'seeds': [101, 102, 103, 104, 105], 'training_pool': 'first 1024 original train by hashed id as previously registered', 'training_pairs': 'all retrieved candidates for selected training queries; no positive insertion; labels only selected queries', 'candidate_pool': 'union of top500 char_name,text_desc,image_image,image_desc; shared for all learned rerankers', 'features': '4 CLIP cosine, entity image presence, branch reciprocal ranks and membership; no entity id or answer name lookup', 'models': 'LogisticRegression C1 balanced standardized; HGB depth2 100iterations balanced; no dev-based hyperparameter tuning', 'evaluation': 'original dev exploratory; final test untouched; stratify unseen gold identities and missing images', 'label_budget': 'all labels, no extra validation split queried'}
(r / 'work' / 'preregistration_wikimel_fewshot_fusion.json').write_text(json.dumps(reg, indent=2))

def load(n):
    a = np.load(f / (n + '.npy')).astype(np.float32)
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-08)
    return a
D = load('entity_name_desc')
E = load('entity_name_attr')
I = load('entity_image')
mask = np.load(f / 'entity_image_mask.npy')
cache = {}
for split in ['train', 'dev']:
    rr = json.load(open(f / (split + '_metadata.json'), encoding='utf-8'))
    sel = sorted(range(len(rr)), key=lambda i: hashlib.sha256(str(rr[i]['id']).encode()).hexdigest())[:1024] if split == 'train' else list(range(len(rr)))
    rr = [rr[i] for i in sel]
    T = load(split + '_text')[sel]
    Q = load(split + '_image')[sel]
    branches = []
    for name in ['char_name', 'text_desc', 'image_image', 'image_desc']:
        a = np.load(o / f'{split}_{name}.npz')
        assert np.array_equal(a['query_id'], [x['id'] for x in rr])
        branches.append(a['top'])
    xs = []
    ys = []
    cs = []
    ptr = [0]
    for i, row in enumerate(rr):
        cand = np.unique(np.concatenate([a[i] for a in branches]))
        cand = cand[cand >= 0]
        features = [D[cand] @ T[i], E[cand] @ T[i], I[cand] @ Q[i], D[cand] @ Q[i], I[cand] @ T[i], mask[cand].astype(float)]
        for a in branches:
            lookup = {int(e): j for j, e in enumerate(a[i]) if e >= 0}
            rank = np.array([lookup.get(int(e), 100000) for e in cand])
            features.extend([1 / (61 + rank), (rank < 500).astype(float)])
        x = np.stack(features, axis=1).astype(np.float32)
        y = (cand == row['answer_id']).astype(np.int8)
        xs.append(x)
        ys.append(y)
        cs.append(cand)
        ptr.append(ptr[-1] + len(cand))
    cache[split] = (np.concatenate(xs), np.concatenate(ys), np.concatenate(cs), np.array(ptr), rr)
    print('features', split, cache[split][0].shape, flush=True)
X, Y, C, P, rr = cache['train']
DX, DY, DC, DP, dr = cache['dev']
results = []
for budget in reg['budgets']:
    for seed in reg['seeds']:
        rng = np.random.default_rng(seed)
        selected = rng.permutation(len(rr))[:budget]
        ix = np.concatenate([np.arange(P[q], P[q + 1]) for q in selected])
        x = X[ix]
        y = Y[ix]
        positive_query_ids = [rr[q]['id'] for q in selected]
        for name in ['logistic', 'hgb']:
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight='balanced', max_iter=500, random_state=seed)) if name == 'logistic' else HistGradientBoostingClassifier(max_depth=2, max_iter=100, max_leaf_nodes=4, class_weight='balanced', l2_regularization=1, early_stopping=False, random_state=seed)
            t = time.perf_counter()
            model.fit(x, y)
            fit = time.perf_counter() - t
            t = time.perf_counter()
            scores = model.predict_proba(DX)[:, 1]
            predict = time.perf_counter() - t
            ranks = []
            for q in range(len(dr)):
                sl = slice(DP[q], DP[q + 1])
                ii = np.lexsort((DC[sl], -scores[sl]))
                yy = DY[sl][ii]
                hit = np.flatnonzero(yy)
                ranks.append(int(hit[0]) + 1 if len(hit) else len(D) + 1)
            ranks = np.array(ranks)
            np.savez_compressed(out / f'{name}_{budget}_{seed}.npz', gold_rank=ranks, train_query_ids=positive_query_ids, dev_query_ids=[z['id'] for z in dr])
            result = dict(method=name, budget=budget, seed=seed, fit_seconds=fit, predict_ms_per_query=predict / len(dr) * 1000, **{f'R@{k}': float((ranks <= k).mean()) for k in [1, 10, 50, 100, 500]})
            results.append(result)
            print(result, flush=True)
            (out / 'summary.json').write_text(json.dumps(results, indent=2))
