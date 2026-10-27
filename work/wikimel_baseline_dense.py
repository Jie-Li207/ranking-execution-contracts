import os, sys, json, time, hashlib, datetime
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(2)
r = Path(__import__('os').environ.get('ICDE_STUDY_ROOT', str(Path(__file__).resolve().parents[1])))
f = r / 'data' / 'wikimel' / 'features'
o = r / 'results' / 'B_wikimel'
o.mkdir(exist_ok=True, parents=True)
reg = {'time': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'methods': ['text_attr', 'text_desc', 'image_image', 'image_desc', 'text_image', 'fusion_desc_image_a25', 'fusion_desc_image_a50', 'fusion_desc_image_a75', 'fusion_text_cross_a50', 'fusion_text_cross_a25'], 'definition': 'a is image-image weight; text_cross uses query-image to entity-description; missing-image vectors zero, no sentinel imputation; normalized stored vectors renormalized float32', 'budgets': [1, 10, 50, 100, 500], 'test': 'never loaded', 'precision': 'all retrieval float32 CPU 2 BLAS threads', 'timing': 'concurrent pilot timings not final performance evidence; isolated benchmark required'}
(r / 'work' / 'preregistration_wikimel_dense_v1.json').write_text(json.dumps(reg, indent=2))

def load(n):
    a = np.load(f / (n + '.npy')).astype(np.float32)
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-08)
    return a
E = load('entity_name_attr')
D = load('entity_name_desc')
I = load('entity_image')
mask = np.load(f / 'entity_image_mask.npy')
summary = []
for split in ['train', 'dev']:
    rr = json.load(open(f / (split + '_metadata.json'), encoding='utf-8'))
    sel = sorted(range(len(rr)), key=lambda i: hashlib.sha256(str(rr[i]['id']).encode()).hexdigest())[:1024] if split == 'train' else list(range(len(rr)))
    rr = [rr[i] for i in sel]
    T = load(split + '_text')[sel]
    Q = load(split + '_image')[sel]
    gold = np.array([x['answer_id'] for x in rr])
    queryids = [x['id'] for x in rr]
    qmask = np.load(f / (split + '_image_mask.npy'))[sel]
    methods = {n: {'top': np.zeros((len(rr), 500), np.int32), 'scores': np.zeros((len(rr), 500), np.float32), 'gold_rank': np.zeros(len(rr), np.int32)} for n in reg['methods']}
    t0 = time.perf_counter()
    for start in range(0, len(rr), 24):
        end = min(start + 24, len(rr))
        scores = {'text_attr': T[start:end] @ E.T, 'text_desc': T[start:end] @ D.T, 'image_image': Q[start:end] @ I.T, 'image_desc': Q[start:end] @ D.T, 'text_image': T[start:end] @ I.T}
        for a in [0.25, 0.5, 0.75]:
            scores[f'fusion_desc_image_a{int(a * 100)}'] = (1 - a) * scores['text_desc'] + a * scores['image_image']
        for a in [0.25, 0.5]:
            scores[f'fusion_text_cross_a{int(a * 100)}'] = (1 - a) * scores['text_desc'] + a * scores['image_desc']
        for name, S in scores.items():
            if name in ['image_image', 'text_image']:
                S[:, ~mask] = -np.inf
            if name in ['image_image', 'image_desc']:
                S[~qmask[start:end]] = -np.inf
            ids = np.argpartition(-S, 499, axis=1)[:, :500]
            for j in range(len(S)):
                ii = ids[j]
                ii = ii[np.lexsort((ii, -S[j, ii]))]
                finite = np.isfinite(S[j, ii])
                methods[name]['top'][start + j] = np.where(finite, ii, -1)
                methods[name]['scores'][start + j] = S[j, ii]
                g = gold[start + j]
                gs = S[j, g]
                rank = int(np.sum(S[j] > gs) + np.sum((S[j] == gs) & (np.arange(len(E)) < g)) + 1) if np.isfinite(gs) else len(E) + 1
                methods[name]['gold_rank'][start + j] = rank
        if start % 240 == 0:
            print(split, start, len(rr), 'seconds', round(time.perf_counter() - t0, 1), flush=True)
    for name, v in methods.items():
        np.savez_compressed(o / f'{split}_{name}.npz', **v, gold=gold, query_id=queryids)
        row = {'method': name, 'split': split, 'n': len(rr), **{f'R@{k}': float((v['gold_rank'] <= k).mean()) for k in reg['budgets']}, 'R100_gold_has_image': float((v['gold_rank'][mask[gold]] <= 100).mean()), 'R100_gold_no_image': float((v['gold_rank'][~mask[gold]] <= 100).mean())}
        summary.append(row)
        print(row, flush=True)
(o / 'dense_summary.json').write_text(json.dumps(summary, indent=2))
