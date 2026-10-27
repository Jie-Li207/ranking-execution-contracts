import os, json, time, hashlib, datetime, collections
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(2)
r = Path(__import__('os').environ.get('ICDE_STUDY_ROOT', str(Path(__file__).resolve().parents[1])))
f = r / 'data' / 'wikidiverse' / 'features'
o = r / 'results' / 'B_full'
reg = {'time': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'reason': 'After observing missing-image score shift on development data; exploratory controls, not confirmatory test', 'baselines': ['RRF60 char+text', 'RRF60 char+text+image', 'balanced union char+text', 'balanced union char+text+image', 'masked average at alpha .25 .5', 'image similarity centered by corpus image mean at alpha .25 .5', 'within-image-presence zscore at alpha .25 .5', 'image residual relative to entity text at alpha .25 .5'], 'explicit': 'All are baseline controls; no novel mechanism claimed.'}
(r / 'work' / 'preregistration_fusion_controls.json').write_text(json.dumps(reg, indent=2))

def load(n):
    a = np.load(f / (n + '.npy')).astype(np.float32)
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-08)
    return a
D = load('entity_name_desc')
I = load('entity_image')
mask = np.load(f / 'entity_image_mask.npy')
mean = I[mask].mean(0)
summary = []
for split in ['train', 'dev']:
    rr = json.load(open(f / (split + '_metadata.json'), encoding='utf-8'))
    sel = sorted(range(len(rr)), key=lambda i: hashlib.sha256(str(rr[i]['id']).encode()).hexdigest())[:1024] if split == 'train' else list(range(len(rr)))
    rr = [rr[i] for i in sel]
    gold = np.array([x['answer_id'] for x in rr])
    ids = [x['id'] for x in rr]
    T = load(split + '_text')[sel]
    Q = load(split + '_image')[sel]
    qm = np.load(f / (split + '_image_mask.npy'))[sel]
    methods = {}

    def save(name, top, rank=None):
        if rank is None:
            rank = np.array([np.where(top[i] == g)[0][0] + 1 if g in top[i] else len(D) + 1 for i, g in enumerate(gold)])
        np.savez_compressed(o / f'{split}_{name}.npz', top=top, gold_rank=rank, query_id=ids, gold=gold)
        row = {'split': split, 'method': name, **{f'R@{k}': float((rank <= k).mean()) for k in [1, 10, 50, 100, 500]}, 'R100_missing': float((rank[~mask[gold]] <= 100).mean()), 'R100_present': float((rank[mask[gold]] <= 100).mean())}
        summary.append(row)
        print(row, flush=True)
    for branches in [['char_name', 'text_desc'], ['char_name', 'text_desc', 'image_image'], ['char_name', 'text_desc', 'image_desc'], ['char_name', 'text_desc', 'image_image', 'image_desc']]:
        arr = [{k: a[k] for k in ['top', 'query_id']} for b in branches for a in [np.load(o / f'{split}_{b}.npz')]]
        assert all((np.array_equal(a['query_id'], ids) for a in arr))
        rrf = np.full((len(rr), 500), -1, np.int32)
        union = rrf.copy()
        for i in range(len(rr)):
            weights = collections.defaultdict(float)
            seen = set()
            seq = []
            for a in arr:
                for rank, e in enumerate(a['top'][i]):
                    if e >= 0:
                        weights[int(e)] += 1 / (61 + rank)
            ordered = sorted(weights, key=lambda e: (-weights[e], e))[:500]
            rrf[i, :len(ordered)] = ordered
            for tup in zip(*[a['top'][i] for a in arr]):
                for e in tup:
                    if e >= 0 and e not in seen:
                        seen.add(e)
                        seq.append(e)
                    if len(seq) >= 500:
                        break
                if len(seq) >= 500:
                    break
            union[i, :len(seq)] = seq
        suffix = '_'.join(branches)
        save('rrf_' + suffix, rrf)
        save('union_' + suffix, union)
    names = [f'{mode}_{int(a * 100)}' for mode in ['masked', 'centered', 'zscore', 'residual'] for a in [0.25, 0.5]]
    tops = {n: np.zeros((len(rr), 500), np.int32) for n in names}
    ranks = {n: np.zeros(len(rr), np.int32) for n in names}
    for start in range(0, len(rr), 24):
        end = min(start + 24, len(rr))
        tt = T[start:end] @ D.T
        ii = Q[start:end] @ I.T
        it = Q[start:end] @ D.T
        center = ii - (Q[start:end] @ mean)[:, None] * mask[None, :]
        resid = (ii - it) * mask[None, :]
        zi = (ii - ii[:, mask].mean(1, keepdims=True)) / np.maximum(ii[:, mask].std(1, keepdims=True), 1e-06)
        zi[:, ~mask] = 0
        zt = (tt - tt.mean(1, keepdims=True)) / np.maximum(tt.std(1, keepdims=True), 1e-06)
        for a in [0.25, 0.5]:
            ss = {'masked': ((1 - a) * tt + a * ii) / (1 - a + a * mask), 'centered': (1 - a) * tt + a * center, 'zscore': (1 - a) * zt + a * zi, 'residual': tt + a * resid}
            for mode, S in ss.items():
                name = f'{mode}_{int(a * 100)}'
                ix = np.argpartition(-S, 499, axis=1)[:, :500]
                for j in range(len(S)):
                    x = ix[j]
                    tops[name][start + j] = x[np.lexsort((x, -S[j, x]))]
                    g = gold[start + j]
                    ranks[name][start + j] = int(np.sum(S[j] > S[j, g]) + np.sum((S[j] == S[j, g]) & (np.arange(len(D)) < g)) + 1)
    for name in names:
        save(name, tops[name], ranks[name])
(o / 'fusion_controls_summary.json').write_text(json.dumps(summary, indent=2))
