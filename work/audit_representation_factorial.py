import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
from pathlib import Path
import sys
import json
import time
import hashlib
import warnings
from types import SimpleNamespace
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(os.environ.get('ICDE_STUDY_ROOT', str(Path(__file__).resolve().parents[1])))
STUDY = ROOT
sys.path.insert(0, str(STUDY / 'work'))
import audit_reranking_grid as upstream
upstream.ROOT = STUDY
OUT = ROOT / 'results/rev10'
torch.set_num_threads(2)
threadpool_limits(2)
warnings.filterwarnings('ignore', message='To copy construct from a tensor')

def representations():
    a = np.load(OUT / 'independent_features.npz')
    take, gallery, gold = [a[k] for k in ['query_indices', 'gallery', 'gold']]
    f = STUDY / 'data/wikidiverse/features'
    ct, ci = [upstream.unit(np.load(f / name, mmap_mode='r')[gallery].astype(np.float32)) for name in ['entity_name_desc.npy', 'entity_image.npy']]
    qt, qi = [upstream.unit(np.load(f / name, mmap_mode='r')[take].astype(np.float32)) for name in ['test_text.npy', 'test_image.npy']]
    for name, t, im, tq, iq in [('CC', ct, ci, qt, qi),
                               ('MC', a['gallery_text'], ci, a['query_text'], qi),
                               ('CR', ct, a['gallery_image'], qt, a['query_image']),
                               ('MR', a['gallery_text'], a['gallery_image'], a['query_text'], a['query_image'])]:
        yield name, upstream.unit(np.concatenate([t, im], axis=1)), upstream.unit(np.concatenate([tq, iq], axis=1)), take, gallery, gold

def main():
    assert (OUT / 'protocol.json').exists()
    classes, cheb = upstream.load_sources()
    allmeta = json.loads((STUDY / 'data/wikidiverse/features/test_metadata.json').read_text(encoding='utf-8'))
    summary = []
    for name, G, Q, take, gallery, gold in representations():
        start = time.perf_counter()
        meta = [allmeta[i] for i in take]
        ranks = np.empty((5, len(Q), 20), np.int32)
        configs = ['identity', 'averageQE', 'similarity-cubedQE', 'SG0', 'SG.15']
        sg = [(classes['MDescAug'](M=400, beta=b), classes['RerankwMDA'](M=400, beta=b)) for b in [0, .15]]
        XT = torch.from_numpy(G)
        with torch.inference_mode():
            for begin in range(0, len(Q), 4):
                q = Q[begin:begin + 4]
                scores = q @ G.T
                ids = np.stack([np.lexsort((gallery, -s))[:400] for s in scores])
                ranks[0, begin:begin + len(q)] = gallery[ids[:, :20]]
                local = G[ids]
                for index, weighted in [(1, False), (2, True)]:
                    weight = np.maximum(np.sum(local[:, :10] * q[:, None, :], axis=2), 0) ** 3 if weighted else np.ones((len(q), 10), np.float32)
                    qe = upstream.unit(q + np.sum(local[:, :10] * weight[:, :, None], axis=1))
                    ss = np.einsum('bmd,bd->bm', local, qe)
                    rr = np.stack([ii[np.lexsort((gallery[ii], -s))[:20]] for ii, s in zip(ids, ss)])
                    ranks[index, begin:begin + len(q)] = gallery[rr]
                for index, (aug, rerank) in enumerate(sg, 3):
                    rt = torch.from_numpy(ids.T.copy())
                    rr = rerank(rt, *aug(XT, torch.from_numpy(q), rt)).T.numpy()[:, :20]
                    ranks[index, begin:begin + len(q)] = gallery[rr]
                if begin % 256 == 0:
                    print(name, 'rerank', begin, 'seconds', round(time.perf_counter() - start, 1), flush=True)
        hits = np.any(ranks[:, :, :5] == gold[None, :, None], axis=2)
        rows = []
        for k, config in enumerate(configs):
            delta = hits[k].astype(float) - hits[0]
            rows.append({'operator': config, 'R5': float(hits[k].mean()), 'delta_pp': float(delta.mean() * 100),
                         'CI95_delta_pp': upstream.bootstrap(delta, meta),
                         'top1_changes': int(np.sum(ranks[k, :, 0] != ranks[0, :, 0]))})
        np.savez_compressed(OUT / f'factorial_{name}.npz', ranks=ranks, query_indices=take, gallery=gallery, gold=gold)
        cfg = SimpleNamespace(GCR=SimpleNamespace(WITH_GPU=False, BETA1=.2, BETA2=.2, LAMBDA1=4.2, LAMBDA2=4.2, SCALE=1, MODE='normal'), COMMON=SimpleNamespace(VERBOSE=False))
        cohorts = []
        ref, refmargin = None, None
        with torch.inference_mode():
            for size, reverse in [(1024, False), (64, False), (1024, True)]:
                ranked = np.empty((len(Q), 20), np.int32)
                margin = np.empty(len(Q))
                order = np.arange(len(Q))[::-1] if reverse else np.arange(len(Q))
                tick = time.perf_counter()
                for begin in range(0, len(Q), size):
                    ix = order[begin:begin + size]
                    features = torch.from_numpy(np.vstack([Q[ix], G]).astype(np.float64))
                    labels = np.zeros((len(features), 2), np.int64)
                    for _ in range(3):
                        features = cheb(cfg, features, labels)
                    ss = (features[:len(ix)] @ features[len(ix):].T).numpy()
                    assert np.isfinite(ss).all()
                    pos = np.stack([np.lexsort((gallery, -s))[:20] for s in ss])
                    ranked[ix] = gallery[pos]
                    margin[ix] = np.take_along_axis(ss, pos[:, :1], axis=1)[:, 0] - np.take_along_axis(ss, pos[:, 1:2], axis=1)[:, 0]
                if ref is None:
                    ref, refmargin = ranked.copy(), margin.copy()
                change = ranked[:, 0] != ref[:, 0]
                row = {'batch': size, 'reverse': reverse, 'top1_changes': int(change.sum()),
                       'ordered10_changes': int(np.any(ranked[:, :10] != ref[:, :10], axis=1).sum()),
                       'set10_changes': int(np.any(np.sort(ranked[:, :10], axis=1) != np.sort(ref[:, :10], axis=1), axis=1).sum()),
                       'strict_first_changes': int((change & (margin > 1e-8) & (refmargin > 1e-8)).sum()),
                       'R5': float(np.any(ranked[:, :5] == gold[:, None], axis=1).mean()),
                       'seconds': time.perf_counter() - tick}
                cohorts.append(row)
                np.savez_compressed(OUT / f'factorial_{name}_cohort{size}_reverse{int(reverse)}.npz', ranks=ranked, margin=margin, gold=gold, query_indices=take, gallery=gallery)
                print(name, 'cohort', row, flush=True)
        result = {'representation': name, 'query_count': len(Q), 'gallery_count': len(G),
                  'gallery_gold_coverage': float(np.isin(gold, gallery).mean()), 'reranking': rows, 'cohort': cohorts,
                  'total_seconds': time.perf_counter() - start}
        summary.append(result)
        (OUT / 'factorial_partial.json').write_text(json.dumps(summary, indent=2))
        print(name, 'COMPLETE', [(r['operator'], round(100 * r['R5'], 2)) for r in rows], flush=True)
    (OUT / 'factorial_summary.json').write_text(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
