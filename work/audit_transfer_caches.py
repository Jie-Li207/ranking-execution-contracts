import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
from pathlib import Path
import json
import time
import numpy as np
import torch
from types import SimpleNamespace
from audit_reranking_grid import unit, load_sources, bootstrap
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rev7'
torch.set_num_threads(2)

def main():
    protocol = {'date': '2026-10-28', 'scope': 'Fixed follow-up replication on the two existing author-cache datasets; no new held-out confirmation', 'datasets': ['WikiMEL', 'RichpediaMEL'], 'representations': ['text', 'multimodal'], 'configurations': ['cosine', 'SuperGlobal beta=0 M=400', 'SuperGlobal beta=.15 M=400', 'ChebGR-singleton lambda=4.2 M=400 temperature=.2 rounds=3'], 'selection': 'No dataset-specific parameter selection; report all four, including identity selected by previous WikiDiverse development exercise', 'inputs': 'Released OpenMEL cached CLIP text/image descriptors and complete entity tables; cached text includes upstream captioning; no claim of second-encoder validation', 'primary': 'Recall@5 with paired image-cluster bootstrap 95% intervals and top-one changes'}
    (OUT / 'transfer_caches_protocol.json').write_text(json.dumps(protocol, indent=2))
    classes, cheb = load_sources()
    allresults = []
    cfg = SimpleNamespace(GCR=SimpleNamespace(WITH_GPU=False, BETA1=0.2, BETA2=0.2, LAMBDA1=4.2, LAMBDA2=4.2, SCALE=1, MODE='normal'), COMMON=SimpleNamespace(VERBOSE=False))
    for ds in protocol['datasets']:
        f = ROOT / 'data/author_cache' / ds
        meta = json.loads((f / 'queries.json').read_text(encoding='utf-8'))
        gold = np.load(f / 'gold.npy')
        for mode in protocol['representations']:
            T = unit(np.load(f / 'entity_text.npy').astype(np.float32))
            X = T if mode == 'text' else unit(np.concatenate([T, unit(np.load(f / 'entity_image.npy').astype(np.float32))], axis=1))
            T = unit(np.load(f / 'query_text.npy').astype(np.float32))
            Q = T if mode == 'text' else unit(np.concatenate([T, unit(np.load(f / 'query_image.npy').astype(np.float32))], axis=1))
            del T
            XT = torch.from_numpy(X)
            ranks = np.empty((4, len(Q), 20), np.int32)
            sg = [(classes['MDescAug'](M=400, beta=b), classes['RerankwMDA'](M=400, beta=b)) for b in [0, 0.15]]
            start = time.perf_counter()
            with torch.inference_mode():
                for begin in range(0, len(Q), 4):
                    q = Q[begin:begin + 4]
                    scores = q @ X.T
                    ids = np.argpartition(-scores, 399, axis=1)[:, :400]
                    ids = np.stack([ii[np.lexsort((ii, -ss[ii]))] for ii, ss in zip(ids, scores)])
                    ranks[0, begin:begin + len(q)] = ids[:, :20]
                    rt = torch.from_numpy(ids.T.copy())
                    for ci, (a, b) in enumerate(sg, 1):
                        inter = a(XT, torch.from_numpy(q), rt)
                        ranks[ci, begin:begin + len(q)] = b(rt, *inter).T.numpy()[:, :20]
                    for j, ii in enumerate(ids):
                        feat = torch.from_numpy(np.vstack([q[j], X[ii]]))
                        labels = np.zeros((401, 2), np.int64)
                        for _ in range(3):
                            feat = cheb(cfg, feat, labels)
                        assert torch.isfinite(feat).all()
                        ss = (feat[0] @ feat[1:].T).numpy()
                        ranks[3, begin + j] = ii[np.lexsort((ii, -ss))[:20]]
                    if begin % 512 == 0:
                        print(ds, mode, begin, len(Q), 'seconds', round(time.perf_counter() - start, 1), flush=True)
            hit = np.any(ranks[:, :, :5] == gold[None, :, None], axis=2)
            rows = [{'method': name, 'R5': float(hit[i].mean()), 'delta_R5_pp': float(100 * np.mean(hit[i].astype(float) - hit[0])), 'CI95_delta_pp': bootstrap(hit[i].astype(float) - hit[0], meta), 'top1_changes': int(np.sum(ranks[i, :, 0] != ranks[0, :, 0]))} for i, name in enumerate(protocol['configurations'])]
            record = {'dataset': ds, 'mode': mode, 'queries': len(Q), 'gallery': len(X), 'rows': rows, 'total_seconds': time.perf_counter() - start}
            np.savez_compressed(OUT / f'transfer_{ds}_{mode}.npz', ranks=ranks, gold=gold)
            (OUT / f'transfer_{ds}_{mode}.json').write_text(json.dumps(record, indent=2))
            allresults.append(record)
            print('COMPLETE', ds, mode, rows, flush=True)
            del X, XT, Q
    (OUT / 'transfer_caches_summary.json').write_text(json.dumps(allresults, indent=2))
if __name__ == '__main__':
    main()
