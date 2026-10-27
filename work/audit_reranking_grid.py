import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
from pathlib import Path
import sys
import argparse
import ast
import hashlib
import json
import time
import types
import warnings
from types import SimpleNamespace
import numpy as np
import torch
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rev7'
OUT.mkdir(parents=True, exist_ok=True)
threadpool_limits(2)
torch.set_num_threads(2)
warnings.filterwarnings('ignore', message='To copy construct from a tensor')

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4194304), b''):
            h.update(block)
    return h.hexdigest()

def unit(array):
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-08)

def load_sources():
    classes = {}
    for name in ['MDescAug', 'RerankwMDA']:
        path = ROOT / 'vendor/SuperGlobal/modules/reranking' / (name + '.py')
        module = types.ModuleType(name)
        exec(compile(path.read_text().replace('.cuda()', '.to(X.device)'), str(path), 'exec'), module.__dict__)
        classes[name] = getattr(module, name)
    path = ROOT / 'vendor/ChebGR/utils/re_ranking.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fast_mergesetfeat4'))
    ns = {'torch': torch, 'np': np, 'time': time}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
    return (classes, ns['fast_mergesetfeat4'])

def configurations():
    rows = [{'name': 'cosine', 'M': 400}]
    for M in [50, 100, 400]:
        rows.extend([{'name': method, 'M': M, 'k': 10} for method in ['AQE', 'alphaQE']])
        rows.extend([{'name': 'SuperGlobal', 'M': M, 'beta': beta} for beta in [0.0, 0.05, 0.15, 0.3]])
    for M in [100, 400]:
        rows.extend([{'name': 'ChebGR-singleton', 'M': M, 'lambda': lam, 'temperature': 0.2, 'rounds': 3} for lam in [4.2, 6.0]])
    return rows

def protocol():
    configs = configurations()
    record = {'date_local': '2026-10-28', 'stage': 'Fixed follow-up sensitivity protocol before these new outcomes; original test outcomes were already known, so this is not new independent confirmatory data.', 'representations': ['text', 'multimodal'], 'selection': '128 development queries ordered by SHA256(id); all 2078 non-NIL test queries; no test-based grid selection', 'development_selection': 'highest development Recall@5 including cosine identity; ties choose the earlier configuration', 'configurations': configs, 'primary': ['paired Recall@5 difference', 'image-cluster bootstrap 95% interval with 2000 draws', 'top1 changes', 'candidate coverage'], 'timing': 'two CPU threads; full-KB initial scan separated from fixed-candidate reranking; no end-to-end speedup claim', 'limits': 'ChebGR is an explicitly singleton-cohort transfer to a fixed candidate pool, not a reproduction of trained person-ReID features. AQE and alphaQE are explicit simple equation baselines. Full parameter grid is reported.', 'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in [ROOT / 'vendor/SuperGlobal/modules/reranking/MDescAug.py', ROOT / 'vendor/SuperGlobal/modules/reranking/RerankwMDA.py', ROOT / 'vendor/ChebGR/utils/re_ranking.py']}}
    path = OUT / 'reranking_protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == record
    else:
        path.write_text(json.dumps(record, indent=2))
    return configs

def bootstrap(delta, meta):
    groups = {}
    for i, row in enumerate(meta):
        groups.setdefault(row['imgPath'] or f'missing-{i}', []).append(i)
    counts = np.array([len(v) for v in groups.values()])
    sums = np.array([delta[v].sum() for v in groups.values()])
    rng = np.random.default_rng(20261028)
    draws = rng.integers(len(counts), size=(2000, len(counts)))
    return (100 * np.quantile(sums[draws].sum(1) / counts[draws].sum(1), [0.025, 0.975])).tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['text', 'multimodal'], required=True)
    args = parser.parse_args()
    configs = protocol()
    classes, cheb = load_sources()
    f = ROOT / 'data/wikidiverse/features'
    text = unit(np.load(f / 'entity_name_desc.npy').astype(np.float32))
    image = unit(np.load(f / 'entity_image.npy').astype(np.float32))
    X = text if args.mode == 'text' else unit(np.concatenate([text, image], axis=1))
    del text, image
    XT = torch.from_numpy(X)
    sg = {i: (classes['MDescAug'](M=c['M'], beta=c['beta']), classes['RerankwMDA'](M=c['M'], beta=c['beta'])) for i, c in enumerate(configs) if c['name'] == 'SuperGlobal'}
    summaries = []
    chosen = None
    for split in ['dev', 'test']:
        meta = json.loads((f / (split + '_metadata.json')).read_text(encoding='utf-8'))
        take = sorted(range(len(meta)), key=lambda i: hashlib.sha256(str(meta[i]['id']).encode()).hexdigest())[:128] if split == 'dev' else list(range(len(meta)))
        meta = [meta[i] for i in take]
        T = unit(np.load(f / (split + '_text.npy')).astype(np.float32))[take]
        I = unit(np.load(f / (split + '_image.npy')).astype(np.float32))[take]
        Q = T if args.mode == 'text' else unit(np.concatenate([T, I], axis=1))
        ranks = np.empty((len(configs), len(Q), 20), np.int32)
        candidates = np.empty((len(Q), 400), np.int32)
        timings = np.zeros(len(configs))
        scan = 0.0
        started = time.perf_counter()
        for begin in range(0, len(Q), 4):
            q = Q[begin:begin + 4]
            tick = time.perf_counter()
            scores = q @ X.T
            ids = np.argpartition(-scores, 399, axis=1)[:, :400]
            ids = np.stack([row[np.lexsort((row, -ss[row]))] for row, ss in zip(ids, scores)])
            scan += time.perf_counter() - tick
            candidates[begin:begin + len(q)] = ids
            ranks[0, begin:begin + len(q)] = ids[:, :20]
            with torch.inference_mode():
                for ci, c in enumerate(configs[1:], 1):
                    tick = time.perf_counter()
                    M = c['M']
                    if c['name'] == 'SuperGlobal':
                        a, b = sg[ci]
                        rt = torch.from_numpy(ids.T.copy())
                        intermediate = a(XT, torch.from_numpy(q), rt)
                        rr = b(rt, *intermediate).T.numpy()[:, :20]
                    elif c['name'] in ['AQE', 'alphaQE']:
                        local = X[ids[:, :M]]
                        weight = np.ones((len(q), 10), np.float32) if c['name'] == 'AQE' else np.maximum(np.sum(local[:, :10] * q[:, None, :], axis=2), 0) ** 3
                        query = unit(q + np.sum(local[:, :10] * weight[:, :, None], axis=1))
                        newscore = np.einsum('bmd,bd->bm', local, query)
                        rr = np.stack([ii[np.lexsort((ii, -ss))[:20]] for ii, ss in zip(ids[:, :M], newscore)])
                    else:
                        cfg = SimpleNamespace(GCR=SimpleNamespace(WITH_GPU=False, BETA1=0.2, BETA2=0.2, LAMBDA1=c['lambda'], LAMBDA2=c['lambda'], SCALE=1, MODE='normal'), COMMON=SimpleNamespace(VERBOSE=False))
                        rr = []
                        for qi, ii in zip(q, ids[:, :M]):
                            feat = torch.from_numpy(np.vstack([qi, X[ii]]))
                            labels = np.zeros((len(feat), 2), np.int64)
                            for _ in range(3):
                                feat = cheb(cfg, feat, labels)
                            assert torch.isfinite(feat).all()
                            ss = (feat[0] @ feat[1:].T).numpy()
                            rr.append(ii[np.lexsort((ii, -ss))[:20]])
                        rr = np.stack(rr)
                    ranks[ci, begin:begin + len(q)] = rr
                    timings[ci] += time.perf_counter() - tick
            if begin % 128 == 0:
                print(args.mode, split, begin, len(Q), 'seconds', round(time.perf_counter() - started, 1), flush=True)
        gold = np.array([z['answer_id'] for z in meta])
        hits = np.any(ranks[:, :, :5] == gold[None, :, None], axis=2)
        if split == 'dev':
            chosen = int(np.argmax(hits.mean(1)))
        rows = []
        for ci, c in enumerate(configs):
            delta = hits[ci].astype(float) - hits[0]
            row = {**c, 'config_index': ci, 'queries': len(Q), 'R5': float(hits[ci].mean()), 'delta_R5_pp': float(100 * delta.mean()), 'R1': float(np.mean(ranks[ci, :, 0] == gold)), 'top1_changes': int(np.sum(ranks[ci, :, 0] != ranks[0, :, 0])), 'rerank_seconds': float(timings[ci]), 'development_selected': ci == chosen}
            if split == 'test':
                row['CI95_delta_pp'] = bootstrap(delta, meta)
            rows.append(row)
        result = {'mode': args.mode, 'split': split, 'queries': len(Q), 'gallery': len(X), 'scan_seconds': scan, 'total_seconds': time.perf_counter() - started, 'selected_on_dev': chosen, 'candidate_R_at_M': {str(M): float(np.mean(np.any(candidates[:, :M] == gold[:, None], axis=1))) for M in [50, 100, 400]}, 'rows': rows}
        np.savez_compressed(OUT / f'reranking_{args.mode}_{split}.npz', ranks=ranks, candidates=candidates, query_indices=take, gold=gold)
        (OUT / f'reranking_{args.mode}_{split}.json').write_text(json.dumps(result, indent=2))
        summaries.append(result)
        print('COMPLETE', args.mode, split, 'dev_selected', configs[chosen], flush=True)
    (OUT / f'reranking_{args.mode}_summary.json').write_text(json.dumps(summaries, indent=2))
if __name__ == '__main__':
    main()
