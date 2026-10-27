import os
for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[k] = '2'
from pathlib import Path
import sys, json, time, hashlib, types, warnings
import numpy as np, torch
from threadpoolctl import threadpool_limits
threadpool_limits(2)
torch.set_num_threads(2)
r = Path(__file__).resolve().parents[1]
out = r / 'results/rev3'
out.mkdir(parents=True, exist_ok=True)
mods = {}
patch = []
for name in ['MDescAug', 'RerankwMDA']:
    p = r / 'vendor/SuperGlobal/modules/reranking' / f'{name}.py'
    s = p.read_text(encoding='utf-8')
    edited = s.replace('.cuda()', '.to(X.device)')
    m = types.ModuleType(name)
    exec(compile(edited, str(p), 'exec'), m.__dict__)
    mods[name] = getattr(m, name)()
    patch.append({'file': name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'device_edits': s.count('.cuda()')})
warnings.filterwarnings('ignore', message='To copy construct from a tensor')

def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-08)

def native(X, Q, ranks):
    with torch.inference_mode():
        args = mods['MDescAug'](X, Q, ranks)
        second = mods['RerankwMDA'](ranks, *args)
    return (np.stack([x.numpy() for x in args[0]]), second.T.numpy())

def independent(x, q, ids):
    x = x[ids]
    sim = x @ x.T
    nn = np.argsort(-sim, axis=1, kind='stable')[:, :10]
    w = np.take_along_axis(sim, nn, axis=1) * 0.15
    w[:, 0] = 1.0
    v = np.sum(x[nn] * w[:, :, None], axis=1) / w.sum(1)[:, None]
    s = v @ q
    ix = np.argsort(-s, kind='stable')
    proto = np.max(v[ix[:10]], axis=0)
    v2 = (s[ix] + v[ix] @ proto) / 2
    return (ids[ix], ids[ix[np.argsort(-v2, kind='stable')]])

def boot(rows, key, base):
    clusters = {}
    for z in rows:
        clusters.setdefault(z['cluster'], []).append(float(z[key]) - float(z[base]))
    sums = np.array([sum(x) for x in clusters.values()])
    n = np.array([len(x) for x in clusters.values()])
    rng = np.random.default_rng(20261026)
    vals = []
    for _ in range(2000):
        ix = rng.integers(len(n), size=len(n))
        vals.append(sums[ix].sum() / n[ix].sum())
    return (100 * np.quantile(vals, [0.025, 0.975])).tolist()
f = r / 'data/wikidiverse/features'
for name in ['test_text.npy', 'test_image.npy']:
    assert (f / name).exists(), 'Encode held-out test first.'
text = unit(np.load(f / 'entity_name_desc.npy').astype(np.float32))
image = unit(np.load(f / 'entity_image.npy').astype(np.float32))
all_summary = []
validate = []
for mode in ['text', 'multimodal']:
    X = text if mode == 'text' else unit(np.concatenate([text, image], axis=1))
    Xt = torch.from_numpy(X)
    for split in ['dev', 'test']:
        meta = json.load(open(f / (split + '_metadata.json'), encoding='utf-8'))
        T = unit(np.load(f / (split + '_text.npy')).astype(np.float32))
        I = unit(np.load(f / (split + '_image.npy')).astype(np.float32))
        Q = T if mode == 'text' else unit(np.concatenate([T, I], axis=1))
        qis = sorted(range(len(meta)), key=lambda i: hashlib.sha256(str(meta[i]['id']).encode()).hexdigest())
        qis = qis[:32] if split == 'dev' else list(range(len(meta)))
        rows = []
        start = time.perf_counter()
        total_scan = 0
        total_rerank = 0
        for begin in range(0, len(qis), 4):
            take = qis[begin:begin + 4]
            q = Q[take]
            t = time.perf_counter()
            s = q @ X.T
            ids = np.argpartition(-s, 399, axis=1)[:, :400]
            ids = np.stack([ii[np.lexsort((ii, -ss[ii]))] for ii, ss in zip(ids, s)])
            total_scan += time.perf_counter() - t
            t = time.perf_counter()
            a, b = native(Xt, torch.from_numpy(q), torch.from_numpy(ids.T.copy()))
            total_rerank += time.perf_counter() - t
            if split == 'dev':
                for j, qi in enumerate(take):
                    a2, b2 = independent(X, Q[qi], ids[j])
                    validate.append({'mode': mode, 'qi': qi, 'stage1_equal': np.array_equal(a[j], a2), 'stage2_equal': np.array_equal(b[j], b2), 'top10_equal': np.array_equal(b[j, :10], b2[:10])})
                    assert np.array_equal(b[j, :10], b2[:10])
                aa, bb = native(Xt, torch.from_numpy(q[::-1].copy()), torch.from_numpy(ids[::-1].T.copy()))
                assert np.array_equal(bb[::-1], b)
                aa, bb = native(Xt, torch.from_numpy(q[:1]), torch.from_numpy(ids[:1].T.copy()))
                assert np.array_equal(bb[0], b[0])
            for j, qi in enumerate(take):
                z = {'qi': qi, 'id': str(meta[qi]['id']), 'gold': int(meta[qi]['answer_id']), 'cluster': meta[qi]['imgPath'] or f'missing-{qi}', 'initial': ids[j, :20].tolist(), 'augmentation': a[j, :20].tolist(), 'full': b[j, :20].tolist(), 'candidate_gold': bool(meta[qi]['answer_id'] in ids[j])}
                for method in ['initial', 'augmentation', 'full']:
                    for B in [1, 5, 10]:
                        z[f'{method}_{B}'] = z['gold'] in z[method][:B]
                rows.append(z)
            if begin % 256 == 0:
                print('SuperGlobal', mode, split, begin, len(qis), round(time.perf_counter() - start, 1), flush=True)
        (out / f'superglobal_{mode}_{split}_queries.json').write_text(json.dumps(rows, indent=2))
        summary = {'mode': mode, 'split': split, 'queries': len(rows), 'entities': len(X), 'candidate_count': 400, 'elapsed_s': time.perf_counter() - start, 'total_scan_s': total_scan, 'total_rerank_s': total_rerank, 'candidate_recall': np.mean([z['candidate_gold'] for z in rows]), 'methods': {}}
        for method in ['initial', 'augmentation', 'full']:
            summary['methods'][method] = {'R' + str(B): float(np.mean([z[f'{method}_{B}'] for z in rows])) for B in [1, 5, 10]}
            summary['methods'][method]['top1_changed'] = sum((z[method][0] != z['initial'][0] for z in rows))
            for B in [1, 5, 10]:
                summary['methods'][method][f'B{B}_gained_lost'] = [sum((z[f'{method}_{B}'] and (not z[f'initial_{B}']) for z in rows)), sum((z[f'initial_{B}'] and (not z[f'{method}_{B}']) for z in rows))]
            if split == 'test':
                summary['methods'][method]['R5_minus_initial_CI_pp'] = boot(rows, f'{method}_5', 'initial_5')
        all_summary.append(summary)
        print(json.dumps(summary), flush=True)
(out / 'superglobal_summary.json').write_text(json.dumps(all_summary, indent=2))
(out / 'superglobal_validation.json').write_text(json.dumps({'source': patch, 'independent_equations': validate, 'cohort_partition_and_permutation': 'PASS on 32 hashed dev queries per input mode', 'scope': 'Released reranking stages transferred to frozen CLIP; no trained SuperGlobal encoder reproduction'}, indent=2))
