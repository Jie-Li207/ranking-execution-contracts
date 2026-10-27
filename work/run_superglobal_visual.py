import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '2'
from pathlib import Path
import sys, json, time, pickle, types, ast, warnings
import numpy as np, torch
from threadpoolctl import threadpool_limits
threadpool_limits(2)
torch.set_num_threads(2)
r = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(r / 'work/deps'))
out = r / 'results/rev3'
mods = {}
for name in ['MDescAug', 'RerankwMDA']:
    p = r / 'vendor/SuperGlobal/modules/reranking' / f'{name}.py'
    m = types.ModuleType(name)
    exec(compile(p.read_text().replace('.cuda()', '.to(X.device)'), str(p), 'exec'), m.__dict__)
    mods[name] = getattr(m, name)()
e = types.ModuleType('evaluate')
p = r / 'vendor/diffusion/evaluate.py'
exec(compile(p.read_text().replace('np.in1d(', 'np.isin('), str(p), 'exec'), e.__dict__)
warnings.filterwarnings('ignore', message='To copy construct from a tensor')
ns = {'np': np, 'torch': torch, 'mods': mods}
tree = ast.parse((r / 'work/run_superglobal.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ['native', 'independent']], type_ignores=[]), 'superglobal_functions', 'exec'), ns)
run = ns['native']
ind = ns['independent']
allres = []
for ds in ['oxford5k', 'paris6k']:
    d = r / 'data/rev3_visual'
    Q = np.load(d / (ds + '_query.npy'))
    X = np.load(d / (ds + '_gallery.npy'))
    g = pickle.load(open(d / f'gnd_{ds}.pkl', 'rb'))
    s = Q @ X.T
    rank = np.argsort(-s, axis=1)
    full = rank.copy()
    aug = rank.copy()
    v = []
    times = []
    Xt = torch.from_numpy(X)
    for i in range(0, len(Q), 4):
        ids = rank[i:i + 4, :400]
        t = time.perf_counter()
        a, b = run(Xt, torch.from_numpy(Q[i:i + 4]), torch.from_numpy(ids.T.copy()))
        times.append(time.perf_counter() - t)
        full[i:i + 4, :400] = b
        aug[i:i + 4, :400] = a
        aa, bb = run(Xt, torch.from_numpy(Q[i:i + 4][::-1].copy()), torch.from_numpy(ids[::-1].T.copy()))
        assert np.array_equal(bb[::-1], b)
        for j in range(min(len(b), max(0, 32 - i))):
            aa2, bb2 = ind(X, Q[i + j], ids[j])
            v.append({'query': i + j, 'stage1_top10_equal': np.array_equal(aa2[:10], a[j, :10]), 'stage2_top10_equal': np.array_equal(bb2[:10], b[j, :10])})
        aa, bb = run(Xt, torch.from_numpy(Q[i:i + 1]), torch.from_numpy(ids[:1].T.copy()))
        assert np.array_equal(bb[0], b[0])
    results = {}
    for name, rr in [('direct', rank), ('augmentation', aug), ('full', full)]:
        mm, ap, _, _ = e.compute_map(rr.T, g['gnd'])
        results[name] = {'mAP': float(mm), 'top1_changed': int(np.sum(rr[:, 0] != rank[:, 0])), 'top10_set_changed': sum((set(a[:10]) != set(b[:10]) for a, b in zip(rank, rr)))}
        np.savez_compressed(out / f'superglobal_{ds}_{name}.npz', ranks=rr.astype(np.int32), aps=ap)
    result = {'dataset': ds, 'queries': len(Q), 'gallery': len(X), 'descriptor': 'published Siamac global 512d', 'results': results, 'rerank_total_s': sum(times), 'native_vs_independent': v, 'cohort_and_permutation_check': 'PASS', 'descriptor_norm_range': [float(np.linalg.norm(X, axis=1).min()), float(np.linalg.norm(X, axis=1).max())]}
    allres.append(result)
    print(json.dumps(result), flush=True)
(out / 'superglobal_visual_summary.json').write_text(json.dumps(allres, indent=2))
