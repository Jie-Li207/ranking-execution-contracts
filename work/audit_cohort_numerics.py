import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
from pathlib import Path
import json
import hashlib
import time
import numpy as np
import torch
from types import SimpleNamespace
from audit_reranking_grid import unit, load_sources
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rev7'
torch.set_num_threads(2)

def graph_support(Q, G, lam):
    Z = torch.from_numpy(np.vstack([Q, G]))
    norm = (Z * Z).sum(1, keepdim=True)
    dist = torch.clamp(-2 * Z @ Z.T + norm + norm.T, min=0)
    threshold = torch.clamp(dist.mean(1, keepdim=True) - lam * dist.std(1, keepdim=True), min=float(torch.diag(dist).max()))
    keep = dist <= threshold
    return (keep[len(Q):, len(Q):].numpy(), threshold[len(Q):, 0].numpy())

def main():
    protocol = {'date': '2026-10-28', 'trigger': 'Native float32 full-cohort reversal changes 7 first results and 155 top-ten sets at lambda=3; distinguish numerical tie sensitivity from cohort dependence', 'fixed_control': 'Same normalized input bytes, source function and parameters; convert to float64 before graph execution; no revised parameter selection', 'report': ['float64 full/64/reversed-full outcomes for lambda 3,4.2,6', 'strict first-result changes with both top-two score margins >1e-8', 'first-round gallery adjacency and moment threshold changes', 'gallery-only cosine batching control'], 'status': 'Post-outcome diagnostic; original float32 results retained'}
    protocol['additional_fixed_repeats'] = {'lambda': 4.2, 'batch': 64, 'permutation_seeds': [20261030, 20261031], 'chosen_before_double_precision_outcomes': True}
    (OUT / 'cohort_controls_protocol.json').write_text(json.dumps(protocol, indent=2))
    base = np.load(OUT / 'lambda3.0_batch1024_reverse0.npz')
    take = base['query_indices']
    gallery = base['gallery']
    gold = base['gold']
    f = ROOT / 'data/wikidiverse/features'
    G = unit(np.concatenate([unit(np.load(f / 'entity_name_desc.npy')[gallery].astype(np.float32)), unit(np.load(f / 'entity_image.npy')[gallery].astype(np.float32))], axis=1)).astype(np.float64)
    Q = unit(np.concatenate([unit(np.load(f / 'test_text.npy')[take].astype(np.float32)), unit(np.load(f / 'test_image.npy')[take].astype(np.float32))], axis=1)).astype(np.float64)
    _, cheb = load_sources()
    records = []
    mechanisms = []
    with torch.inference_mode():
        for lam in [3.0, 4.2, 6.0]:
            fulladj, fulltau = graph_support(Q, G, lam)
            partadj, parttau = graph_support(Q[:64], G, lam)
            mechanisms.append({'lambda': lam, 'changed_directed_gallery_edges': int(np.sum(fulladj != partadj)), 'full_directed_gallery_edges': int(fulladj.sum()), 'partition_directed_gallery_edges': int(partadj.sum()), 'changed_gallery_thresholds': int(np.sum(abs(fulltau - parttau) > 1e-12)), 'max_threshold_change': float(np.max(abs(fulltau - parttau)))})
            cfg = SimpleNamespace(GCR=SimpleNamespace(WITH_GPU=False, BETA1=0.2, BETA2=0.2, LAMBDA1=lam, LAMBDA2=lam, SCALE=1, MODE='normal'), COMMON=SimpleNamespace(VERBOSE=False))
            ref = None
            refmargin = None
            schedule = [(1024, False, None), (64, False, None), (1024, True, None)]
            if lam == 4.2:
                schedule.extend([(64, False, 20261030), (64, False, 20261031)])
            for size, reverse, seed in schedule:
                rank = np.empty((1024, 20), np.int32)
                margin = np.empty(1024)
                order = np.random.default_rng(seed).permutation(1024) if seed is not None else np.arange(1024)[::-1] if reverse else np.arange(1024)
                tick = time.perf_counter()
                for begin in range(0, 1024, size):
                    ix = order[begin:begin + size]
                    feat = torch.from_numpy(np.vstack([Q[ix], G]))
                    labels = np.zeros((len(feat), 2), np.int64)
                    for _ in range(3):
                        feat = cheb(cfg, feat, labels)
                    scores = (feat[:len(ix)] @ feat[len(ix):].T).numpy()
                    assert np.isfinite(scores).all()
                    pos = np.stack([np.lexsort((gallery, -ss))[:20] for ss in scores])
                    rank[ix] = gallery[pos]
                    margin[ix] = np.take_along_axis(scores, pos[:, :1], axis=1)[:, 0] - np.take_along_axis(scores, pos[:, 1:2], axis=1)[:, 0]
                if ref is None:
                    ref = rank.copy()
                    refmargin = margin.copy()
                changed = rank[:, 0] != ref[:, 0]
                filename = f'cohort64_lambda{lam}_batch{size}_reverse{int(reverse)}' + ('' if seed is None else f'_seed{seed}') + '.npz'
                record = {'lambda': lam, 'batch': size, 'reverse': reverse, 'seed': seed, 'file': filename, 'R5': float(np.mean(np.any(rank[:, :5] == gold[:, None], axis=1))), 'top1_changes': int(changed.sum()), 'top10_set_changes': int(np.any(np.sort(rank[:, :10], axis=1) != np.sort(ref[:, :10], axis=1), axis=1).sum()), 'well_separated_top1_changes': int(np.sum(changed & (margin > 1e-08) & (refmargin > 1e-08))), 'seconds': time.perf_counter() - tick}
                records.append(record)
                np.savez_compressed(OUT / filename, ranks=rank, margin=margin, gold=gold, query_indices=take, gallery=gallery)
                print(record, flush=True)
                (OUT / 'cohort_controls_partial.json').write_text(json.dumps(records, indent=2))
    (OUT / 'cohort_controls_summary.json').write_text(json.dumps({'rows': records, 'mechanisms': mechanisms}, indent=2))
if __name__ == '__main__':
    main()
