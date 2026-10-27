import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
from pathlib import Path
import json
import hashlib
import time
import sqlite3
import numpy as np
import torch
from types import SimpleNamespace
from audit_reranking_grid import unit, load_sources, digest
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rev7'
torch.set_num_threads(2)

def main():
    protocol = {'date': '2026-10-28', 'status': 'Follow-up sensitivity on previously evaluated test inputs, not independent confirmatory evidence', 'queries': 'First 1024 non-NIL WikiDiverse test IDs by SHA256', 'gallery': 'Fixed union of the initial top-five candidates of all selected queries from full 132460-entity scan; no gold supplementation', 'parameters': {'lambda': [3.0, 4.2, 6.0], 'temperature': 0.2, 'rounds': 3}, 'cohorts': [1024, 256, 64], 'orders': ['hash', 'reverse'], 'primary': ['changed top-one IDs', 'changed top-ten sets', 'Recall@5', 'changed joined entity-type fields'], 'cache': 'Replay full and 64-query result requests twice; query-only key versus ordered-cohort, vectors, source, parameters and gallery fingerprint; compare each hit with freshly computed expected output', 'timing': 'Two CPU threads, graph and join stages separately; controlled local replay, not a deployed service or throughput claim', 'source_sha256': digest(ROOT / 'vendor/ChebGR/utils/re_ranking.py')}
    pp = OUT / 'cohort_protocol.json'
    if pp.exists():
        assert json.loads(pp.read_text()) == protocol
    else:
        pp.write_text(json.dumps(protocol, indent=2))
    f = ROOT / 'data/wikidiverse/features'
    allmeta = json.loads((f / 'test_metadata.json').read_text(encoding='utf-8'))
    take = np.array(sorted(range(len(allmeta)), key=lambda i: hashlib.sha256(str(allmeta[i]['id']).encode()).hexdigest())[:1024])
    meta = [allmeta[i] for i in take]
    gold = np.array([z['answer_id'] for z in meta])
    initial = np.load(OUT / 'reranking_multimodal_test.npz')['candidates'][take]
    gallery = np.unique(initial[:, :5])
    G = unit(np.concatenate([unit(np.load(f / 'entity_name_desc.npy')[gallery].astype(np.float32)), unit(np.load(f / 'entity_image.npy')[gallery].astype(np.float32))], axis=1))
    Q = unit(np.concatenate([unit(np.load(f / 'test_text.npy')[take].astype(np.float32)), unit(np.load(f / 'test_image.npy')[take].astype(np.float32))], axis=1))
    _, cheb = load_sources()
    kb = json.loads((ROOT / 'data/wikidiverse/cleaned/WikiDiverse/kb_entity.json').read_text(encoding='utf-8'))
    db = sqlite3.connect(':memory:')
    db.execute('create table entity (id integer primary key, qid text, name text, type text)')
    db.executemany('insert into entity values (?,?,?,?)', [(int(i), str(kb[i]['qid']), str(kb[i]['entity_name']), str(kb[i].get('instance', ''))) for i in gallery])
    del kb
    outputs = {}
    timings = {}
    records = []
    started = time.perf_counter()
    basehash = hashlib.sha256(G.tobytes() + gallery.tobytes() + Q.tobytes() + protocol['source_sha256'].encode()).hexdigest()
    with torch.inference_mode():
        for lam in protocol['parameters']['lambda']:
            cfg = SimpleNamespace(GCR=SimpleNamespace(WITH_GPU=False, BETA1=0.2, BETA2=0.2, LAMBDA1=lam, LAMBDA2=lam, SCALE=1, MODE='normal'), COMMON=SimpleNamespace(VERBOSE=False))
            for size, reverse in [(1024, False), (256, False), (64, False), (1024, True), (64, True)]:
                name = f'lambda{lam}_batch{size}_reverse{int(reverse)}'
                rank = np.empty((len(Q), 20), np.int32)
                order = np.arange(len(Q))[::-1] if reverse else np.arange(len(Q))
                tick = time.perf_counter()
                for begin in range(0, len(Q), size):
                    ix = order[begin:begin + size]
                    feat = torch.from_numpy(np.vstack([Q[ix], G]))
                    labels = np.zeros((len(feat), 2), np.int64)
                    for _ in range(3):
                        feat = cheb(cfg, feat, labels)
                    assert torch.isfinite(feat).all()
                    scores = (feat[:len(ix)] @ feat[len(ix):].T).numpy()
                    rank[ix] = np.stack([gallery[np.lexsort((gallery, -ss))[:20]] for ss in scores])
                seconds = time.perf_counter() - tick
                outputs[name] = rank
                timings[name] = seconds
                ref = outputs[f'lambda{lam}_batch1024_reverse0']
                tick = time.perf_counter()
                joined = [db.execute('select qid,name,type from entity where id=?', (int(i),)).fetchone() for i in rank[:, 0]]
                refjoined = [db.execute('select qid,name,type from entity where id=?', (int(i),)).fetchone() for i in ref[:, 0]]
                join_seconds = time.perf_counter() - tick
                record = {'name': name, 'lambda': lam, 'batch': size, 'reverse': reverse, 'queries': len(Q), 'gallery': len(G), 'R5': float(np.mean(np.any(rank[:, :5] == gold[:, None], axis=1))), 'top1_changes': int(np.sum(rank[:, 0] != ref[:, 0])), 'top10_set_changes': int(np.sum(np.any(np.sort(rank[:, :10], axis=1) != np.sort(ref[:, :10], axis=1), axis=1))), 'joined_type_changes': sum((a[2] != b[2] for a, b in zip(joined, refjoined))), 'graph_seconds': seconds, 'join_pair_seconds': join_seconds}
                records.append(record)
                np.savez_compressed(OUT / (name + '.npz'), ranks=rank, query_indices=take, gold=gold, gallery=gallery)
                (OUT / 'cohort_partial.json').write_text(json.dumps(records, indent=2))
                print(record, flush=True)
    cache_results = []
    for lam in protocol['parameters']['lambda']:
        for guarded in [False, True]:
            cache = {}
            hits = misses = wrong = 0
            tick = time.perf_counter()
            for size in [1024, 64, 1024, 64]:
                expected = outputs[f'lambda{lam}_batch{size}_reverse0']
                for begin in range(0, len(Q), size):
                    ix = np.arange(begin, min(begin + size, len(Q)))
                    scope = hashlib.sha256((basehash + str(lam)).encode() + ix.tobytes()).hexdigest() if guarded else ''
                    for i in ix:
                        key = (int(i), scope)
                        if key in cache:
                            hits += 1
                            wrong += int(not np.array_equal(cache[key], expected[i, :10]))
                        else:
                            misses += 1
                            cache[key] = expected[i, :10].copy()
            cache_results.append({'lambda': lam, 'guarded': guarded, 'requests': 4096, 'hits': hits, 'misses': misses, 'wrong_top10_hits': wrong, 'payload_bytes': sum((v.nbytes for v in cache.values())), 'lookup_seconds': time.perf_counter() - tick})
    summary = {'protocol': protocol, 'gallery': len(G), 'queries': len(Q), 'gallery_gold_coverage': float(np.mean(np.isin(gold, gallery))), 'baseline_R5': float(np.mean(np.any(initial[:, :5] == gold[:, None], axis=1))), 'rows': records, 'cache': cache_results, 'total_seconds': time.perf_counter() - started}
    (OUT / 'cohort_summary.json').write_text(json.dumps(summary, indent=2))
    print('COMPLETE', summary['total_seconds'], flush=True)
if __name__ == '__main__':
    main()
