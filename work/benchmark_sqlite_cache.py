import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
from pathlib import Path
import sys
import json
import time
import hashlib
import sqlite3
import warnings
from types import SimpleNamespace
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(os.environ.get('ICDE_STUDY_ROOT', str(Path(__file__).resolve().parents[1])))
STUDY = ROOT
OUT = ROOT / 'results/rev10/sqlite_service_v2'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(STUDY / 'work'))
import audit_reranking_grid as upstream
upstream.ROOT = STUDY
torch.set_num_threads(2)
threadpool_limits(2)
warnings.filterwarnings('ignore', message='To copy construct from a tensor')

def main():
    protocol = json.loads((ROOT / 'results/rev10/protocol.json').read_text())
    base = np.load(STUDY / 'results/rev7/cohort64_lambda4.2_batch1024_reverse0.npz')
    take, gallery = base['query_indices'], base['gallery']
    f = STUDY / 'data/wikidiverse/features'
    G = upstream.unit(np.concatenate([upstream.unit(np.load(f / name, mmap_mode='r')[gallery].astype(np.float32)) for name in ['entity_name_desc.npy', 'entity_image.npy']], axis=1)).astype(np.float64)
    Q = upstream.unit(np.concatenate([upstream.unit(np.load(f / name, mmap_mode='r')[take].astype(np.float32)) for name in ['test_text.npy', 'test_image.npy']], axis=1)).astype(np.float64)
    kb = json.loads((STUDY / 'data/wikidiverse/cleaned/WikiDiverse/kb_entity.json').read_text(encoding='utf-8'))
    entity = [(int(i), str(kb[i]['qid']), str(kb[i]['entity_name']), str(kb[i].get('instance', ''))) for i in gallery]
    entity_types = {row[0]: row[3] for row in entity}
    del kb
    _, cheb = upstream.load_sources()
    cfg = SimpleNamespace(GCR=SimpleNamespace(WITH_GPU=False, BETA1=.2, BETA2=.2, LAMBDA1=4.2, LAMBDA2=4.2, SCALE=1, MODE='normal'), COMMON=SimpleNamespace(VERBOSE=False))
    namespace = hashlib.sha256(G.tobytes() + Q.tobytes() + gallery.tobytes() + take.tobytes() + json.dumps(vars(cfg.GCR), sort_keys=True).encode() + (STUDY / 'vendor/ChebGR/utils/re_ranking.py').read_bytes()).digest()

    def execute(ix):
        with torch.inference_mode():
            features = torch.from_numpy(np.vstack([Q[ix], G]))
            labels = np.zeros((len(features), 2), np.int64)
            for _ in range(3):
                features = cheb(cfg, features, labels)
            scores = (features[:len(ix)] @ features[len(ix):].T).numpy()
            assert np.isfinite(scores).all()
        return gallery[np.stack([np.lexsort((gallery, -s))[:10] for s in scores])].astype('<i4')

    records = []
    for seed in protocol['service_control']['seeds']:
        for stable in [0, .5, 1]:
            first = [np.arange(i, i + 64) for i in range(0, 1024, 64)]
            kept = int(stable * 16)
            rng = np.random.default_rng(seed)
            shuffled = rng.permutation(np.arange(kept * 64, 1024))
            second = first[:kept] + [shuffled[i:i + 64] for i in range(0, len(shuffled), 64)]
            batches = first + second
            expected_path = OUT / f'expected_seed{seed}_stable{stable}.npz'
            reference = np.stack([execute(ix) for ix in batches])
            np.savez_compressed(expected_path, ranks=reference, cohorts=np.stack(batches), query_indices=take, gallery=gallery)
            policies = ['none', 'query', 'cohort']
            rng.shuffle(policies)
            for policy in policies:
                name = f'seed{seed}_stable{stable}_{policy}'
                path = OUT / (name + '.sqlite')
                assert not path.exists(), name
                db = sqlite3.connect(path)
                db.execute('pragma journal_mode=WAL')
                db.execute('pragma synchronous=NORMAL')
                db.execute('create table entity(id integer primary key, qid text, name text, type text)')
                db.executemany('insert into entity values(?,?,?,?)', entity)
                db.execute('create table request(batch integer, position integer, query_index integer, primary key(batch,position))')
                db.executemany('insert into request values(?,?,?)', [(b, j, int(i)) for b, ix in enumerate(batches) for j, i in enumerate(ix)])
                db.execute('create table result_cache(key blob primary key, ids blob not null) without rowid')
                db.commit()
                initial_pages = db.execute('pragma page_count').fetchone()[0]
                page_bytes = db.execute('pragma page_size').fetchone()[0]
                hits = misses = computed = 0
                returned = np.empty((32, 64, 10), np.int32)
                joined_types = []
                stage_times, request_ms = [], []
                all_start = time.perf_counter()
                for b in range(32):
                    start = time.perf_counter()
                    ix = np.array([row[0] for row in db.execute('select query_index from request where batch=? order by position', (b,))])
                    scope = hashlib.sha256(namespace + take[ix].tobytes()).digest() if policy == 'cohort' else namespace
                    keys = [hashlib.sha256(scope + np.int64(take[i]).tobytes()).digest() for i in ix]
                    result = np.empty((len(ix), 10), np.int32)
                    absent = []
                    for j, key in enumerate(keys):
                        row = None if policy == 'none' else db.execute('select ids from result_cache where key=?', (key,)).fetchone()
                        if row is None:
                            misses += 1
                            absent.append(j)
                        else:
                            result[j] = np.frombuffer(row[0], dtype='<i4')
                            hits += 1
                    graph_time = 0.
                    if absent:
                        tick = time.perf_counter()
                        fresh = execute(ix)
                        graph_time = time.perf_counter() - tick
                        computed += 1
                        for j in absent:
                            result[j] = fresh[j]
                            if policy != 'none':
                                db.execute('insert into result_cache values(?,?)', (keys[j], fresh[j].tobytes()))
                        db.commit()
                    types = [db.execute('select type from entity where id=?', (int(i),)).fetchone()[0] for i in result[:, 0]]
                    elapsed = time.perf_counter() - start
                    returned[b] = result
                    joined_types.extend(types)
                    request_ms.extend([elapsed * 1000] * len(ix))
                    stage_times.append({'batch': b, 'seconds': elapsed, 'graph_seconds': graph_time, 'misses': len(absent)})
                wall = time.perf_counter() - all_start
                expected_types = [entity_types[int(i)] for i in reference[:, :, 0].flatten()]
                flat, ref = returned.reshape(-1, 10), reference.reshape(-1, 10)
                payload = db.execute('select coalesce(sum(length(key)+length(ids)),0) from result_cache').fetchone()[0]
                cache_pages = (db.execute('pragma page_count').fetchone()[0] - initial_pages) * page_bytes
                db.execute('pragma wal_checkpoint(TRUNCATE)')
                db.close()
                row = {'seed': seed, 'stable_cohort_fraction': stable, 'policy': policy, 'requests': 2048,
                       'hits': hits, 'misses': misses, 'graph_calls': computed,
                       'incorrect_ordered10': int(np.any(flat != ref, axis=1).sum()),
                       'incorrect_set10': int(np.any(np.sort(flat, axis=1) != np.sort(ref, axis=1), axis=1).sum()),
                       'incorrect_first': int(np.sum(flat[:, 0] != ref[:, 0])),
                       'incorrect_joined_type': sum(a != b for a, b in zip(joined_types, expected_types)),
                       'wall_seconds': wall, 'latency_p50_ms': float(np.median(request_ms)),
                       'latency_p95_ms': float(np.quantile(request_ms, .95)), 'payload_bytes': payload,
                       'cache_page_bytes': cache_pages, 'sqlite_total_bytes': path.stat().st_size,
                       'stage_times': stage_times}
                assert row['incorrect_ordered10'] == 0 if policy != 'query' else True
                np.savez_compressed(OUT / (name + '.npz'), ranks=returned, latency_ms=request_ms)
                records.append(row)
                (OUT / 'partial.json').write_text(json.dumps(records, indent=2))
                print({k: v for k, v in row.items() if k != 'stage_times'}, flush=True)
    (OUT / 'summary.json').write_text(json.dumps({'protocol': protocol['service_control'], 'rows': records}, indent=2))

if __name__ == '__main__':
    main()
