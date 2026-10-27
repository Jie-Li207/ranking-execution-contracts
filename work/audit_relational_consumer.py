from pathlib import Path
import json
import hashlib
import sqlite3
import numpy as np
from contract_checks import compare_outputs, rewrite_decision, dependency_key
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rev7'
kb = json.loads((ROOT / 'data/wikidiverse/cleaned/WikiDiverse/kb_entity.json').read_text(encoding='utf-8'))
db = sqlite3.connect(':memory:')
db.execute('create table entity (id integer primary key, qid text, name text, type text)')
base = np.load(OUT / 'cohort64_lambda4.2_batch1024_reverse0.npz')
gallery = base['gallery']
take = base['query_indices']
db.executemany('insert into entity values (?,?,?,?)', [(int(i), str(kb[i]['qid']), str(kb[i]['entity_name']), str(kb[i].get('instance', ''))) for i in gallery])
del kb
clean = json.loads((ROOT / 'results/rev4/image_quality_audit.json').read_text())['clean_indices']
cleanmask = np.isin(take, clean)
records = []
rows = []
for lam in [3.0, 4.2, 6.0]:
    full = np.load(OUT / f'cohort64_lambda{lam}_batch1024_reverse0.npz')['ranks']
    part = np.load(OUT / f'cohort64_lambda{lam}_batch64_reverse0.npz')['ranks']
    fulltypes = [db.execute('select type from entity where id=?', (int(i),)).fetchone()[0] for i in full[:, 0]]
    parttypes = [db.execute('select type from entity where id=?', (int(i),)).fetchone()[0] for i in part[:, 0]]
    report = compare_outputs(full, part, take, 10)
    rows.append({'lambda': lam, 'comparison': report, 'decision': rewrite_decision(report), 'joined_type_changes': sum((a != b for a, b in zip(fulltypes, parttypes))), 'clean_queries': int(cleanmask.sum()), 'clean_top1_changes': int(np.sum((full[:, 0] != part[:, 0]) & cleanmask)), 'clean_top10_set_changes': int(np.sum(np.any(np.sort(full[:, :10], axis=1) != np.sort(part[:, :10], axis=1), axis=1) & cleanmask))})
    for guarded in [False, True]:
        cache = {}
        counters = {'hits': 0, 'misses': 0, 'wrong_top1_hits': 0, 'wrong_top10_order_hits': 0, 'wrong_top10_set_hits': 0, 'wrong_joined_type_hits': 0}
        for size in [1024, 64, 1024, 64]:
            expected = full if size == 1024 else part
            for begin in range(0, 1024, size):
                cohort = take[begin:begin + size]
                scope = hashlib.sha256(cohort.tobytes() + str(lam).encode()).hexdigest() if guarded else ''
                for i in range(begin, begin + size):
                    key = (int(take[i]), scope)
                    if key in cache:
                        cached = cache[key]
                        counters['hits'] += 1
                        counters['wrong_top1_hits'] += int(cached[0] != expected[i, 0])
                        counters['wrong_top10_order_hits'] += int(not np.array_equal(cached, expected[i, :10]))
                        counters['wrong_top10_set_hits'] += int(not np.array_equal(np.sort(cached), np.sort(expected[i, :10])))
                        ct = db.execute('select type from entity where id=?', (int(cached[0]),)).fetchone()[0]
                        et = db.execute('select type from entity where id=?', (int(expected[i, 0]),)).fetchone()[0]
                        counters['wrong_joined_type_hits'] += int(ct != et)
                    else:
                        counters['misses'] += 1
                        cache[key] = expected[i, :10].copy()
        records.append({'lambda': lam, 'guarded': guarded, 'requests': 4096, 'cached_ID_payload_bytes': sum((a.nbytes for a in cache.values())), **counters})
report = {'implementation': 'Actual in-process result-cache replay and SQLite primary-key lookups; same fixed dataset/source/parameters for each replay; guarded cohort scope added to that fixed namespace', 'rows': rows, 'cache': records, 'limitations': 'Constructed repeated-request trace over real inputs, not production traffic; cache payload excludes Python and fingerprint overhead; conditioned image-quality slice retains the original surrounding cohort'}
(OUT / 'consumer_summary.json').write_text(json.dumps(report, indent=2))
print(json.dumps({'rows': [{k: v for k, v in r.items() if k != 'comparison'} for r in rows], 'cache': records}, indent=2))
