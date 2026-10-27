from pathlib import Path
import sys, json, time, hashlib, collections
from difflib import SequenceMatcher
import numpy as np
r = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(r / 'work/deps'))
from rapidfuzz import process, fuzz
out = r / 'results/rev3'
out.mkdir(parents=True, exist_ok=True)
summ = []
for ds in ['WikiMEL', 'RichpediaMEL']:
    f = r / 'data/author_cache' / ds
    queries = json.load(open(f / 'queries.json', encoding='utf-8'))
    kb = json.load(open(f / 'kb_metadata.json', encoding='utf-8'))
    names = [x['entity_name'] for x in kb]
    lens = np.array([len(x) for x in names])
    exact = {s: i for i, s in enumerate(names)}
    cache = {}
    rows = []
    start = time.perf_counter()
    negative_scans = 0
    comparisons = 0
    for qi, q in enumerate(queries):
        s = q['mentions']
        if s not in cache:
            witness = exact.get(s)
            tested = 0
            if witness is None:
                for t, score, i in process.extract(s, names, scorer=fuzz.ratio, limit=30):
                    tested += 1
                    if SequenceMatcher(None, s, t).ratio() >= 0.5:
                        witness = i
                        break
            if witness is None:
                m = len(s)
                possible = np.flatnonzero(4 * np.minimum(m, lens) >= m + lens)
                negative_scans += 1
                for i in possible:
                    tested += 1
                    if SequenceMatcher(None, s, names[i]).ratio() >= 0.5:
                        witness = int(i)
                        break
            comparisons += tested
            cache[s] = {'witness': witness, 'matched': witness is not None, 'exact_tests': tested}
            if witness is not None:
                assert SequenceMatcher(None, s, names[witness]).ratio() >= 0.5
        rows.append({'qi': qi, 'query_id': str(q.get('id', qi)), **cache[s]})
        if qi % 1000 == 0:
            print(ds, qi, len(queries), round(time.perf_counter() - start, 1), flush=True)
    old = json.load(open(r / 'results/author_reproduction' / ds / 'query_results.json'))
    mismatch = [z['qi'] for z, w in zip(rows, old) if z['matched'] != w['name_filter_matched']]
    assert not mismatch
    (out / f'{ds}_native_branch_certificates.json').write_text(json.dumps(rows, indent=2))
    v = {'dataset': ds, 'queries': len(rows), 'matched': sum((z['matched'] for z in rows)), 'fallback': sum((not z['matched'] for z in rows)), 'unique_mentions': len(cache), 'exhaustive_length_feasible_scans': negative_scans, 'exact_SequenceMatcher_tests': comparisons, 'mismatches': mismatch, 'seconds': time.perf_counter() - start, 'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [f / 'queries.json', f / 'kb_metadata.json']}}
    summ.append(v)
    print(json.dumps(v), flush=True)
(out / 'native_branch_verification.json').write_text(json.dumps({'status': 'PASS', 'datasets': summ, 'proof': 'similar_strings is SequenceMatcher.ratio; positive witnesses suffice; negative cases exhaust every name not excluded by its exact length upper bound. Repairs to image handling, adjacency containers and inner-loop variables cannot change this text-only predicate.', 'limit': 'Certifies coverage, not the complete original upstream model or its published accuracy.'}, indent=2))
