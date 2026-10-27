import os, time, json, datetime, itertools
from pathlib import Path
import numpy as np
from enumeration_oracle import enumerate_roots
r = Path(os.environ.get('ICDE_STUDY_ROOT', str(Path(__file__).resolve().parents[1])))
out = r / 'results/enumerated'
out.mkdir(parents=True, exist_ok=True)
protocol = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'trigger': 'Full second-formulation audit found original HiGHS status0/gap0 value lower by3.8743019104003906e-7 for RichpediaMEL qi2536 B10. Gold hit unchanged in this case.', 'action': 'Enumerate every subset of <=10 roots for ALL directed graph conditions, with exact conditional top-child optimization; retain original MILP records unchanged.', 'unchanged': ['input graphs', 'query sets', 'budgets', 'thresholds', 'hit metrics', '1e-7 strict/tie guard', 'primary four contrasts', 'bootstrap procedure'], 'numerical': 'float64 sums of saved weights; exhaustive finite root-subset optimization, no solver stopping criterion', 'not_claimed': 'new scalable general graph algorithm'}
(r / 'work/protocol_numerical_correction.json').write_text(json.dumps(protocol, indent=2))
rng = np.random.default_rng(77241)
maxerr = 0.0
tests = 0
for case in range(200):
    nr = int(rng.integers(1, 5))
    nc = int(rng.integers(1, 6))
    rt = list(range(nr))
    s = np.round(rng.random(nr), 1)
    ed = [(u, v, float(np.round(rng.random(), 1))) for u in rt for v in range(nr, nr + nc) if rng.random() < 0.7]
    gold = int(rng.integers(nr + nc + 1))
    B = int(rng.integers(1, nr + nc + 1))
    solutions = []
    for mask in range(1 << nr + nc):
        sel = [v for v in range(nr + nc) if mask & 1 << v]
        if len(sel) > B:
            continue
        val = sum((s[u] for u in sel if u < nr))
        ok = True
        for v in sel:
            if v >= nr:
                cand = [w for u, t, w in ed if t == v and u in sel]
                if not cand:
                    ok = False
                    break
                val += max(cand)
        if ok:
            solutions.append((val, sel))
    en = enumerate_roots(rt, s, ed, [B], gold=gold)[0]
    best = max((v for v, x in solutions))
    err = abs(en['value'] - best)
    assert err < 1e-12
    maxerr = max(maxerr, err)
    force = max((v for v, x in solutions if 0 in x))
    assert abs(en['forced_first_value'] - force) < 1e-12
    can = any((gold in x and best - v <= 1e-07 for v, x in solutions))
    must = not any((gold not in x and best - v <= 1e-07 for v, x in solutions))
    assert can == en['gold_can_appear'] and must == en['gold_must_appear'], (case, en, solutions)
    tests += 1
(out / 'validation.json').write_text(json.dumps({'entity_subset_enumeration_cases': tests, 'max_error': maxerr, 'seed': 77241}, indent=2))
totals = []
for parent in ['author_objective', 'objective_v2']:
    for d in sorted((r / 'results' / parent).glob('*')):
        if not (d / 'inputs.npz').exists():
            continue
        a = dict(np.load(d / 'inputs.npz'))
        old = json.load(open(d / 'query_results.json'))
        by = {}
        for z in old:
            by.setdefault(z['qi'], []).append(z)
        threshold = json.load(open(d / 'protocol.json'))['threshold']
        lookup = {int(u): j for j, u in enumerate(a['unique'])}
        qm = {int(q): i for i, q in enumerate(a['selected'])}
        records = []
        diff = []
        changed_hits = 0
        start = time.perf_counter()
        for num, (qi, zz) in enumerate(by.items()):
            i = qm[qi]
            keep = a['scores'][i] > threshold
            rt = a['roots'][i][keep].astype(int).tolist()
            s = a['scores'][i][keep]
            ed = [(u, int(v), float(w)) for u in rt for v, w in zip(a['nn'][lookup[u]], a['dd'][lookup[u]]) if int(v) not in rt and w > threshold]
            t = time.perf_counter()
            en = enumerate_roots(rt, s, ed, [z['B'] for z in zz], gold=int(a['gold'][i]))
            elapsed = 1000 * (time.perf_counter() - t)
            for oldrow, x in zip(zz, en):
                z = oldrow.copy()
                z['milp_original_status'] = z['status']
                z['milp_original_exact_flag'] = z['exact']
                z['milp_solve_ms_original'] = z.pop('solve_ms', None)
                z.update(status=0, exact=True, opt_value=x['value'], optimal_set=x['set'], opt_hit=int(a['gold'][i]) in x['set'], forced_first_status=0, forced_first_value=x['forced_first_value'], strict_first_exclusion=x['strict_first_exclusion'], opt_hit_min=x['gold_must_appear'], opt_hit_max=x['gold_can_appear'], gold_tie_status=0, oracle='complete_root_subset_enumeration', enumeration_graph_ms=elapsed)
                assert z['opt_value'] >= z['greedy_value'] - 1e-07
                if oldrow['opt_value'] is not None:
                    delta = z['opt_value'] - oldrow['opt_value']
                    if abs(delta) > 1e-07:
                        diff.append({'qi': qi, 'B': z['B'], 'delta': delta, 'original_status': oldrow['status'], 'original_gold_hit': oldrow['opt_hit'], 'new_gold_hit': z['opt_hit']})
                changed_hits += int(z['opt_hit'] != oldrow['opt_hit'])
                records.append(z)
            if num % 500 == 0:
                print(d.name, num, len(by), round(time.perf_counter() - start, 1), flush=True)
        target = out / d.name
        target.mkdir(parents=True, exist_ok=True)
        summary = []
        for B in [1, 5, 10, 20]:
            z = [x for x in records if x['B'] == B]
            summary.append({'B': B, 'n': len(z), 'exact_n': len(z), 'greedy_suboptimal_n': sum((x['opt_value'] - x['greedy_value'] > 1e-07 for x in z)), 'mean_relative_gap': float(np.mean([(x['opt_value'] - x['greedy_value']) / x['opt_value'] for x in z])), 'optimal_recall': float(np.mean([x['opt_hit'] for x in z])), 'optimal_recall_min': float(np.mean([x['opt_hit_min'] for x in z])), 'optimal_recall_max': float(np.mean([x['opt_hit_max'] for x in z])), 'gold_tie_certified_n': len(z), 'greedy_recall': float(np.mean([x['greedy_hit'] for x in z])), 'root_recall_KminB10': float(np.mean([x['root_hit'] for x in z])), 'strict_first_exclusion_n': sum((x['strict_first_exclusion'] for x in z)), 'forced_first_certified_n': len(z), 'enumeration_graph_ms_median': float(np.median([x['enumeration_graph_ms'] for x in z]))})
        (target / 'query_results.json').write_text(json.dumps(records))
        (target / 'summary.json').write_text(json.dumps(summary, indent=2))
        (target / 'correction_comparison.json').write_text(json.dumps({'input': str(d), 'records': len(records), 'values_changed_above_guard': diff, 'selected_gold_hits_changed': changed_hits}, indent=2))
        totals.append({'run': d.name, 'graphs': len(by), 'values_changed': len(diff), 'hits_changed': changed_hits, 'seconds': time.perf_counter() - start})
        print('FINISHED', totals[-1], flush=True)
(out / 'run_status.json').write_text(json.dumps(totals, indent=2))
