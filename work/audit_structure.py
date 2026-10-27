import json, os, ast, heapq
from pathlib import Path
import numpy as np
r = Path(os.environ.get('ICDE_STUDY_ROOT', str(Path(__file__).resolve().parents[1])))
d = r / 'results/revision_review'
report = []
dom = []
src = r / 'vendor/OpenMEL-main/maximum_spanning.py'
t = ast.parse(src.read_text(encoding='utf-8'))
ns = {'heapq': heapq}
exec(compile(ast.Module(body=[x for x in t.body if isinstance(x, (ast.FunctionDef, ast.ClassDef)) and x.name in ['MaxHeap', 'find', 'union', 'maximum_spanning_tree']], type_ignores=[]), str(src), 'exec'), ns)
for ds, folder, parent in [('WikiDiverse', 'wikidiverse_native3_t0.6_nall', 'objective_v2'), ('WikiMEL', 'WikiMEL_author3_t0.6_nall', 'author_objective'), ('RichpediaMEL', 'RichpediaMEL_author3_t0.6_nall', 'author_objective')]:
    rows = json.load(open(d / (ds + '_query_results.json')))
    a = dict(np.load(r / 'results' / parent / folder / 'inputs.npz'))
    lookup = {int(u): j for j, u in enumerate(a['unique'])}
    qm = {int(q): i for i, q in enumerate(a['selected'])}
    maps = {K: {z['qi']: z for z in rows if z['method'] == 'blend_raw' and z['param'] == 1 and (z['B'] == 5) and (z['K'] == K)} for K in [5, 10]}
    edge = [0, 0]
    nodes = [0, 0]
    changed = 0
    setchanged = 0
    for qi in maps[5]:
        i = qm[qi]
        rt = a['roots'][i].tolist()
        ss = []
        for k, K in enumerate([5, 10]):
            ed = {(u, int(v)) for u in rt for v, w in zip(a['nn'][lookup[u]][:K], a['dd'][lookup[u]][:K]) if int(v) not in rt and w > 0.6}
            ss.append(ed)
            edge[k] += len(ed)
            nodes[k] += len(set(rt) | {v for u, v in ed})
        changed += ss[0] != ss[1]
        setchanged += set(maps[5][qi]['selected']) != set(maps[10][qi]['selected'])
    report.append({'dataset': ds, 'n': len(maps[5]), 'K': [5, 10], 'edges': edge, 'reachable_entities': nodes, 'graphs_changed': changed, 'unscaled_B5_optimal_sets_changed': setchanged, 'original_same_512_root_R5': float(np.mean([z['root_hit'] for z in maps[5].values()])), 'R5_K5': float(np.mean([z['hit'] for z in maps[5].values()])), 'R5_K10_same_subset': float(np.mean([maps[10][qi]['hit'] for qi in maps[5]]))})
    for meth in ['attenuated', 'clipped']:
        for B in [5, 10]:
            z = [x for x in rows if x['method'] == meth and x['K'] == 10 and (x['B'] == B) and (x['partition'] == 'evaluate')]
            nativehits = []
            for x in z:
                i = qm[x['qi']]
                rt = a['roots'][i].tolist()
                s = a['scores'][i].tolist()
                adj = {}
                ww = {}
                for u, p in zip(rt, s):
                    ed = [(int(v), p * float(w) if meth == 'attenuated' else min(p, float(w))) for v, w in zip(a['nn'][lookup[u]], a['dd'][lookup[u]]) if int(v) not in rt and w > 0.6]
                    adj[u] = [v for v, w in ed]
                    ww[u] = [w for v, w in ed]
                rank = ns['maximum_spanning_tree'](-1, rt, s, adj, ww, B + 1)[0][1:]
                nativehits.append(x['gold'] in rank)
            dom.append({'dataset': ds, 'method': meth, 'B': B, 'n': len(z), 'ordered_native_recall': float(np.mean(nativehits)), 'enumeration_canonical_recall': float(np.mean([x['hit'] for x in z])), 'gold_min_recall': float(np.mean([x['gold_must'] for x in z])), 'gold_max_recall': float(np.mean([x['gold_can'] for x in z])), 'strict_first_exclusion': sum((x['strict'] for x in z)), 'ordered_hits': int(sum(nativehits))})
(d / 'neighbor_structure.json').write_text(json.dumps(report, indent=2))
(d / 'dominated_tie_audit.json').write_text(json.dumps(dom, indent=2))
print(json.dumps(report, indent=2))
print(json.dumps(dom, indent=2))
