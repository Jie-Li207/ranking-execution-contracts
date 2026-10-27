from pathlib import Path
import json, hashlib, shutil, numpy as np, time
from PIL import Image
r = Path(__file__).resolve().parents[1]
d = r / 'data/wikidiverse/cleaned/WikiDiverse'
out = r / 'results/rev4'
q = json.load(open(d / 'wiki_diverse_test.json', encoding='utf-8'))
kb = json.load(open(d / 'kb_entity.json', encoding='utf-8'))
z = json.load(open(r / 'results/rev3/superglobal_multimodal_test_queries.json'))
cache = {}
start = time.perf_counter()
catalog = json.load(open(out / 'placeholder_catalog.json'))
known = catalog['pixel_sha256']
placeholder_path = str(r / catalog['source_relative'])

def quality(p):
    key = str(p)
    if key not in cache:
        try:
            aa = np.asarray(Image.open(p).convert('RGB'))
            h = hashlib.sha256(aa.tobytes()).hexdigest()
            cache[key] = {'valid': True, 'placeholder': h == known, 'flat': int(aa.max()) == int(aa.min()), 'pixel_sha256': h, 'shape': list(aa.shape)}
        except Exception as e:
            cache[key] = {'valid': False, 'error': str(e)}
    c = cache[key]
    return c['valid'] and (not c['placeholder']) and (not c['flat'])

def entitypath(e):
    names = kb[e]['image_list']
    return d / 'kb_image' / names[0] if names else d / 'absent'
conditions = {'lost': lambda x: x['initial_1'] and (not x['full_5']), 'gained': lambda x: not x['initial_5'] and x['full_5'], 'retained': lambda x: x['initial_1'] and x['full_1'], 'candidate_miss': lambda x: not x['candidate_gold']}
result = []
for kind, fn in conditions.items():
    eligible = []
    for x in z:
        if not fn(x):
            continue
        pp = [d / 'mention_image' / q[x['qi']]['imgPath']] + [entitypath(e) for e in [x['gold']] + [x[k][0] for k in ['initial', 'augmentation', 'full']]]
        if all((quality(p) for p in pp)):
            eligible.append(x)
    if not eligible:
        print('NO ELIGIBLE', kind)
        continue
    x = min(eligible, key=lambda y: hashlib.sha256(y['id'].encode()).hexdigest())
    row = {'kind': kind, 'eligible_n': len(eligible), 'selection': 'minimum SHA256(query id) within outcome stratum with all first-source images present, decodable, non-flat and not the known missing-image placeholder; availability condition only, no visual-aesthetic choice', 'result': x, 'query': q[x['qi']], 'query_image': str(d / 'mention_image' / q[x['qi']]['imgPath']), 'entities': [{'role': role, 'metadata': kb[e], 'image': str(entitypath(e))} for role, e in zip(['gold', 'initial', 'augmentation', 'full'], [x['gold']] + [x[k][0] for k in ['initial', 'augmentation', 'full']])]}
    result.append(row)
    print(kind, x['id'], q[x['qi']]['mentions'], [(a['role'], a['metadata']['entity_name']) for a in row['entities']], flush=True)
clean = []
counts = {'queries': len(q), 'query_placeholder': 0, 'query_flat': 0, 'gold_placeholder': 0, 'gold_flat': 0, 'query_invalid': 0, 'gold_invalid': 0}
for i, x in enumerate(z):
    qp = d / 'mention_image' / q[i]['imgPath']
    gp = entitypath(x['gold'])
    a = quality(qp)
    b = quality(gp)
    clean.append(a and b)
    for tag, p in [('query', qp), ('gold', gp)]:
        v = cache[str(p)]
        counts[tag + '_invalid'] += not v['valid']
        if v['valid']:
            counts[tag + '_placeholder'] += v['placeholder']
            counts[tag + '_flat'] += v['flat']
metrics = []
for mode in ['text', 'multimodal']:
    rows = json.load(open(r / f'results/rev3/superglobal_{mode}_test_queries.json'))
    chosen = [x for x, m in zip(rows, clean) if m]
    metrics.append({'mode': mode, 'queries': len(chosen), 'initial_R5': np.mean([x['initial_5'] for x in chosen]), 'augmentation_R5': np.mean([x['augmentation_5'] for x in chosen]), 'full_R5': np.mean([x['full_5'] for x in chosen]), 'full_gains': sum((x['full_5'] and (not x['initial_5']) for x in chosen)), 'full_losses': sum((x['initial_5'] and (not x['full_5']) for x in chosen))})
(out / 'visual_cases.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
(out / 'image_quality_audit.json').write_text(json.dumps({'counts': counts, 'clean_metrics': metrics, 'clean_indices': [i for i, b in enumerate(clean) if b], 'known_placeholder_pixel_sha256': known, 'known_placeholder_source': placeholder_path, 'scope': 'Post-hoc exact-known-placeholder and flat-image diagnostic. Original embeddings, metrics, and model selection unchanged. Does not certify arbitrary image semantics or all placeholder variants.', 'seconds': time.perf_counter() - start}, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'counts': counts, 'clean_metrics': metrics}, indent=2), flush=True)
