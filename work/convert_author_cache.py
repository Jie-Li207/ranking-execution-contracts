import os
os.environ['OMP_NUM_THREADS'] = '2'
import argparse, zipfile, json, ijson, hashlib, datetime
from pathlib import Path
import numpy as np
p = argparse.ArgumentParser()
p.add_argument('dataset', choices=['WikiMEL', 'RichpediaMEL'])
args = p.parse_args()
r = Path(__import__('os').environ.get('ICDE_STUDY_ROOT', str(Path(__file__).resolve().parents[1])))
assert (r / 'work/preregistration_confirmatory_native.json').exists()
archive = r / 'archives' / (args.dataset + '.zip')
out = r / 'data/author_cache' / args.dataset
out.mkdir(parents=True, exist_ok=True)
z = zipfile.ZipFile(archive)
names = [x.filename for x in z.infolist() if x.filename.endswith('.json') and (not x.filename.startswith('__MACOSX'))]

def find(s):
    candidates = [n for n in names if n.rsplit('/', 1)[-1].lower() == s.lower()]
    assert len(candidates) == 1, (s, candidates)
    return candidates[0]

def read(s):
    return json.load(z.open(find(s)))
tests = [n for n in names if n.rsplit('/', 1)[-1].lower().endswith('_test.json')]
assert len(tests) == 1
queries = json.load(z.open(tests[0]))
kb = read('kb_entity.json')
qid = read('qid2id.json')
print('schema', len(kb), len(queries), 'querykeys', list(queries[0]), 'kbkeys', list(kb[0]), flush=True)
for i, e in enumerate(kb):
    assert int(qid[e['qid']]) == i
checks = {}
N = len(kb)
Q = len(queries)

def save(name, arr):
    a = np.asarray(arr, dtype=np.float32)
    b = a.astype(np.float16)
    error = float(np.max(np.abs(a - b.astype(np.float32)))) if a.size else 0
    checks[name] = {'shape': list(a.shape), 'half_roundtrip_max_abs': error}
    np.save(out / (name + '.npy'), b if error == 0 else a)

def vec(v):
    a = np.asarray(v, dtype=np.float32).reshape(-1, 512)
    assert a.shape[0] >= 1
    return a[0]
text = np.empty((N, 512), np.float32)
with z.open(find('entity_text_clip.json')) as f:
    count = 0
    for i, v in enumerate(ijson.items(f, 'item', use_float=True)):
        assert i < N
        text[i] = vec(v)
        count = i + 1
assert count == N
save('entity_text', text)
del text
print('entity text done', flush=True)
needed = {}
for i, e in enumerate(kb):
    if e['image_list']:
        needed.setdefault(e['image_list'][0], []).append(i)
images = np.zeros((N, 512), np.float32)
mask = np.zeros(N, bool)
with z.open(find('entity_image_clip.json')) as f:
    for key, v in ijson.kvitems(f, '', use_float=True):
        if key in needed:
            images[needed[key]] = vec(v)
            mask[needed[key]] = True
save('entity_image', images)
np.save(out / 'entity_image_mask.npy', mask)
del images
print('entity image done', int(mask.sum()), flush=True)
qimages = np.zeros((Q, 512), np.float32)
qmask = np.zeros(Q, bool)
need = {}
for i, q in enumerate(queries):
    name = q['imgPath']
    name = name if name.endswith('.jpg') else name.rsplit('.', 1)[0] + '.jpg'
    if len(name) > 1:
        need.setdefault(name, []).append(i)
with z.open(find('mention_image_clip.json')) as f:
    for key, v in ijson.kvitems(f, '', use_float=True):
        if key in need:
            qimages[need[key]] = vec(v)
            qmask[need[key]] = True
save('query_image', qimages)
np.save(out / 'query_image_mask.npy', qmask)
del qimages
qt = np.empty((Q, 512), np.float32)
with z.open(find('mention_add_cap_clip.json' if args.dataset == 'WikiMEL' else 'mention_add_caption_clip.json')) as f:
    count = 0
    for i, v in enumerate(ijson.items(f, 'item', use_float=True)):
        assert i < Q
        qt[i] = vec(v)
        count = i + 1
assert count == Q
save('query_text', qt)
(out / 'queries.json').write_text(json.dumps(queries, ensure_ascii=False), encoding='utf-8')
(out / 'kb_metadata.json').write_text(json.dumps([{'qid': e['qid'], 'entity_name': e['entity_name'], 'image_list': e['image_list']} for e in kb], ensure_ascii=False), encoding='utf-8')
np.save(out / 'gold.npy', np.array([int(qid[q['answer']]) for q in queries]))
report = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'checks': checks, 'entities': N, 'queries': Q, 'valid_entity_images': int(mask.sum()), 'valid_query_images': int(qmask.sum()), 'missing_query_image_paths': [queries[i]['imgPath'] for i in range(Q) if not qmask[i]], 'note': 'Published cached LLM inputs; generation quality not independently reproduced. No retrieval outcomes computed by this conversion.'}
(out / 'manifest.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != 'missing_query_image_paths'}, indent=2), flush=True)
