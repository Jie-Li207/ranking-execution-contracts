from pathlib import Path
import json, hashlib, collections
r = Path(__file__).resolve().parents[1]
f = r / 'data/wikidiverse/features'
images = r / 'data/wikidiverse/cleaned/WikiDiverse/mention_image'
out = r / 'results/rev3'
meta = {s: json.load(open(f / (s + '_metadata.json'), encoding='utf-8')) for s in ['train', 'dev', 'test']}
hashes = {}
for z in [x for seq in meta.values() for x in seq]:
    name = z['imgPath']
    p = images / name
    if name and name not in hashes and p.is_file():
        hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
earlier = {hashes[z['imgPath']] for s in ['train', 'dev'] for z in meta[s] if z['imgPath'] in hashes}
oldid = {str(z['id']) for s in ['train', 'dev'] for z in meta[s]}
oldgold = {z['answer_id'] for s in ['train', 'dev'] for z in meta[s]}
test = meta['test']
overlap = [i for i, z in enumerate(test) if hashes.get(z['imgPath']) in earlier]
dupids = [i for i, z in enumerate(test) if str(z['id']) in oldid]
seen = [i for i, z in enumerate(test) if z['answer_id'] in oldgold]
groups = collections.Counter((hashes.get(z['imgPath'], f'missing{i}') for i, z in enumerate(test)))
clean = set(range(len(test))) - set(overlap) - set(dupids)
report = {'queries': len(test), 'image_hash_overlap_prior': len(overlap), 'query_id_overlap_prior': len(dupids), 'gold_identity_seen_prior': len(seen), 'test_repeated_image_clusters': sum((v > 1 for v in groups.values())), 'clean_queries': len(clean), 'overlap_indices': overlap, 'id_overlap_indices': dupids, 'scope': 'Frozen CLIP and no label training in this extension; prior identity presence is described, not automatically leakage. Foundation-model pretraining membership unknown.', 'clean_slice': []}
for mode in ['text', 'multimodal']:
    rows = json.load(open(out / f'superglobal_{mode}_test_queries.json'))
    rr = [z for z in rows if z['qi'] in clean]
    report['clean_slice'].append({'mode': mode, 'n': len(rr), 'initial_R5': sum((z['initial_5'] for z in rr)) / len(rr), 'full_R5': sum((z['full_5'] for z in rr)) / len(rr)})
(out / 'holdout_integrity.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if not k.endswith('indices')}, indent=2))
