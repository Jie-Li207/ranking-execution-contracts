import os, sys, json, time, hashlib
from pathlib import Path
for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[k] = '2'
r = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(r / 'work/deps'))
os.environ['TORCH_HOME'] = str(r / 'models')
import numpy as np, torch, clip
from PIL import Image
from urllib.parse import unquote
torch.set_num_threads(2)
d = r / 'data/wikidiverse/cleaned/WikiDiverse'
f = r / 'data/wikidiverse/features'
out = r / 'results/rev3'
out.mkdir(parents=True, exist_ok=True)
assert (r / 'work/preregistration_rev3.json').exists()
if (f / 'test_text.npy').exists() or (f / 'test_image.npy').exists():
    raise SystemExit('Refusing to overwrite first-use test encodings; inspect existing manifest.')
qid = json.load(open(d / 'qid2id.json', encoding='utf-8'))
raw = json.load(open(d / 'wiki_diverse_test.json', encoding='utf-8'))
rows = [z for z in raw if z['answer'] in qid]
meta = [{'id': z['id'], 'answer_id': qid[z['answer']], 'imgPath': z['imgPath'], 'mentions': z['mentions'], 'sentence': z['sentence']} for z in rows]
(f / 'test_metadata.json').write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
model, prep = clip.load(str(r / 'models/ViT-B-32.pt'), device='cpu', jit=False)
model.eval()
start = time.perf_counter()
tt = []
ii = []
mask = []
fail = []
with torch.inference_mode():
    for i in range(0, len(rows), 32):
        batch = rows[i:i + 32]
        tokens = clip.tokenize([unquote(z['mentions']) + ' [SEP] ' + z['sentence'] for z in batch], truncate=True)
        x = model.encode_text(tokens).float()
        x /= x.norm(dim=-1, keepdim=True).clamp_min(1e-08)
        tt.append(x.numpy().astype(np.float16))
        ims = []
        valid = []
        for j, z in enumerate(batch):
            try:
                with Image.open(d / 'mention_image' / z['imgPath']) as im:
                    ims.append(prep(im.convert('RGB')))
                    valid.append(j)
            except Exception as e:
                fail.append({'id': z['id'], 'error': str(e)})
        a = np.zeros((len(batch), 512), np.float16)
        b = np.zeros(len(batch), bool)
        if ims:
            x = model.encode_image(torch.stack(ims)).float()
            x /= x.norm(dim=-1, keepdim=True).clamp_min(1e-08)
            a[valid] = x.numpy().astype(np.float16)
            b[valid] = True
        ii.append(a)
        mask.append(b)
        if i % 256 == 0:
            print('test encoded', i, len(rows), 'seconds', round(time.perf_counter() - start, 1), flush=True)
np.save(f / 'test_text.npy', np.concatenate(tt))
np.save(f / 'test_image.npy', np.concatenate(ii))
np.save(f / 'test_image_mask.npy', np.concatenate(mask))
record = {'device': 'cpu', 'threads': 2, 'dtype': 'fp32 encoder, normalized float16 storage', 'queries': len(rows), 'NIL_excluded': len(raw) - len(rows), 'elapsed_s': time.perf_counter() - start, 'failed_images': fail, 'hashes': {str(p.relative_to(r)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [d / 'wiki_diverse_test.json', f / 'test_text.npy', f / 'test_image.npy', r / 'models/ViT-B-32.pt', r / 'work/preregistration_rev3.json']}}
(out / 'test_encoding_manifest.json').write_text(json.dumps(record, indent=2))
print('finished', len(rows), round(record['elapsed_s'], 1), flush=True)
