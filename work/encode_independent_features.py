import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
from pathlib import Path
import json
import hashlib
import time
from urllib.parse import unquote
import requests
import numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights
from transformers import AutoTokenizer, AutoModel
from PIL import Image

ROOT = Path(os.environ.get('ICDE_STUDY_ROOT', str(Path(__file__).resolve().parents[1])))
STUDY = ROOT
OUT = ROOT / 'results/rev10'
MODEL = ROOT / 'models'
os.environ['HF_HOME'] = str(MODEL / 'hf')
os.environ['TORCH_HOME'] = str(MODEL / 'torch')
torch.set_num_threads(2)

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(4194304), b''):
            h.update(block)
    return h.hexdigest()

def download(url, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        part = p.with_suffix(p.suffix + '.part')
        with requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with part.open('wb') as f:
                for block in response.iter_content(1048576):
                    f.write(block)
        part.replace(p)
    return {'url': url, 'bytes': p.stat().st_size, 'sha256': digest(p)}

def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)

def main():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    revision = protocol['representation_control']['text_revision']
    base = f'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/{revision}/'
    receipts = {}
    for name in ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt']:
        receipts[name] = download(base + name, MODEL / 'minilm' / name)
    receipts['resnet18'] = download('https://download.pytorch.org/models/resnet18-f37072fd.pth', MODEL / 'resnet18.pth')
    assert receipts['resnet18']['sha256'].startswith('f37072fd')
    (OUT / 'model_receipts.json').write_text(json.dumps(receipts, indent=2))
    base_arrays = np.load(STUDY / 'results/rev7/cohort64_lambda4.2_batch1024_reverse0.npz')
    take, gallery, gold = (base_arrays[k] for k in ['query_indices', 'gallery', 'gold'])
    data = STUDY / 'data/wikidiverse/cleaned/WikiDiverse'
    kb = json.loads((data / 'kb_entity.json').read_text(encoding='utf-8'))
    meta = json.loads((STUDY / 'data/wikidiverse/features/test_metadata.json').read_text(encoding='utf-8'))
    rows = [meta[i] for i in take]
    records = rows + [kb[i] for i in gallery]
    texts = [unquote(z['mentions']) + ' [SEP] ' + z['sentence'] for z in rows]
    texts += [str(kb[i]['entity_name']) + ' [SEP] ' + str(kb[i].get('desc', '')) for i in gallery]
    tok = AutoTokenizer.from_pretrained(MODEL / 'minilm', local_files_only=True)
    encoder = AutoModel.from_pretrained(MODEL / 'minilm', local_files_only=True).eval()
    vectors = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(texts), 32):
            batch = tok(texts[start:start + 32], padding=True, truncation=True, max_length=256, return_tensors='pt')
            hidden = encoder(**batch).last_hidden_state
            mask = batch['attention_mask'].unsqueeze(-1)
            vector = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            vectors.append(unit(vector.numpy()))
            if start % 512 == 0:
                print('text', start, len(texts), round(time.perf_counter() - started, 1), flush=True)
    text = np.vstack(vectors).astype(np.float32)
    del encoder
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(MODEL / 'resnet18.pth', map_location='cpu', weights_only=True))
    model.fc = torch.nn.Identity()
    model.eval()
    prep = ResNet18_Weights.IMAGENET1K_V1.transforms()
    paths = [data / 'mention_image' / z['imgPath'] for z in rows]
    for i in gallery:
        ims = kb[i].get('image_list') or []
        paths.append(data / 'kb_image' / ims[0] if ims else None)
    image = np.zeros((len(paths), 512), np.float32)
    valid = np.zeros(len(paths), bool)
    failures = []
    with torch.inference_mode():
        for start in range(0, len(paths), 16):
            ims, ids = [], []
            for i in range(start, min(start + 16, len(paths))):
                try:
                    if paths[i] is None:
                        raise FileNotFoundError('No first-source image')
                    with Image.open(paths[i]) as im:
                        ims.append(prep(im.convert('RGB')))
                    ids.append(i)
                except Exception as e:
                    failures.append({'index': i, 'error': str(e)})
            if ims:
                image[ids] = unit(model(torch.stack(ims)).numpy())
                valid[ids] = True
            if start % 256 == 0:
                print('image', start, len(paths), round(time.perf_counter() - started, 1), flush=True)
    target = OUT / 'independent_features.npz'
    np.savez_compressed(target, query_indices=take, gallery=gallery, gold=gold,
                        query_text=text[:len(rows)], gallery_text=text[len(rows):],
                        query_image=image[:len(rows)], gallery_image=image[len(rows):], image_valid=valid)
    manifest = {'queries': len(rows), 'gallery': len(gallery), 'seconds': time.perf_counter() - started,
                'device': 'cpu', 'threads': 2, 'features_sha256': digest(target),
                'protocol_sha256': digest(OUT / 'protocol.json'), 'image_failures': failures,
                'model_receipts': receipts}
    (OUT / 'encoding_manifest.json').write_text(json.dumps(manifest, indent=2))
    print('COMPLETE', manifest['seconds'], 'failures', len(failures), flush=True)

if __name__ == '__main__':
    main()
