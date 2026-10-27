import os, sys, json, time, hashlib, datetime, collections, subprocess, psutil
from pathlib import Path
r = Path(__import__('os').environ.get('ICDE_STUDY_ROOT', str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(r / 'work' / 'deps'))
os.environ['HF_HOME'] = str(r / 'models' / 'hf')
os.environ['TORCH_HOME'] = str(r / 'models')
os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np, torch, clip
from PIL import Image
from urllib.parse import unquote
from concurrent.futures import ThreadPoolExecutor
torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.3)
d = r / 'data' / 'wikimel' / 'WikiMEL'
out = r / 'data' / 'wikimel' / 'features'
out.mkdir(parents=True, exist_ok=True)
entities = json.load(open(d / 'kb_entity_desc.json', encoding='utf-8'))
qid = json.load(open(d / 'qid2id.json', encoding='utf-8'))
rows = {s: json.load(open(d / f'WIKIMEL_{s}.json', encoding='utf-8')) for s in ['train', 'dev']}
reg = dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(), previous='preregistration_B_full_v1.json', text_context='OpenAI native 77 tokens, truncate, name [SEP] attr; additional name [SEP] desc baseline, both registered before scores. This differs from MIMIC 40-token training and is not an exact MIMIC reproduction.', images='First declared entity image as official evaluation loader; native CLIP center crop, not stretch. Missing/corrupt stored zero vectors with explicit mask; all actual presence audited.', query_subset='All original non-NIL dev; train discovery first 1024 by SHA256 of id; later training selection pre-registered independently.', precision='Cooperative device selection: CPU fp32 when any other Python GPU process active, otherwise CLIP CUDA fp16; normalized float32 stored float16. Device changes logged; numerical control required.', test='Do not read or score test queries.')
if not (r / 'work' / 'preregistration_B_wikimel_v1.json').exists():
    (r / 'work' / 'preregistration_B_wikimel_v1.json').write_text(json.dumps(reg, indent=2))
model, preprocess = clip.load(str(r / 'models' / 'ViT-B-32.pt'), device='cpu', jit=False)
model.eval()
device = 'cpu'
last_check = 0

def choose_device():
    global device, last_check, model
    if time.monotonic() - last_check < 25:
        return
    last_check = time.monotonic()
    other = []
    try:
        pp = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        for row in pp.splitlines():
            try:
                pid = int(row.strip())
                proc = psutil.Process(pid)
                if pid != os.getpid() and 'python' in proc.name().lower():
                    other.append(pid)
            except (ValueError, psutil.Error):
                pass
    except Exception:
        other = ['unavailable-monitor']
    wanted = 'cpu' if other else 'cuda'
    if wanted != device:
        if wanted == 'cpu':
            model.float().cpu()
            torch.cuda.empty_cache()
        else:
            model.cuda()
            clip.model.convert_weights(model)
        device = wanted
        print('DEVICE', device, 'other GPU Python', other, flush=True)
manifest = {'model_sha256': hashlib.sha256((r / 'models' / 'ViT-B-32.pt').read_bytes()).hexdigest(), 'source_hashes': {}, 'failures': [], 'timings': {}, 'counts': {}}
for f in ['kb_entity_desc.json', 'qid2id.json', 'WIKIMEL_train.json', 'WIKIMEL_dev.json']:
    manifest['source_hashes'][f] = hashlib.sha256((d / f).read_bytes()).hexdigest()

def texts(name, seq):
    dst = out / (name + '.npy')
    if dst.exists():
        print('reuse', name, flush=True)
        return
    t = time.perf_counter()
    a = np.lib.format.open_memmap(str(dst) + '.partial', mode='w+', dtype=np.float16, shape=(len(seq), 512))
    with torch.inference_mode():
        for i in range(0, len(seq), 128):
            choose_device()
            tok = clip.tokenize(seq[i:i + 128], truncate=True).to(device)
            z = model.encode_text(tok).float()
            z /= z.norm(dim=-1, keepdim=True).clamp_min(1e-08)
            a[i:i + len(z)] = z.cpu().numpy()
            if i % 8192 == 0:
                print(name, i, len(seq), round(time.perf_counter() - t, 1), flush=True)
    a.flush()
    del a
    os.replace(str(dst) + '.partial', dst)
    manifest['timings'][name] = time.perf_counter() - t

def images(name, paths):
    dst = out / (name + '.npy')
    if dst.exists():
        print('reuse', name, flush=True)
        return
    t = time.perf_counter()
    a = np.lib.format.open_memmap(str(dst) + '.partial', mode='w+', dtype=np.float16, shape=(len(paths), 512))
    a[:] = 0
    mask = np.zeros(len(paths), bool)

    def read(x):
        try:
            if x is None:
                return None
            with Image.open(x) as im:
                return preprocess(im.convert('RGB'))
        except Exception as e:
            manifest['failures'].append({'path': str(x), 'error': str(e)})
            return None
    with ThreadPoolExecutor(max_workers=3) as pool, torch.inference_mode():
        for i in range(0, len(paths), 64):
            zz = list(pool.map(read, paths[i:i + 64]))
            valid = [j for j, x in enumerate(zz) if x is not None]
            if valid:
                choose_device()
                z = model.encode_image(torch.stack([zz[j] for j in valid]).to(device)).float()
                z /= z.norm(dim=-1, keepdim=True).clamp_min(1e-08)
                a[np.array(valid) + i] = z.cpu().numpy()
                mask[np.array(valid) + i] = True
            if i % 4096 == 0:
                print(name, i, len(paths), round(time.perf_counter() - t, 1), flush=True)
    a.flush()
    del a
    os.replace(str(dst) + '.partial', dst)
    np.save(out / (name + '_mask.npy'), mask)
    manifest['timings'][name] = time.perf_counter() - t
    manifest['counts'][name] = {'total': len(paths), 'valid': int(mask.sum())}
texts('entity_name_attr', [unquote(e['entity_name']) + ' [SEP] ' + e.get('attr', '') for e in entities])
texts('entity_name_desc', [unquote(e['entity_name']) + ' [SEP] ' + e.get('desc', '') for e in entities])
images('entity_image', [d / 'kb_image' / e['image_list'][0] if e['image_list'] else None for e in entities])
for s, rr in rows.items():
    rr = [x for x in rr if x['answer'] in qid]
    (out / (s + '_metadata.json')).write_text(json.dumps([{'id': x['id'], 'answer_id': qid[x['answer']], 'imgPath': x['imgPath'], 'mentions': x['mentions'], 'sentence': x['sentence']} for x in rr], ensure_ascii=False), encoding='utf-8')
    texts(s + '_text', [unquote(x['mentions']) + ' [SEP] ' + x['sentence'] for x in rr])
    images(s + '_image', [d / 'mention_image' / x['imgPath'] for x in rr])
manifest['finished'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print('FINISHED', manifest['counts'], flush=True)
