import faulthandler
faulthandler.enable()
import os, sys, time, json, random, hashlib, datetime, ast, subprocess, psutil, argparse
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
from pathlib import Path
from urllib.parse import unquote
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch
from torch import nn
import torch.nn.functional as F
r = Path(__import__('os').environ.get('ICDE_STUDY_ROOT', str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(r / 'work' / 'deps'))
import clip
from PIL import Image
ap = argparse.ArgumentParser()
ap.add_argument('--budget', type=int, default=128)
ap.add_argument('--seed', type=int, default=101)
ap.add_argument('--epochs', type=int, default=30)
args = ap.parse_args()
torch.set_num_threads(2)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
out = r / 'results' / 'fastmel_lowshot_resumable' / f'b{args.budget}_s{args.seed}'
out.mkdir(exist_ok=True, parents=True)
d = r / 'data' / 'wikidiverse' / 'cleaned' / 'WikiDiverse'
source = r / 'vendor' / 'FASTMEL-main'
es = json.load(open(d / 'kb_entity.json', encoding='utf-8'))
qid = json.load(open(d / 'qid2id.json', encoding='utf-8'))
train = json.load(open(d / 'wiki_diverse_train.json', encoding='utf-8'))
dev = json.load(open(d / 'wiki_diverse_dev.json', encoding='utf-8'))
hn = json.load(open(source / 'data' / 'WikiDiverse' / 'wiki_diverse_train_100cands-BM25.json', encoding='utf-8'))
assert len(hn) == len(train)
for a, b in zip(train, hn):
    assert (a['sentence'], a['mentions'], a['answer'], a['imgPath']) == (b['sentence'], b['mentions'], b['answer'], b['imgPath'])
pool = sorted(range(len(train)), key=lambda i: hashlib.sha256(str(train[i]['id']).encode()).hexdigest())[:1024]
sel = np.random.default_rng(args.seed).permutation(len(pool))[:args.budget]
selected = [pool[i] for i in sel]
reg = {'time': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'method': 'FAST-MEL SIGIR2026 architecture, low-shot reimplementation with original pooling class', 'budget': args.budget, 'seed': args.seed, 'epochs': args.epochs, 'batch': 8, 'lr': 1e-05, 'adam_eps': 0.0001, 'weight_decay': 0.0001, 'text_length': 40, 'supervision': 'exact same query subset as simple fusion baseline; no additional validation labels', 'hard_negatives': 'one uniformly selected from authors released first10 BM25 negatives, per query per step', 'image_preprocess': 'authors stretch224 then CLIP processor, random entity view during train and first during eval; missing masks as authors; random pixel fallback', 'backbone': 'same official CLIP-B32 weights via OpenAI implementation; text sequence ln_final and vision sequence before ln_post match HF feature definitions; pooling class loaded unmodified by AST', 'departures': ['batch8 instead of original64 due laptop memory; fixed30epochs without dev-based early stopping', 'standalone trainer repairs missing-import/invalid conditional in released Lightning source; native OpenAI backend instead of transformers', 'AMP FP16 gradients with FP32 master parameters'], 'training_query_ids': [train[i]['id'] for i in selected], 'test_access': 'none; only dev evaluated', 'source_pool_sha256': hashlib.sha256((source / 'codes' / 'model' / 'model_pooling.py').read_bytes()).hexdigest()}
(out / 'protocol.json').write_text(json.dumps(reg, indent=2))
base, preprocess = clip.load(str(r / 'models' / 'ViT-B-32.pt'), device='cpu', jit=False)
base.float()
node = next((n for n in ast.parse((source / 'codes' / 'model' / 'model_pooling.py').read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'MLPWeightedPooling'))
ns = {'torch': torch, 'nn': nn}
exec(compile(ast.Module(body=[node], type_ignores=[]), 'original_pool', 'exec'), ns)

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.clip_model = base
        self.text_proj = nn.Linear(512, 512)
        self.vision_proj = nn.Linear(768, 512)
        self.pool = ns['MLPWeightedPooling'](512, 256)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def forward(self, tok, pix, mask, empty):
        m = self.clip_model
        x = m.token_embedding(tok) + m.positional_embedding[:tok.shape[1]]
        x = x.permute(1, 0, 2)
        for b in m.transformer.resblocks:
            b.attn_mask = m.build_attention_mask()[:tok.shape[1], :tok.shape[1]]
        x = m.transformer(x).permute(1, 0, 2)
        x = m.ln_final(x)
        v = m.visual
        z = v.conv1(pix)
        z = z.reshape(z.shape[0], z.shape[1], -1).permute(0, 2, 1)
        cls = v.class_embedding.to(z.dtype) + torch.zeros(z.shape[0], 1, z.shape[-1], dtype=z.dtype, device=z.device)
        z = torch.cat([cls, z], 1) + v.positional_embedding.to(z.dtype)
        z = v.ln_pre(z)
        z = v.transformer(z.permute(1, 0, 2)).permute(1, 0, 2)
        xx = self.text_proj(x)
        zz = self.vision_proj(z)
        both = torch.cat([zz, xx], dim=1)
        valid = torch.cat([~empty[:, None].expand(-1, zz.shape[1]), mask], dim=1)
        return F.normalize(self.pool(both, valid).float(), dim=-1)
model = Encoder()
groups = [{'params': [p for n, p in model.named_parameters() if 'bias' not in n and 'ln_' not in n], 'weight_decay': 0.0001}, {'params': [p for n, p in model.named_parameters() if 'bias' in n or 'ln_' in n], 'weight_decay': 0}]
opt = torch.optim.AdamW(groups, lr=1e-05, eps=0.0001)
scaler = torch.amp.GradScaler('cuda')
device = 'cpu'
last_check = 0

def other_gpu_python():
    try:
        pp = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        other = []
        for z in pp.splitlines():
            try:
                pid = int(z.strip())
                pr = psutil.Process(pid)
                if pid != os.getpid() and 'python' in pr.name().lower():
                    other.append(pid)
            except (ValueError, psutil.Error):
                pass
        return other
    except Exception:
        return ['monitor-unavailable']

def move(dev):
    global device
    model.to(dev)
    for state in opt.state.values():
        for k, v in state.items():
            if torch.is_tensor(v) and k != 'step':
                state[k] = v.to(dev)
    device = dev
    if dev == 'cpu':
        torch.cuda.empty_cache()

def reserve_gpu():
    global last_check
    if device == 'cuda' and time.monotonic() - last_check < 15:
        return
    last_check = time.monotonic()
    other = other_gpu_python()
    if other:
        if device == 'cuda':
            move('cpu')
            print('yielded GPU', other, flush=True)
        while other:
            print('waiting for GPU', other, flush=True)
            time.sleep(15)
            other = other_gpu_python()
    if device != 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.55)
        move('cuda')
        print('GPU acquired', flush=True)

def tokenize(s):
    ids = [49406] + clip.clip._tokenizer.encode(s) + [49407]
    if len(ids) > 40:
        ids = ids[:40]
        ids[-1] = 49407
    n = len(ids)
    return (torch.tensor(ids + [0] * (40 - n)), torch.tensor([True] * n + [False] * (40 - n)))
et = [tokenize(unquote(x['entity_name']) + ' <|endoftext|> ' + x.get('desc', '')) for x in es]
tt = [tokenize(unquote(x['mentions']) + ' <|endoftext|> ' + x['sentence']) for x in train]
dt = [tokenize(unquote(x['mentions']) + ' <|endoftext|> ' + x['sentence']) for x in dev]

@lru_cache(maxsize=1024)
def image_file(path):
    try:
        with Image.open(path) as im:
            return preprocess(im.convert('RGB').resize((224, 224), Image.Resampling.LANCZOS))
    except Exception:
        return None

def records(rows, kind, training=False):
    toks = []
    masks = []
    pics = []
    empty = []
    for i in rows:
        ent = es[i] if kind == 'entity' else train[i] if kind == 'train' else dev[i]
        t, m = et[i] if kind == 'entity' else tt[i] if kind == 'train' else dt[i]
        imgs = ent['image_list'] if kind == 'entity' else [ent['imgPath']] if ent['imgPath'] else []
        name = (random.choice(imgs) if training else imgs[0]) if imgs else ''
        path = d / ('kb_image' if kind == 'entity' else 'mention_image') / name
        pix = image_file(str(path)) if name else None
        toks.append(t)
        masks.append(m)
        pics.append(pix if pix is not None else torch.rand(3, 224, 224))
        empty.append(not bool(imgs))
    return (torch.stack(toks).to(device), torch.stack(pics).to(device), torch.stack(masks).to(device), torch.tensor(empty, device=device))
reserve_gpu()
losses = []
t0 = time.perf_counter()
checkpoint = out / 'trained_weights.pt'
if checkpoint.exists():
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state['model'])
    torch.set_rng_state(state['rng'].cpu())
    random.setstate(state['python_rng'])
    print('resumed trained weights', flush=True)
else:
    for epoch in range(args.epochs):
        model.train()
        order = np.random.default_rng(args.seed + epoch).permutation(selected)
        epoch_loss = []
        for pos in range(0, len(order), 8):
            reserve_gpu()
            batch = order[pos:pos + 8].tolist()
            g = [qid[train[i]['answer']] for i in batch]
            negative = [qid[random.choice([z for z in hn[i]['cands'][:10] if z != train[i]['answer']])] for i in batch]
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16, cache_enabled=False):
                e = model(*records(g + negative, 'entity', True))
                q = model(*records(batch, 'train', True))
                score = model.logit_scale.exp().clamp(max=100) * q @ e.T
                loss = F.cross_entropy(score, torch.arange(len(batch), device=device))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            epoch_loss.append(float(loss.detach()))
            del loss, e, q, score
        losses.append(float(np.mean(epoch_loss)))
        print('epoch', epoch + 1, 'loss', losses[-1], 'seconds', round(time.perf_counter() - t0, 1), flush=True)
        (out / 'training.json').write_text(json.dumps({'losses': losses, 'elapsed_seconds': time.perf_counter() - t0}, indent=2))
    torch.save({'model': model.state_dict(), 'rng': torch.get_rng_state(), 'python_rng': random.getstate()}, checkpoint)
model.eval()
image_file.cache_clear()
features = {}
with torch.inference_mode():
    for kind, N in [('entity', len(es)), ('dev', len(dev))]:
        feature_path = out / (kind + '_embeddings.npy')
        progress = out / (kind + '_progress.json')
        completed = json.load(open(progress))['completed'] if progress.exists() else 0
        v = np.lib.format.open_memmap(feature_path, mode='r+' if feature_path.exists() else 'w+', dtype=np.float32, shape=(N, 512))
        for start in range(completed, N, 64):
            reserve_gpu()
            with torch.autocast('cuda', dtype=torch.float16, cache_enabled=False):
                a = model(*records(list(range(start, min(start + 64, N))), kind)).cpu().numpy()
            v[start:start + len(a)] = a
            v.flush()
            progress.write_text(json.dumps({'completed': start + len(a), 'total': N}))
            if start % 8192 == 0:
                print('encode', kind, start, N, flush=True)
        features[kind] = v
rank = []
E = features['entity']
Q = features['dev']
gold = np.array([qid[x['answer']] for x in dev])
torch.set_num_threads(2)
for start in range(0, len(Q), 32):
    S = Q[start:start + 32] @ E.T
    for j in range(len(S)):
        g = gold[start + j]
        rank.append(int(np.sum(S[j] > S[j, g]) + np.sum((S[j] == S[j, g]) & (np.arange(len(E)) < g)) + 1))
rank = np.array(rank)
np.savez_compressed(out / 'development_ranks.npz', rank=rank, query_ids=[x['id'] for x in dev], gold=gold)
summary = {'budget': args.budget, 'seed': args.seed, 'epochs': args.epochs, **{f'R@{k}': float((rank <= k).mean()) for k in [1, 10, 50, 100, 500]}, 'MRR': float((1 / rank).mean()), 'elapsed_seconds': time.perf_counter() - t0}
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
print('FINISHED', summary, flush=True)
