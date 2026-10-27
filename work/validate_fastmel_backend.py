import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '2'
import ast, sys, json, hashlib, types
from pathlib import Path
import numpy as np, torch
from torch import nn
import torch.nn.functional as F
r = Path(__import__('os').environ.get('ICDE_STUDY_ROOT', str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(r / 'work/deps'))
import clip
torch.set_num_threads(2)
torch.manual_seed(787)
base, _ = clip.load(str(r / 'models/ViT-B-32.pt'), device='cpu', jit=False)
base.float().eval()
src = r / 'vendor/FASTMEL-main/codes/model/model_pooling.py'
tree = ast.parse(src.read_text())
poolnode = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MLPWeightedPooling'))
ns = {'torch': torch, 'nn': nn}
exec(compile(ast.Module(body=[poolnode], type_ignores=[]), 'nativepool', 'exec'), ns)
ourtree = ast.parse((r / 'work/fastmel_lowshot.py').read_text())
ournode = next((n for n in ourtree.body if isinstance(n, ast.ClassDef) and n.name == 'Encoder'))
own = {'torch': torch, 'nn': nn, 'np': np, 'F': F, 'base': base, 'ns': ns}
exec(compile(ast.Module(body=[ournode], type_ignores=[]), 'ourencoder', 'exec'), own)
model = own['Encoder']().eval()
captured = {}

def text_hook(m, a, out):
    captured['text'] = out.detach().clone()

def vision_hook(m, a, out):
    captured['vision'] = out.detach().permute(1, 0, 2).clone()
h1 = base.ln_final.register_forward_hook(text_hook)
h2 = base.visual.transformer.register_forward_hook(vision_hook)
pix = torch.randn(3, 3, 224, 224)
tokens = clip.tokenize(['a photograph of a bird', 'a named person in a street', 'a church'])
mask = tokens != 0
empty = torch.tensor([False, True, False])
checks = {}
with torch.inference_mode():
    reference_t = base.encode_text(tokens)
    manual_t = captured['text'][torch.arange(3), tokens.argmax(-1)] @ base.text_projection
    checks['text_global_max_abs'] = float((reference_t - manual_t).abs().max())
    reference_v = base.encode_image(pix)
    manual_v = base.visual.ln_post(captured['vision'][:, 0]) @ base.visual.proj
    checks['vision_global_max_abs'] = float((reference_v - manual_v).abs().max())
    ours = model(tokens[:, :40], pix, mask[:, :40], empty)
    hidden = {k: v.clone() for k, v in captured.items()}

class Fake(nn.Module):

    def __init__(self, key, dim):
        super().__init__()
        self.key = key
        self.config = types.SimpleNamespace(hidden_size=dim)

    def forward(self, **kwargs):
        return types.SimpleNamespace(last_hidden_state=hidden[self.key])
nativeclass = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Encoder'))

class Portable(ast.NodeTransformer):

    def visit_Constant(self, node):
        return ast.copy_location(ast.Constant('cpu'), node) if node.value == 'cuda' else node
nativeclass = Portable().visit(nativeclass)
ast.fix_missing_locations(nativeclass)
auto = types.SimpleNamespace(from_pretrained=lambda _: types.SimpleNamespace(text_model=Fake('text', 512), vision_model=Fake('vision', 768)))
native_ns = {'torch': torch, 'nn': nn, 'AutoModel': auto, 'MLPWeightedPooling': ns['MLPWeightedPooling']}
exec(compile(ast.Module(body=[nativeclass], type_ignores=[]), 'original_encoder_cpu', 'exec'), native_ns)
args = types.SimpleNamespace(pretrained_model='supplied hidden states', model=types.SimpleNamespace(embed_dim=512, pooling_type='mlp', use_vision=True))
native = native_ns['Encoder'](args).eval()
native.text_proj[0].load_state_dict(model.text_proj.state_dict())
native.vision_proj[0].load_state_dict(model.vision_proj.state_dict())
native.mlp_pooling.load_state_dict(model.pool.state_dict())
with torch.inference_mode():
    ref = F.normalize(native(tokens[:, :40], pix, attention_mask=mask[:, :40], empty_img_flag=empty).float(), dim=-1)
    checks['native_forward_fusion_max_abs'] = float((ours - ref).abs().max())
assert max(checks.values()) < 1e-06
report = {'checks': checks, 'source_sha256': hashlib.sha256(src.read_bytes()).hexdigest(), 'test': 'deterministic synthetic tensor algebra; no data accuracy claims', 'native_modification': 'only literal cuda device changed to cpu; identical supplied hidden states and head weights', 'remaining_departures': ['OpenAI vs HF transformer backend not a full bitwise HF checkpoint comparison', 'batch8 versus64, fixed30epochs, AMP fp16', 'our optimizer excludes ln_ from decay; native source excludes LayerNorm by name. Recorded as backend reimplementation, not bit-exact reproduction.']}
(r / 'results/fastmel_lowshot/backend_validation.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
