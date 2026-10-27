import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '2'
import sys, argparse, json, datetime, hashlib, importlib.util
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(2)
from directed_objective_core import run_directed
p = argparse.ArgumentParser()
p.add_argument('dataset', choices=['WikiMEL', 'RichpediaMEL'])
p.add_argument('--mode', type=int, choices=[3, 4], default=3)
p.add_argument('--threshold', type=float, default=0.6)
p.add_argument('--limit', type=int, default=0)
args = p.parse_args()
r = Path(__import__('os').environ.get('ICDE_STUDY_ROOT', str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(r / 'work/deps'))
sp = importlib.util.spec_from_file_location('native', r / 'vendor/OpenMEL-main/maximum_spanning.py')
native = importlib.util.module_from_spec(sp)
sp.loader.exec_module(native)
tag = f'{args.dataset}_author{args.mode}_t{args.threshold:g}_n{args.limit or 'all'}'
out = r / 'results/author_objective' / tag
out.mkdir(parents=True, exist_ok=True)
a = np.load(r / 'results/author_reproduction' / args.dataset / f'controlled_inputs_{args.mode}.npz')
meta = json.load(open(r / 'data/author_cache' / args.dataset / 'queries.json', encoding='utf-8'))
for i, q in enumerate(meta):
    q['id'] = str(q.get('id', i))
selected = np.arange(len(meta))
if args.limit:
    selected = np.array(sorted(selected, key=lambda i: hashlib.sha256(str(meta[i]['id']).encode()).hexdigest())[:args.limit])
reg = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'frozen_protocol_sha256': hashlib.sha256((r / 'work/preregistration_confirmatory_native.json').read_bytes()).hexdigest(), 'threshold': args.threshold, 'B': [1, 5, 10, 20], 'strict_tolerance': 1e-07, 'scope': 'unfiltered controlled operator intervention using author cached expanded features; NOT repaired native name-filtered pipeline', 'mode': args.mode, 'limit': args.limit}
(out / 'protocol.json').write_text(json.dumps(reg, indent=2))
np.savez_compressed(out / 'inputs.npz', roots=a['roots'][selected], scores=a['scores'][selected], gold=a['gold'][selected], selected=selected, unique=a['unique'], nn=a['nn'], dd=a['dd'])
run_directed(a['roots'], a['scores'], a['gold'], a['unique'], a['dd'], a['nn'], meta, selected, args.threshold, reg, out, native, tag)
