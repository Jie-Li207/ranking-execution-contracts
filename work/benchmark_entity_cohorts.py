import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
from pathlib import Path
import json
import time
import threading
import numpy as np
import torch
import psutil
from types import SimpleNamespace
from audit_reranking_grid import unit, load_sources
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rev7'
torch.set_num_threads(2)
a = np.load(OUT / 'lambda4.2_batch1024_reverse0.npz')
take = a['query_indices']
gallery = a['gallery']
f = ROOT / 'data/wikidiverse/features'
G = unit(np.concatenate([unit(np.load(f / 'entity_name_desc.npy')[gallery].astype(np.float32)), unit(np.load(f / 'entity_image.npy')[gallery].astype(np.float32))], axis=1))
Q = unit(np.concatenate([unit(np.load(f / 'test_text.npy')[take].astype(np.float32)), unit(np.load(f / 'test_image.npy')[take].astype(np.float32))], axis=1))
_, cheb = load_sources()
cfg = SimpleNamespace(GCR=SimpleNamespace(WITH_GPU=False, BETA1=0.2, BETA2=0.2, LAMBDA1=4.2, LAMBDA2=4.2, SCALE=1, MODE='normal'), COMMON=SimpleNamespace(VERBOSE=False))
records = []
process = psutil.Process()
peak = [process.memory_info().rss]
stop = threading.Event()

def monitor():
    while not stop.wait(0.005):
        peak[0] = max(peak[0], process.memory_info().rss)
thread = threading.Thread(target=monitor)
thread.start()
try:
    with torch.inference_mode():
        for repeat, order in enumerate([[1024, 64], [64, 1024], [1024, 64]]):
            for size in order:
                tick = time.perf_counter()
                ranks = []
                for begin in range(0, 1024, size):
                    feat = torch.from_numpy(np.vstack([Q[begin:begin + size], G]))
                    labels = np.zeros((len(feat), 2), np.int64)
                    for _ in range(3):
                        feat = cheb(cfg, feat, labels)
                    scores = (feat[:size] @ feat[size:].T).numpy()
                    ranks.extend([gallery[np.lexsort((gallery, -ss))[:20]] for ss in scores])
                expected = np.load(OUT / f'lambda4.2_batch{size}_reverse0.npz')['ranks']
                assert np.array_equal(ranks, expected)
                records.append({'repeat': repeat, 'batch': size, 'seconds': time.perf_counter() - tick})
                print(records[-1], flush=True)
finally:
    stop.set()
    thread.join()
(OUT / 'cohort_timing.json').write_text(json.dumps({'rows': records, 'peak_process_rss_MiB': peak[0] / 2 ** 20, 'method': 'Three alternating paired repetitions, two CPU threads, float32 native source, full graph build and scoring; loaded descriptors, no first-stage KB scan or image encoding; peak process RSS includes runtime', 'matrix_N_full': len(Q) + len(G)}, indent=2))
