from pathlib import Path
import json
import numpy as np
import torch
from audit_reranking_grid import unit, load_sources
from contract_checks import compare_outputs, rewrite_decision
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rev7'
torch.set_num_threads(2)
classes, _ = load_sources()
records = []
f = ROOT / 'data/wikidiverse/features'
with torch.inference_mode():
    for mode in ['text', 'multimodal']:
        arrays = np.load(OUT / f'reranking_{mode}_dev.npz')
        ix = arrays['query_indices']
        ids = arrays['candidates']
        T = unit(np.load(f / 'entity_name_desc.npy').astype(np.float32))
        X = T if mode == 'text' else unit(np.concatenate([T, unit(np.load(f / 'entity_image.npy').astype(np.float32))], axis=1))
        T = unit(np.load(f / 'dev_text.npy')[ix].astype(np.float32))
        Q = T if mode == 'text' else unit(np.concatenate([T, unit(np.load(f / 'dev_image.npy')[ix].astype(np.float32))], axis=1))
        XT = torch.from_numpy(X)
        for M in [50, 100, 400]:
            outputs = []
            augmentation = []
            for start in range(0, len(Q), 4):
                rt = torch.from_numpy(ids[start:start + 4].T.copy())
                result = classes['MDescAug'](M=M, beta=0)(XT, torch.from_numpy(Q[start:start + 4]), rt)
                augmentation.extend([row.numpy()[:20] for row in result[0]])
                rr = [classes['RerankwMDA'](M=M, beta=beta)(rt, *result).T.numpy()[:, :20] for beta in [0, 0.05, 0.15, 0.3]]
                assert all((np.array_equal(rr[0], z) for z in rr[1:]))
                outputs.extend(rr[0])
            report = compare_outputs(ids[:, :20], np.array(outputs), ix, 20)
            aug = compare_outputs(ids[:, :20], np.array(augmentation), ix, 20)
            records.append({'mode': mode, 'M': M, 'fixed_augmentation_beta': 0, 'final_beta_values': [0, 0.05, 0.15, 0.3], 'final_beta_outputs_equal': True, 'identity_comparison': report, 'augmentation_identity_comparison': aug, 'decision': rewrite_decision(report)})
(OUT / 'neutralization_checks.json').write_text(json.dumps(records, indent=2))
print([(z['mode'], z['M'], z['identity_comparison']['ordered_changes']) for z in records])
