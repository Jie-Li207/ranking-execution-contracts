from pathlib import Path
import os
import json

ROOT = Path(os.environ.get("ICDE_STUDY_ROOT", str(Path(__file__).resolve().parents[1])))
OUT = ROOT / "results/rev10"
OUT.mkdir(parents=True, exist_ok=True)
protocol = {'date_local': '2026-10-28',
 'status': 'Follow-up fixed before new outcomes; previously inspected WikiDiverse test queries, not '
           'independent confirmatory data',
 'representation_control': {'query_indices': 'Existing 1024 hash-selected test queries',
                            'gallery': 'Existing fixed 2472-entity second-stage gallery, CLIP-selected '
                                       'without gold supplementation; same IDs in all conditions',
                            'factorial': ['CLIP text + CLIP image',
                                          'MiniLM text + CLIP image',
                                          'CLIP text + ResNet18 image',
                                          'MiniLM text + ResNet18 image'],
                            'text_model': 'sentence-transformers/all-MiniLM-L6-v2',
                            'text_revision': '1110a243fdf4706b3f48f1d95db1a4f5529b4d41',
                            'text_pooling': 'attention-mask mean, max_length=256, L2 normalize',
                            'visual_model': 'torchvision ResNet18 IMAGENET1K_V1; global average pooled 512d '
                                            'features; prescribed weight transforms',
                            'fusion': 'L2-normalize each modality; concatenate; L2-normalize; decoding '
                                      'failures zeroed and masks retained',
                            'rerankers': ['identity',
                                          'averageQE first10',
                                          'similarity-cubedQE first10',
                                          'SuperGlobal beta0',
                                          'SuperGlobal beta0.15'],
                            'reranker_M': 400,
                            'cohort_operator': 'Released Cheb-GR three rounds, temperature .2, lambda4.2, '
                                               'float64',
                            'cohorts': ['full1024', 'hash64', 'reversed-full1024'],
                            'primary': ['paired conditional-gallery Recall@5 difference and image-cluster '
                                        'bootstrap95CI (2000 replicates)',
                                        'changed first IDs',
                                        'changed ordered top10 and top10 sets',
                                        'strict first changes with both score gaps >1e-8'],
                            'not_claimed': ['full-KB retrieval quality for new encoders',
                                            'new encoder superiority',
                                            'fresh holdout confirmation',
                                            'causal identification of representation training differences']},
 'service_control': {'implementation': 'Local SQLite request/result-cache/entity-join pipeline, operator '
                                       'executed on every miss; full actual cohort retained even when only '
                                       'some results miss',
                     'workloads': 'Real 1024 benchmark queries; constructed two-pass schedules with 0, 50, '
                                  'or100 percent unchanged 64-query cohorts; changed cohort members '
                                  'regrouped by seeded permutation',
                     'seeds': [2026102801, 2026102802, 2026102803],
                     'policies': ['no cache', 'query-only invalid control', 'query+ordered-cohort cache'],
                     'operators': ['Cheb-GR lambda4.2 float64'],
                     'primary': ['incorrect returned ordered top10 and joined type counts against '
                                 'same-cohort fresh execution',
                                 'hit rate',
                                 'wall time including cache lookup, graph execution, insertion, entity joins',
                                 'request latency p50/p95',
                                 'SQLite pages and key+ID payload'],
                     'conditions': 'Two CPU threads, paired policy orders randomized; descriptors resident, '
                                   'no encoder or initial KB scan in timing',
                     'not_claimed': 'Production traffic frequency or concurrency/throughput guarantee'},
 'evidence': {'separate_small_bundle': 'numeric per-query outputs, correction records, table/figure mapping, '
                                       'SHA256 and verifier; no raw images, weights or full data caches',
              'verification': 'reaggregation distinguished from fresh inference; public input retrieval '
                              'probes distinguished from full downloads'}}
path = OUT / "protocol.json"
if path.exists():
    assert json.loads(path.read_text()) == protocol, "Existing protocol differs; preserve it and use a fresh output root."
else:
    path.write_text(json.dumps(protocol, indent=2))
print(str(path))
