# Ranking and Execution Contracts in Graph-Enhanced Multimodal Retrieval

Core implementation for the experimental study of rooted candidate selection, graph-based reranking, and cohort-dependent diffusion. This repository contains executable source and dependency specifications. Data, model checkpoints, descriptors, generated results, and caches are obtained or generated locally.

## Installation

Use Python 3.12 and run commands from the repository root. The reference environment used PyTorch 2.5.1, CUDA 12.1 for optional encoder training, and two CPU threads for retrieval experiments.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python prepare.py
python work/audit_enumeration.py
```

Activate `.venv` before the installation command (`.venv\Scripts\activate` on Windows; `source .venv/bin/activate` on Linux). `prepare.py` downloads pinned upstream source files, verifies SHA256 hashes, and writes the experiment configurations. `--config-only` writes configurations without downloads. The first enumeration run checks 200 small graphs against exhaustive entity-subset search; it does not require a dataset. Running it again after graph construction evaluates all available graph conditions.

All generated files stay below the repository root unless `ICDE_STUDY_ROOT` is explicitly set for the scripts supporting that variable. Use a drive with sufficient free space. Full image encoding and training require substantially more disk space and time than the core algorithms. The scripts do not stop other processes. The low-label trainer requires CUDA and waits when it detects a competing Python GPU workload.

## Native ANN contracts and bounded update workloads

This extension starts from public SIFT1M descriptors, independently of the entity encoders and historical prediction arrays. hnswlib 0.8.0 may require a local C++ compiler; the reference Windows build used the installed Microsoft toolchain. Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to `1` before numerical imports when running the verifier.

```bash
python work/prepare_native.py
python work/acquire_native_inputs.py
python work/audit_native_ann.py --backend faiss --scale 100000 --phase development
python work/audit_native_ann.py --backend faiss --scale 100000 --phase confirmation
python work/audit_native_ann.py --backend hnswlib --scale 100000 --phase development
python work/audit_native_ann.py --backend hnswlib --scale 100000 --phase confirmation
python work/audit_native_ann.py --backend faiss_sq8 --scale 1000000 --phase development
python work/audit_native_ann.py --backend faiss_sq8 --scale 1000000 --phase confirmation
python work/audit_native_ann.py --backend hnswlib --scale 1000000 --phase development
python work/audit_native_ann.py --backend hnswlib --scale 1000000 --phase confirmation
python work/audit_budget_guard.py
python work/benchmark_budget_service.py
python work/benchmark_budget_service.py --cache-bytes 65536
python work/benchmark_budget_service.py --cache-bytes 65536 --invalid-only
python work/verify_native_evidence.py
```

The dataset is 525,128,288 bytes; SHA256 is `dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984`. Downloading verifies this hash. Approximately 2 GB of local space covers descriptors, four serialized indexes and run outputs; reserve additional working memory and disk headroom. Builds stream blocks of 10,000 vectors. Direct 8-bit storage asserts that every descriptor coordinate is an integer in [0,255], so it does not quantize these source values. All configurations are fixed, including those crossing hnswlib's recommended `ef >= k` boundary; this is not a library-bug claim.

The source-derived guard requires identical query/index/search dependencies, equal `max(ef,k)` and `max(ef,K)`, and a strict distance gap at the retained prefix cut. Both native and reused results use distance-then-ID canonical order. Its proof covers pinned hnswlib searches without filters or deletions. It preserves the native small-k answer, not exact nearest-neighbor truth. No guard is asserted for arbitrary Faiss modes.

The service runs native hnswlib plus SQLite, 4,096 requests per trace, three seeds, one/four readers, LRU eviction, two attribute updates and 256 real-vector insertions. Writes occur at epoch barriers. A lock serializes the native mutable-ef configuration and search. Request schedules and revision fields are synthetic. Retained payload caps count serialized keys and distance/ID/revision arrays, excluding Python containers and shared inputs. Exact-k IDs isolate attribute invalidation; the guarded policy additionally permits cross-budget reuse. The invalid policy intentionally ignores versions and the prefix guard. Result differences are measured against fresh native searches for each actual k and epoch, not against a changed fixed-K interface. Existing output directories are rejected to preserve failed runs.

The evidence archive adds `native_20261030/`, including development and confirmation outputs, per-request caches/joins, build receipts, protocols, corrections and failure logs. After extracting the archive, run `python work/verify_native_evidence.py evidence/native_20261030`. That command independently recalculates stored numerical checks; the full command sequence above performs new native builds and searches. Neither constitutes independent external replication. The original entry-only trace records, corrected reference failure and equal-byte extension remain distinguishable.

## Implementation map

| Component | Entry points |
|---|---|
| Exact directed selection and optimal-set diagnostics | `work/enumeration_oracle.py`, `work/directed_objective_core.py`, `work/audit_enumeration.py` |
| Undirected flow formulation | `work/undirected_tree_oracle_tight.py`, `work/undirected_objective_audit_tight.py` |
| Bounded expansion and physical plans | `work/lazy_coherence_plan.py`, `work/benchmark_prefix_plan_v3.py`, `work/benchmark_native_retrieval.py` |
| Original cached-input retrieval | `work/convert_author_cache.py`, `work/author_cache_reproduction.py`, `work/author_objective_audit.py` |
| Calibration and ranking controls | `work/audit_ranking_controls.py`, `work/audit_joint_calibration.py`, `work/audit_structure.py` |
| SuperGlobal transfer and visual controls | `work/run_superglobal.py`, `work/run_superglobal_visual.py` |
| Diffusion, cache scope, and cohort partitions | `work/audit_diffusion.py`, `work/audit_cohort_partitions.py` |
| Split integrity and image quality | `work/audit_holdout.py`, `work/audit_image_quality.py`, `work/analyze_image_quality.py` |

The remaining scripts construct the matched-input baselines, encode public images, or run the low-label controls needed by these experiments. Output directory names preserve the experiment identifiers used during the study.

## Contract checks and follow-up experiments

`work/contract_checks.py` compares aligned ordered results and constructs cache fingerprints from source revision, parameters, IDs and descriptor bytes. A finite pass is reported as not falsified on the tested inputs; universal admission requires an explicit proved guard. The directed optimizer requires disjoint root and child entity sets and rejects inputs outside that domain.

After obtaining the WikiDiverse descriptors and author caches using the commands below, run the following sequentially with two CPU threads:

```bash
python work/audit_reranking_grid.py --mode multimodal
python work/audit_reranking_grid.py --mode text
python work/audit_entity_cohorts.py
python work/audit_transfer_caches.py
python work/audit_cohort_numerics.py
python work/audit_relational_consumer.py
python work/audit_neutralization.py
python work/benchmark_entity_cohorts.py
python work/verify_contract_results.py results/rev7
```

The grid reports every fixed configuration and selects using 128 development queries, including an identity option. These are follow-up analyses of previously evaluated test inputs. Cheb-GR uses the released function at pinned commit `a3f67722b7c87fe3c4f1e0b80055b9487875a3ad`; the experiments transfer that function to frozen entity descriptors and do not reproduce its trained person-reidentification encoder. The 1,024-query cohort workload fixes a 2,472-entity union of initial top-five candidates across every partition. This is a second-stage pool after a full-KB scan, not a full-KB dense graph. Float32, float64, arrival-order and seeded-partition outputs are retained separately. Cache replay is a controlled local trace over real queries, not a production-service measurement.

The source archive contains no checkpoints, data caches, generated results or historical logs. A separate numerical evidence asset is available from the [revision 12 release](https://github.com/Jie-Li207/ranking-execution-contracts/releases/tag/rev12-evidence). The [revision 10 release](https://github.com/Jie-Li207/ranking-execution-contracts/releases/tag/rev10-evidence) remains available unchanged. It contains compact predictions, timings, graph weights and paired correction records, without images or model weights. Extract it to an `evidence` directory and run:

```bash
python work/verify_contract_results.py evidence/rev7
python work/verify_evidence.py evidence
python work/verify_evidence.py evidence --solve
python work/verify_repeat.py evidence
python work/export_evidence_tables.py evidence
```

The first two commands reaggregate stored predictions and verify hashes; they do not rerun encoders. `--solve` additionally performs new exact optimization on 128 saved graphs per primary dataset and all four budgets. The exporter writes source tables and measured-figure data under `evidence/generated`; the evidence map identifies each supported manuscript display. The revision 11 archive preserves the historical outputs and adds `repeat_20261029/`: a complete operator repeat on the same stored representations and all 27 executing SQLite runs. `verify_repeat.py` checks repeated output arrays, workload identities, cache outcomes and measured latency summaries. The exporter uses the repeat timings when present. These checks are distinct from full encoder inference and from independent external replication.

## Crossed encoders and executing SQLite workload

After the WikiDiverse acquisition, encoding and entity-cohort commands below, run:

```bash
python work/prepare_followup.py
python work/encode_independent_features.py
python work/audit_representation_factorial.py
python work/benchmark_sqlite_cache.py
```

These commands write `results/rev10/` and download approximately 140 MB of pinned MiniLM and ResNet18 weights under `models/`. They use two CPU threads and do not use the GPU. The four conditions cross CLIP/MiniLM text with CLIP/ResNet18 images; all use the same 1,024 previously evaluated queries and 2,472 gallery IDs. This gallery was selected by CLIP, so the replacement-encoder results are conditional-gallery controls, not full-KB retrieval accuracy or fresh holdout confirmation. Each candidate ranking is recomputed within that fixed gallery.

The SQLite runner executes Cheb-GR on actual cache misses, writes result rows and joins returned IDs to public entity types. It varies the fraction of unchanged cohorts over three seeded two-pass workloads. No-cache, invalid query-only caching and ordered-cohort caching are compared against fresh same-cohort references. Output directories must be new; existing databases are rejected rather than silently reused. Timings include database operations and the second-stage graph, but exclude encoding, the first KB scan, arrivals and concurrency. The workload is constructed from real query records; it is not a production trace. Allocated page growth replaces the optional `dbstat` interface unavailable on the original workstation; the failed first attempt is retained in the separate evidence archive.

## Strict admission guard

The traversal-to-sort admission check uses `0 <= edge_weight <= parent_score` without a positive tolerance. A previous `1e-7` relaxation admitted a two-entity counterexample. Source and numerical records from that version remain in the historical evidence archive. The corrected guard is conservative about self-edges: their rounding overshoots can cause fallback even when ignoring the self-edge would be safe.

```bash
python work/audit_strict_guard.py evidence/repeat_20261029/native_traces --output results/strict_guard_verification.json
```

Run `prepare.py` first to fetch the pinned native traversal used by this check. The audit extracts the actual guard and flat-selection function from the public runner, checks three tolerance counterexamples, replays 358 saved native traces including every fallback query, and checks 4,000 synthetic prefixes. The revision 11 evidence also contains fresh native/compiled latency measurements with the strict guard; the table exporter selects those measurements when present. These follow-up measurements preserve the original query subset and repetition protocol and do not overwrite the historical measurements.

## Public inputs

| Input | Source and acquisition |
|---|---|
| WikiMEL and RichpediaMEL author embeddings | [OpenMEL](https://github.com/DAISYzxy/OpenMEL); `python work/fetch_public_inputs.py WikiMEL` and `python work/fetch_public_inputs.py RichpediaMEL` |
| Cleaned WikiDiverse and WikiMEL images and splits | [MIMIC project](https://pengfei-luo.github.io/MIMIC/) and [release repository](https://github.com/pengfei-luo/MIMIC); `python work/fetch_cleaned_inputs.py WikiDiverse --probe` checks current access, and the command without `--probe` downloads and verifies the archive; repeat with `WikiMEL`, then unpack the nested image archives |
| CLIP ViT-B/32 weights | [CLIP](https://github.com/openai/CLIP); `python work/fetch_public_inputs.py clip` |
| Oxford5k and Paris6k Siamac descriptors and ground truth | [Diffusion release](https://github.com/fyang93/diffusion); `python work/fetch_visual_inputs.py` |

Downloads are verified against the hashes in the source code. The visual descriptor archives total approximately 530 MB; they are downloaded only when that command is run. No weights or datasets are distributed in this repository. Public inputs and upstream implementations retain their original terms.

Expected cleaned-data layout:

```text
data/wikidiverse/cleaned/WikiDiverse/
  kb_entity.json
  qid2id.json
  wiki_diverse_train.json
  wiki_diverse_dev.json
  wiki_diverse_test.json
  kb_image/
  mention_image/
data/wikimel/WikiMEL/
  kb_entity_desc.json
  qid2id.json
  WIKIMEL_train.json
  WIKIMEL_dev.json
  kb_image/
  mention_image/
```

The cleaned-data share password stated by the MIMIC authors is `kdd2023`. Preserve the released entity ordering, IDs, image filenames, and train/dev/test splits. Missing or corrupt images are represented by zero vectors with explicit masks; they are not silently replaced by successful examples.

## Cached-input and objective experiments

```bash
python work/fetch_public_inputs.py WikiMEL
python work/fetch_public_inputs.py RichpediaMEL
python work/convert_author_cache.py WikiMEL
python work/convert_author_cache.py RichpediaMEL
python work/author_cache_reproduction.py WikiMEL
python work/author_cache_reproduction.py RichpediaMEL
python work/author_objective_audit.py WikiMEL --mode 3
python work/author_objective_audit.py RichpediaMEL --mode 3
python work/author_objective_audit.py WikiMEL --mode 4
python work/author_objective_audit.py RichpediaMEL --mode 4
python work/audit_enumeration.py
python work/undirected_objective_audit_tight.py --input results/author_objective/WikiMEL_author3_t0.6_nall --limit 128
python work/undirected_objective_audit_tight.py --input results/author_objective/RichpediaMEL_author3_t0.6_nall --limit 128
python work/benchmark_native_retrieval.py WikiMEL
python work/benchmark_native_retrieval.py RichpediaMEL
python work/audit_native_trace.py
python work/audit_native_coverage.py
```

For threshold sensitivity, run each dataset/mode combination again with `--threshold 0.55 --limit 256` and `--threshold 0.65 --limit 256`, then rerun `audit_enumeration.py`. The primary threshold is 0.6; budgets are 1, 5, 10, and 20. Threshold subsets use the first 256 SHA256-ordered query IDs.

Canonical directed outcomes are written to `results/enumerated/`. Initial MILP outcomes remain in `results/author_objective/` or `results/objective_v2/`; a zero reported gap alone is not used to certify the final directed reference. Enumeration checks all subsets of at most ten roots. Undirected timeouts remain unresolved and must not be counted as exact solutions. The strict/tie comparison guard is 1e-7.

The cached-input reproduction uses the released expanded text and image embeddings; it does not regenerate captions or reproduce the full published system score. The unfiltered objective intervention is a separate condition from native name-filtered retrieval. Three-channel and four-channel aggregation must also remain separate conditions.

## Frozen features and low-label controls

After obtaining the cleaned datasets:

```bash
python work/fetch_public_inputs.py clip
python work/encode_full_wikidiverse.py
python work/encode_full_wikimel.py
python work/baseline_lexical.py
python work/baseline_dense.py
python work/fusion_controls.py
python work/fewshot_fusion_baseline.py
python work/coherence_operator_audit.py
python work/wikimel_baseline_lexical.py
python work/wikimel_baseline_dense.py
python work/wikimel_fusion_controls.py
python work/wikimel_fewshot_fusion_baseline.py
python work/wikimel_coherence_operator_audit.py
python work/validate_fastmel_backend.py
python work/fastmel_lowshot.py --budget 128 --seed 101 --epochs 30
python work/fastmel_lowshot_resumable.py --budget 128 --seed 102 --epochs 30
python work/fastmel_lowshot.py --budget 128 --seed 103 --epochs 30
python work/objective_replication_v2.py --dataset wikidiverse --roots native3
python work/objective_replication_v2.py --dataset wikimel --roots native3
python work/objective_replication_v2.py --dataset wikidiverse --roots paper4
python work/objective_replication_v2.py --dataset wikidiverse --roots fast128
python work/objective_replication_fast_seeds.py --dataset wikidiverse --roots fast128 --fast-seed 101 --limit 256
python work/objective_replication_fast_seeds.py --dataset wikidiverse --roots fast128 --fast-seed 102 --limit 256
python work/objective_replication_fast_seeds.py --dataset wikidiverse --roots fast128 --fast-seed 103 --limit 256
python work/audit_enumeration.py
python work/audit_ranking_controls.py
python work/audit_joint_calibration.py
python work/audit_structure.py
```

The low-label control uses 128 identical query labels per seed, batch size 8, and 30 fixed epochs without test-based tuning. It uses the released FAST-MEL pooling class with a native CLIP backend. This is an explicitly adapted low-label control, not a bitwise reproduction of the authors' batch-64 training. Seed 102 used the resumable path after a native runtime failure; that recovery path is retained. Checkpoints are local outputs and are excluded from version control.

Calibration is fitted on the specified training subset; control choices use the development partition. Test outcomes must not be used to select the channel combination, threshold, graph weight, or exclusion rule. The image-quality slice below is exploratory and does not replace the original full test set.

## Physical plans

```bash
python work/benchmark_prefix_plan_v3.py --dataset wikidiverse --scale 8192 --queries 32
python work/benchmark_prefix_plan_v3.py --dataset wikidiverse --scale 32768 --queries 32
python work/benchmark_prefix_plan_v3.py --dataset wikidiverse --queries 32
python work/benchmark_prefix_plan_v3.py --dataset wikidiverse --queries 64
python work/benchmark_prefix_plan_v3.py --dataset wikimel --queries 64
```

Runs use three interleaved repetitions and retain per-query timings and ordered-output checks. Use the same 32 queries across scales for the matched scaling comparison. The 64-query full-corpus run is a separate workload. CPU threads, Faiss BLAS switching, cache state, candidate budgets, and ties are controlled. Timings are hardware dependent. Native cached-input latency excludes raw image encoding and caption generation; isolated graph-stage latency must not be described as end-to-end speedup.

## Reranking and cohort experiments

```bash
python work/encode_wikidiverse_test.py
python work/run_superglobal.py
python work/audit_holdout.py
python work/audit_image_quality.py
python work/analyze_image_quality.py
python work/fetch_visual_inputs.py
python work/run_superglobal_visual.py
python work/audit_diffusion.py --dataset oxford5k --certified
python work/audit_diffusion.py --dataset paris6k --certified
python work/audit_cohort_partitions.py --dataset oxford5k --certified
python work/audit_cohort_partitions.py --dataset paris6k --certified
```

SuperGlobal uses the pinned released reranking stages with 400 candidates, nine neighbors excluding self, and beta 0.15. Text-only and normalized text/image concatenation are both reported. The initial cosine ranking, descriptor augmentation, and complete reranking share identical frozen descriptors and candidates. This transfers the reranker; it does not reproduce the trained SuperGlobal encoder. Oxford/Paris use the original public Siamac descriptors as visual controls, not newly trained zero-shot features.

Diffusion uses truncation 1000, kd 50, kq 10, alpha 0.99, and gamma 3. `--certified` uses CG tolerance 1e-8 with at most 200 iterations and a float64 direct solve of the same matrix when the residual is insufficient. Residuals and fallbacks are recorded. The default command without `--certified` retains the original shorter solver budget for numerical diagnostics; it is not the certified paper comparison.

The cohort experiment preserves the complete gallery and changes only query grouping. It includes the fixed hash partition, two repeated partitions (seeds 20261028 and 20261029), fresh query permutation, and the original gallery-only control. The intentionally stale-cache run violates the cache namespace precondition and is reported separately. Cohort dependence of a transductive method is not itself an implementation error. Use fresh output/cache directories when changing the dataset, source, solver, or graph parameters.

## Interpretation and reproducibility

Results are generated under `results/`; images and arrays are not Git-tracked. Keep negative, failed, and unresolved outcomes. Optimal-set inclusion is not a ranking metric, repeated partitions reuse the same queries, and cross-dataset differences do not identify a causal mechanism. Confidence intervals quantify the specified query/image-cluster variation, not variation across machines or foundation-model pretraining datasets.

This source release does not bundle historical arrays or manuscript assets in the Git tree. The separate evidence release makes compact historical outputs and numerical corrections inspectable. On October 29, 2026, the historical numerical-evidence ZIP was downloaded completely through its unauthenticated public link and its SHA256 was verified. The public verifier passed 1,577 checks, including 1,024 new exact budget solves over 256 saved graphs, and the contract verifier passed 119 checks. These are numerical-artifact checks. The separate October 28 source-dataset connectivity probes retrieved only archive headers; they are not new full-file downloads of the source datasets. Acquisition scripts verify complete download hashes when executed. Public data availability and the cost of full encoding/training remain separate from checking the small exact-selection algorithms.

## Upstream implementations

| Project | Pinned commit | Use |
|---|---|---|
| [OpenMEL](https://github.com/DAISYzxy/OpenMEL) | `5ebbebfe7e56513df8561178a93026d88103c730` | Original rooted expansion and cached-input contract |
| [FAST-MEL](https://github.com/to2002td-cpu/FASTMEL) | `bf9fea428cb9f7fb107238df04b6391da247ea96` | Pooling class and released training negatives |
| [SuperGlobal](https://github.com/ShihaoShao-GH/SuperGlobal) | `86946964b907e6f28a7264add2c15640fae30009` | Descriptor augmentation and reranking |
| [Diffusion](https://github.com/fyang93/diffusion) | `5f5c6b3dae1bf887c686c0f424869b417d632481` | Released graph solver and official evaluation code |

Upstream source is downloaded into the ignored `vendor/` directory with hash verification. Upstream license files are retained when downloaded. Device and API compatibility edits are applied in memory and recorded by the experiment runners.
