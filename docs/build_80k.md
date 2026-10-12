# Build the public 80k adaptation dataset

Use the [command-line guide](../finetuning/README.md) or Step 3 of the
[notebook](../notebooks/finetuning_arbiter.ipynb). The same `finetuning.build_80k`
module is used by both.

## What is built

Exactly **80,000 training decisions** and **8,000 validation decisions**, using pinned
public training splits. Files are generated locally under `results/adaptation/80k_typed/`,
which is ignored by Git. No paid API or LLM labeling is required.

| Group | Train | Validation | Source |
|---|---:|---:|---|
| Typed/workflows | 32,000 | 3,200 | Official Typed + procedural state reconstruction, evidence sufficiency, state perturbation |
| NLI | 24,000 | 2,400 | Translated XNLI train, 15 languages |
| Policy decisions | 24,000 | 2,400 | Procedural policy applicability, policy uncertainty and taxonomy routing |
| Historical replay | 0 (removed) | 0 | Benchmark replay (Banking77/MASSIVE) removed to eliminate bias toward historical suites |

Train type totals: **48,800 choice**, **21,600 noul**, **9,600 score**.
Policy contributes 12k choice and 12k noul; Typed workflows contribute 12.8k choice, 9.6k noul, 9.6k score.
These are policy and workflow structured questions, not newly annotated Fast Decisions corpus.

This is the **typed-boost-80k-v1** recipe. It doubles the volume of typed decisions to 32,000 (40% of the corpus)
and removes the historical replay tasks.

## Source pins

| Dataset | Revision |
|---|---|
| [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | `d0e2f0c42fef86cc15d1688d25a19f5ba7c85b18` |
| [tasksource/procedural-typed-decisions](https://huggingface.co/datasets/tasksource/procedural-typed-decisions) | `609513a3faddf729123266efe9861456fe748f95` |
| [facebook/xnli](https://huggingface.co/datasets/facebook/xnli) | `b8dd5d7af51114dbda02c0e3f6133f332186418e` |

License fields are source metadata, not blanket clearance for redistribution. In
particular, the pinned XNLI source does not declare a license in its card; the builder
records `unspecified-see-upstream-terms`. Check upstream terms before publishing data
or derived releases. Native soft targets are preserved; procedural hard targets are
not replaced with model predictions.

## Selection and checks

1. Keep source train splits only. Group scenario questions before assigning splits.
2. Select official Typed examples first, then fill its budget with procedural examples.
3. Keep all 15 NLI translations of a selected premise together. Training has 1,600
   decisions per language; validation has 160. Relations are nearly equally represented.
   Selection shuffles the first sufficient shard(s), not the complete NLI corpus.
5. Reject contexts matching supplied held-out benchmark cases, cross-split context
   overlaps and duplicate requests. Do not modify the benchmark files.
6. Check each selected request with the real inference formatter and tokenizer. Reject
   over-budget inputs; never silently truncate. Default maximum length is 1,024 tokens.
7. Require exact quotas, schema validity and final train/validation/heldout audit before
   marking the build complete. Missing data causes an error, not oversampling.

The manifest records sources, file hashes, group counts, type/family counts and rejection
counts. A repeated build reuses outputs only when hashes and configuration match.
If an incomplete output directory remains after interruption, use a fresh destination.
Use scripts/prepare_adaptation.py for verified reuse, including recipe quotas and output hashes.

## What the checks do not prove

- No semantic paraphrase or template-overlap audit is performed automatically.
- Published procedural examples are synthetic, not newly human-annotated policies.
- The base may already have seen some of these source datasets in previous training.
- Grouped validation is not necessarily an unseen-rule or out-of-distribution test.
- XNLI train translations vary in quality; 24k translated decisions do not mean 24k
  independent English premise/hypothesis situations.
- Meeting the quotas and passing tokenization does not establish a benchmark gain.

The builder creates a reproducible starting mixture. Validate the adapted model on the
unchanged benchmarks, inspect failures, and report retention as well as improvements.
