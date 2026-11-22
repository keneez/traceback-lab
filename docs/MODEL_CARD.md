**Traceback Lab model card**

Serving query language is English. The interface is Russian. The retrieval target is an existing Python function in a fixed source snapshot. Historical positives are functions modified by a selected patch.

## Bundled Compact-64

The default installation includes a 64-dimensional deep averaging dual encoder with 8192 hashed token buckets, two residual tanh projections and L2 normalization. Symmetric contrastive learning uses in-batch negatives, token dropout and a penalty on the distance between query and code projections. The release contains float16 parameters in NPZ format. NumPy performs inference in float32 without a transformer runtime.

Training used 4,984 deduplicated docstring/implementation pairs from CPython 3.12.14. Another 525 pairs from disjoint top-level standard-library modules supported model selection. Initialization and regularization variants were compared on that validation task. The released ninth epoch reached MRR 0.2944 against 0.2810 at random initialization. These two numbers measure docstring retrieval. The historical localization experiment is a separate task.

On 25 non-demo historical cases from four external projects, Compact-64 alone scored 3/25 on Hit@5. The combination with BM25 and graph scored 12/25. BM25 with stack and graph scored 13/25 and is the UI default when the service loads Compact-64. The experiment is small and the text source history is unverified. Dataset selection, individual results and grouped intervals are in the report. [Training journal](../src/traceback_lab/assets/compact/training.json) · [Historical evaluation](../reports/compact_evaluation.json)

## Transformer pipeline

| Component | Initialization and behavior |
|---|---|
| Bi-encoder | microsoft/unixcoder-base, encoder-only prefix, official full attention mask and mean pooling, L2 normalization |
| Overview | 512 tokens including specials, up to 160 header tokens, head/tail truncation |
| Query | 256 tokens including specials, prioritizing title and observable diagnostics |
| Reranker | microsoft/codebert-base with a learned scalar classification head |
| Pair | 128 query tokens, 48 metadata tokens, up to 332 body tokens, four special tokens |
| Windows | Up to three, overlap 96 tokens, selected with query terms and confirmed frames |
| Aggregation | Log-mean-exp |
| Calibration | Logistic regression, event is top1 matching a historical changed function |

The downloaded UniXcoder checkpoint is pretrained. A full-corpus fine-tuned model remains an experiment to run on the user's chosen corpus. Both training paths are exercised on small local test networks with save/reload and inference checks.

The bi-encoder objective distributes target probability across known positives. Batch masks preserve known positives and exclude ambiguous code duplicates. Mining uses training queries. The random continuation baseline should use the same optimizer update budget as the hard-negative continuation.

Training metadata records case IDs, corpus manifest, hyperparameters, updates, loss and validation metrics. Models are published in immutable checkpoint directories. A current.json pointer is replaced after a complete save.

Cosine similarities, RRF scores and cross-encoder logits describe ranking. UI probabilities require a compatible calibrator, examples of both outcomes and independent threshold groups.

The three supplied demonstrations are an integration sample. The evaluation report preserves all outcomes, including weaker first-place accuracy. Larger test sets and unseen repositories require their own experiment.

Source documentation is published by the model authors. [UniXcoder](https://huggingface.co/microsoft/unixcoder-base) [CodeBERT](https://huggingface.co/microsoft/codebert-base)
