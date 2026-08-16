# Final training validation

Validation date: 2026-08-16.

## Pass rule

A dataset passes only when adapted nDCG@10, Recall@10, and MRR@10 are each
strictly greater than raw base-encoder cosine retrieval on the same queries,
corpus, and top-100 candidate sets. Test splits did not influence training or
best-epoch selection.

## Configuration

- Base encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Revision: `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`
- Projector: 384×384, identity initialization
- Hypernetwork hidden dimension: 128
- Dropout: 0
- Positive: deterministic highest-relevance document per query
- Negatives: 63 raw dense-retriever hard negatives from top 100
- Objective: `mean(max(0, 0.05 + base_margin - adapted_margin))`
- Optimizer: Adam, learning rate `1e-4`, weight decay `1e-5`
- Batch size: 32
- Maximum epochs: 30
- Patience: 7
- Seed: 42
- Selection: validation mean of nDCG@10, Recall@10, and MRR@10

| Dataset | Documents | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: |
| SciFact | 5,183 | 688 | 121 | 300 |
| NFCorpus | 3,633 | 2,202 | 388 | 323 |
| ArguAna | 8,674 | 981 | 210 | 210 |

SciFact and NFCorpus retain their official test sets. ArguAna uses a seeded
70/15/15 split because it provides one qrels split. Five qrels referenced
documents absent from the distributed ArguAna corpus and were excluded before
splitting, leaving 1,401 usable queries.

## Validation results

| Dataset | Metric | Base | Adapted | Delta | Status |
| --- | --- | ---: | ---: | ---: | --- |
| SciFact | nDCG@10 | 0.66543 | 0.70725 | +0.04182 | PASS |
|  | Recall@10 | 0.80165 | 0.82645 | +0.02479 | PASS |
|  | MRR@10 | 0.62181 | 0.67170 | +0.04990 | PASS |
| NFCorpus | nDCG@10 | 0.33103 | 0.35069 | +0.01966 | PASS |
|  | Recall@10 | 0.14909 | 0.16284 | +0.01375 | PASS |
|  | MRR@10 | 0.51950 | 0.54446 | +0.02496 | PASS |
| ArguAna | nDCG@10 | 0.37648 | 0.43545 | +0.05897 | PASS |
|  | Recall@10 | 0.80952 | 0.88571 | +0.07619 | PASS |
|  | MRR@10 | 0.24056 | 0.29189 | +0.05134 | PASS |

## Held-out test results

| Dataset | Metric | Base | Adapted | Delta | Status |
| --- | --- | ---: | ---: | ---: | --- |
| SciFact | nDCG@10 | 0.64508 | 0.69166 | +0.04658 | PASS |
|  | Recall@10 | 0.78333 | 0.84333 | +0.06000 | PASS |
|  | MRR@10 | 0.60472 | 0.64800 | +0.04328 | PASS |
| NFCorpus | nDCG@10 | 0.31727 | 0.31990 | +0.00263 | PASS |
|  | Recall@10 | 0.15499 | 0.15581 | +0.00082 | PASS |
|  | MRR@10 | 0.50765 | 0.50751 | -0.00014 | FAIL |
| ArguAna | nDCG@10 | 0.37703 | 0.40933 | +0.03230 | PASS |
|  | Recall@10 | 0.79048 | 0.82857 | +0.03810 | PASS |
|  | MRR@10 | 0.24700 | 0.27586 | +0.02886 | PASS |

The strict all-metrics result is 2/3 held-out datasets. NFCorpus is recorded as
a failure because its MRR delta is negative.

## Checkpoint

Path: `checkpoints/validated/minilm/imrnns-minilm-scifact.pt`

SHA-256: `53a89e54d233a0e966c5ca11a6ef08f9bf6c0283d27678167b7422515fec5c30`

Best SciFact validation epoch: 16.
