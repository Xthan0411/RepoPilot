# RepoPilot BM25 Baseline: p2.synthetic-smoke

## Run

- Repository: `fixture.baseline`
- Commit: `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`
- Tasks: 3
- Sources / chunks: 5 / 10
- Retrieval depth: 100 chunks
- BM25 k1 / b: 1.5 / 0.75
- Indexing latency: 4 ms
- Mean retrieval latency: 0.000 ms

## Aggregate localization

| Recall@1 | Recall@3 | Recall@5 | MRR | Relevant tasks |
|---:|---:|---:|---:|---:|
| 1.000000 | 1.000000 | 1.000000 | 1.000000 | 3/3 |

## Tasks

| Task | Split | First relevant rank | R@1 | R@3 | R@5 |
|---|---|---:|---:|---:|---:|
| `baseline-001` | dev | 1 | 1.000000 | 1.000000 | 1.000000 |
| `baseline-002` | dev | 1 | 1.000000 | 1.000000 | 1.000000 |
| `baseline-003` | holdout | 1 | 1.000000 | 1.000000 | 1.000000 |

> This report is evaluator-owned. Gold files are used only after retrieval;
> expected patch content is never included in the query, corpus, index, or report.
