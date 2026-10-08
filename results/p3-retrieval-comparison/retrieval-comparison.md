# P3 Retrieval Comparison

- Benchmark: `p2.synthetic-smoke`
- Repository: `fixture.baseline`
- Commit: `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`
- Fixed Top-K: `100`
- Tasks: `3`
- Rewritten tasks: `1`
- Evidence: `deterministic_fake_smoke`

> This synthetic Fake run validates mechanics and comparison accounting. It does not measure real-model retrieval quality.

## Mode summary

| Mode | Recall@1 | Recall@3 | Recall@5 | MRR | Index ms | Query ms | Provider calls | Retries | Failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 0 | 0 | 0 | 0 |
| dense | 0.000 | 0.667 | 1.000 | 0.361 | 0 | 0 | 3 | 0 | 0 |
| hybrid | 0.667 | 1.000 | 1.000 | 0.833 | 0 | 0 | 3 | 0 | 0 |

## Hybrid rank changes

| Task | Baseline | Baseline rank | Hybrid rank | Outcome |
| --- | --- | ---: | ---: | --- |
| baseline-001 | bm25 | 1 | 2 | regressed |
| baseline-001 | dense | 3 | 2 | improved |
| baseline-002 | bm25 | 1 | 1 | unchanged |
| baseline-002 | dense | 4 | 1 | improved |
| baseline-003 | bm25 | 1 | 1 | unchanged |
| baseline-003 | dense | 2 | 1 | improved |
