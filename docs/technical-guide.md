# RepoPilot technical guide

This reference describes implemented commands and contracts in the publication snapshot.
For current progress, see [development status](development-status.md).

## Development setup

Requirements:

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

Install the locked development environment:

```bash
uv sync --locked
```

Run the current quality checks:

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy src/repopilot
```

The same commands run in ordinary GitHub CI with Python 3.12, read-only repository
permissions, and no model secrets or Docker services.

The optional local Jina runtime is deliberately excluded from the ordinary environment:

```bash
uv sync --locked --extra jina
```

This installs the large SentenceTransformers/PyTorch runtime but does not download or load
model weights until the explicit Jina adapter is first called.

## P0 smoke CLI

Run the deterministic vertical slice:

```bash
uv run repopilot smoke \
  --manifest tests/fixtures/p0_smoke_manifest.json \
  --output-dir results/p0-smoke \
  --config configs/default.yaml
```

This writes `result.json` and `trace.json` and prints the `FinalResult`. Exit code `0`
means the smoke pipeline completed; the result remains `stopped/agent_stopped` because no
tool, patch, test, or Verifier is active. The command rejects real providers and any
Sandbox network mode.

## P1 Agent fixture and replay CLI

Run the complete deterministic Agent loop:

```bash
uv run repopilot run-fixture \
  --manifest tests/fixtures/p1_agent_manifest.json \
  --output-dir results/p1-fixture \
  --config configs/default.yaml
```

Verify the resulting Trace offline:

```bash
uv run repopilot replay \
  --trace results/p1-fixture/trace.json
```

The fixture command performs contained repository reads and scripted Fake Sandbox behavior;
it does not call a model, network, host subprocess, or Docker. Replay performs no Provider,
Registry, repository, or Sandbox operation.

## P2 BM25 localization baseline

Run the complete deterministic sparse-retrieval evaluation:

```bash
uv run repopilot eval-bm25 \
  --manifest tests/fixtures/p2_bm25_baseline_manifest.json \
  --output-dir results/p2-baseline \
  --config configs/eval.yaml
```

The command builds one repository corpus and BM25 index, constructs each query only from
Agent-visible Issue title/body, deduplicates chunk hits to file rankings, and only then
joins evaluator-owned `gold_files`. It writes `chunks.jsonl`, `bm25-index.json`,
`tasks.jsonl`, `tasks.csv`, `summary.md`, and `baseline.json`.

The tracked first report contains three intentionally separable synthetic fixture tasks
and scores Recall@1/3/5 and MRR of `1.0`. This validates the metric/artifact pipeline and
expected-patch isolation; it is not evidence of 100% real-world localization quality or
Agent task success.

## P3 retrieval comparison smoke

Regenerate the tracked, network-free BM25/Dense/Hybrid mechanism comparison:

```bash
uv run python scripts/run_p3_retrieval_comparison_smoke.py
```

This uses the frozen P2 task set and corpus, real BM25 logic, deterministic Fake
embeddings, RRF Hybrid, and one Fake structured rewrite. It writes versioned JSON plus a
human-readable table to `results/p3-retrieval-comparison/`. The report records all three
tasks, the fixed Top-K, indexing/query latency, calls/retries/failures, and every Hybrid
rank change relative to BM25 and Dense. Expected patch text is absent.

The current Fake smoke contains both improved and regressed Hybrid rank cases. Those
directions validate comparison and bad-case accounting; they must not be quoted as
real-model gains.

## P5 production repair CLI

Run one Agent-only task with an already-local, immutable Docker image:

```bash
uv run repopilot repair \
  --task path/to/task.json \
  --sandbox-image python@sha256:<64-hex-digest> \
  --provider-script path/to/scripted-provider.json \
  --output-dir results/repair \
  --config configs/default.yaml
```

`--task` must be a strict `TaskSpec` with exactly one local repository path; evaluator
gold fields, URLs, symlinked inputs, path escape, and malformed or oversized JSON are
rejected. The default Fake profile requires a versioned Provider script whose request IDs
match the run's decision order. For an OpenAI profile, omit `--provider-script`; the API
key is read only from the environment.

The output directory must be empty and must not be a symlink. A completed run writes
`result.json`, replayable `trace.json`, `artifacts.json`, and hash-addressed patch/test
streams below `artifacts/`. CLI exit `0` means the workflow completed and wrote a result;
inspect `FinalResult.status` and `termination_reason` to determine whether the repair was
verified or safely stopped.

## Configuration

RepoPilot ships `configs/default.yaml` and `configs/eval.yaml`. Runtime settings are merged
in this order:

```text
safe code defaults < YAML profile < environment variables < explicit overrides
```

The default profile uses the deterministic Fake provider. Selecting the OpenAI provider
requires `OPENAI_API_KEY` and a canonical pinned `gpt-*` model ID; mutable family or
`latest` aliases are rejected. Enabling LangSmith tracing requires `LANGSMITH_API_KEY`.

API keys and the PostgreSQL URL are represented as secrets and are redacted from safe
configuration snapshots. Configuration loading does not connect to a model, database, or
Docker.

The OpenAI decision boundary adds:

- `REPOPILOT_OPENAI_REASONING_EFFORT` (`medium` by default; `none`, `low`, `medium`,
  `high`, `xhigh`, or `max`)
- `REPOPILOT_OPENAI_MAX_OUTPUT_TOKENS` (default `4096`, allowed range `1`–`128000`)
- `REPOPILOT_OPENAI_REQUEST_TIMEOUT_SECONDS` (default `120`, allowed range `1`–`600`)
- `REPOPILOT_OPENAI_MAX_RETRIES` (default `2`, allowed range `0`–`5`)
- `REPOPILOT_OPENAI_RETRY_BACKOFF_MS` (default `500`, allowed range `0`–`10000`)
- `REPOPILOT_OPENAI_PROMPT_MAX_CHARS` (default `50000`, allowed range
  `4096`–`200000`)
- four explicit input, cached-input, cache-write, and output per-million-token USD rates
  used for auditable estimates

The adapter disables SDK retries, does not stream, sends `store=false`, and never places
the API key in its payload. Transient retry delays are capped at 30 seconds. Pricing is
configuration rather than a timeless constant so a run retains the rates used for its
estimate.

Read-only repository access is bounded by:

- `REPOPILOT_REPOSITORY_MAX_FILE_BYTES` (default `1000000`)
- `REPOPILOT_REPOSITORY_MAX_READ_BYTES` (default `65536`)
- `REPOPILOT_REPOSITORY_MAX_READ_LINES` (default `400`)
- `REPOPILOT_REPOSITORY_MAX_LIST_ENTRIES` (default `2000`)

Registry and source inspection add:

- `REPOPILOT_TOOL_MAX_OUTPUT_BYTES` (default `65536`, hard maximum `1000000`)
- `REPOPILOT_SEARCH_MAX_MATCHES` (default `100`, hard maximum `1000`)
- `REPOPILOT_SEARCH_MAX_QUERY_LENGTH` (default `256`, hard maximum `2000`)
- `REPOPILOT_MAX_REPLANS` (default `1`, allowed range `0`–`10`)

Python and text chunking add:

- `REPOPILOT_CHUNK_MAX_LINES` (default `200`, hard maximum `2000`)
- `REPOPILOT_CHUNK_OVERLAP_LINES` (default `20`, hard maximum `500` and less than max lines)

Sparse retrieval adds:

- `REPOPILOT_BM25_K1` (default `1.5`, allowed range greater than `0` through `5`)
- `REPOPILOT_BM25_B` (default `0.75`, allowed range `0`–`1`)
- `REPOPILOT_BM25_DEFAULT_TOP_K` (default `10`, allowed range `1`–`100`)

Unified retrieval and RRF add:

- `REPOPILOT_RETRIEVAL_MODE` (`bm25` by default; also `dense` or `hybrid`)
- `REPOPILOT_RETRIEVAL_DEFAULT_TOP_K` (default `10`, allowed range `1`–`100`)
- `REPOPILOT_HYBRID_CANDIDATE_TOP_K` (default `100`, allowed range `1`–`100`)
- `REPOPILOT_HYBRID_RRF_K` (default `60`, allowed range `1`–`1000`)

The Hybrid candidate depth must not be smaller than the common default output depth.
Environment values override YAML, and explicit `load_settings` overrides remain highest
priority.

One-shot query rewriting adds:

- `REPOPILOT_QUERY_REWRITE_ENABLED` (`false` by default)
- `REPOPILOT_QUERY_REWRITE_MIN_HITS` (default `3`, allowed range `1`–`100`)
- `REPOPILOT_QUERY_REWRITE_MAX_CONTEXT_CHUNKS` (default `5`, allowed range `0`–`10`)

Rewriting runs only when enabled and the initial hit count is below the threshold. It can
run at most once, receives at most the configured number of bounded repository chunks
marked as untrusted data, and can return only the allowlisted query fields. It cannot
request tools, commands, permissions, policy, or configuration changes.

Sandbox policy limits add:

- `REPOPILOT_SANDBOX_NETWORK` (`none` by default; optional `setup_only`)
- `REPOPILOT_SANDBOX_CPU_LIMIT` (default `1.0`, greater than `0` through `8`)
- `REPOPILOT_SANDBOX_MEMORY_LIMIT_MB` (default `1024`, range `128`–`8192`)
- `REPOPILOT_SANDBOX_PIDS_LIMIT` (default `256`, range `16`–`1024`)
- `REPOPILOT_SANDBOX_TMPFS_LIMIT_MB` (default `256`, range `16`–`2048`)
- `REPOPILOT_SANDBOX_COMMAND_TIMEOUT_SECONDS` (default `300`, range `1`–`3600`)
- `REPOPILOT_SANDBOX_MAX_OUTPUT_BYTES` (default `1000000`, maximum `10000000`)
- `REPOPILOT_SANDBOX_MAX_ARTIFACT_BYTES` (default `20000000`, maximum `100000000`)

tmpfs cannot exceed memory and captured output cannot exceed the complete artifact limit.
Image, executable, environment, privilege, capability, mount, rootfs, and
no-new-privileges controls are hard invariants rather than environment overrides.

Dense indexing and retrieval add:

- `REPOPILOT_DENSE_DEFAULT_TOP_K` (default `10`, allowed range `1`–`100`)
- `REPOPILOT_EMBEDDING_PROVIDER` (`fake` by default; optional `jina_local`)
- `REPOPILOT_EMBEDDING_MODEL` and `REPOPILOT_EMBEDDING_REVISION`
- `REPOPILOT_EMBEDDING_BATCH_SIZE` (default `32`, allowed range `1`–`256`)
- `REPOPILOT_EMBEDDING_MAX_RETRIES` (default `2` extra attempts, allowed range `0`–`5`)
- `REPOPILOT_EMBEDDING_RETRY_BACKOFF_MS` (default `100`, exponential between attempts)
- `REPOPILOT_EMBEDDING_MAX_LENGTH` (default `2048`, hard maximum `8192`)
- `REPOPILOT_EMBEDDING_NORMALIZE` (default `true`)
- `REPOPILOT_EMBEDDING_CACHE_DIR` (default `artifacts/embedding-cache`)
- `REPOPILOT_EMBEDDING_TRUST_REMOTE_CODE` (default `false`)

Selecting `jina_local` requires the exact model
`jinaai/jina-embeddings-v2-base-code`, an immutable 40-character revision, dimension
`768`, and explicit `REPOPILOT_EMBEDDING_TRUST_REMOTE_CODE=true`. A mutable `main`
revision is rejected.

The implementation accepts canonical POSIX paths beneath one trusted root. It never follows
repository symlinks, does not expose `.git`, `.hg`, `.svn`, or `.env`, and has no write,
delete, patch, Git, or command-execution method. Binary, invalid UTF-8, oversized, or
partially readable source files are skipped with typed reasons rather than treated as
complete search input.

## Agent state and checkpoints

P1 uses a LangGraph `StateGraph` whose channels mirror the validated `AgentState` model.
Working-memory entries, Observations, consumed tool-call IDs, tool results, failure
records, and action fingerprints use deterministic append-once reducers; plans, retrieved
context, patch/test snapshots, budgets, and termination state are replaced explicitly.
Partial updates are previewed against all `AgentState` invariants before a checkpoint is
written.

The current checkpointer is `InMemorySaver`. It supports deterministic thread isolation and
recovery while the Python process and saver object remain alive; it is not disk or database
persistence. `thread_id` is a required per-run argument rather than an environment
variable. Checkpoint deserialization uses an explicit RepoPilot type allowlist and passes
LangGraph strict MsgPack mode.

## Repository ingestion

`repopilot.retrieval.ingest_repository` accepts a `ReadOnlyRepository`, validated
repository ID, and 40-character lowercase commit SHA. It returns a deterministic
`RepositoryManifest` containing:

- stable `source_id` derived from repository ID, commit, and canonical POSIX path;
- complete source `content_hash` and byte size;
- the effective versioned ignore policy;
- typed skipped entries for symlinks, binary files, invalid UTF-8, and oversized files;
- a canonical SHA-256 covering the complete Manifest.

The default ignore policy excludes `.env`, Git/Mercurial/Subversion metadata, `.venv`,
`.tox`, `__pycache__`, and common Python tool caches. Existing Repository denied names are
always retained. `REPOPILOT_REPOSITORY_MAX_FILE_BYTES` and
`REPOPILOT_REPOSITORY_MAX_LIST_ENTRIES` remain the relevant environment limits; P2-S1
adds no configuration or environment variable.

A truncated listing fails instead of emitting a partial index. Source text and absolute
paths are never stored in the Manifest. The supplied commit is trusted metadata for an
already prepared immutable checkout: ingestion does not execute Git and does not claim to
verify a mutable working tree against that SHA.

## Python AST-aware chunking

`repopilot.retrieval.chunk_python_repository` first verifies the Manifest digest, then
re-reads each lowercase `.py` source and requires its size and complete hash to match. It
uses Python's standard-library AST and never imports or executes repository code.

Valid sources produce deterministic module, class, function, async-function, method, and
nested-symbol chunks. Decorators are included in symbol ranges; nonblank module-level
regions outside top-level definitions remain searchable module chunks. Parent symbols and
nested symbols intentionally overlap so retrieval can select either container context or
the precise leaf.

Whitespace-only files produce an explicit empty result. Syntax errors produce marked text
fallback windows rather than false AST claims. Units longer than the configured line
budget use deterministic overlap. Every chunk carries repository/commit/path, qualified
symbol, exact line range, source ID/hash, chunk content/hash, strategy metadata, and a
stable canonical ID.

## Text chunks and JSONL corpus

`TextChunker` owns lowercase Markdown, RST, text, TOML, YAML, JSON, INI, and CFG sources.
Markdown uses ATX heading sections outside fenced code; plain text and configuration use
line windows. Configuration is indexed as inert text and is never parsed or executed.
Whitespace-only sources remain explicit empty source results.

`RepositoryChunkCorpusBuilder` combines Python, repository text, and optional Agent-visible
`TaskSpec` Issue title/body. Incremental reuse requires exact source ID/hash, processor
version, and window-policy fingerprint. Empty sources are reusable because the corpus
header retains snapshots independently of chunk records.

`write_chunk_corpus` writes deterministic JSONL: the first versioned record contains
repository/commit and sorted source snapshots; subsequent versioned records contain
sorted `CodeChunk` values. `read_chunk_corpus` rejects invalid documents, unsupported
versions, mixed repository/commit identity, duplicate chunk IDs, unsorted records, bad
content hashes, and inconsistent source ownership.

## BM25 sparse retrieval

`build_bm25_index` consumes one validated `ChunkCorpus` and emits an immutable index bound
to its repository ID, commit, and chunk fingerprint. Tokenization preserves exact
identifiers and paths while splitting common code boundaries. Source, path, and qualified
symbol fields use fixed 1/2/3 integer weights.

`BM25Retriever.search` returns shared `RetrievalHit` values with sparse rank, score,
matched-term/path/symbol reason, and stable `chunk_id` tie-breaking. Blank or punctuation-
only queries fail explicitly; no lexical match is a successful empty result. Index JSON
is deterministic, versioned, atomically written, and SHA-256 protected.

This remains a standalone retrieval baseline. The current Agent tool has not been silently
switched from literal search. P3 reuses the same query and metric boundary for Dense and
Hybrid comparisons.

## P3 vector storage and Embedding

Start the contained local database and run the real storage integration contract:

```bash
docker compose up -d --wait pgvector
REPOPILOT_TEST_DATABASE_URL=postgresql://repopilot:repopilot@localhost:55433/repopilot \
  uv run pytest -m integration -q
```

`VectorStore` has deterministic in-memory and Psycopg implementations with the same
initialize/upsert/replace/list/search/delete lifecycle. The migration pins schema version
and embedding dimension and adds an HNSW cosine expression index. Values are parameterized
and database errors are reduced to safe typed codes without DSNs, SQL, source text, or
vectors.

The Embedding identity includes provider, model, immutable revision, dimension,
normalization, and maximum input length. Its SHA-256 fingerprint is used in cache keys and
vector namespaces. The local cache adds the chunk content hash, so equal content reuses a
vector across chunks and repositories while any model or policy change forces a miss.
Each entry is size-bounded, schema-versioned, integrity-protected, and atomically written.
Corruption fails closed instead of silently calling the model again.

The default Fake creates stable normalized hash vectors without files, network, or model
dependencies. The Jina adapter loads lazily, passes the configured maximum sequence
length, checks the reported output dimension, and validates every finite response vector.
Because the official model uses custom code, RepoPilot requires explicit consent and a
pinned model revision before loading it. Ordinary tests use the Fake or an injected inert
encoder and never download model weights.

`DenseRetriever` embeds each normalized query once, searches only the exact
repository/commit/model/version/dimension namespace, maps cosine distance to similarity
with `1 - distance`, and orders by score descending with `chunk_id` ascending as the exact
tie break. Blank/oversized queries, Top-K outside `1`–`100`, zero/non-finite/wrong-dimension
vectors, identity mismatches, Provider failures, storage failures, and malformed backend
results have typed failures. No match is a successful empty result.

The Dense evaluator deliberately reuses P2's Issue-title/body query, retrieval depth,
chunk-to-file first-occurrence deduplication, task splits, gold files, Recall@1/3/5, and
MRR. Gold is joined only after retrieval, and Provider calls/retries are reported
separately for indexing and queries. The deterministic Fake proves the pipeline and
accounting only; it is not a real embedding-quality result.

`HybridRetriever` keeps one `search(query, top_k, metadata_filter)` call shape for all
three configured modes. Hybrid mode retrieves up to the configured candidate depth from
both branches and computes `1 / (rrf_k + rank)` per available branch. It never adds raw
BM25 and cosine scores, which have incompatible scales. Original sparse/dense ranks and
scores remain in each hit; final ties use `chunk_id` ascending.

`RetrievalFilter` supports canonical path prefixes, case-normalized languages, and
`all/tests_only/non_tests_only`. Values within one filter dimension are ORed; path,
language, and test scope are ANDed. Test classification recognizes `test/`, `tests/`,
`__tests__/`, `test_*`, `*_test`, and `conftest.py` paths. The corpus compiles these
predicates into validated chunk IDs. BM25 filters before scoring and pgvector receives a
bound `ANY(...)` allowlist before cosine ordering; rejected paths or language strings
never become SQL.

`QueryRewriteController` preserves the original query and makes no Provider call when
disabled or when initial evidence is sufficient. Weak evidence may trigger exactly one
correlated structured response containing separate implementation and test queries.
Retrieved repository text is count/size bounded and labeled
`untrusted_repository_text`; the response schema has no tool, command, permission,
policy, or configuration surface. A second attempt is stopped as `limit_reached`.

`evaluate_retrieval_comparison` requires all three configured modes and applies one
validated effective query equally to each arm. It joins gold only after retrieval,
retains failed tasks as zero-score samples, and emits deterministic JSON/Markdown with
sample count, Recall@1/3/5, MRR, indexing/query latency, Provider calls/retries, failure
codes, and Hybrid improved/regressed/unchanged cases.

## P4 Docker Sandbox integration

The ordinary suite injects a Fake Docker client and never contacts the daemon. To run the
real prepare/inspect/destroy contract with the already-local P4-S2 fixture image:

```bash
REPOPILOT_TEST_SANDBOX_IMAGE=alpine@sha256:d9e853e87e55526f6b2917df91a2115c36dd7c696a35be12163d44e6e2a4b6bc \
  uv run pytest tests/integration/test_docker_lifecycle_integration.py -q
```

The value must be an immutable repository digest already present in Docker; the adapter
does not pull mutable tags. This environment variable belongs only to the opt-in
integration test. Runtime image selection is passed by trusted application/deployment
code, not read from RepoPilot configuration.

The test prepares two simultaneous task containers and proves distinct IDs/workspaces,
the inspect-enforced security posture, filtered copying, idempotent removal, and absence
after cleanup. The Alpine fixture only proves lifecycle controls.

Run the P4-S3 execution contract with the local digest-pinned Python fixture:

```bash
REPOPILOT_TEST_SANDBOX_PYTHON_IMAGE=python@sha256:6d43704baacd1bfbe7c295d7f13079d5d8104ed33568873133f8fc69980419df \
  uv run pytest tests/integration/test_docker_execution_integration.py -q
```

This test proves a real setup-only network transition and post-setup disconnection,
shell-free argv execution, nonzero exit preservation, preview truncation with complete
artifact retention, line-range reading, runtime symlink rejection, timeout-driven
container deletion, artifact availability after deletion, and final absence from Docker.
The digest shown is the arm64 image used for the recorded local Gate; another platform
must supply its already-local official Python RepoDigest rather than falling back to a
mutable tag.

Run the P4-S4 core adversarial suite with the same already-local Python fixture:

```bash
REPOPILOT_TEST_SANDBOX_PYTHON_IMAGE=python@sha256:6d43704baacd1bfbe7c295d7f13079d5d8104ed33568873133f8fc69980419df \
  uv run pytest -m security -q
```

The suite uses only temporary fake secret and host-file canaries; it does not inspect real
user credentials. Without the environment variable the host-free policy attacks still
run and the real-Docker proof is skipped. The test image variable is not a runtime policy
override.

The same command also runs the P4-S5 resource-abuse tests. The reproducible recorded
configuration and cleanup outcomes are stored in
[`results/p4-sandbox-security/cleanup-audit.md`](../results/p4-sandbox-security/cleanup-audit.md)
and its machine-readable JSON companion.

## Security

- Never commit `.env` or real API keys.
- `.env.example` contains variable names and safe local defaults only.
- [`docs/threat-model.md`](threat-model.md) defines Sandbox assets, adversaries,
  trust boundaries, hard policy invariants, typed denials, audit fields, residual risks,
  the P4-S2 lifecycle proof, P4-S3 execution boundary, P4-S4 adversarial evidence, P4-S5
  resource-abuse evidence, and residual Docker boundary.
- P4 proves local containment for the tested command, path, network, secret, timeout,
  PID, CPU, memory, output, artifact, and cleanup cases. It is not a VM boundary or a
  production multi-tenant security certification.
- Hugging Face remote model code is disabled by default. Enabling it requires an immutable
  revision and explicit configuration after reviewing the model source.
- Real-world performance and success-rate claims require a larger fixed external task set;
  the tracked synthetic smoke baseline is labeled accordingly.
