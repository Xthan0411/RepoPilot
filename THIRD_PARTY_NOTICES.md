# Third-party notices / 第三方来源说明

RepoPilot original code is MIT-licensed. Upstream issue excerpts and evaluator patches in
`data/manifests/p7-v1/source-rows.json`, the issue text in
`results/p7-holdout/agent-inputs-v1.json`, and evaluator regressions in
`results/p7-holdout/regression-tests-v1.json` remain subject to their upstream licenses; the
RepoPilot license does not relicense them. The frozen manifest records each base commit,
license blob identity, PR/issue URL, and source-row identity.

## Evaluation sources

- SWE-bench Lite dataset: [source](https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite),
  frozen revision `69611d31007e1c6731db8bd5b5c3f2d33f5bab6e`.
- Reference: Carlos E. Jimenez et al., *SWE-bench: Can Language Models Resolve Real-World
  GitHub Issues?*, ICLR 2024.
- SWE-bench harness: [project](https://github.com/SWE-bench/SWE-bench),
  MIT license text retained in [swe-bench.txt](third_party/licenses/swe-bench.txt), fetched
  at the recorded harness revision `726c5461e2ef52d83cf1ea2107870a8bb3328d57`.
- No single dataset-wide license is substituted for the selected repositories' licenses.

## Selected upstream repositories

| Repository | License | Recorded copyright / notice | Exact upstream license | Included text |
| --- | --- | --- | --- | --- |
| [django/django](https://github.com/django/django) | BSD-3-Clause | Copyright (c) Django Software Foundation and individual contributors. | [pinned source](https://github.com/django/django/blob/44c24bf02835323d5418512ebe8e76166739ebf8/LICENSE) | [django.txt](third_party/licenses/django.txt) |
| [mwaskom/seaborn](https://github.com/mwaskom/seaborn) | BSD-3-Clause | Copyright (c) 2012-2021, Michael L. Waskom. | [pinned source](https://github.com/mwaskom/seaborn/blob/4a9e54962a29c12a8b103d75f838e0e795a6974d/LICENSE.md) | [seaborn.txt](third_party/licenses/seaborn.txt) |
| [pallets/flask](https://github.com/pallets/flask) | BSD-3-Clause | Copyright 2010 Pallets. | [pinned source](https://github.com/pallets/flask/blob/4c288bc97ea371817199908d0d9b12de9dae327e/LICENSE.rst) | [flask.txt](third_party/licenses/flask.txt) |
| [pydata/xarray](https://github.com/pydata/xarray) | Apache-2.0 | Licensed to the xarray developers under the Apache License, Version 2.0. | [pinned source](https://github.com/pydata/xarray/blob/a64cf2d5476e7bbda099b34c40b7be1880dbd39a/LICENSE) | [xarray.txt](third_party/licenses/xarray.txt) |
| [pytest-dev/pytest](https://github.com/pytest-dev/pytest) | MIT | Copyright (c) 2004 Holger Krekel and others. | [pinned source](https://github.com/pytest-dev/pytest/blob/6995257cf470d2143ad1683824962de4071c0eb7/LICENSE) | [pytest.txt](third_party/licenses/pytest.txt) |
| [sympy/sympy](https://github.com/sympy/sympy) | BSD-3-Clause | Copyright (c) 2006-2020 SymPy Development Team. | [pinned source](https://github.com/sympy/sympy/blob/cffd4e0f86fefd4802349a9f9b19ed70934ea354/LICENSE) | [sympy.txt](third_party/licenses/sympy.txt) |


The complete license texts are included without altering the upstream copyright or
disclaimer terms. Xarray and pytest each contribute two selected cases whose license
blobs match, so each identical license is included once. SymPy's license file also
contains its own bundled third-party notices, retained in full.

## Runtime dependencies

Runtime and development dependencies are declared in `pyproject.toml` and locked in
`uv.lock`; their code and model weights are not vendored in this snapshot. Installing or
downloading them uses their respective upstream terms. This is source attribution,
not a claim of exhaustive legal or transitive-dependency license review.

Key external components include [LangGraph](https://github.com/langchain-ai/langgraph),
[LangChain Core](https://github.com/langchain-ai/langchain),
[OpenAI Python SDK](https://github.com/openai/openai-python),
[Pydantic](https://github.com/pydantic/pydantic),
[Psycopg](https://github.com/psycopg/psycopg),
[pgvector](https://github.com/pgvector/pgvector),
[Sentence Transformers](https://github.com/huggingface/sentence-transformers),
[Transformers](https://github.com/huggingface/transformers) and
[PyTorch](https://github.com/pytorch/pytorch). The Docker daemon, task images, model
weights and dependency sources are not distributed by this repository.

The optional [Jina code embedding model](https://huggingface.co/jinaai/jina-embeddings-v2-base-code/tree/516f4baf13dec4ddddda8631e019b5737c8bc250)
and its [remote implementation](https://huggingface.co/jinaai/jina-bert-v2-qk-post-norm/tree/3baf9e3ac750e76e8edd3019170176884695fb94)
are revision-pinned; remote code requires explicit trust configuration. Consult the
upstream terms before downloading or redistributing them. RepoPilot's MIT license
does not grant rights to those weights, images or remote code.

## 中文说明

公开快照保留 SWE-bench Lite 选定任务的来源信息和评测补丁，各项目沿用原许可，不被统一改成 RepoPilot 的 MIT。上述许可文本从对应固定 commit 获取，并与数据集 Manifest 中的 Git blob SHA 核对；没有打包依赖库源码或模型权重。
