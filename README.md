# Harmonizer

Agentic Harmonized System (HS) tariff-code classification, built entirely on
open-source tooling and open-weight models. Give it a product description,
photo, or invoice line and it returns a defensible HTS code with the duty
rate, a confidence score, and citations to real customs rulings.

Everything runs locally. No paid APIs are used anywhere in this project.

## Status

Retrieval layer plus the first agent node. The US tariff schedule and 22k
CBP rulings are in Postgres with bge-m3 embeddings and an in-memory BM25
index. Fused retrieval feeds a hierarchical vote over the nearest past
rulings, which is the no-model floor, and a Qwen3 8B adjudicator that argues
for one candidate with a cited precedent. Every stage is scored on the same
held-out rulings so each addition has a measured lift. A cross-encoder
reranker was measured and did not beat fusion, so it stays opt-in. Reports live in `evals/reports/`.

## Data sources

| Source | Endpoint | Use |
|---|---|---|
| USITC Harmonized Tariff Schedule | `https://hts.usitc.gov/reststop/exportList?from=0101&to=9999&format=JSON&styles=false` | Full US schedule with duty columns |
| CBP CROSS rulings search | `https://rulings.cbp.gov/api/search` | Ruling metadata and assigned tariff codes (gold labels) |
| CBP CROSS ruling text | `https://rulings.cbp.gov/api/ruling/{number}` | Full ruling text for retrieval and evals |

Both are US government works in the public domain.

## Quickstart

Requirements: Docker, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com).

```bash
docker compose up -d           # Postgres 17 with pgvector, published on host port 5434
cd apps/api && uv sync && cd ../..
ollama pull bge-m3 && ollama pull qwen3:8b
./hz db migrate                # apply sql/*.sql, idempotent
./hz hts ingest                # USITC schedule, downloaded if no --file is given
./hz cross index               # ruling metadata, newest first, NY and HQ collections
./hz cross fetch-text          # full ruling text, resumable
./hz index hts                 # embed every numbered tariff line
./hz index rulings             # distil each ruling to a product description, embed it
./hz eval split                # 100 dev + 1,000 test rulings, deterministic
./hz eval baseline --split test
./hz eval recall --split test    # recall@k per retrieval mode, the ceiling for later stages
./hz eval rerank-ablation        # does a cross-encoder help? measured on the dev split
./hz eval vote --split test      # nearest ruling vs flat and hierarchical voting over neighbours
./hz eval adjudicate --split dev # Qwen3 adjudicator vs the voting floor, paired per item
```

## Models

All inference is local through Ollama. No paid or hosted APIs are used.

| Model | Licence | Role |
|---|---|---|
| bge-m3 | MIT | Embeddings for tariff lines and rulings, served by Ollama |
| bge-reranker-v2-m3 | Apache-2.0 | Cross-encoder reranking, opt-in only; measured and not adopted, weights cached in `data/hf` |
| Qwen3 8B | Apache-2.0 | Adjudicator: picks a code among retrieved candidates with a cited ruling, structured output, thinking off |

## Layout

```
apps/api/        Python package (FastAPI later), ingestion, index, evals
apps/api/sql/    schema, applied by `hz db migrate`
evals/reports/   committed eval results
data/            raw downloads and logs, gitignored
```

## License

Apache-2.0. See LICENSE.
