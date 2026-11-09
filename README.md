# Harmonizer

Harmonized System (HS) tariff-code classification for US imports, built on
open-source tooling and open-weight models and run entirely on one laptop.
Give it a product description and it returns an HTS code, the tariff line and
general duty rate that code lands on, the past customs ruling it relied on, a
confidence and a short rationale.

```
$ ./hz classify "stainless steel vacuum-insulated water bottle, 500 ml, screw cap, keeps drinks hot or cold"
9617.00.10.00  confidence 0.95
  Vacuum flasks and other vacuum vessels, complete; parts thereof other than
  glass inners: > Vessels: > Having a capacity not exceeding 1 liter
  general duty: 7.2%
  precedent: N353266 (2026-01-28) The tariff classification of a
  vacuum-insulated stainless steel water bottle from China
  The product is a stainless steel vacuum-insulated water bottle, which is a
  double-walled vacuum-sealed container designed to keep drinks hot or cold.
  This matches the description in ruling N353266, which classified a similar
  vacuum-insulated stainless steel water bottle under 9617.00.10.00. The
  General Rule of Interpretation (GRI) 3(a) applies, as the specific
  description of a vacuum-insulated stainless steel water bottle takes
  precedence over more general classifications.
  28.7s, 2758 prompt tokens
```

No paid APIs are used anywhere in this project. The tariff schedule and the
rulings are US government works in the public domain; the models are MIT and
Apache-2.0 licensed.

## Status

v1 is complete and the project is closed. It is a command-line pipeline with
an evaluation harness, not a service: there is no API, no user interface and
no verifier or abstention stage. "What it does not do" below is the honest
boundary. Everything that is here was measured on held-out rulings before it
was kept.

## How it works

1. **Data.** The full US Harmonized Tariff Schedule (35,804 rows, 19,949
   ten-digit lines, hierarchy rebuilt from the export's indent levels) and
   22,349 CBP CROSS rulings with their full text. Rulings that assign exactly
   one tariff code (10,316) form the precedent pool; each is distilled to its
   merchandise paragraph with the codes masked out.
2. **Retrieval.** bge-m3 embeddings in Postgres with pgvector and an in-memory
   BM25 index, fused with reciprocal rank fusion. Fusion runs over both the
   past rulings and the tariff lines.
3. **Vote.** The twenty nearest past rulings vote on the code, settled level by
   level from chapter down to the ten-digit line. This is the no-model floor
   that every later stage has to beat.
4. **Adjudicate.** Qwen3 8B through Ollama gets the product, the vote as a
   stated prior, the twenty rulings with snippets and the ten nearest tariff
   lines, and must answer in a fixed JSON shape with a code copied from the
   candidates, the ruling it relies on, a confidence and a rationale. An
   answer outside the candidate set falls back to the vote and is flagged.

## Results

Evaluation set: 1,100 New York rulings dated 2025 or later with a single
ten-digit code, split deterministically into 100 dev and 1,000 test. Retrieval
is leak-free: an item may only see rulings dated before it, never itself.
Top-1 accuracy is measured at every level of the code because a chapter
match and a ten-digit match are very different outcomes.

| Stage | Split | 2 digits | 4 | 6 | 10 |
|---|---|---|---|---|---|
| Nearest tariff line, dense | test | 48.1% | 30.0% | 15.9% | 5.0% |
| Nearest past ruling, dense | test | 64.5% | 47.8% | 37.0% | 26.4% |
| Nearest past ruling, BM25 + dense fused | test | 70.1% | 53.2% | 41.3% | 29.2% |
| Hierarchical vote over 20 fused rulings | test | 75.1% | 56.1% | 43.4% | 30.3% |
| Qwen3 8B adjudicator | dev | 78.0% | 63.0% | 51.0% | 31.0% |
| Hierarchical vote, same 100 dev items | dev | 72.0% | 56.0% | 45.0% | 28.0% |

What the table says:

- **Past rulings beat the tariff text by a wide margin.** Matching a product to
  how customs actually classified similar goods is far more reliable than
  matching it to the legal text of the schedule.
- **Fusion beats either retriever alone.** BM25 beats dense search on rulings
  and dense beats BM25 on tariff lines; fusing them is best for both.
- **Voting across neighbours helps most at the chapter level** and adds two
  points at six digits over the single nearest ruling.
- **The adjudicator adds about six points at every level** on the dev split,
  with every one of 100 answers valid JSON and a code from the candidate set.
  It agreed with the vote 82% of the time, fixed six wrong votes at six digits
  and broke none. Its self-reported confidence carries signal: 59.3% accuracy
  at six digits when confidence was 0.90 or above, 17.6% between 0.70 and
  0.89, 0% below.
- **The ceiling is retrieval recall.** The correct six-digit subheading is
  among the twenty fused rulings 72.6% of the time on test (79.4% at fifty).
  When it is not there, no adjudicator can find it.

A negative result, kept on purpose: a cross-encoder reranker
(bge-reranker-v2-m3) never beat the fused order on dev in five
configurations, at worst dropping six-digit top-1 from 44% to 26%. New York
rulings share template sentences that a cross-encoder rewards and that BM25's
inverse document frequency discounts. The reranker stays in the code as an
opt-in mode so the ablation is reproducible.

Two caveats on the adjudicator number. It comes from the 100-item dev split
only: the 1,000-item test run takes about six hours on the laptop and was not
completed, though a partial pass stopped at 370 items was tracking 54.1% at
six digits against the 43.4% floor. And the prompt is the first one written,
untuned, so it is a floor for what the model can do rather than a ceiling.

Cost on an M2 Pro with 16 GB: about 18 to 29 seconds and 2,700 to 2,900
prompt tokens per classification. All reports are in `evals/reports/`, with
per-item rows for error analysis.

## Quickstart

Requirements: Docker, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com),
and about 16 GB of memory for the 8B model.

```bash
docker compose up -d           # Postgres 17 with pgvector, published on host port 5434
cd apps/api && uv sync && cd ../..
ollama pull bge-m3 && ollama pull qwen3:8b
./hz db migrate                # apply sql/*.sql, idempotent
./hz hts ingest                # USITC schedule, downloaded if no --file is given
./hz cross index               # ruling metadata, newest first, NY and HQ collections
./hz cross fetch-text          # full ruling text, resumable; the longest step
./hz index hts                 # embed every numbered tariff line
./hz index rulings             # distil each ruling to a product description, embed it
./hz classify "men's short-sleeve t-shirt, knitted, 100% cotton"
```

Building the corpus from scratch takes a few hours, most of it fetching ruling
text from CBP and embedding it. A classification takes about half a minute,
including loading the indexes.

To reproduce the evaluation:

```bash
./hz eval split                  # 100 dev + 1,000 test rulings, deterministic
./hz eval baseline --split test  # nearest tariff line and nearest ruling, per retrieval mode
./hz eval recall --split test    # recall@k per retrieval mode, the ceiling for later stages
./hz eval rerank-ablation        # the cross-encoder ablation, dev split
./hz eval vote --split test      # nearest ruling vs flat and hierarchical voting
./hz eval adjudicate --split dev # adjudicator vs the voting floor, paired per item
```

## Models

All inference is local through Ollama. No paid or hosted APIs are used.

| Model | Licence | Role |
|---|---|---|
| bge-m3 | MIT | Embeddings for tariff lines and rulings |
| Qwen3 8B | Apache-2.0 | Adjudicator: picks a code among retrieved candidates with a cited ruling, structured output, thinking off |
| bge-reranker-v2-m3 | Apache-2.0 | Cross-encoder reranking, opt-in only; measured and not adopted |

## Data sources

| Source | Endpoint | Use |
|---|---|---|
| USITC Harmonized Tariff Schedule | `https://hts.usitc.gov/reststop/exportList?from=0101&to=9999&format=JSON&styles=false` | Full US schedule with duty columns |
| CBP CROSS rulings search | `https://rulings.cbp.gov/api/search` | Ruling metadata and assigned tariff codes (gold labels) |
| CBP CROSS ruling text | `https://rulings.cbp.gov/api/ruling/{number}` | Full ruling text for retrieval and evals |

Both are US government works in the public domain.

## What it does not do

- It is a CLI, not a service. There is no API, no queue and no review interface.
- It takes text only. No photos, no invoice parsing.
- It covers the US schedule only.
- It reports confidence but never acts on it: there is no abstention and no
  hand-off to a human reviewer.
- There is no verifier step checking chapter and section notes or exclusions.
- The precedent pool is the 10,316 rulings that assign exactly one code.
  Rulings covering several products are indexed but never used as precedents.
- A classification from this tool is not legal advice. Binding rulings come
  from CBP.

## What I would build next

In the order the results point to:

1. **Recall before reasoning.** The right subheading is missing from the
   candidates 27% of the time, which caps every downstream stage. Query
   rewriting and multi-code rulings split into one precedent per product are
   the obvious moves.
2. **Abstain.** The confidence signal is real. Answering only at 0.90 and above
   and routing the rest to review would raise precision on the answered set.
3. **Verify.** A second pass that reads the chapter and section notes for the
   chosen heading and can send the case back.
4. **Finish the 1,000-item adjudicator run** so the headline number is on the
   test split.
5. **Serve it.** FastAPI, a review queue and a small UI for the human in the loop.

## Layout

```
apps/api/        Python package: ingestion, index, retrieval, vote, adjudicator, evals
apps/api/sql/    schema, applied by `hz db migrate`
evals/reports/   committed eval results
data/            raw downloads, model caches and logs, gitignored
```

## License

Apache-2.0. See LICENSE.
