# PLAN.md — Voice Screening Agent

**Historical implementation roadmap, with the shipped canonical contract recorded below.**
Derived from `00_TASK_BRIEF.md`, `QUESTION.md`, `data/corpus/*`, and `CLAUDE.md`.

## 0. Canonical shipped contract (supersedes historical phase notes)

- Runtime: Python 3.11, one FastAPI process, static `static/index.html`, LangGraph with
  `InMemorySaver`, launched with `uv run uvicorn app.main:app --port 8000`.
- Voice: ElevenLabs Scribe `scribe_v1` for STT and `eleven_multilingual_v2` for TTS. Uploads retain
  their real filename/MIME and accept bounded MP3, WebM, MP4, and M4A input.
- Judge: Groq `openai/gpt-oss-120b` with low reasoning and strict JSON Schema. Pydantic enforces
  score 1-5 and evidence/status invariants. Provider failures map to useful 502/503 API errors;
  only final TTS retains the intentional text-only degradation path.
- RAG: local FastEmbed `BAAI/bge-small-en-v1.5` plus NumPy cosine ranking. All five rubric
  competencies from `data/corpus/00_scoring_rubric.md` are mandatory evaluation scope; semantic
  top-1 retrieval over five reference docs chooses supplemental context only and never decides
  what may be reported missing.
- State: checkpointed values are plain JSON-compatible dictionaries; Pydantic models are rebuilt
  at node boundaries.
- Data: rubric/reference inputs are under `data/corpus/`; supplied samples are under `data/audio/`.
- Verification: `uv run pytest` is the offline suite. `uv run python scripts/eval_samples.py` is
  the paced live gate and invokes real STT, Groq graph calls, and English/Arabic TTS.
- Docker: optional and not implemented; the native `uv` path is the only supported run contract.

Everything from section 1 onward is retained as a **historical phase plan and decision log**.
Where it conflicts with section 0 or the shipped code, section 0 and the code take precedence.
Where this plan departs from `CLAUDE.md`, the deviation is marked **[DEVIATION]** with its reason.

---

## 1. Project overview

A voice agent that runs one technical screening turn end to end:

> asks one .NET question aloud → transcribes a code-mixed Egyptian-Arabic/English spoken answer →
> retrieves the relevant rubric criteria → builds an evidence-backed coverage map →
> **decides deterministically whether to ask one follow-up** → scores 1–5 with a one-line
> justification → speaks the verdict back in the candidate's language.

### What the graders actually score (brief §"How we evaluate"), and where this plan spends time

| Rank | Criterion | Where it is earned in this plan |
|---|---|---|
| 1 | **It runs** from a clean checkout | Phase 0 + §12 run contract; verified by fresh-clone rehearsal in Phase 8 |
| 2 | **Agentic design** — a real branch | `policy.py` + conditional edge + `interrupt()`/resume (Phase 6) |
| 3 | **Coding-agent fluency** | `CLAUDE.md` (already strong) + `data/docs/PROMPT_LOG.md` written *during* the build |
| 4 | **Quality gate** — sane retrieval, regression signal | `scripts/eval_samples.py` + `tests/golden.json` (Phases 1, 2, 7) |
| 5 | **Voice & communication** | STT/TTS measurement gates (Phases 1, 4) + README |

Explicitly **not** built (brief: "we do not care"): auth, DB, visual polish, deployment,
streaming/barge-in/VAD, vector DB, CI, multi-turn conversation beyond the single follow-up.

### Time budget

~4h total. **At 3h, freeze features and verify** — announce crossing that line.
Phase 5 (browser mic) is hard-capped at 45 min with a documented fallback.

---

## 2. Architecture

### 2.1 One-screen view

```
Browser (static/index.html, MediaRecorder)
   │  GET  /api/question         → question text + spoken question (base64 mp3)
   │  POST /api/screening/start  ── audio blob ─────────────┐
   │  POST /api/screening/{tid}/resume ── audio blob ───────┤
   ▼                                                        ▼
FastAPI (app/main.py)  ── one process, sync `def` handlers on the threadpool
   │
   ├─ providers/stt.py      transcribe(bytes, filename, mime) -> Transcript
   │                        (audio never enters the graph; the graph resumes with TEXT)
   │
   ├─ graph.invoke(...) / graph.invoke(Command(resume=text), ...)   [thread_id keyed]
   │        │
   │        │   LangGraph StateGraph + InMemorySaver
   │        │
   │        ├── summarize ──► retrieve ──► analyze ──┬── (probe) ─► compose_probe ─► probe*
   │        │        ▲                               │                                 │
   │        │        └───────────────────────────────┼─────────────────────────────────┘
   │        │                                        └── (score) ─► score ─► END
   │        │   * probe suspends via interrupt(); resumes on the NEXT HTTP request
   │        │
   │        ├─ judge.py       LLM stages (summarize / coverage / followup / score) — no langgraph
   │        ├─ retrieval.py   NumPy cosine over 10 title-augmented chunks — no LLM
   │        └─ policy.py      decide() — pure Python, no LLM, no I/O, no imports from providers
   │
   ├─ providers/tts.py      speak(text, language) -> mp3 bytes
   └─ obs.py                one JSON line per event on the `screener` logger → stdout
```

### 2.2 Fixed technology decisions and why the alternatives were rejected

| Area | Choice | Why | Rejected alternative + the condition that would flip it |
|---|---|---|---|
| Runtime | Python 3.11 via `uv` | `uv` installs the interpreter, erasing "wrong Python" as a reviewer failure mode; same commands on macOS and Windows | System venv + `requirements.txt` — flip if the reviewer's org forbids `uv` |
| Backend | FastAPI, one process | One `uv run uvicorn` command, no build step | Streamlit/Gradio — flip if the UI needed to be more than a mic button; they hide the HTTP boundary that `interrupt()`/resume depends on |
| UI | one `static/index.html` + `MediaRecorder` | No npm, no bundler, no `node_modules` on a clean checkout | React/Vite — flip only if UI were graded; it is explicitly not |
| Orchestration | LangGraph `StateGraph`, fenced to 5 primitives | The loop is genuinely **cyclic** and **suspends for human input mid-run** (§4.4) | Hand-rolled `while` loop + session dict — flip if there were no cycle and no suspend; then the framework buys nothing |
| Session state | `InMemorySaver` keyed by `thread_id` | Resume in place across two HTTP requests instead of rebuilding state | Redis/SQLite checkpointer — flip the moment there is a second process or a restart-survival requirement (**named limit**, see §14) |
| Retrieval | in-memory embeddings + NumPy cosine | 6 tiny docs → 10 chunks. A vector DB is operational weight with no recall benefit at this size | Chroma/pgvector/FAISS — flip above ~10⁴ chunks or when the index must outlive the process |
| Providers | one OpenAI key: STT + LLM + TTS + embeddings | One credential = one setup step in the README | Groq (fast Whisper) / Deepgram / ElevenLabs — flip if the Phase-1 or Phase-4 measurement gates fail (§6.3, §8.3) |
| STT model | `gpt-transcribe` (measured, fallback ladder) | Highest-accuracy current model; supports `prompt`, `keywords`, and returns `languages[]` — a **list**, which is the only API-native way to express code-mixing | local Faster-Whisper — rejected outright: ~1.5 GB first-run download + CTranslate2 install variance destroys the three-command clean checkout, the strongest asset in this submission |
| LLM | `gpt-5.6-terra` via `responses.parse(text_format=…)` | Structured outputs against Pydantic; cheapest of the 5.6 family at $2/M in | JSON mode + regex over prose — rejected: unparseable failures are silent |
| Embeddings | `text-embedding-3-small`, default dims | Sufficient for 10 chunks; query is always **English** (§7.2) so multilingual embedding quality is a non-issue by construction | `-3-large` — flip only if the Phase-1 distractor margin is thin |
| Logging | stdlib `logging`, one JSON object per line | Reconstructable with `grep` + `jq` and nothing else; zero deps | structlog / loguru / OpenTelemetry — banned by `CLAUDE.md` §9 and unjustifiable for one process |

### 2.3 Enforceable module boundaries (grep-checkable at Definition of Done)

- **No provider SDK imported outside `app/providers/`** → swapping STT vendors touches one file.
- **No model-ID string literal outside `app/config.py`.**
- **`app/policy.py`**: no LLM call, no I/O, no import from `providers/` or `langgraph`.
- **`app/judge.py`**: zero `langgraph` imports. Plain args in, Pydantic out. *This is the
  load-bearing constraint* — it keeps the graded decision logic testable without a graph runtime
  and keeps the framework reversible.
- **`app/graph.py`**: state, nodes, edges, checkpointer. No prompts, no thresholds, no provider
  calls. **No node body over ~10 lines**; timing/transition logs come only from `@traced_node`.
- New dependency requires a one-line justification. Default: **no**. `uv add`, never `pip install`,
  never hand-edit `pyproject.toml`; commit `uv.lock` in the same commit.

---

## 3. Folder structure and responsibilities

```
app/
  __init__.py      # EMPTY — so `import app.policy` works with no API key (offline tests)
  main.py          # FastAPI routes; transcribe → invoke/resume graph → assemble → speak
  config.py        # every model ID + tunable as a named constant; reads env, never raises
  schemas.py       # Pydantic contracts (§5) + Evaluation.from_state() + Evaluation.spoken_text()
  obs.py           # ~40 lines: setup(), log_event(), @traced_node, thread-id contextvar
  providers/
    __init__.py
    stt.py         # transcribe(...) -> Transcript ; vocab hint ; language derivation
    tts.py         # speak(text, language) -> bytes ; per-language voice instructions
    llm.py         # structured JSON calls only: complete(system, user, model_cls) ; embed(texts)
  retrieval.py     # chunk data/corpus/ at competency boundaries ; retrieve(claims_summary_en) -> [Chunk]
  judge.py         # 4 LLM stages + 4 prompt constants. NO langgraph import.
  policy.py        # decide(coverage, followups_used) -> Decision. Pure. NO langgraph import.
  graph.py         # StateGraph: state, nodes, edges, checkpointer. Nothing else.
static/index.html  # mic → upload → transcript on screen → audio reply. ~100 lines.
scripts/eval_samples.py
tests/test_policy.py
tests/golden.json
data/corpus/       # 6 reference docs — provided input, not built
data/audio/        # 3 sample answers — provided input, not built
data/docs/PROMPT_LOG.md
artifacts/         # gitignored; eval_<n>.json, run.log, summary.md
PLAN.md  README.md  CLAUDE.md  pyproject.toml  uv.lock  .env.example
Dockerfile  .dockerignore      # Phase 9 only
```

### 3.1 [DEVIATION] `corpus/`, `audio/`, `docs/` moved under `data/`

`CLAUDE.md` and the rest of this plan originally referenced `corpus/`, `audio/`, and
`docs/PROMPT_LOG.md` at the repo root. User override for a cleaner top level: input assets
(`data/corpus/`, `data/audio/` — provided, never built) and generated documentation
(`data/docs/PROMPT_LOG.md`) now live under one `data/` folder, separating "things the task
handed us" and "things we wrote about the build" from the actual application code (`app/`,
`static/`, `scripts/`, `tests/`). Every path reference below is updated accordingly; no code
existed yet that touched the old paths, so this is a pure documentation/layout change with zero
runtime impact.

### 3.2 [DEVIATION] post-hoc restructure into layered subpackages (after Phase 8)

The flat `app/` layout above (§3) shipped through Phase 8 and passed the quality gate. A
follow-up architecture pass reorganized it into subpackages once the flat layout started
accumulating real SRP/coupling issues (mixed-concern `main.py`, one `providers/llm.py` wrapping
two unrelated SDKs, hardcoded prompts contradicting `CLAUDE.md`'s own "Prompt Rules"). New layout:

```
prompts/                        # summarize.txt, analyze_coverage.txt, write_followup.txt, score.txt
app/
  main.py                       # FastAPI() + static mount + logging/obs bootstrap + include_router — nothing else
  api/{health.py, routes.py}    # split out of main.py
  core/{config.py, obs.py, question.py}
  graph/{build.py, judge.py, policy.py}   # build.py = old graph.py, renamed to avoid the graph/graph.py collision
  rag/retrieval.py
  providers/{llm.py, embeddings.py, stt.py, tts.py}   # embeddings.py split out of llm.py (fastembed vs. Groq are different integrations)
  models/schemas.py
```

Every prompt moved from a Python string constant into `prompts/*.txt`, loaded via the same
`lru_cache` + `Path.read_text()` idiom `judge.py` already used for `load_question()`/the rubric
levels block — closing an until-then-undocumented gap against `CLAUDE.md`'s prompt rule. This was
a pure move: every function body, prompt string, threshold, and API contract is unchanged;
verified via `scripts/eval_samples.py` (byte-identical golden scores) and `uv run pytest`.
Top-level `agents/`/`services/`/`rag/` siblings (`CLAUDE.md`'s literal wording) were considered and
rejected in favor of one importable `app/` root with internal subpackages — this is a single
deployable FastAPI service, not a multi-package monorepo.

---

## 4. Execution flow

### 4.1 Turn 1 (always)

1. `GET /api/question` → question text from `QUESTION.md` + TTS mp3 (cached in-process after first call).
2. Candidate records → `POST /api/screening/start` (multipart `audio`).
3. Handler: mint `thread_id = uuid4().hex` → set the obs contextvar → `log_event("session.start", …)`.
4. `providers/stt.transcribe(bytes, filename, mime)` → `Transcript(text, language)`.
   **Transcription happens in the route handler, never in a node** — the graph resumes with *text*,
   so all audio handling stays inside `providers/`.
5. Guard: empty/whitespace transcript → `422` with a friendly message; **no graph run started**.
6. `graph.invoke({"thread_id":…, "language":…, "turns":[text], "followups_used":0}, config)`.
7. Branch on the result:
   - `"__interrupt__" in result` → suspended. Read the payload
     (`result["__interrupt__"][0].value`) for the follow-up question; read the partial state via
     `graph.get_state(config).values` for `coverage` / `retrieved` / `decision`.
     `score` is `None`. **Spoken text = the follow-up question.**
   - otherwise → complete. `score` is `1..5`. **Spoken text = score line + justification.**
8. `providers/tts.speak(spoken_text, language)` → mp3 → base64 into the JSON response.

### 4.2 Turn 2 (only if the graph suspended)

9. Candidate records the follow-up answer → `POST /api/screening/{thread_id}/resume`.
10. Same STT path → `graph.invoke(Command(resume=text), config)` on the **same `thread_id`**.
11. The `probe` node returns; `operator.add` merges the new turn; the graph cycles
    `probe → summarize` and re-analyzes the **merged** transcript.
12. `policy.decide` now sees `followups_used == 1` → hard cap → `score`. Guaranteed termination.
13. Assemble `Evaluation`, speak, `log_event("session.end", …)`.

### 4.3 Batch flow (`scripts/eval_samples.py`) — no server, no human

Same graph, same interrupt. When a run suspends, record `followup_question` in the artifact, then
`graph.invoke(Command(resume=NO_FOLLOWUP_SENTINEL), config)` so every sample reaches a score.
**The interrupt is never disabled for batch runs** — that would test a different graph from the one
demoed, and the artifact showing *both* the probe that fired and the final verdict is precisely the
evidence that the branch is live.

### 4.4 Why a graph runtime at all — the three sentences to say out loud

- **`probe → summarize` is a real cycle.** "Re-analyze the merged transcript" is an *edge*, not
  re-entrant handler code.
- **`interrupt()` models the suspend correctly.** The turn is a paused computation awaiting human
  input, not a session dict rebuilt on the next request.
- **The `operator.add` reducer on `turns` merges the transcript.** One reducer, doing real work.

…and the fence is the argument: `StateGraph`, `TypedDict` state, `add_conditional_edges`,
`interrupt()`/`Command(resume=…)`, one checkpointer. **No** subgraphs, tool-calling agents,
multi-agent handoff, streaming modes, a second reducer, or any `langchain` chain/retriever/vector
store. Unused framework surface reads as résumé-driven architecture; **minimal usage is the argument.**

> **The graph routes; `policy.py` decides.**

---

## 5. LangGraph workflow

### 5.1 State model

```python
class ScreeningState(TypedDict):
    thread_id: str                              # carried in state so nodes can log (§10)
    language: str                               # "ar" | "en" | "ar-en" — set once by STT
    turns: Annotated[list[str], operator.add]   # the ONLY reducer: turns accumulate
    claims_summary_en: str
    retrieved: list[Chunk]
    coverage: list[CoverageItem]
    decision: Decision | None
    followups_used: int
    score: int | None
    justification: str | None
    followup_question: str | None
```

`language` is written once from STT and **read** by every node and by TTS. Never re-inferred; a
probe turn never resets it — the follow-up is asked in the language of the first answer.

### 5.2 Nodes and edges

| Node | Reads | Calls | Writes |
|---|---|---|---|
| `summarize` | `turns`, `followup_question` | `judge.summarize()` | `claims_summary_en` |
| `retrieve` | `claims_summary_en` | `retrieval.retrieve()` | `retrieved` |
| `analyze` | `turns`, `retrieved`, `language`, `followups_used` | `judge.analyze_coverage()` **then** `policy.decide()` | `coverage`, `decision` |
| `compose_probe` | `decision.target`, `language` | `judge.write_followup()` | `followup_question` |
| `probe` | `followup_question` | `interrupt()` | `turns` (+1), `followups_used` (+1) |
| `score` | `turns`, `coverage`, `retrieved`, `language` | `judge.score()` | `score`, `justification` |

```
START → summarize → retrieve → analyze ─┬─(probe)→ compose_probe → probe → summarize   (cycle)
                                        └─(score)→ score → END
```

`policy.decide()` is called **inside `analyze_node`** and its result stored in state; the router
only reads it. Conditional-edge functions cannot write state, and `compose_probe` needs
`decision.target` — so deciding in the router would lose it.

### 5.3 [DEVIATION] `probe` is split into `compose_probe` + `probe`

`CLAUDE.md` §4 shows one `probe_node` that calls `judge.write_followup()` and *then* `interrupt()`.
The official LangGraph docs state: *"the node restarts from the beginning of the node where the
interrupt was called"* — all code before `interrupt()` runs again on resume.

Consequence of the literal version: on resume, `write_followup()` fires a **second LLM call**, and
because the resume value is matched positionally the *new* question text is what lands in state.
The artifact and the log would then record question **B** while the candidate actually answered
question **A**. That is a silent correctness bug in the exact feature graded second.

**Fix (minimal):** the LLM call moves to its own node `compose_probe`, whose delta is checkpointed
*before* the suspend. `probe` becomes side-effect-free and idempotent under replay:

```python
def probe_node(state):
    answer = interrupt({"question": state["followup_question"],
                        "target": state["decision"].target})
    return {"turns": [answer], "followups_used": state["followups_used"] + 1}
```

Cost: one extra 2-line node and one extra edge. `graph.py` stays the size `CLAUDE.md` specifies,
every node stays a thin adapter, and the fence is untouched. **Interview line:** *"The node that
suspends does nothing but suspend, because LangGraph replays it from the top on resume."*

### 5.4 Diagram

Exported with `graph.get_graph().draw_mermaid()` and pasted into the README — generated from the
compiled graph, so it cannot drift from the code. A hand-drawn diagram is a Definition-of-Done
failure.

---

## 6. RAG pipeline

### 6.1 Corpus and the planted distractors

`QUESTION.md` covers a slow endpoint, async/await, and EF Core. Therefore:

- **Relevant:** `00_scoring_rubric.md`, `01_async_concurrency.md`, `03_data_access_ef_core.md`, `05_performance_memory.md`
- **Distractors — must never drive scoring:** `02_dependency_injection.md`, `04_api_design_security.md`

The rubric itself says *"Only score competencies the question actually touches."* Penalising a
candidate for omitting JWT refresh tokens or service lifetimes is **wrong**, and is the exact
failure this corpus is built to catch.

### 6.2 Chunking → exactly 10 chunks

| Source | Chunks | `competency` |
|---|---|---|
| `00_scoring_rubric.md` split at `## Competency:` | 5 | the heading text verbatim |
| `01_async_concurrency.md` | 1 | `Async & concurrency` |
| `02_dependency_injection.md` | 1 | `Dependency injection` *(distractor — no rubric competency)* |
| `03_data_access_ef_core.md` | 1 | `Data access (EF Core)` |
| `04_api_design_security.md` | 1 | `API design & security` *(distractor)* |
| `05_performance_memory.md` | 1 | `Caching & performance` |

Two deliberate choices:

- **The `## Levels` block of the rubric is NOT a chunk.** The 1–5 scale is universal, not a
  competency; as a chunk it would compete for a top-k slot against real competencies. It is loaded
  verbatim and **always injected into the scoring prompt**. *Defence: the level scale is not
  something you retrieve — you always need it.*
- **Reference notes are labelled with the rubric competency they support** via a small hand-authored
  map in `retrieval.py`. Over a fixed 6-file corpus this is *data, not logic*. `# TODO(tradeoff):`
  derive from headings if the corpus grows.
- Chunk text is **title-augmented** (`"{competency}\n\n{body}"`) before embedding — standard, free,
  and it widens the distractor margin.

### 6.3 Query, ranking, and the measurement that sets `k`

- **Embed and retrieve on `claims_summary_en`, never the raw transcript.** An Arabic transcript
  against an English corpus is a real embedding-quality risk; stage 2 emits this field anyway, so
  this removes the cross-lingual problem at **zero cost**. Say that sentence in the walkthrough.
- Cosine similarity, NumPy, one matrix. Index built once at import (10 embeddings, one API call).
- `retrieve()` returns `list[Chunk]` **with scores**, surfaced in the API response so it is visible
  on screen during the demo.
- `competencies_in_scope = distinct competency across retrieved` — this is what the coverage stage
  is allowed to judge.

**`RETRIEVAL_K` is chosen from data, not guessed.** Phase 1 prints the full 10-row ranked table for
a hand-written stand-in summary and for each real sample summary, and records **the margin between
the worst relevant chunk and the best distractor** in `PROMPT_LOG.md`. Starting point `k = 5`.
- If the margin is comfortable → ship `k = 5`.
- If a distractor creeps into top-5 → prefer lowering `k` to 4 over adding a magic threshold; a
  relative floor (`score ≥ 0.6 × top_score`) is the second resort and must be justified by the
  measured numbers.
- Flat single index over all 10 chunks — **not** "retrieve over rubric chunks only." The latter
  makes distractor exclusion true *by construction*, which is not evidence of retrieval quality and
  an interviewer would rightly call it out.

---

## 7. STT pipeline

### 7.1 Contract

```python
def transcribe(audio: bytes, filename: str, mime: str) -> Transcript   # .text, .language
```

`app/config.py` holds the model ID and vocabulary hint. Provider SDK appears nowhere else.

### 7.2 Rules (non-negotiable)

- **Never pass a language parameter to STT.** Auto-detect. Pinning `language="ar"` mangles
  "N+1 queries" and "thread pool" into transliterated noise.
- **Vocabulary hint via `prompt=`** (verified present in the installed SDK 2.52.0 signature, along
  with `keywords` and `languages`):
  `"N+1 queries, thread pool, AsNoTracking, EF Core, IQueryable, async/await, indexes, profiler, APM, Redis, ConfigureAwait, CancellationToken"`
- **File identity matters.** The SDK infers format from the filename, so upload as a tuple
  `(filename, bytes, mime)` using the browser's real MIME type. Chrome sends
  `audio/webm;codecs=opus`; **Safari sends `audio/mp4`** — both are accepted by the API, and the
  reviewers are on macOS (the `.DS_Store` files say so). Getting this wrong is a classic silent 400.

### 7.3 Language derivation → `"ar" | "en" | "ar-en"`

`gpt-transcribe` returns `languages: list[{code}]` — a **list**, verified in
`openai.types.audio.Transcription`. That is the only API-native signal for code-mixing, and it is
the primary input. It is combined with a deterministic Arabic-script ratio over the transcript:

| Signal | → `language` |
|---|---|
| Arabic-char ratio ≥ 0.60 and some Latin technical tokens present | `ar-en` |
| Arabic-char ratio ≥ 0.60, no meaningful Latin | `ar` |
| Arabic-char ratio < 0.10 | `en` |
| otherwise | `ar-en` |

~8 lines, no LLM, unit-testable, and it survives an empty `languages[]`. **Defence:** *"No STT API
returns a code-mixed label; my schema needs one, so I compute it from the script distribution."*

### 7.4 Measurement gate (Phase 1) — this decides the provider

Transcribe all three samples and check **one thing**: do English technical terms survive as Latin
script, or come back transliterated ("ثريد بول" for "thread pool")? Record the answer in
`PROMPT_LOG.md`. Fallback ladder, in order, stopping at the first that passes:

1. `gpt-transcribe` + `prompt=` vocabulary hint ← default
2. `gpt-transcribe` + `keywords=[…]` (a documented alternative hinting channel)
3. `gpt-4o-transcribe` + `prompt=` (well-trodden, stable `json` response)
4. `whisper-1` + `response_format="verbose_json"` (legacy, but a guaranteed `language` string)
5. Only then: swap `providers/stt.py` to Deepgram

Steps 1–4 are a one-constant change in `config.py`, which is the point of the boundary.

---

## 8. TTS pipeline

### 8.1 Contract

```python
def speak(text: str, language: str) -> bytes   # mp3
```

Model `gpt-4o-mini-tts-2025-12-15` (latest snapshot; ~35% lower multilingual WER than the previous
generation and the only family supporting `instructions`). Returned to the browser as **base64 in
the JSON response** — no server-side audio store, no second round trip, ~80 KB for three sentences.

### 8.2 Steering Arabic output

The `instructions` parameter (supported only on `gpt-4o-mini-tts*`, not `tts-1*`) is used per
language, e.g. for `ar` / `ar-en`:

> *"Speak natural Egyptian Arabic at a calm, professional interviewer pace. Pronounce English
> technical terms in English, not transliterated Arabic."*

Voice candidates to A/B in Phase 4: `marin` and `cedar` (documented as the highest quality),
falling back to `alloy`.

### 8.3 Measurement gate (Phase 4) — **TTS is a gate, not a checkbox**

Generate the Arabic verdict to a file and **listen to it** before leaving Phase 4. It is the last
thing the reviewers hear in a required recording. If OpenAI renders Arabic with a distracting
anglophone accent that `instructions` + a voice swap cannot fix, swap `providers/tts.py` to
ElevenLabs and note which shipped and why. *(Requires the human — see §17.)*

### 8.4 What gets spoken

| Turn type | Spoken text |
|---|---|
| Probing turn (`score is None`) | the `followup_question`, in the candidate's language |
| Final turn | `SCORE_LINE[language].format(score) + " " + justification` |

`SCORE_LINE` is two constants in `config.py` (`"التقييم {n} من 5."` / `"Score {n} out of 5."`).
The brief requires the score to be *spoken*; a fixed label is a constant, not a translation, so
§"never write English and translate" is respected. Total spoken output stays under ~3 sentences —
**this is voice, not a report.**

---

## 9. Decision pipeline

### 9.1 The staging rule

**Banned:** `if len(transcript) < N`, keyword matching, word counts, STT-confidence thresholds. Any
length- or keyword-based branch fails this task outright.

**Equally banned:** one LLM call producing the coverage map, the score, and the follow-up decision
together — that contaminates the score with a question not yet asked and leaves the decision
untestable.

| # | Stage | Node | Logic in | LLM? |
|---|---|---|---|---|
| 1 | Transcribe | *(route handler, pre-graph)* | `providers/stt.py` | — |
| 2 | Summarize → `claims_summary_en` | `summarize` | `judge.py` | yes |
| 3 | Retrieve on the summary | `retrieve` | `retrieval.py` | no |
| 4 | Coverage map, then decide | `analyze` | `judge.py` **+** `policy.py` | coverage only |
| 5 | **Route: probe or score** | **conditional edge** | reads `state["decision"]` | **no** |
| 6a | Write follow-up, then suspend | `compose_probe` → `probe` | `judge.py` | yes (in `compose_probe`) |
| 6b | Score + justification | `score` | `judge.py` | yes |

> **The LLM supplies judgment; code supplies policy.** Stage 4's coverage answers *what did they
> cover?* — open-ended, model work. `policy.decide()` answers *given that coverage, do we probe?* —
> a rule you can read, test, and defend.

This sentence goes in the README so the deterministic policy reads as deliberate design, not a
hardcoded shortcut.

### 9.2 Coverage contract

For each competency in scope: `covered | partial | missing`, plus a **verbatim quote from the
transcript as evidence** (`null` only when `missing`). Evidence is mandatory — it is what stops the
model asserting coverage it cannot support.

`judge.analyze_coverage()` post-validates before returning (in `judge.py`, never in the graph):
- competency set returned **must equal** the set requested → extras dropped, omissions added as
  `missing`/`None`. Policy never sees a malformed map.
- evidence is checked for verbatim presence in the merged transcript. **Mismatches are counted and
  logged as a number, never mutated** — silently downgrading `covered`→`partial` on a
  normalisation artefact would systematically punish Arabic answers. `# TODO(tradeoff):` promote to
  a downgrade once Phase 7 data shows evidence is reliably verbatim. Listed as a known limitation
  in the README.

### 9.3 `policy.decide()` — implemented verbatim from `CLAUDE.md` §4

```python
def decide(coverage: list[CoverageItem], followups_used: int) -> Decision:
    if followups_used >= 1:                      # hard cap, in code
        return Decision(probe=False, reason="follow-up budget spent")
    gaps = [c for c in coverage if c.status in ("partial", "missing")]
    if not gaps:
        return Decision(probe=False, reason="all competencies covered — score now")
    if len([c for c in coverage if c.status == "missing"]) >= 2:
        return Decision(probe=False, reason="weak across the board; one probe cannot move it")
    if len(gaps) == 1:
        return Decision(probe=True, target=gaps[0].competency,
                        reason=f"single gap in {gaps[0].competency}; a probe may move the score")
    return Decision(probe=False, reason="diffuse gaps; score now")
```

**Defence:** probe only where a probe could plausibly change the outcome. All covered → they earned
it. Two or more missing → one question cannot rescue the answer, and interrogating a struggling
candidate is bad product behaviour. The cap lives **in code** because prompts drift, and an
infinite probe loop during the live demo is unrecoverable.

**Noted disagreement (one sentence, then built as specified):** "≥2 partials, 0 missing" is a
decent answer with two soft spots where a probe arguably *could* move a 3 to a 4, and the current
rule scores it immediately — but the rule as written is coherent, and tuning a policy to make a
demo fire more often is exactly the smell this file exists to avoid. See §16-R1 for the
consequence and its mitigation.

`reason` is always populated on **both** branches and logged on **every** run.

---

## 10. Prompt architecture

Four prompts, module-level constants at the top of `judge.py` (nowhere else). Every call returns a
Pydantic model via `client.responses.parse(text_format=…)` — **no regex over prose, no free-text
parsing.** Separate models per stage so each is independently testable.

| Stage | Input | Output model | Key instructions |
|---|---|---|---|
| `summarize` | interview question, `turns`, optional `followup_question` | `ClaimsSummary(claims_summary_en)` | English only, ≤80 words, preserve technical terms verbatim, no judgement, consolidate multiple turns into one summary |
| `analyze_coverage` | question, `turns`, retrieved chunks, **explicit competency list**, `language` | `CoverageReport(items: list[CoverageItem])` | judge **only** the listed competencies; **never** penalise for topics outside the list; evidence must be a verbatim transcript quote; `null` evidence only when `missing`; **mention ≠ coverage (see below)** |
| `write_followup` | target competency + its rubric text, `language`, prior summary | `Followup(question)` | exactly one question, ≤25 words, in the candidate's language, technical terms in English, probes the gap without giving the answer away |
| `score` | question, `turns`, `coverage`, retrieved rubric chunks, **the `## Levels` block**, `language` | `Verdict(score: int, justification: str)` | echo the rubric's level wording; **one line**; **generate directly in the candidate's language — never write English and translate**; keep technical terms in English ("الـ thread pool", "الـ N+1 queries") |

Three prompt-level rules that carry real weight:

- The **coverage** prompt receives the competency list explicitly and is told the scope is closed.
  This is the second line of defence behind retrieval against the planted distractors.
- **Mention ≠ coverage** (added during Phase 2 golden-labelling, generic across all competencies,
  not just EF Core — the exact rule found while hand-labelling `tests/golden.json` and confirmed
  by re-checking all 3 samples against it). The `analyze_coverage` prompt must instruct:
  - A competency is `covered`/`partial` only when the transcript demonstrates actual
    understanding, reasoning, or correct application of it — never for naming a technology or
    concept alone. A bare mention with no explanation of why/how it applies does not clear the
    bar for `partial`.
  - **An explicit statement of uncertainty or lack of knowledge about a competency overrides any
    mention of it** — e.g. "I don't know EF Core," "I'm not familiar with X," "I've never used
    X," "I'm not sure how this affects things under load." Mark that competency `missing`, not
    `partial`, even though the concept was named. Applies uniformly to all 5 competencies, not
    only the one it was first noticed on.
  - Evidence quotes must support the reasoning/application itself, not merely the sentence a
    keyword happens to appear in. If the only support for `partial`/`covered` is a bare keyword
    mention with no demonstrated reasoning, the status is `missing`.
- The **score** prompt is told the answer may include a follow-up turn and to judge the merged
  answer as a whole.

Translation is never used: stiff MSA sounds nothing like an Egyptian engineer, and that is what the
reviewers hear.

---

## 11. API design

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/health` | — | `{"status":"ok","openai_key_present":bool}` |
| `GET` | `/` | — | `static/index.html` |
| `GET` | `/api/question` | — | `{"question":str,"audio_b64":str}` (TTS cached in-process) |
| `POST` | `/api/screening/start` | multipart `audio` | `TurnResponse` |
| `POST` | `/api/screening/{thread_id}/resume` | multipart `audio` | `TurnResponse` |
| `GET` | `/api/graph.mmd` | — | mermaid text from `draw_mermaid()` *(dev convenience for the README; 3 lines)* |

```python
class TurnResponse(BaseModel):
    status: Literal["awaiting_followup", "complete"]
    thread_id: str
    transcript: str                 # shown on screen — a brief requirement
    evaluation: Evaluation          # exactly the §5 shape, unmodified
    spoken_text: str
    audio_b64: str
```

**[DEVIATION]** `CLAUDE.md` §5 calls `Evaluation` "the API response". It is **nested** inside
`TurnResponse` instead, because the transport needs three things `Evaluation` must not absorb: the
raw transcript (must be on screen), the audio, and the `thread_id` that must round-trip for resume.
`Evaluation`'s own field list is untouched.

**Handlers are declared `def`, not `async def`** — every call inside is blocking I/O (OpenAI SDK,
graph invoke), so FastAPI runs them on the threadpool and the event loop is never stalled. One line
of knowledge, and the failure it prevents (a frozen server on the second concurrent request) is
invisible until the demo.

---

## 12. Error handling

| Failure | Handling | Rationale |
|---|---|---|
| `OPENAI_API_KEY` missing | `config.py` reads it, **never raises**. Providers raise a clear `RuntimeError` on first use. `/health` reports presence. | `tests/test_policy.py` must import and pass with **no key** |
| Empty / whitespace transcript | `422` + friendly message, **before** any graph run | Don't burn LLM calls or leave a half-run thread |
| Unsupported / oversized audio (>25 MB) | `413`/`415` with the accepted list | Fast, legible failure |
| Structured-output parse failure or refusal | one inline retry in `providers/llm.py`, then raise | Not a retry *framework* (§banned) — one `for` loop |
| Any node exception | `@traced_node` logs `node.error` with exception **type** (never the message body — messages can carry transcript fragments) and re-raises → `500` | Keeps the log leak-proof |
| Resume on an unknown/expired `thread_id` | `404` "session not found — start a new screening (state is in-process)" | Turns the `InMemorySaver` limit into a legible message instead of a stack trace |
| Resume when the graph is not suspended | `409` | Guards double-submit from the browser |
| TTS failure | return the turn with `audio_b64: ""` and a `tts_error` flag; UI shows text | **Never lose a completed evaluation to a speech failure** |
| Windows console + Arabic | `obs.setup()` does `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` | Logs use `ensure_ascii=False`; on a cp1252 console the first Arabic follow-up question would otherwise raise `UnicodeEncodeError` **inside the logger** |
| All file writes | explicit `encoding="utf-8"` | Same reason, for artifacts and transcripts |

---

## 13. Logging & observability

`app/obs.py`, ~40 lines, **stdlib `logging` only**. One JSON object per line on the `screener`
logger, so a session is reconstructable with `grep` + `jq` and nothing else. Configured once in
`main.py` with `logging.basicConfig(level=INFO)` **before** uvicorn starts, keeping the `screener`
namespace separate from uvicorn's access log. Output to **stdout — no file handler**, so it behaves
identically under uvicorn and in the container.

```python
_BANNED = {"text", "transcript", "turns", "answer", "evidence", "audio",
           "api_key", "authorization", "prompt"}   # raises if a caller passes one
```

`thread_id` lives in `ScreeningState` so nodes log without extra plumbing; providers read it from a
`contextvars.ContextVar` set by the route handler / eval script, so `providers/*.py` need no new
arguments. `seq` is a per-thread counter — with interrupt/resume spanning two HTTP requests,
wall-clock timestamps alone do not guarantee readable ordering.

### Exactly these events. Nothing per-token, per-chunk, or inside a loop.

| Event | Fields beyond `thread_id`, `seq` |
|---|---|
| `session.start` | `language`, `source` (`"mic"` \| `"sample:Answer 1.mp3"`) |
| `node.exit` | `node`, `elapsed_ms` — from the decorator, covers every transition |
| `provider.call` | `provider`, `model`, `kind` (`stt`\|`llm`\|`tts`\|`embed`), `latency_ms` |
| `retrieval.done` | `doc_ids`, `scores`, `k` |
| `coverage.done` | `statuses` (`{competency: status}`), `evidence_unverified` (count) |
| `decision.made` | `probe`, `target`, `reason`, `followups_used` |
| `interrupt.raised` | `target`, `followup_question` |
| `interrupt.resumed` | `turn_index`, `answer_chars` |
| `session.end` | `score`, `followups_used`, `total_elapsed_ms` |
| `node.error` | `node`, `exc`, `elapsed_ms` |

**Never logged, at any level:** audio bytes · API keys / `Authorization` headers / `.env` values ·
**the transcript, in whole or in part** · evidence quotes · full prompt bodies. Log *derived* facts:
lengths, counts, statuses, competency names, `doc_id`s, scores. The candidate's words live in
`artifacts/eval_<n>.json`, not in the log stream. `followup_question` is the one permitted text
field, because it is agent-generated and it is what makes a probe reconstructable.

**Three events carry the walkthrough:** `decision.made` is the replayable audit trail for the
branch graded second; `retrieval.done` proves the distractor exclusion held on **live** runs, not
just inside the gate; `provider.call` latencies answer "where does the time go?" without a profiler.

```bash
uv run uvicorn app.main:app --port 8000 | tee app.log
grep '"thread_id": "abc123"' app.log | jq -s 'sort_by(.seq)'
```
must yield the full node path, the decision and its reason, and the final score.
`scripts/eval_samples.py` additionally writes `artifacts/run.log`, so every gate run leaves a
replayable trace next to its verdicts.

---

## 14. Quality gate

`scripts/eval_samples.py` runs all three files in `data/audio/`, writes `artifacts/eval_<n>.json` +
`artifacts/run.log` + a markdown table for the README, and **exits non-zero on any failed
assertion.** It imports the graph directly — **no running server required.**

### Assertions

- [ ] every sample returns `1 <= score <= 5` and a non-empty `justification`
- [ ] `02_dependency_injection` and `04_api_design_security` absent from **every** top-k
- [ ] the **spoken text's** language matches `state["language"]` — Arabic-char ratio `> 0.30` when
      `language in ("ar","ar-en")`, `< 0.05` when `"en"`. *(Asserting on the text handed to TTS is
      the honest machine-checkable form; the audio itself is verified once by ear in Phase 4.)*
- [ ] `decision.reason` non-empty on every run, **both** branches
- [ ] `score` is `None` on any probing turn's intermediate state
- [ ] `tests/test_policy.py` passes — all-covered, one-gap, two-missing, budget-spent. **Offline,
      no API key.**
- [ ] **score within ±1 of `tests/golden.json`**

### `tests/golden.json` — the regression baseline

Mandatory, hand-labelled, and **labelled before the scorer exists** (Phase 2, before Phase 3) so the
baseline cannot be contaminated by the pipeline's own output. Per sample: expected score, expected
competencies touched, and a one-line note. Without a hand-labelled baseline you cannot distinguish
improvement from drift — and that baseline is the answer to the brief's required *"how would you
know it got worse."*

### README regression note (2–3 sentences, verbatim intent)

> The metric is **score agreement with `tests/golden.json` within ±1 across all three samples**,
> plus **zero distractor documents in top-k**. On any change — a prompt edit, a model swap, a `k`
> change — I re-run `scripts/eval_samples.py` and diff the committed `artifacts/eval_<n>.json`
> against the previous run: a shifted score, a flipped coverage status, a changed `decision.reason`,
> or a distractor appearing in `retrieved` each localise the regression to one stage. The
> per-stage artifact is what makes that diff readable — a single end-to-end score would tell me
> something broke but not where.

### Named limits, stated by me rather than found by them

- **`InMemorySaver` is single-process.** A restart or a second worker loses in-flight sessions;
  the resume route returns a legible `404` rather than a stack trace. Swap to a
  SQLite/Redis checkpointer — a one-line change at `compile()` — the moment there is a second
  process. *A named limit reads as understanding; a found limit reads as an oversight.*
- Evidence quotes are verified, counted, and **not** enforced (§9.2).
- No auth, no rate limiting, no cost ceiling per session — out of scope per the brief.

---

## 15. Testing strategy

| Layer | What | Runs offline? |
|---|---|---|
| `tests/test_policy.py` | the four `decide()` branches + the hard cap; the file the interviewer will be shown | **yes, no key** |
| `tests/test_policy.py` (2 extra cases) | language derivation `ar` / `en` / `ar-en` from script ratio | **yes** |
| `scripts/eval_samples.py` | the integration test — the whole graph on real audio, with a live interrupt | no |
| `tests/golden.json` | data, not a test: the drift baseline | — |
| grep checks (Definition of Done) | `judge.py`/`policy.py` contain zero `langgraph` imports; no model-ID literal outside `config.py`; no provider SDK outside `providers/` | **yes** |

**No unit tests beyond these** (`CLAUDE.md` §9). The gate is the rest of the suite. Mocking the LLM
to test `judge.py` would test the mock.

---

## 16. Risks & mitigations

| # | Risk | Impact | Mitigation | Recovery if it fires |
|---|---|---|---|---|
| **R1** | **The probe branch never fires on any of the 3 samples** (§9.3: "exactly one gap" is narrow across 4–5 competencies) | **High** — Definition of Done requires the follow-up to fire on one input and correctly skip on another, both demonstrable | Measured at Phase 6 as soon as coverage output exists. Do **not** tune the policy to force it. | (a) Record **one extra short answer** that leaves a single gap (e.g. strong on EF Core + diagnostics, silent on caching), commit as `data/audio/Answer 4 (probe demo).mp3`, add to the gate; (b) fire it live from the mic in the recording; (c) `tests/test_policy.py` proves all four branches regardless |
| **R2** | STT transliterates English technical terms | High — poisons retrieval *and* coverage evidence | Measured in Phase 1 before any graph exists; 4-rung fallback ladder (§7.4), each rung a one-constant change | Rung 5: Deepgram in `providers/stt.py` only |
| **R3** | A distractor doc enters top-k | High — the exact failure the corpus is built to catch | Title-augmented chunks; `k` chosen from measured margins in Phase 1, not guessed | Lower `k` to 4; relative floor only if the numbers demand it |
| **R4** | `interrupt()` replay re-runs the follow-up LLM call | High — silent wrong-question-in-artifact bug | **Already designed out** via §5.3 `compose_probe` split | — |
| **R5** | Pydantic objects in `ScreeningState` fail checkpointer serialization | Medium — blocks the whole graph | Verified in Phase 3 the first time a graph runs (LangGraph's `JsonPlusSerializer` handles Pydantic; confirm, don't assume) | Store `model_dump()` dicts in state and re-hydrate in `judge.py` |
| **R6** | Browser mic capture eats the evening | **High** — the classic way this task dies | Phase 5 is **hard-capped at 45 min / ~100 lines** and comes *after* a working verdict | Ship `<input type="file" accept="audio/*">`. The brief sanctions it ("live from the mic **or your chosen input**"), it takes five minutes, the full loop is preserved. Record the swap under README → Assumptions |
| **R7** | Arabic TTS has a distracting anglophone accent | Medium — it is the last thing reviewers *hear* | `instructions` steering + `marin`/`cedar` A/B, verified by ear in Phase 4 | Swap `providers/tts.py` to ElevenLabs; note which shipped and why |
| **R8** | `UnicodeEncodeError` logging Arabic on a Windows console | Medium — crashes *inside* the logger, mid-demo | `sys.stdout.reconfigure(encoding="utf-8")` in `obs.setup()`; `encoding="utf-8"` on every file write | — |
| **R9** | Safari (macOS reviewers) sends `audio/mp4`, not `webm` | Medium — silent 400 on the reviewers' machine, works on mine | Pass the browser's real MIME + matching filename to the SDK (§7.2); README says Chrome, but Safari works | — |
| **R10** | Blocking OpenAI calls in `async def` handlers freeze the server | Medium — invisible until a concurrent request in the demo | Handlers declared `def` (§11) | — |
| **R11** | Time overrun | High | 3h feature freeze, announced. Phases 8–9 are the drop zone | Ship without Docker (§18) and without the extra probe sample; never ship without the gate |
| **R12** | Golden labels can't be assigned by ear by the implementer | Medium — undermines the baseline's independence | Labels are drafted from Phase-1 transcripts **before** the scorer exists, then **confirmed by the human by listening** (§17). README states plainly how they were assigned | If unconfirmed, say so in the README rather than implying a listen that did not happen |

---

## 17. Human-in-the-loop checkpoints (cannot be automated)

| When | What the human must do | Blocks |
|---|---|---|
| Before Phase 1 | Put a working `OPENAI_API_KEY` in `.env` | everything with a network call |
| Phase 2 | **Listen to all three samples** and confirm/adjust the drafted `tests/golden.json` scores | the honesty of the regression baseline |
| Phase 4 | **Listen to the Arabic verdict mp3.** Intelligible? Accent acceptable? | R7 / the recording |
| Phase 5 | Grant mic permission in Chrome and confirm the loop | Definition of Done |
| Phase 6 | If R1 fires: record one short answer that leaves exactly one gap | probe demonstrability |
| Phase 8 | Record the 2–3 min screen capture | a **required** submission artifact |

---

## 18. Docker strategy — Phase 9 only, after everything else passes natively

A wrapper over the same app. It does **not** replace the `uv` path; the README documents both,
`uv` first.

```dockerfile
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev          # cached layer; deps before source
COPY . .
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- `--frozen` so a stale lockfile fails the build **loudly** instead of silently re-resolving.
- Manifests copied before source so code edits don't bust the dependency layer.
- **Never bake `.env` or a key into the image.** `--env-file` at run time; `.dockerignore` lists
  `.env`, `.venv/`, `.git/`, `artifacts/`, `__pycache__/`, `data/audio/` is **kept** (the gate runs
  in-container).
- Bind `0.0.0.0` inside the container, still browse to `http://localhost:8000`.
- Done means `docker run` reaches the same working loop **and** the gate passes in-container.
- **If the image fights back for >20 min, ship the `uv` path and move Docker to "what's next."**
  A working native run beats a broken container.

Live walkthrough runs **natively with `--reload`**, never in Docker — the session ends with them
asking for a small change, and a two-second hot reload beats an image rebuild.

---

## 19. Development phases

Risk-ordered inside `CLAUDE.md` §2's build order. **Each phase must RUN before the next begins.
Commit per phase, with a message saying what now runs. Paste the actual output — "it should work"
is not a status.**

---

### Phase 0 — Foundations & skeleton *(~20 min)*

**Goal:** the server boots and serves a page from a clean checkout; nothing needs a key.

- **Create:** `.env.example`, `app/__init__.py` (empty), `app/config.py`, `app/schemas.py`,
  `app/obs.py`, `app/main.py` (`/health`, `/` → static), `app/providers/__init__.py`,
  `static/index.html` (stub), `data/docs/PROMPT_LOG.md`
- **Modify:** `.gitignore` (add `.pytest_cache/`, confirm `artifacts/`, `*.log`)
- **Expected output:** `curl localhost:8000/health` → `200 {"status":"ok",…}`
- **Validate:** `uv sync --frozen` from clean; `uv run uvicorn app.main:app --port 8000`;
  `uv run python -c "import app.policy"`-style import check works with **no** `.env`
- **Risks:** `config.py` raising on a missing key → breaks offline tests. **Recovery:** read env
  with `os.getenv`, never assert at import.
- **DoD:** `/health` 200 · page serves · `uv sync --frozen` clean · committed

---

### Phase 1 — Providers + the two measurements that could force a swap *(~45 min)*

**Goal:** de-risk STT and retrieval **before** any graph exists. Produces the transcripts Phase 2
needs.

- **Create:** `app/providers/stt.py`, `app/providers/tts.py`, `app/providers/llm.py`,
  `app/retrieval.py`
- **Modify:** `config.py` (model IDs, `RETRIEVAL_K`, vocab hint), `data/docs/PROMPT_LOG.md`
- **Throwaway (scratchpad, not committed):** a script that (a) transcribes all 3 samples and dumps
  text, (b) embeds a hand-written stand-in `claims_summary_en` plus each real one and prints the
  **full 10-row ranked table with scores**
- **Expected output:** three transcripts on stdout; two ranked tables; a recorded number for
  *"worst relevant chunk score − best distractor score."*
- **Validate:** § 7.4 transliteration check — do "thread pool", "N+1", "AsNoTracking" survive in
  Latin script? Record the verdict and the chosen rung in `PROMPT_LOG.md`. Set `RETRIEVAL_K` from
  the measured margin (§6.3).
- **Risks:** R2, R3. **Recovery:** the 4-rung STT ladder; lower `k`.
- **DoD:** transcripts readable and code-mixed · distractors below the cut with a recorded margin ·
  `PROMPT_LOG.md` updated · committed

---

### Phase 2 — `tests/golden.json`, labelled before the scorer exists *(~15 min, human-gated)*

**Goal:** an uncontaminated regression baseline.

- **Create:** `tests/golden.json` — per sample: `expected_score`, `expected_competencies`, `note`,
  `labelled_from` (`"transcript + listen"`)
- **Validate:** **the human listens to all three samples** and confirms or adjusts each score
- **Risks:** R12. **Recovery:** state plainly in the README how labels were assigned
- **DoD:** three hand-labelled entries committed **before** `judge.py` scores anything

---

### Phase 3 — Linear graph on files: `summarize → retrieve → analyze → score` *(~60 min)* ⟵ **the priority**

**Goal:** a real JSON verdict from `data/audio/Answer 1.mp3`. **No probe branch yet.** A straight
`analyze → score` path proves retrieval, coverage, scoring, and language handling before any
suspend/resume machinery exists.

- **Create:** `app/judge.py` (4 prompts + 4 stages), `app/policy.py`, `tests/test_policy.py`,
  `app/graph.py` (4 nodes, linear), `scripts/eval_samples.py` (v1: one sample, print JSON)
- **Modify:** `main.py` (`POST /api/screening/start`), `obs.py` (`@traced_node` wired)
- **Note:** `analyze_node` calls `policy.decide()` and stores the `Decision` **from Phase 3
  onward** — it is logged and returned, just not yet routed on. Phase 6 then adds *one conditional
  edge to working code.*
- **Expected output:** a JSON verdict on stdout with `claims_summary_en`, `retrieved` + scores,
  `coverage` + evidence, `decision.reason`, `score`, `justification` in the answer's language
- **Validate:** `uv run python scripts/eval_samples.py` prints it · `uv run pytest` passes offline
  with no key · grep: zero `langgraph` in `judge.py`/`policy.py` · **R5 check:** the checkpointer
  round-trips Pydantic state
- **Risks:** R5; coverage returning competencies that weren't requested. **Recovery:** dicts in
  state; the §9.2 post-validation
- **DoD:** real verdict printed and **pasted into the commit message** · `pytest` green offline ·
  distractors absent from `retrieved`

---

### Phase 4 — TTS gate *(~20 min, human-gated)*

**Goal:** the question and one Arabic verdict, spoken and **listened to**.

- **Create:** the `GET /api/question` route; write both mp3s to `artifacts/`
- **Modify:** `providers/tts.py` (per-language `instructions`, voice constant), `config.py`
- **Validate:** **listen.** Is the Arabic intelligible? Are "thread pool" / "N+1" pronounced in
  English? Is the accent acceptable?
- **Risks:** R7. **Recovery:** voice A/B (`marin`/`cedar`/`alloy`) → then ElevenLabs
- **DoD:** both files listened to · chosen voice + instructions recorded in `PROMPT_LOG.md`

---

### Phase 5 — Browser loop *(hard cap: 45 min / ~100 lines)*

**Goal:** mic → upload → transcript on screen → audio reply, in Chrome.

- **Modify:** `static/index.html` only
- **Expected output:** click record, speak, see the transcript, see coverage + retrieved scores,
  hear the reply
- **Validate:** end to end in Chrome on `http://localhost:8000` (**`getUserMedia` needs a secure
  context — browsers grant it on `localhost` but refuse on a LAN IP over plain HTTP, and the failure
  is silent**)
- **Risks:** R6, R9. **Recovery:** **at the 45-minute mark, stop and ship
  `<input type="file" accept="audio/*">`.** Record the swap under README → Assumptions rather than
  spending Phase 6's budget.
- **DoD:** one full loop completed in the browser, whichever input shipped

---

### Phase 6 — The branch *(~40 min)*

**Goal:** a real decision: conditional edge, `compose_probe` + `probe`, `interrupt()`/resume across
**two HTTP requests**, cycle back to `summarize`.

- **Create:** `compose_probe_node`, `probe_node`, `route_after_analysis`,
  `POST /api/screening/{thread_id}/resume`
- **Modify:** `graph.py` (one `add_conditional_edges` + two edges), `main.py`,
  `scripts/eval_samples.py` (batch interrupt handling per §4.3), `static/index.html` (second record
  button)
- **Expected output:** one input probes, another skips; `decision.reason` logged on both;
  `score is None` on the probing turn; the resumed run scores the merged transcript
- **Validate:** `curl` the two-request flow on one `thread_id` · `interrupt.raised` +
  `interrupt.resumed` in the log · re-run and confirm `compose_probe` fires **once**, not twice
  (§5.3)
- **Risks:** **R1** — measure immediately. **Recovery:** §16-R1 (a)/(b)/(c); never tune the policy
- **DoD:** fires on one input, correctly skips on another, both demonstrable · resume works across
  two requests · `graph.py` still has no node body over ~10 lines

---

### Phase 7 — Quality gate hardening *(~30 min)*

**Goal:** `scripts/eval_samples.py` enforces every §14 assertion and exits non-zero on failure.

- **Modify:** `scripts/eval_samples.py` (all 3 samples, `artifacts/eval_<n>.json`,
  `artifacts/run.log`, `artifacts/summary.md`), `tests/test_policy.py` (language-derivation cases)
- **Expected output:** exit 0, three artifacts, a markdown table ready to paste into the README
- **Validate:** deliberately break something (drop `k` to 10) and confirm the gate **fails loudly**
- **Risks:** golden ±1 fails. **Recovery:** if the pipeline is right and the label was wrong, fix
  the label **and say so in `PROMPT_LOG.md`** — never silently
- **DoD:** exits 0 · every §14 box ticked · artifacts committed

---

### Phase 8 — README, PROMPT_LOG, recording *(~35 min)*

- **Create/Modify:** `README.md` (every §20 row), `data/docs/PROMPT_LOG.md` (final pass — it was written
  *during* the build; reconstructed at the end, it looks reconstructed)
- **Validate:** **clone to a fresh directory and follow your own README literally**, not by reading
  it · every Definition-of-Done grep passes · the leak grep: search `app.log` for a distinctive
  phrase from a sample answer → **zero hits**
- **DoD:** fresh clone runs in <5 min · 2–3 min recording captured · §20 complete

---

### Phase 9 — Docker *(only if time remains; ≤20 min)*

- **Create:** `Dockerfile`, `.dockerignore`
- **Validate:** `docker build` → `docker run --env-file .env -p 8000:8000` reaches the same loop;
  the gate passes in-container; no key in the image
- **Recovery:** over 20 min → ship the `uv` path, move Docker to README → "what's next"

---

## 20. README checklist

| Section | Content | Source |
|---|---|---|
| Quickstart | the five commands of §21, `pip install uv` line **kept** | §21 |
| Architecture → diagram | mermaid, pasted from `graph.get_graph().draw_mermaid()` | §5.4 |
| Architecture → "why LangGraph, and where it stops" | the three bullets, **plus the scope fence as what I refused to hand it** | §4.4 |
| Architecture → one sentence | *"The graph routes; `policy.py` decides."* | §4.4 |
| How the follow-up decision works | prose + the `decide()` source **inline** + *"The LLM supplies judgment; code supplies policy."* | §9 |
| Known limit | `InMemorySaver` is single-process | §14 |
| Retrieval | competency chunking, `claims_summary_en` as the query, distractors excluded, **measured scores** | §6 |
| Quality gate | the results table from `artifacts/summary.md` | §14 |
| Regression note | metric + the artifact I would diff, 2–3 sentences | §14 |
| Voice | STT transliteration finding, TTS voice + why, Arabic-in/Arabic-out with English terms preserved | §7, §8 |
| Assumptions · what I cut · what's next | incl. any Phase-5 file-upload swap | §16, §19 |
| Rejected roads, one line each **with the condition that would flip it** | vector DB, Groq, local Whisper, Streamlit, a UI framework | §2.2 |

Two framings, each stated **once**:
- *A named limit reads as understanding; a found limit reads as an oversight.* — which is why the
  `InMemorySaver` boundary is in my text, not theirs.
- *Write it as scoping, not justification:* here is the narrow thing the framework does for me, and
  here is everything I refused to hand it.

---

## 21. Run contract

The native `uv` path is **primary** — what the reviewers run first and what is used in the live
session. Must work from a clean clone, identically on macOS and Windows:

```bash
pip install uv                               # if not already installed
uv sync                                      # installs Python 3.11 + deps from uv.lock
cp .env.example .env                         # add OPENAI_API_KEY
uv run uvicorn app.main:app --port 8000      # open http://localhost:8000
uv run python scripts/eval_samples.py        # quality gate
```

- **Keep the `pip install uv` line.** A `uv: command not found` on line one is a criterion-1 failure
  for the sake of one line.
- No venv activation, no shell-specific forks. Commit `uv.lock`; regenerate with `uv lock` only when
  deps change; never edit by hand.
- Commit `.env.example`. **Never** commit `.env` or a key.
- Bind `127.0.0.1`, browse to `http://localhost:8000`.
- Verify by cloning to a fresh directory and following the README **literally**.

---

## 22. Demo checklist (2–3 min recording + live walkthrough)

**Recording — one take, in this order:**
1. `uv run uvicorn app.main:app --reload --port 8000 | tee app.log`, browser on `localhost:8000`.
2. Press play → the agent **speaks the question**.
3. Answer **in Egyptian Arabic with English technical terms**, deliberately leaving **one** gap.
4. Transcript appears on screen; retrieved chunks + scores visible (**distractors absent — point at
   it**).
5. The agent **speaks a follow-up question in Arabic**. Answer it.
6. The agent **speaks the verdict in Arabic**: score + one-line justification.
7. Cut. Under 3 minutes.

**Live walkthrough — have these on screen, ready:**
- `app/policy.py` (one screen) and `tests/test_policy.py` passing offline.
- The mermaid diagram next to `app/graph.py`.
- A `grep '"thread_id": "…"' app.log | jq -s 'sort_by(.seq)'` that replays a **probing** session.
- `artifacts/summary.md` + `tests/golden.json` side by side.
- The leak grep: a distinctive phrase from a sample answer → **zero hits in the log**.
- Running with `--reload` so their "make one small change" lands in two seconds.

---

## 23. Interview talking points — the ten questions to expect

| They ask | Answer in one breath |
|---|---|
| *Why a graph runtime for one question and one optional probe?* | Two things it does that a handler can't state cleanly: `probe → summarize` is a real **cycle**, and `interrupt()` models a **suspend across two HTTP requests** as a paused computation, not a rebuilt session dict. Plus one reducer doing real work. Everything else I refused to hand it — no subgraphs, no agents, no chains, no vector store. |
| *Isn't a hardcoded `if`-chain a shortcut?* | The LLM supplies judgment, code supplies policy. Coverage is open-ended model work; *given that coverage, do we probe?* is a rule I can read, test offline, and defend. The cap lives in code because prompts drift and an infinite probe loop in a live demo is unrecoverable. |
| *Why not one LLM call for coverage + decision + score?* | It contaminates the score with a question not yet asked, and it makes the decision untestable. Three stages, three contracts, three failure surfaces I can localise. |
| *Cosine over 10 chunks — that's not really RAG.* | It is retrieval with a measured distractor margin; a vector DB at this size is operational weight with no recall benefit. Here are the actual scores, and here's the corpus size at which I'd flip. |
| *You hardcoded the doc→competency map.* | Six-file fixed corpus; that mapping is **data**, not logic. `TODO(tradeoff)` marks deriving it from headings if the corpus grows. |
| *How do you know the score is right?* | `tests/golden.json`, hand-labelled **before the scorer existed**, asserted at ±1. That ordering is the whole point — otherwise the baseline is just the pipeline agreeing with itself. |
| *How would you know a change made it worse?* | Re-run the gate, diff `artifacts/eval_<n>.json`: a shifted score, a flipped coverage status, a changed `decision.reason`, or a distractor in `retrieved` each localise the regression to one stage. |
| *What breaks first in production?* | `InMemorySaver` — single process, no restart survival; it's a one-line swap at `compile()`. Then STT on heavy code-mixing. Then the absence of a per-session cost ceiling. |
| *Show me the branch actually firing.* | Log replay for one `thread_id`: `decision.made` → `interrupt.raised` → `interrupt.resumed` → `session.end`, plus the artifact showing both the probe and the final verdict. |
| *Where does the time go?* | `provider.call` latencies, no profiler needed — STT dominates, then the coverage call. |

Two things worth volunteering unprompted, because they show the reading was real:
- **The corpus has planted distractors** (`02`, `04`); the rubric says *"only score competencies the
  question actually touches"*, so excluding them is a correctness requirement, not tidiness — and
  the gate asserts it on every run.
- **`interrupt()` replays its node from the top on resume**, so the LLM call that writes the
  follow-up lives in the node *before* the suspend. Found in the docs, not in production.

---

## 24. Future improvements — only if time remains, otherwise README → "what's next"

Ordered by value per hour, none started before Phase 8 is complete:

1. **SQLite checkpointer** — one line at `compile()`, removes the single-process limit.
2. **Enforce evidence grounding** — downgrade `covered` → `partial` when the quote isn't verbatim
   in the transcript (§9.2), once the counter shows it's safe.
3. **Per-competency sub-scores** in `Evaluation`, so the 1–5 is auditable rather than asserted.
4. **A second golden sample per score band** — three samples is a thin baseline.
5. **Streaming TTS** for a lower perceived latency in the verdict.
6. **Cost/latency budget per session**, surfaced from `provider.call` totals.

---

## 25. Working agreement

- Run it after every phase and **paste the actual output**. "It should work" is not a status.
- Commit per phase; the message says **what now runs**.
- Update `data/docs/PROMPT_LOG.md` **as you go** — key prompts, dead ends, course corrections. It is a
  **graded artifact**.
- On ambiguity: take the shipping option, log it under README → Assumptions, continue. The brief
  already delegated these decisions.
- Keep the tradeoff list running as you build — it becomes "what I cut" and the rejected roads.
  Written at the end, it is guesswork about your own reasoning.
- **No refactoring for elegance before Phase 7 passes.**
- Precedence when rules conflict: **1. It runs → 2. The follow-up branch is a real decision →
  3. Retrieval is sane → 4. Everything else.** Choosing "works now" over "better design" is always
  correct here — leave a `# TODO(tradeoff):` and move on.
