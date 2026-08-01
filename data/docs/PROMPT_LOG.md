# Prompt Log

Key prompts, dead ends, and course corrections, written during the build.

## Phase 0 — Foundations & skeleton

- `CLAUDE.md`/`PLAN.md` fix a single OpenAI key for STT + LLM + TTS + embeddings ("one credential
  = one setup step"). Overridden for this build: **Deepgram** (STT), **Groq** (LLM judge stages),
  **ElevenLabs** (TTS), and a **local `sentence-transformers` model** (`all-MiniLM-L6-v2`) for
  retrieval embeddings, since none of the three hosted providers serve an embedding endpoint.
  One-sentence disagreement, then building as specified: four credentials instead of one is a real
  setup-cost regression for a clean checkout, and Groq's OpenAI-compatible endpoint needs checking
  in Phase 1 for structured-output reliability versus OpenAI's `responses.parse`. `.env.example`
  and `app/config.py` reflect the new keys; the local embedding dependency is added in Phase 1
  (`retrieval.py` doesn't exist yet).
- Phase 0's validate step in `PLAN.md` §19 says `import app.policy` as the offline-import smoke
  test; `policy.py` doesn't exist until Phase 3, so the equivalent check here is
  `import app.config; import app.schemas` with no `.env` present — same property (nothing raises
  at import time on a missing key), against the modules that actually exist yet.
- `app/config.py` swapped raw `os.getenv` for `pydantic_settings.BaseSettings(env_file=".env")`.
  This wasn't just style: `python-dotenv` was a declared dependency but nothing ever called
  `load_dotenv()`, so `.env` was never actually being read. Fixed while keeping the flat
  `GROQ_API_KEY`/`DEEPGRAM_API_KEY`/`ELEVENLABS_API_KEY` module constants `CLAUDE.md` §3 asks for.
- Repo restructure: `corpus/`, `audio/`, and `docs/` moved under one `data/` folder
  (`data/corpus/`, `data/audio/`, `data/docs/PROMPT_LOG.md`), separating provided input assets and
  generated documentation from application code (`app/`, `static/`, `scripts/`, `tests/`). No code
  existed yet that referenced the old paths, so this was a documentation/layout-only change — every
  path mention in `CLAUDE.md`, `PLAN.md`, and `README.md` was updated to match, and `PLAN.md` §3.1
  records it as a `[DEVIATION]` from the paths those docs originally assumed.

## Phase 1 — Providers + measurement gates

`PLAN.md`'s own Phase 1 section (STT/TTS/LLM/retrieval specifics) was still written entirely
against OpenAI (`gpt-transcribe`, `responses.parse`, `text-embedding-3-small`), which doesn't match
the Deepgram/Groq/ElevenLabs stack chosen in Phase 0. Re-derived concretely against the actual
providers, checked against each one's current docs (2026-07-31) rather than assumed:

- **Embeddings: switched Phase 0's `sentence-transformers` choice to `fastembed`.** Same underlying
  model family (`BAAI/bge-small-en-v1.5`), ONNX runtime instead of PyTorch — confirmed zero `torch`
  in the install (16 packages, ~30s). Protects the "clean checkout in <5 min" criterion, which is
  graded #1, at no quality cost for 10 tiny chunks.
- **Groq via the existing `openai` pip dependency**, pointed at `base_url=https://api.groq.com/openai/v1`
  — no new SDK needed. Model `openai/gpt-oss-120b`, Groq's current flagship for constrained-decoding
  structured outputs (`response_format={"type":"json_schema","strict":true}`).
- **New deps via `uv add`:** `deepgram-sdk`, `elevenlabs`, `fastembed`. `uv.lock` regenerated.

### Measurement 1 — STT language mode (Deepgram `nova-3`)

Transcribed all 3 samples in `data/audio/` twice each: auto-detect vs pinned `language="ar"` (both
with the same `keyterm` vocabulary hint). **Finding, not assumption:**

- **Pinned `language="ar"` returned an empty transcript on all 3 samples** — not a usable mode at
  all with this SDK/parameter combination (no exception raised; `results.channels[0].alternatives[0].transcript`
  came back empty every time). Root cause not chased further — time-boxed per `CLAUDE.md`'s
  precedence rule ("it runs" > perfect quality); flagged under "what's next" instead.
- **Auto-detect (`detect_language=True`, no `language=`) is the only mode that produces output**,
  and it correctly transcribes the two Arabic-heavy samples in Arabic script.
- **But English technical terms still get transliterated into Arabic script** under auto-detect
  (e.g. "الكويرز", "كاشين بريدس", "اندكس" for "queries"/"caching Redis"/"index") even with the
  `keyterm` hint active. This is the exact failure `CLAUDE.md` §7 warned about for OpenAI Whisper —
  it reproduces on Deepgram too, and the one mitigation tried (`keyterm`) did not fix it. Sample 3
  additionally showed one outright mis-transcription ("ore fore maries", likely a garbled "N+1
  queries"). **Decision: ship `DEEPGRAM_LANGUAGE_MODE = "auto"`** — it's the only working mode —
  and record the transliteration gap as a known, measured limitation rather than paper over it.
  `# TODO(tradeoff):` try Deepgram's older `keywords=` param alongside `keyterm=`, or reconsider
  Deepgram specifically for STT, if this regresses scoring quality in practice.

### Measurement 2 — Groq structured-output reliability

Ran `llm.complete()` four times against a nested-list schema (`{items: [{competency, status,
evidence}]}` — the actual shape Phase 3's coverage stage will use) on `openai/gpt-oss-120b` with
strict `json_schema` mode. **4/4 valid, schema-conformant responses** — the community-reported
flakiness on this model did not reproduce for this schema shape. Shipping strict `json_schema` as
the primary path; the one-retry fallback already coded in `llm.py` stays as a safety net, not the
expected path.

### Measurement 3 — Retrieval margin → `RETRIEVAL_K`

Built the 10-chunk index (5 rubric competencies + 5 reference notes) and ran the full ranked table
against (a) a hand-written stand-in `claims_summary_en` and (b) the raw transcript of `Answer 1.mp3`
(stage-2 `summarize` doesn't exist yet, so the raw transcript stands in for it here). Both
distractor docs (`02_dependency_injection`, `04_api_design_security`) ranked 8th–9th of 10 in both
runs — comfortably outside `k=5`, with a real margin at the actual cutoff (~0.09 between the 5th
and 6th-ranked chunks on both queries). **`RETRIEVAL_K = 5` confirmed from data, not guessed** — no
change from the Phase 0 placeholder. Note: the "Tradeoff & communication" competency ranked last
(below both distractors) on the real-transcript query — a genuinely relevant competency can rank
weakly if a transcript doesn't touch it; worth watching once `judge.py` exists, but is a different
risk than distractor inclusion and not a Phase 1 blocker.

## Phase 1.5 — Deepgram dropped after a real Phase-2 failure; STT provider re-benchmarked

Starting Phase 2 (golden.json labelling), I transcribed all 3 samples through the shipped
Deepgram path to draft labels. **This exposed a failure Phase 1's own measurement had missed:**
Deepgram (`nova-3`, `detect_language=True`) returned a 2-4 word transcript for each of the two
Arabic-heavy samples, despite each being ~37.5s of real, continuous speech — at plausible-looking
confidence (0.76-0.83), i.e. it silently misidentified the language as `en` and discarded almost
everything else rather than failing loudly. `nova-2`/`nova-2-general` were tried too: no better,
and one sample came back at 0.0 confidence with an empty transcript. Root cause: Arabic genuinely
isn't in Nova-3's (or Nova-2's) code-switch language list — `config.py` had already flagged this
as a risk before it was measured, but Phase 1's own transliteration check apparently ran against
different/earlier output than this session reproduced, and understated how bad it was. Pinning
`language="ar"` was not retried here since Phase 1 already recorded it returning empty transcripts
on all 3 samples — no reason to expect that changed.

This is a real trigger of `PLAN.md` §16-R2 ("STT transliterates English technical terms" — turned
out to be worse than transliteration, it was near-total content loss), not a hypothetical. Per the
user's explicit direction: do not proceed on a degraded STT path, since transcription quality is
the acceptance criterion (task brief: "two of our test answers are Egyptian Arabic mixed with
English technical terms — handle that") that the follow-up decision, retrieval, scoring, and the
quality gate all sit downstream of.

**Benchmarked 3 alternatives against all 3 samples**, criteria: Arabic transcription quality (does
it actually transcribe vs. drop content), English technical-term preservation as Latin script (not
transliterated — `CLAUDE.md` §7), latency, integration simplicity (new credential? new SDK?), and
reliability across samples:

| Provider / model | New credential? | Answer 2/3 (Arabic) quality | English terms preserved? | Latency |
|---|---|---|---|---|
| Deepgram `nova-3`/`nova-2` (shipped) | — (already had it) | **Fails** — 2-4 words from 37s of speech | N/A (mostly not transcribed) | ~1-2s |
| Groq `whisper-large-v3` (OpenAI-compatible endpoint, existing `GROQ_API_KEY`) | No | Full, coherent Arabic; some awkward phrasing | Yes, Latin script (`query`, `index`, `Redis`, `async`) | ~1.3-2.6s |
| Groq `whisper-large-v3-turbo` | No | Answer 2 fine; **Answer 3 hallucinate-translated into broken English** mid-transcript (a known Whisper-turbo failure mode on code-switched audio) | Inconsistent — sometimes fine, sometimes translated away entirely | ~0.8-1.7s |
| **ElevenLabs `scribe_v1`** (existing `ELEVENLABS_API_KEY`) | **No** | **Best** — clean, natural, punctuated Egyptian Arabic on both samples; correct per-sample `language_code` (`eng`/`ara`) at 0.86-0.98 confidence | Yes, consistently, Latin script throughout | ~2-3.6s |

**Chosen: ElevenLabs Scribe (`scribe_v1`).** Best transcript quality by a clear margin, and it
needs no new credential — `ELEVENLABS_API_KEY` was already held for TTS. **Bonus consequence:**
this drops Deepgram as a dependency entirely, so the stack goes from 3 credentials
(Groq/Deepgram/ElevenLabs) to **2** (Groq/ElevenLabs) — fewer moving parts than either Phase 0's
deviation or `CLAUDE.md`'s original 1-key OpenAI plan achieved for STT specifically, at
correctness the deviation didn't have. `keyterms=` (Scribe's vocabulary-hint parameter) carries
the same domain-term list Deepgram's `keyterm` used. `# TODO(tradeoff):` Groq Whisper stays a
credible fallback if Scribe's cost/quota ever becomes a problem — noted, not acted on.

**Changes:** `app/providers/stt.py` rewritten against
`elevenlabs.client.ElevenLabs.speech_to_text.convert` (same `transcribe(audio: bytes) ->
Transcript` contract — no caller changes); `app/config.py` drops `DEEPGRAM_*` constants and
`DEEPGRAM_API_KEY`, adds `ELEVENLABS_STT_MODEL` and `STT_VOCAB_HINT` (renamed from the
Deepgram-specific `DEEPGRAM_KEYTERMS`, unchanged content), renames `ELEVENLABS_MODEL` →
`ELEVENLABS_TTS_MODEL` now that ElevenLabs serves two roles; `app/main.py` drops the
`deepgram_key_present` health-check field; `pyproject.toml`/`uv.lock` drop `deepgram-sdk` (`uv
remove deepgram-sdk`); `.env.example` drops `DEEPGRAM_API_KEY`. Re-ran the transcription smoke
test post-swap — all 3 samples now transcribe correctly, and language derivation tags them `en` /
`ar-en` / `ar-en` as expected.

## Phase 2 — `tests/golden.json`, cross-validated before the scorer exists

Per user direction, added a temporary `OPENAI_API_KEY` (not wired into any shipped provider) and
ran `gpt-4o-transcribe` and `whisper-1` against all 3 samples as a **quality cross-check on the
golden-labelling process only** — highest-quality transcript for a one-time hand-labelling
exercise, independent of what ships in `app/providers/stt.py`. Compared against the already-shipped
ElevenLabs Scribe output and the earlier Groq `whisper-large-v3` benchmark run.

**Result: all 4 independent engines (ElevenLabs Scribe, Groq whisper-large-v3, OpenAI
gpt-4o-transcribe, OpenAI whisper-1) agree in substance** on both Arabic-heavy samples — same
content, same technical terms, only minor formatting differences (spacing/capitalization around
English terms embedded in Arabic). This 4-way agreement is treated as strong evidence the
transcript content is trustworthy for labelling, standing in for a literal human ear-listen per
`PLAN.md` §17/§16-R12 ("if unconfirmed, say so rather than implying a listen that did not happen").
**`labelled_from` in `tests/golden.json` states this plainly** — cross-validated transcript +
rubric-based review, not a human ear-listen. OpenAI was not adopted as the shipped STT provider;
ElevenLabs Scribe (already chosen in Phase 1.5) remains the production path.

**Scored each sample against `QUESTION.md` + `00_scoring_rubric.md` + the 3 relevant reference
docs**, with a second-pass check per label (justified by rubric wording? only relevant
competencies considered? covered/missing calls correct? defensible in the live walkthrough?):

- **Answer 1 (English) → 3.** Systematic, measure-first (APM/logs), strong EF Core (N+1, missing
  index, `AsNoTracking`, explicitly "checks the SQL we are generating"), solid caching
  (in-memory→Redis, invalidation rule, correctly distinguishes stale-tolerant data). **Never
  mentions async/await once**, despite the question naming it explicitly and the omission being
  specifically about behavior *under load* — the exact scenario asked about. That single,
  question-relevant gap is what caps this at 3 rather than 4; everything else present is
  senior-level. Single gap (Async & concurrency) → `policy.decide()` should probe.
- **Answer 2 (Arabic) → 2.** Touches Diagnostic method, Data access (EF Core), Async & concurrency,
  and Caching & performance, but every one shallowly: no measurement step, no EF-specific
  terminology, an explicit self-admission of not understanding async's effect under load, no
  invalidation story, and zero tradeoff reasoning. Matches the rubric's "names a concept but can't
  apply it" closely — not a 1 (there's real, relevant content), not a 3 (nothing rises to "correct
  on the common cases"). Diffuse gaps (4 partial, 1 missing, not a single gap and not ≥2 missing) →
  `policy.decide()` should skip and score now.
- **Answer 3 (Arabic) → 1.** "Restart the server," then "add more RAM / get a stronger server" — a
  near-verbatim match for the rubric's own stated 1–2 anti-pattern, plus an explicit disclaim of
  EF Core/database-query knowledge and no mention of async or caching at all. ≥2 missing →
  `policy.decide()` should skip.

Together the 3 samples exercise all 3 of `policy.decide()`'s non-trivial branches (single-gap
probe, diffuse-gaps skip, weak-across-the-board skip) on real audio, ahead of `tests/test_policy.py`
covering the same branches synthetically in Phase 3.

### Follow-up verification — the shipped STT function itself, against OpenAI's actual strongest model

Re-ran the comparison one more time, more rigorously: called `app.providers.stt.transcribe()`
directly (the real shipped function, not a bypass script) against OpenAI, and used
`client.models.list()` to find OpenAI's actual current flagship rather than assuming
`gpt-4o-transcribe` still was one - the account exposes `gpt-transcribe`, one generation newer.

**Content accuracy:** effectively identical between the two on all 3 samples - same facts, same
structure, nothing dropped either side.

**Technical-term preservation (the actual bar, `CLAUDE.md` §7):** the shipped ElevenLabs Scribe
path kept every technical term in Latin script, on every sample, with zero exceptions (`database`,
`queries`, `code`, `server` x3, `index`, `caching`, `Redis`, `async`, `load`, `entity framework`).
**OpenAI `gpt-transcribe` transliterated several of the exact same words into Arabic script**
(`database`->`داتابيس`, `queries`->`كويريز`, `code`->`كود`, `server`->`سيرفر`/`السيرفر` all three
times it was said) while leaving others in English - i.e. it reproduces the precise failure mode
this whole STT search was meant to avoid. It also mis-heard "APM" as "EPM" on Answer 1, where
Scribe got the acronym right.

**Conclusion: no change to `app/providers/stt.py`.** ElevenLabs Scribe outperforms even OpenAI's
current flagship transcription model on the metric that actually matters for this task (consistent
Latin-script technical terms inside Arabic output), confirming the Phase 1.5 decision rather than
just re-affirming it by assumption.

### Second-pass correction to `tests/golden.json`

Per user request, re-derived every label word-by-word against the rubric text (not against my own
earlier summary) to check for errors rather than re-stating prior conclusions. Found one real
mistake and left everything else standing on tighter rubric citations:

- **Answer 2's `Tradeoff & communication` was wrongly marked `missing`.** The transcript has a
  quotable line - "I'll start with the database, probably because that's the thing most likely to
  be slow" - which states an assumption and names what they'd do first + why: literally what this
  competency asks for, even if thin. Since the coverage contract requires evidence to be `null`
  *only* when `missing`, and a real quote exists here, `missing` was never a defensible label -
  corrected to `partial`. `expected_competencies` for Answer 2 grows from 4 to all 5; the score
  (2) and the policy branch (diffuse gaps, skip) are both unchanged, since missing-count moves
  from 1 to 0 (still < 2) and gap-count stays 5 (still not exactly 1).
- **Answer 1's score (3) re-confirmed on a tighter citation**: rubric level 3 is literally "correct
  on the common cases; some gaps *under load*," and async/thread-pool starvation is specifically
  the *under-load* failure mode - so the total omission of async is a textual match for level 3,
  not just a subjective deduction from an otherwise-4 answer. Level 4 explicitly requires
  DB+async+caching as a set, which is ruled out since async is entirely absent - checked
  word-by-word again to be sure.
- **Answer 3's score (1) re-confirmed, no changes.** One design note surfaced for Phase 3: the
  candidate's "I don't know how EF Core or database queries could affect this" is a quotable line,
  but it is a disclaimer of ignorance, not applied knowledge - kept `Data access (EF Core)` as
  `missing` since coverage should reflect demonstrated understanding, not self-aware ignorance.
  `judge.py`'s `analyze_coverage` prompt should say this explicitly so the LLM doesn't default to
  `partial` merely because *some* quote exists.

### Deliberate scoring policy: mention is not coverage

Per user direction, generalized the note above into an explicit rule and wrote it into `PLAN.md`
§10 as the exact instruction Phase 3's `analyze_coverage` prompt must carry (`judge.py` does not
exist yet - this is a specification, to be implemented verbatim when Phase 3 starts, not a prompt
edit to a file that is not there):

- A competency is `covered`/`partial` only when the transcript demonstrates actual understanding,
  reasoning, or correct application of it - never for naming a technology or concept alone.
- An explicit statement of uncertainty or lack of knowledge about a competency (e.g. "I don't know
  EF Core," "I'm not familiar with X," "I've never used X," "I'm not sure how this affects things
  under load") overrides any mention of it - mark that competency `missing`, not `partial`, even
  though the concept was named.
- Evidence quotes must support the reasoning/application itself, not merely the sentence a keyword
  happens to appear in.
- The rule is generic across all 5 competencies, not special-cased to EF Core (where it was first
  noticed on Answer 3).

**Re-ran the sanity check against all 3 golden labels with this rule applied, quote by quote:**

- Answer 1: no explicit uncertainty disclaimers anywhere; every credited competency has real
  explanatory reasoning attached (not bare mentions). No change.
- Answer 3: `Data access (EF Core)` was already `missing` precisely because of the "I don't know
  entity framework" disclaimer - this rule formalizes a call already made ad hoc. No change.
- **Answer 2: one real change.** The async line - "we should make the code async so blocking
  doesn't happen. Honestly, I'm not very sure how it affects things under load" - is a direct match
  for the explicit-uncertainty clause, and the disclaimed part is specifically the load-related
  mechanism the rubric asks about. `Async & concurrency` moves from `partial` to `missing`.
  `expected_competencies` drops from 5 to 4 (async removed). Score stays 2 (still solidly
  level-2 - "names a concept but can't apply it" - now on cleaner grounds). Policy branch is
  unchanged: missing-count 0->1 (still < 2), gap-count still 5 (4 partial + 1 missing, still not
  exactly 1) -> still "diffuse gaps; score now."

Net effect: Answers 1 and 2 now show the same surface pattern (4 of 5 competencies touched) for
different underlying reasons - Answer 1's one gap is a clean, total omission (never said the word
"async"); Answer 2's one gap is a named-but-disclaimed concept caught specifically by the new rule.
That is a useful, deliberate distinction to have in the regression baseline: it means the coverage
stage has to actually reason about *demonstrated* understanding rather than pattern-matching on
which keywords appear, and the golden set now exercises that difference directly rather than by
accident.

**Golden baseline finalized** as of this pass: `tests/golden.json` (3 entries), the mention-is-not-
coverage rule specified in `PLAN.md` §10, ready for Phase 3 to build `judge.py`/`policy.py`/
`graph.py` against.

## Phase 3 — Linear graph: `summarize → retrieve → analyze → score`

Built `app/policy.py` (verbatim `decide()` from `CLAUDE.md` §4), `tests/test_policy.py` (4
offline cases, no key needed — all pass), `app/judge.py` (4 prompts + 4 stages, zero `langgraph`
imports), `app/graph.py` (linear `StateGraph`, `InMemorySaver`), `scripts/eval_samples.py` v1, and
`POST /api/screening/start` in `main.py`. Added `pythonpath = ["."]` to `pyproject.toml`'s pytest
config (tests couldn't import `app` without it) and `python-multipart` via `uv add` (FastAPI's
`UploadFile` requires it; not a transitive dep of bare `fastapi`).

**R5 resolved by running it, not by assuming:** the checkpointer round-trips `Chunk`/`Decision`
Pydantic objects through `InMemorySaver` with no serialization error — confirmed on the first real
`graph.invoke()`, no `model_dump()` workaround needed.

### Retrieval margin, re-measured on real `claims_summary_en` — a finding, not the Phase 1 guess

Phase 1's margin measurement (PROMPT_LOG "Phase 1" above) used the raw transcript as a stand-in for
`claims_summary_en`, since `judge.py` didn't exist yet, and flagged "Tradeoff & communication"
ranking weakly as something to watch. With the real `summarize()` output now available, printed the
full 10-row ranked table for all 3 samples (`00_scoring_rubric.md`'s 5 competency chunks + 3
competency-tagged reference notes + 2 distractors):

- **"Diagnostic method" and "Tradeoff & communication" are borderline-to-absent from `k=5`** on
  Answers 1/2, because a competency with both a rubric chunk *and* a reference-note chunk (Data
  access, Caching, Async) can occupy 2 of the 5 slots, crowding out competencies that only have a
  rubric chunk (Diagnostic method, Tradeoff & communication).
- **Raising `k` does not safely fix this.** On Answer 3, `02_dependency_injection.md` (a planted
  distractor) ranks 6th at `k=5`'s boundary — comfortably excluded — but only ~0.03–0.05 below the
  5th-ranked legitimate chunk. Any `k` large enough to reliably pull in "Diagnostic method" (rank 6
  on Answers 1/2) also pulls the distractor into Answer 3's top-k on at least some runs. Verified
  this is not hypothetical: an early live run of `POST /api/screening/start` on Answer 3, before the
  temperature fix below, put `02_dependency_injection.md` in `retrieved` — the exact failure this
  corpus is built to catch.
- **Root cause of the run-to-run flip: Groq's default sampling temperature on the `summarize` call.**
  Two consecutive `graph.invoke()` calls on identical audio produced slightly different
  `claims_summary_en` wording, which was enough to flip the 5th/6th-place ranking given how thin
  Answer 3's margin already was. `# TODO(tradeoff):` a cleaner long-term fix is deduplicating the
  rubric-chunk/reference-note pair per competency into one merged chunk (frees slots without
  touching `k` at all) — not attempted now; it changes `retrieval.py`'s chunking contract
  (`PLAN.md` §6.2, a considered Phase-0/1 decision) and Phase 3's job is proving the linear graph,
  not re-deriving the chunking strategy.
- **Fix shipped: `llm.complete()` now defaults to `temperature=0.0`** (`app/providers/llm.py`).
  Re-ran all 3 samples twice more after the fix: **zero distractor leaks across 2 full runs**, all
  3 scores exactly matched `tests/golden.json` (not just within ±1) both times. `RETRIEVAL_K`
  stays at 5 — unchanged, still the right call now that the noise source is fixed rather than
  papered over with a lower/higher `k`.
- **Residual, smaller finding:** even at `temperature=0`, the coverage stage's per-competency
  `partial` vs `missing` call for Answer 2 varied by one competency between the two post-fix runs
  (score and `decide()` outcome unaffected both times). Known behavior of MoE inference backends
  (batching/kernel non-associativity can perturb "greedy" decoding even at `temperature=0`) — noted
  as a limitation for the README rather than chased further; the properties that actually matter for
  the Definition of Done (distractor exclusion, score-within-±1) held on every run.

### Verified end to end

`uv run pytest` — 4/4 offline, no key. `uv run python scripts/eval_samples.py` — real JSON verdict
for Answer 1 (score 3, single gap in Async & concurrency, probe fires — matches golden). All 3
samples cross-checked against `tests/golden.json`: exact score match on all 3, correct `decide()`
branch on all 3 (single-gap probe / diffuse-gaps skip / weak-across-the-board skip — the same three
branches `tests/test_policy.py` covers synthetically, now confirmed on real audio). `grep langgraph
app/judge.py app/policy.py` → zero hits. Arabic verdicts correctly keep technical terms in English
script ("الـ database", "الـ caching", "الـ async", "الـ server", "الـ RAM").

---

## Phase 4 — TTS gate: the question and one Arabic verdict, spoken

`app/providers/tts.py` (ElevenLabs `eleven_multilingual_v2`) was written back in Phase 0/1 but
never called from a route. This phase wired it in and ran the required listening check.

- **`Evaluation.spoken_text()` added to `schemas.py`** as the single place that decides what TTS
  receives: the follow-up question on a probing turn (`score is None`), else `f"{score}/5 —
  {justification}"`. No probe node exists until Phase 6, so only the second branch is reachable
  right now — written both ways anyway since the shape is cheap and this file shouldn't need a
  second pass in Phase 6.
- **`judge._load_question()` → `judge.load_question()`** — made public (kept the
  `@lru_cache(maxsize=1)`) so `main.py` can reuse the existing `QUESTION.md` `>`-line parser for
  the new route instead of duplicating it.
- **`GET /api/question`** added: synthesizes the fixed question in English (no candidate language
  is known yet — nothing has been detected), base64s it, and caches the result in a module-level
  variable on first call rather than at import time (so a missing `ELEVENLABS_API_KEY` doesn't
  break app startup). Verified the cache works: called the route 3 times against a live server,
  `grep '"kind": "tts"'` on the log showed exactly **one** `provider.call` for `thread_id: "-"`.
- **`POST /api/screening/start`** now calls `tts.speak(evaluation.spoken_text(), transcript.language,
  thread_id)` and returns real `spoken_text` / `audio_b64` instead of the hardcoded `""` placeholders.
- **Regression check**: re-ran all 3 `data/audio/` samples through the now-TTS-wired endpoint —
  scores unchanged (Answer 1 → 3, Answer 2 → 2, Answer 3 → 1), still exact matches to
  `tests/golden.json`. `grep -i caching app.log` after the run only matched the *competency label*
  `"Caching & performance"` inside `coverage.done` (a status/count, explicitly allowed) — no
  transcript or justification text leaked, confirming `obs.log_event`'s `_BANNED` field guard holds
  under real TTS traffic too.
- **Listening gate (human-run, not simulated)**: generated `artifacts/question.mp3` (the English
  question) and `artifacts/verdict.mp3` (Answer 2's Arabic verdict — spoken text: `"2/5 — بيذكر الـ
  caching والـ async لكن ما بيطبقهمش ولا عنده منهجية واضحة"`) and listened to both. **Verdict:
  placeholder voice `JBFqnCBsd6RMkjVDRZzb` passes as-is** — Arabic is intelligible, "caching" and
  "async" are spoken in English rather than transliterated, accent is acceptable. No voice A/B
  needed; `config.py`'s comment updated from "placeholder" to record this decision. `artifacts/` is
  gitignored, so the two mp3s used for the listen are not committed — regenerate with the
  `curl`/`Invoke-RestMethod` steps in this phase's plan if the check needs to be repeated (e.g.
  after a future voice change).

---

## Phase 7 — Quality gate hardening

`scripts/eval_samples.py` rewritten from the Phase-3 single-sample throwaway into the real gate:
runs all 3 `data/audio/` samples, checks every `PLAN.md` §14 assertion (score range +
justification, distractor exclusion across **every** `retrieve` call in a session — not just the
final one, since a probing sample retrieves twice — spoken-text language match, non-empty
`decision.reason`, `score is None` mid-probe, golden ±1), writes `artifacts/eval_<n>.json` /
`artifacts/run.log` / `artifacts/summary.md`, and exits non-zero on any failure without stopping
early (every sample still runs, every failure still gets reported). `tests/test_policy.py` gained
3 parametrized cases for `stt._derive_language` (ar / en / ar-en from script ratio).

### `RETRIEVAL_K`: 5 → 4, evidence-based, corrects a Phase 3 conclusion that didn't hold up

Phase 3's log (above) concluded `RETRIEVAL_K = 5` was safe after a `temperature=0.0` fix on the
Groq `summarize` call — "zero distractor leaks across 2 full runs" at the time. Phase 7's first
live gate run at `k=5` **contradicted that**: `02_dependency_injection.md` (a planted distractor)
landed at rank 5 on `Answer 3.mp3` (score 0.6756, only ~0.02 below the 4th-place legitimate chunk
at 0.6964) and the gate correctly failed on it.

Per user request, checked the actual index shape before touching anything (not blind tuning):
**10 chunks total** — 5 from `00_scoring_rubric.md` split at `## Competency:` headings, plus 1
each from the 5 reference docs (`app/retrieval.py::_chunk_rubric`/`_chunk_reference`), of which 2
are the planted distractors (`02_dependency_injection.md`, `04_api_design_security.md`).
Retrieval is a flat cosine index over all 10 chunks, never filtered to rubric-only. At that size,
lowering `k` to 4 is a proportionate, well-justified change, not an arbitrary knob turn.

**Verified before committing to it:** every `retrieval.done` log event across all 3 golden
samples — both retrieves on the probing sample, one each on the two non-probing samples — has a
clean top-4 (no distractor in the first 4 ranks on any of them). Re-ran the full gate at `k=4`:
3/3 samples pass, all scores exactly match `tests/golden.json` (Δ0), `Answer 1` still probes on
the single Async & concurrency gap, `Answer 2`/`Answer 3` still skip. Deliberately broke it back to
`k=10` to confirm the gate fails loudly and names the leaking sample + doc — it did, on all 3
samples, then reverted.

**One residual finding, not chased further:** on one intermediate rerun during this session
(discarded, not the committed artifacts), `Answer 1`'s retrieval ranked the `01_async_concurrency.md`
reference chunk *just* outside the top-4, and the probe didn't fire — a different `claims_summary_en`
wording from the same underlying audio (Groq's structured-output sampling is not perfectly
invariant even at `temperature=0`, per the Phase 3 note above on MoE non-associativity) was enough
to flip a razor-thin ranking. The final committed run doesn't show this, and 3/3 passes, but it
means `k=4` narrows Answer 1's probe-firing margin too, not just Answer 3's distractor margin.
`# TODO(tradeoff):` the structural fix from Phase 3's note — merging each competency's
rubric-chunk + reference-note pair into one chunk — would free a retrieval slot without relying on
`k` alone to separate signal from the two distractors; not attempted here, same reasoning as
Phase 3: it changes the chunking contract and is out of Phase 7's scope.

`.gitignore` updated (`artifacts/*` + explicit re-includes) so `eval_1/2/3.json`, `run.log`, and
`summary.md` are committed as the regression baseline evidence `PLAN.md` §14/§20 call for, while
ad-hoc files from earlier manual Phase 4/5/6 testing (`question.mp3`, `verdict.mp3`, etc.) stay
ignored.

## Phase 8 — README, PROMPT_LOG, recording

`README.md` rewritten from scratch — it was still the original candidate-package stub pointing at
`00_TASK_BRIEF.md`/`QUESTION.md`/`data/`, with none of `PLAN.md` §20's checklist written. Every
fact in the new README is sourced from the actual shipped system (`app/config.py`, `app/graph.py`,
`app/policy.py`, `artifacts/summary.md`, this log) rather than `PLAN.md`'s earlier OpenAI-only
draft sections (§2.2/§7/§8), which were superseded during Phase 0/1 and never matched what
actually shipped. The architecture diagram is pasted verbatim from
`graph.get_graph().draw_mermaid()` run against the real compiled graph, not hand-drawn or
copied from `PLAN.md`'s prose description.

**A real finding surfaced while pulling the quality-gate table into the README, not invented for
this phase:** all 3 committed `artifacts/eval_*.json` show `"decision.probe": false` — the
follow-up branch does not fire on any of the 3 golden samples right now. Checked against
`tests/golden.json`'s own labelling notes, which say Answer 1 has exactly one gap (Async &
concurrency) and *should* trigger a probe. Cross-checked against this log's own Phase 3 entry,
which recorded the probe firing on Answer 1 at the time. Root cause, not a new bug: Phase 7's
`RETRIEVAL_K` fix (5 → 4, to exclude a distractor that had crept into rank 5 on Answer 3) also
pushed the `Async & concurrency` reference chunk just outside Answer 1's top-4 — so only 2
competencies (`Data access (EF Core)`, `Caching & performance`) enter scope for that sample, both
score `covered`, and `decide()` correctly returns "all competencies covered — score now" given
what it was handed. Phase 7's own log already flagged this exact tradeoff as a residual finding
("k=4 narrows Answer 1's probe-firing margin too, not just Answer 3's distractor margin") without
confirming it had actually flipped — this phase confirms it did.

Per user direction, this is **documented, not re-engineered**: `PLAN.md` §16-R1 anticipated
precisely this risk and explicitly says not to tune the policy to force the branch to fire.
Retrieval was left untouched too — re-opening `k` (or the chunking-merge fix flagged as
`# TODO(tradeoff)` since Phase 3) is real engineering work belonging to a future phase, not this
documentation-and-recording one, and doing it under time pressure specifically to make a demo
branch fire is the exact smell `policy.py`'s own module docstring warns against. Instead:

- The README's new "Known limits" section states the finding plainly, with the reasoning, using
  `PLAN.md`'s own framing ("a named limit reads as understanding; a found limit reads as an
  oversight").
- Correctness of the branch itself is unaffected and doubly provable without touching retrieval:
  `tests/test_policy.py`'s 4 offline cases exercise all of `decide()`'s branches directly, and the
  branch is demonstrated firing live from the mic in the submitted recording by deliberately
  giving an answer that leaves exactly one competency gap (the same shape `PLAN.md` §16-R1
  suggests: strong on EF Core + diagnostics, silent on caching).
- Added to README → What's next: a 4th golden sample recorded specifically to leave one gap, wired
  into `scripts/eval_samples.py` + `tests/golden.json`, so the probe is demonstrated on committed
  audio too, not only live — `PLAN.md` §16-R1's mitigation (a), deferred rather than dropped.

**Verification run before committing:** Definition-of-Done greps (`PLAN.md` §2.3) — zero
`langgraph` imports in `app/judge.py`/`app/policy.py`, no provider SDK import outside
`app/providers/`, no model-ID string literal outside `app/config.py` (the one `eleven_multilingual_v2`
hit outside `config.py` is inside a docstring explaining an API constraint, not a literal driving
behavior), every `graph.py` node body a handful of statement-lines, well under the ~10-line
guideline. Leak grep: a distinctive verbatim quote from Answer 1's transcript
("in-memory cache first and then Redis...") greps zero hits in `artifacts/run.log`, confirming
`obs.log_event`'s banned-field guard held under real traffic — it only appears in the committed
`eval_1.json` artifact, which is meant to hold it.

---

## Phase 10 — Performance & architecture audit: measure first, then fix the two things worth fixing

Brief for this phase was explicitly *not* to add features: audit the whole request lifecycle,
rank findings by impact, fix only Medium/High, and report measurements rather than estimates.
Everything below was measured before it was changed.

### Finding 1 (High) — retrieval redundancy was silently deciding the evaluation's scope

Baseline `artifacts/run.log` showed `retrieval.done` returning the same `doc_id` twice per call.
The initial read — "duplicate chunks, cosmetic" — was wrong, and checking it is what made this
phase worthwhile. The index holds *two views* of most competencies: the rubric's criteria line and
a longer reference note. Against a given query they score near-identically, so flat top-4 spent
consecutive slots restating one competency. And because `analyze_coverage()` derives its
competency list from whatever retrieval hands it (`requested = sorted({c.competency for c in
retrieved})`), that redundancy was not wasted context — it was **deciding how much of the rubric
the candidate got assessed against**. Measured over the committed samples, flat top-4 judged 2, 2
and 3 of 5 competencies.

This is the same root cause Phase 7 hit from the other side and Phase 9 chose to document rather
than re-engineer: Answer 1's probe stopped firing because `Async & concurrency` fell outside a
top-4 that was half-spent on `Data access (EF Core)` twice. Phase 9 deferred it on the grounds
that re-opening `k` to force a demo branch to fire is the smell `policy.py` warns against — which
was right. Fixing the redundancy is a different change: it widens scope on merit, and the probe
firing again is a *consequence*, not the objective.

`retrieve()` now returns the best-scoring chunk of each of the top-`k` competencies. Ranking is
untouched and still purely semantic — every chunk, distractors included, competes on cosine alone,
nothing is filtered by document or competency name, so distractor exclusion remains a measured
property rather than an assumption.

### `RETRIEVAL_K`: 4 -> 3, and why not 4

Deduping reaches further down the ranking than flat top-4 did, so `k` had to come down with it or
a distractor rides in on the slack. This was nearly missed: the first measurement, against the
committed `claims_summary_en` values, showed dedup k=4 clean on all three samples. Re-measuring
against a *fresh* set of summaries — the summarizer is not bit-stable, per the MoE decoding note
in Phase 7 — showed dedup k=4 leaking `02_dependency_injection.md` into Answer 3. Tuning `k`
against a single sampling of a non-deterministic upstream stage would have shipped a coin-flip.

Margin from the last selected competency to the best-scoring distractor, both samplings:

|  | flat k=4 (before) | dedup k=3 (shipped) | dedup k=4 |
|---|---|---|---|
| Answer 1 | +0.078 / +0.086 | +0.069 / +0.085 | +0.041 / +0.030 |
| Answer 2 | +0.053 / +0.080 | +0.061 / +0.077 | +0.014 / +0.026 |
| Answer 3 | +0.018 / +0.017 | +0.018 / +0.017 | +0.016 / **LEAK** |

dedup k=3 judges 3 competencies on every sample where flat k=4 managed 2-3, and its distractor
margin is never worse than the scheme it replaces. k=3 is where this corpus's separability runs
out, not a round number.

**Result:** competencies judged 2/2/3 -> 3/3/3; golden scores unchanged (3/2/1, delta=0);
distractors clean on every `retrieval.done` event; and Answer 1 probes again — confirmed twice
through the gate and once end-to-end over HTTP, where it asked *"Can you explain why calling
.Result on an async method inside a controller can cause thread-pool starvation under load?"* and
scored 3 after the merged turn. The README's Phase 9 "known limit" is retired and replaced with
the real remaining one: only 3 of 5 competencies are ever in scope, and `Tradeoff & communication`
never retrieves at all (ranks last, 0.53-0.57, on every sample).

### Finding 2 (High) — ~2.4 s of ONNX cold start was billed to the first candidate

`provider.call` proved it rather than suggesting it: the first corpus embed logged 1169-2382 ms,
every subsequent query embed 16-81 ms. Not a per-call cost — model load plus the one-time corpus
embed, landing inside whoever answered first. It also logged under `thread_id: "-"` in the middle
of a real session, colliding with that session's `seq` counter and breaking the `sort_by(.seq)`
reconstruction the README advertises.

`retrieval.warmup()` now runs in a FastAPI `lifespan` hook (and at the top of the gate). First
`retrieve_node` went **2450 ms -> 36 ms**. Warmup failure is logged and swallowed, not fatal — the
lazy path still works, and a boot that dies because a model download blipped is worse than a slow
first answer.

### Finding 3 (Medium) — provider timeouts were the SDK defaults, i.e. 240 s and 600 s

Surfaced by an actual failure: a transient Groq connect error during baseline measurement threw
`APITimeoutError` out of `score_node` after 34 s and **destroyed the whole 3-sample run**, four
LLM calls of completed work included. Two separate defects behind one symptom.

- No ceiling anywhere. Now `LLM_TIMEOUT_S = 45`, `ELEVENLABS_TIMEOUT_S = 60` — well above the
  measured maxima (STT 3.4-7.3 s, LLM 0.9-16.2 s, TTS 1.2-2.3 s), far below the defaults. A dead
  provider should fail a turn, not pin a worker thread for ten minutes.
- Retries were checked before being "fixed", and did not need fixing: both SDKs already retry
  twice on connection errors and 429/5xx (`elevenlabs/core/http_client.py` defaults `max_retries`
  to 2; the OpenAI client the same). Re-specifying that would only add a second place to get it
  wrong. Left alone, deliberately.
- The gate now isolates each sample, so one provider blip costs one row instead of the run.

### Finding 4 (Medium) — client construction triplicated, one host pooled twice

Three copies of the same lazy-singleton-with-credential-check, and STT and TTS each held their own
`ElevenLabs` client — two connection pools and two TLS handshake paths to the same host with the
same credential. Collapsed into `app/providers/clients.py`: credentials, pooling and timeout
policy stated once. The per-provider modules keep their own domain logic; only client
*construction* moved.

### Finding 5 (Medium) — a TTS failure threw away a completed evaluation

`_finish_turn()` called `tts.speak()` unguarded after the graph had finished, so an ElevenLabs
blip returned a 500 for a turn whose verdict already existed. Now `_speak_or_degrade()` logs
`tts.failed` and returns empty audio; the page renders the text verdict and hides the player.
Losing the audio costs the turn its voice — raising cost the candidate the interview and the graph
state behind it.

### Smaller items, measured

- `_strict_schema()` rebuilt a JSON schema and walked it recursively on every LLM call: **624 us
  -> 0.2 us** cached per model class. Small in absolute terms; free, and it is per-call.
- Corpus embeddings are L2-normalised once at index build, so `retrieve()` is a single matmul
  instead of recomputing a constant norm vector per call.
- `/api/question` replaced a mutable module global + `global` statement with `lru_cache`:
  2.96 s cold -> 0.036 s cached, and no unsynchronised global.
- `retrieval.done` now logs `competencies` alongside `doc_ids`, and the gate's summary table has a
  *Competencies judged* column — this is the quantity that regressed silently once already, and
  nothing in the gate would have shown it moving.

### Considered and deliberately not done

- **Merging `analyze` and `score` into one LLM call** (~1-5 s saved). Rejected: the decision policy
  sits between them and scoring must survive a probe. It would trade the assignment's
  decision/evaluation separation for latency.
- **Dropping `summarize` on English answers** (~1-5 s). Rejected: it is the retrieval query, and
  making the query's provenance depend on detected language is a correctness risk for exactly the
  code-mixed input this project exists to handle.
- **Removing `node.exit` as duplicative of `provider.call`.** For three of four nodes the two
  durations agree within ~2 ms. Kept anyway: it is the only signal for `retrieve`/`probe`, and it
  carries the `node.error` path.
- **Making the routes `async`.** They are `def`, so FastAPI already runs them in a threadpool;
  converting them without making every provider call awaitable would block the event loop. The
  current shape is correct.
- **Bounding `obs`'s per-`thread_id` counters.** Real unbounded growth, but the same lifetime
  problem as `InMemorySaver` beside it; both are fixed by the SQLite-checkpointer item, not by a
  second eviction policy.

---

## Phase 11 — Final-submission remediation and re-review (2026-08-01)

The provisional review found that retrieval was still deciding evaluation scope. The shipped fix
separates the two concerns: all five chunks from `00_scoring_rubric.md` are injected into every
coverage and scoring call. The final cleanup below narrows semantic retrieval to top-1 over
supplemental reference documents only.
`tests/golden.json` now records `expected_probe` and an explicit five-status minimum coverage map.
The live gate checks the initial-audio branch/coverage, the final five-item scope, score tolerance,
distractor exclusion, and actual MP3 bytes from ElevenLabs for every verdict/follow-up.

Other remediation completed in the same pass:

- STT now accepts bytes + real filename + MIME. FastAPI preserves MP3/WebM/MP4/M4A identity,
  rejects empty/mismatched/over-25-MB input, and Safari `video/mp4` is accepted for audio-only MP4.
- Coverage evidence/status and score 1-5 constraints are Pydantic validators. Provider JSON is
  retried once and malformed output raises a sanitized `ProviderOutputError`.
- LangGraph checkpoints now contain plain dictionaries, not Pydantic instances; live runs emitted
  none of the earlier future-serialization warnings.
- Model-derived table values are rendered with created DOM nodes and `textContent`; no
  `innerHTML` remains.
- STT/LLM provider failures map to useful 502/503 API errors. Completed evaluation TTS still uses
  the deliberate text-only degradation path.
- Offline verification expanded from 7 to **36 passing tests**: every policy branch, language
  derivation, schema constraints, mandatory scope, MP3/WebM/MP4 uploads, identity propagation,
  API/provider failures, graph serialization, fixture completeness, and safe UI rendering.

### Provider/model readiness attempts and live outcome

The original `openai/gpt-oss-120b` account bucket still returned 429 after Answer 1 coverage.
Groq's official model/structured-output documentation confirms that `openai/gpt-oss-20b` supports
the same strict JSON Schema mode on a separate quota bucket, so configuration moved to 20B with
`reasoning_effort=low`. Groq's current prompting guidance also recommends placing GPT-OSS
instructions in the user message; doing that improved the Arabic sample's EF status from missing
to partial. A 55-second inter-sample cooldown respects this account's 8K rolling TPM limit.

The last fully provider-available paced run had zero aborted calls, correct scores `3/2/1`, all
five competencies, correct probe behavior (`yes/no/no`), no distractors, and valid English/Arabic
MP3 output. It passed Answers 1 and 3 but failed Answer 2 because Diagnostic method was below its
human-labelled minimum. The final calibration clarifies that inspecting actual endpoint queries is
partial evidence gathering, and the gate now treats `covered` as satisfying a `partial` minimum.

Before that final calibration could be verified, repeated remediation runs exhausted the configured
ElevenLabs account: the latest fresh run returned 401 on Answer 1/2 TTS and then Answer 3 STT.
Those failures are intentionally preserved in `artifacts/summary.md`, `eval_*.json`, and `run.log`.
Current live verdict is therefore **blocked / not submission-ready**, despite the offline 36/36.
Required external action: replenish or replace `ELEVENLABS_API_KEY`, then rerun all three from an
isolated clean copy and require 3/3 with zero provider failures.

The credential-free clean-copy rehearsal itself completed: copied without `.git`, `.venv`, `.env`,
or caches; `uv sync --locked` created Python 3.11.15 with 69 locked packages; `uv run pytest -q`
passed 35/35; and `/health` plus `/` returned HTTP 200. The temporary copy was then removed.

The separately submitted recording is assumed to exist per the task handoff, but no video file is
present in this workspace. Mic input, transcript display, real interrupt/resume, and spoken Arabic
evaluation therefore remain independently unverified here and must be inspected before submission.

Final architecture cleanup after that run: mandatory rubric chunks were removed from the semantic
index entirely. The index now contains only five supplemental reference docs, including both
distractors, and returns top-1; the five rubric competencies remain fixed context. Historical live
rankings put a relevant supplemental doc first for each sample, but this post-credit-exhaustion
change still requires the same fresh live rerun before submission.

Repeated the clean-copy rehearsal after this final change: Python 3.11.15 locked install,
**36/36** tests, and credential-free `/health` + `/` HTTP smoke checks all passed.

## Phase 12 — Model and inference parameters chosen by measurement (2026-08-01)

Prior state: `openai/gpt-oss-20b` at a single global `reasoning_effort="low"`, with only three
inference parameters set at all. The model had been picked under duress — the 120B daily token
allowance was exhausted mid-run — and never re-evaluated on merit. `max_completion_tokens`,
`top_p` and `seed` were unset, and there was no per-stage tuning.

**What the provider actually permits** (docs used to bound the search space, never to pick a
winner). Only `openai/gpt-oss-120b` and `openai/gpt-oss-20b` support strict `json_schema`
decoding on GroqCloud; everything else offers `json_object` at best, which does not enforce our
schema. `kimi-k2*`, `qwen3-32b`, `deepseek-r1-distill-*` are retired; `llama-3.3-70b-versatile`
and `llama-3.1-8b-instant` shut down 2026-08-16; `qwen3.6-27b` is Preview and json-object-only.
`frequency_penalty`, `presence_penalty`, `logprobs`, `logit_bias` are unsupported on every Groq
model. A live canary also disproved a community report that 120B ignores strict schema — it
honoured `json_schema` + `strict` + `reasoning_effort=high` + `seed` together.

**Three findings that came from running the code, not from reading docs.**

1. **Groq charges `prompt_tokens + max_completion_tokens` against the rate-limit bucket before
   generating.** An 8192 ceiling made the *smallest* stage in the pipeline impossible: HTTP 413,
   `Requested 8658, Limit 8000`. Evidence corroborated by a TPD 429 reporting `Requested 1022`
   for a 22-token prompt with a 1000-token ceiling. `max_completion_tokens` is therefore a
   correctness and reliability constraint here, not an efficiency knob.
2. **`reasoning_effort="high"` breaks strict JSON on `analyze_coverage`.** At ceilings below
   ~6144 the model consumes the entire completion budget on reasoning and returns HTTP 400
   `json_validate_failed` with `failed_generation: ""`. Confirmed by raising the ceiling to 6144,
   at which the same call succeeds — but 1587 + 6144 reserves 97% of the 8K/minute bucket.
3. **Free tier is 200K tokens/day *per model*, 8K/minute, shared-nothing between models.** The
   earlier note claiming 20B had "an independent quota bucket" was right about TPD and wrong
   about TPM; corrected in `config.py`.

**Reasoning effort, measured in isolation.** `analyze_coverage` decides both the coverage map and
the probe, so it was run on frozen inputs (fixed transcript, summary and chunks) three times per
level against `Answer 2.mp3`, the sample whose labels sit closest to a boundary:

| effort | ceiling needed | outcome across 3 reps | golden-exact |
|---|---|---|---|
| low | 1280 | covered, partial, covered | 1/3 |
| medium | 2560 | covered, partial, covered | 1/3 |
| high | 3072 → HTTP 400 | — | — |
| high | 6144 | covered, partial, covered | 1/3 |

Identical pattern at every level. Reasoning depth does not move this task, so `low` ships on all
four stages. The residual flip is a genuinely borderline `partial`/`covered` judgement plus MoE
decoding non-determinism (Phase 3's note), not shallow reasoning.

**Model, measured on the production gate.** Three runs each, identical settings:

| | gpt-oss-20b (3 runs) | gpt-oss-120b (4 runs) |
|---|---|---|
| gate sample-passes | 6/9 | **12/12** |
| `Answer 1` / `Answer 3` scores | 3 / 1 every run | 3 / 1 every run |
| `Answer 2` score (golden 2) | 3, 2, 2 | 2, 2, 2, 3 |
| probe agreement | 9/9 | **12/12** |
| coverage agreement | 40/45 | **60/60** |
| distractors retrieved | 0 | 0 |
| JSON/schema failures | 0 | 0 |
| avg latency | **1461 ms/call** | 1740 ms/call |

20B is the faster model and is rejected anyway: it fails `Answer 2.mp3`'s coverage in all three
runs — once crediting `Async & concurrency` as `partial` where the candidate explicitly disclaimed
it, an error in the direction that would wrongly suppress a probe.

Correction worth recording, because the first three 120B runs suggested a stronger claim than the
evidence supports: a fourth run scored `Answer 2.mp3` **3** rather than 2. 120B is therefore *not*
Δ0-on-every-run, and the earlier wording to that effect was wrong. `Answer 2.mp3` sits on the 2/3
rubric boundary and both models wobble there. What separates them is that all four 120B runs stay
inside the gate (12/12, coverage 60/60) while 20B's failures are systematic, not borderline. 120B
ships on that basis, not on being the larger model.

**Ceilings sized from observation.** 2× the longest completion seen across a full gate run
(summarize 158, analyze_coverage 559, write_followup 144, score 179 — reasoning included),
rounded to a multiple of 256. This cut the reserved cost of a gate run from 36,677 to 22,341
tokens, which is what stops a run tripping 429s partway through. `provider.truncated` now fires
explicitly on `finish_reason == "length"` so a ceiling set too low is never misread as the model
returning malformed JSON.

**Also landed:** `top_p=1.0` and `seed` set explicitly; per-call token accounting
(prompt/completion/**reasoning**) on every `provider.call` log line; Groq SDK `max_retries` 2 → 5
(a 429 auto-recovered mid-run that would otherwise have failed a sample), ElevenLabs left at 2
since its failures are auth and quota; `LLM_TIMEOUT_S` 45 → 90 for reasoning-model tail latency.

**Rejected and reverted:** `reasoning_effort` above `low` at any stage (no measurable gain, and
`high` breaks strict JSON); `gpt-oss-20b` (unstable scores); generous token ceilings (no quality
effect, 64% more reserved tokens per run). A benchmark harness written for this phase was deleted
rather than shipped — it deadlocked on its own throttle and mis-modelled the token budget, and
all evidence above comes from `scripts/eval_samples.py`, the production evaluation path.

## Phase 13 — Prompt-caching audit, and a correction to Phase 12's stability claim (2026-08-01)

**Question asked:** is the project actually using Groq prompt caching, and if not, should it?

**Verified from the docs.** Groq supports prompt caching on GPT-OSS 20B, 120B and
Safeguard 20B. It "works automatically on all your API requests with no code changes required",
"is automatically enabled and cannot be manually disabled", needs no `cache_control` breakpoints,
has a 2-hour TTL, requires an exact match on the *beginning* of the prompt, and has a minimum
cacheable length of 128–1024 tokens depending on model. Hits surface as
`prompt_tokens_details.cached_tokens`. Separately, the rate-limits page states cached tokens do
not count towards rate limits — relevant because rate limits, not cost, are what bind here.

**Verified from live responses.** A 377-token probe returned no `prompt_tokens_details` at all
(below the minimum). The real `analyze_coverage` payload, replayed three times, returned
`cached_tokens: 768` of `prompt_tokens: 1481`. So caching was already active and already working
before any change — the answer to "is it unused?" is no.

`llm.complete()` now logs `cached_tokens` alongside the existing token fields, so cache behaviour
is visible from `artifacts/run.log` rather than from a throwaway probe.

**Tried and reverted: static-prefix-first prompt composition.** `analyze_coverage` and `score`
both placed the variable transcript second, leaving the rubric, competency list and calibration
sentence behind it and therefore uncacheable. Reordering so every immutable section precedes the
transcript raised the cached prefix from 768 to 1280 tokens on a replayed payload. But measured
across a real gate run the hit rate was ~2% either way (`analyze_coverage` 0%) — each sample
brings a different transcript and different retrieved chunks, so almost nothing repeats — while
`Answer 1.mp3`, a 3 in every previous run, returned 4. Reverted: no measurable benefit, and it
moved a stable score. Only the instrumentation was kept.

**Correction to Phase 12.** Phase 12 recorded 12/12 gate sample-passes over four 120B runs.
Two further runs of the byte-identical configuration bring the real figure to **17/18 sample-passes,
with five of six runs producing a clean 3/3**. The sixth failed on `Answer 3.mp3`'s
`Tradeoff & communication` returning `partial` where golden requires `missing`. All 18 scores
stayed inside golden ±1 and all 18 probe decisions were correct; the instability is confined to
competency labels. The earlier framing implied more determinism than the evidence supports. The
committed artifacts are one passing run, not a guarantee.

This does not change the model choice: 120B produced a clean gate in five of six runs, 20B in
none of three.
