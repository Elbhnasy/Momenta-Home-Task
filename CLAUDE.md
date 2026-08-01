# Voice Interview Agent — Development Guide

## Project Context

This repository implements the take-home assignment for an AI Engineer (Voice & Agentic Systems).

The goal is to build a working end-to-end voice technical interviewer capable of:

1. Asking a technical interview question using TTS.
2. Receiving a spoken answer.
3. Transcribing speech to text.
4. Deciding whether exactly one follow-up question is needed.
5. Retrieving the relevant evaluation criteria from the provided corpus.
6. Producing a structured evaluation.
7. Speaking the evaluation back using TTS.

The implementation should optimize for a working MVP, not production completeness.

The assignment prioritizes:

- Running software
- Agentic decision making
- RAG
- Voice interaction
- Clean architecture
- Readability
- Fast iteration

Avoid implementing features outside the assignment.

---

# Development Philosophy

When writing code:

- Prefer the simplest working implementation.
- Optimize for clarity over cleverness.
- Keep modules small.
- Keep responsibilities separated.
- Avoid unnecessary abstractions.
- Avoid premature optimization.
- Avoid introducing frameworks unless they simplify development.

If multiple implementations are possible:

Prefer the one that:

- reduces complexity
- is easier to debug
- is easier to explain during the walkthrough

---

# Overall Architecture

Frontend

- Static HTML/CSS/JavaScript served by FastAPI (`static/index.html`)

Backend

- FastAPI

Workflow

- LangGraph

Speech-to-Text

- ElevenLabs Scribe (`scribe_v1`), with filename and MIME preserved

Large Language Model

- Groq (`openai/gpt-oss-120b`, low reasoning, strict JSON Schema, per-stage token ceilings)

Embeddings

- Local FastEmbed `BAAI/bge-small-en-v1.5`

Vector Database

- In-process NumPy cosine retrieval (five supplemental reference documents)

Text-to-Speech

- ElevenLabs

---

# Workflow

The application must always follow this flow.

Question

↓

TTS

↓

Speech Input

↓

STT

↓

LLM coverage analysis + deterministic decision policy

↓

Follow-up?

↓

If yes:

    Ask exactly one follow-up

↓

RAG

↓

Evaluation Agent

↓

Score

↓

TTS

---

# Agent Responsibilities

## Interview Agent

Responsible only for:

- interview flow
- maintaining conversation state
- invoking STT/TTS
- sending answers to downstream agents

Must never perform evaluation.

---

## Decision Agent

The LLM produces a five-competency coverage map. `app/graph/policy.py` then decides whether a
follow-up is required from that map using deterministic, offline-tested rules.

Policy inputs

- LLM-produced coverage
- Follow-up budget used

Policy outputs

- `probe` (boolean)
- `target`
- `reason`

Never use transcript-length or keyword heuristics. The LLM supplies semantic judgment; code
supplies the branching policy.

Only one follow-up question is allowed.

---

## Evaluation Agent

Responsible only for scoring.

Inputs

- Question
- Transcript
- Retrieved rubric

Outputs

- Score (1-5)
- Covered competencies
- Missing competencies
- One-line justification

Evaluation must always use retrieved context.

Never rely on model knowledge alone.

---

# RAG Rules

Always retrieve relevant context before evaluation.

Retrieved rubric has priority over model knowledge.

Always evaluate all five question-relevant rubric competencies. Semantic retrieval selects only
supplemental reference material; answer similarity must never narrow evaluation scope.

Do not inject unrelated documents.

Prefer semantic retrieval over keyword matching.

---

# Prompt Rules

Every prompt must live inside:

prompts/

Never hardcode prompts inside Python files.

Each prompt should perform exactly one task.

Use structured JSON outputs whenever possible.

Avoid prompts requesting chain-of-thought.

---

# Language Rules

Automatically detect the candidate language.

Support:

- English
- Arabic
- Egyptian Arabic mixed with English technical terms

If the candidate primarily answers in Arabic:

Generate the evaluation directly in Arabic.

Do not translate English output afterwards.

---

# Folder Responsibilities

static/

Contains the static browser UI only.

No business logic.

---

app/

FastAPI application.

REST endpoints only.

---

app/graph/

LangGraph nodes.

No UI logic.

---

app/providers/

External services.

Examples:

- STT
- TTS
- LLM
- Embeddings

---

app/rag/

Chunking

Embeddings

Retrieval

Vector Store

---

prompts/

Prompt templates only.

---

data/corpus/

Provided evaluation corpus.

Never modify.

---

data/audio/

Provided sample answers.

Treat them as regression tests.

---

tests/

Quality gate.

Regression tests.

---

# Coding Standards

Use:

- Python 3.11+

Always:

- use type hints

- write small functions

- prefer dependency injection

- keep async where appropriate

- isolate external APIs

Avoid:

- global mutable state

- duplicated prompts

- duplicated retrieval logic

---

# API Rules

All external providers must be wrapped behind services.

Never call providers directly from UI.

Never mix provider-specific logic with business logic.

---

# Error Handling

Every external API call should:

- retry when appropriate
- return useful errors
- fail gracefully

The application should continue operating whenever possible.

---

# Logging

Log:

- STT duration

- Retrieval duration

- LLM latency

- Final score

Never log:

- API keys

- secrets

---

# Quality Gate

Every completed feature must be validated using all provided sample audio files.

Verify:

✓ transcript quality

✓ follow-up decision

✓ retrieved chunks

✓ evaluation score

✓ spoken output

A feature is not complete unless all samples still pass.

---

# Docker

Docker is optional for this assignment and is not implemented. The verified run contract is the
`uv` workflow documented in `README.md`.

---

# Definition of Done

A task is complete only if:

✓ the documented `uv` run contract works

✓ README is updated

✓ Prompts remain synchronized

✓ Existing functionality still works

✓ Sample audio files still evaluate correctly

✓ No duplicated code was introduced

✓ Architecture remains clean

---

# Non Goals

Unless explicitly requested, do NOT implement:

- Authentication
- User accounts
- Streaming audio
- WebRTC
- Deployment
- Database persistence
- Analytics dashboards
- Multi-user support
- Long-term memory
- Agent autonomy beyond this assignment

---

# AI Collaboration Guidelines

When modifying this repository:

1. Read the existing implementation before generating code.

2. Reuse existing modules whenever possible.

3. Preserve folder boundaries.

4. Explain architectural trade-offs before introducing major changes.

5. Prefer incremental improvements over large rewrites.

6. If a requirement is ambiguous, make a reasonable assumption and document it.

7. Always preserve a working application after each change.
