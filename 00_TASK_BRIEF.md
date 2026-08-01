# AI Engineer — Take-Home Task: Build a Running Voice-Agent MVP

**Role:** AI Engineer (Voice & Agentic Systems)
**Time box:** A focused evening's work — roughly **3–4 hours** once you're set up. Please don't spend more. A running-but-rough loop beats a polished half.
**Format:** Async build, then a 30–45 min live walkthrough.

---

## Context

Assume we build voice-driven AI agents for enterprise client. This task asks you to stand up a **running MVP** of one: a voice agent that runs a short technical screening over a small document set, end to end.

We want to see you take a product-shaped problem and ship something that actually works, fast — using coding agents to move across the whole stack.

---

## Starting point

**Begin from whatever you like.** Use any open-source template, starter kit, or scaffold you're comfortable with — we're testing how you *integrate a working loop*, not whether you can hand-write a mic-capture UI or wire an STT client from scratch. Reaching for the right template quickly is part of the skill.

## Setup

Use any STT, TTS, and LLM providers you like (OpenAI, Google, Azure, Deepgram, ElevenLabs, Anthropic, etc.). Use your own free-tier or personal API keys — the task fits comfortably within free/low-cost tiers.

---

## What to build

A **running** app with a UI, a backend, and an agentic voice loop:

1. **A UI that runs** — anything works: a simple web page, Streamlit / Gradio, or even a CLI plus an audio player. The user speaks in, sees the transcript, and hears the agent reply. Polish is explicitly not graded.
2. **The agent asks a question by voice** (one is provided; TTS it out).
3. **The user answers by voice** → you transcribe it (two of our test answers are Egyptian Arabic mixed with English technical terms — handle that).
4. **The agent decides** — based on the answer, does it ask **one** clarifying follow-up, or move on? This branch is the point: we want a real decision, not a linear script.
5. **It retrieves** the relevant criteria from the provided rubric corpus (RAG).
6. **It speaks back** a short scored evaluation (1–5 + one-line justification) **in the input language** (Arabic in → Arabic out).
7. Finally, if any of the requirements above are unclear, proceed with reasonable assumptions and ensure they are clearly documented.

### Definition of done
- From a clean checkout, we can run it with your documented steps in a few minutes.
- The full voice loop works live from the mic (or your chosen input).
- The provided sample answers each produce an evaluation.

---

## You are expected to use a coding agent

Use Claude Code, Cursor, or whatever you actually work in. Fluency with coding agents is part of the role, and part of how you'll hit this quickly. We want to see how you drive one across a full stack.

---

## What to submit

1. **The running repo** — with a `README` that tells us exactly how to run it. If we can't get it running quickly, that counts against you.
2. **A 2–3 minute screen recording** of the loop working end to end (talk → transcript → agent decision → spoken evaluation). Required — it proves it runs.
3. **Your agent artifacts** — the `CLAUDE.md` / `.cursorrules` / config you set up, plus a log or export of the key prompts you used. We read these.
4. **A small quality gate (required):** run the provided sample answers through the pipeline, show the evaluations, and write **2–3 sentences on how you'd know if a change made the system worse.** Keep it short — a sanity check, not a research project.
5. **A short README section** covering: architecture, how the agent decides to follow up, what you cut to fit the time, and what you'd build next.

---

## How we evaluate

In order of what matters most:

1. **It runs** — the full loop works end to end from a clean checkout. For a ship-fast role, this is non-negotiable.
2. **Agentic design** — a real decision in the loop (the follow-up branch), not a straight-line script.
3. **Coding-agent fluency** — how you set up the agent's context and drove it to integrate a working loop quickly.
4. **Quality gate** — sane retrieval, and evidence you'd notice if the system regressed.
5. **Voice & communication** — Arabic / code-mixed handling, a working speak-back, and a README/walkthrough we can follow.

We do **not** care about visual polish, auth, or deployment. Spend the time on a working loop.

---

## Live walkthrough (after submission)

30–45 minutes, screen-share:
- Run it for us and walk through your architecture and one tradeoff.
- We'll ask you to make one small change live, using your agent, so we can see how you work.

Come ready to defend your decisions — that's the interesting part.
