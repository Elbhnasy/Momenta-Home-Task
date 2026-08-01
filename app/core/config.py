from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    GROQ_API_KEY: str | None = None
    ELEVENLABS_API_KEY: str | None = None


_settings = Settings()

GROQ_API_KEY = _settings.GROQ_API_KEY
ELEVENLABS_API_KEY = _settings.ELEVENLABS_API_KEY

# --- STT (ElevenLabs Scribe) ---
# Deepgram (nova-3 and nova-2, detect_language=True) was benchmarked first and dropped:
# on the two Arabic-heavy samples (~37s of real speech each) it returned 2-4 words at
# plausible-looking confidence, having silently misidentified the language as English
# and discarded almost everything else. Re-benchmarked against Groq Whisper
# (whisper-large-v3 / -turbo) and ElevenLabs Scribe; see data/docs/PROMPT_LOG.md for the
# full comparison. Scribe won on transcript quality and needs no new credential — we
# already hold ELEVENLABS_API_KEY for TTS, so this also drops Deepgram as a dependency
# entirely (3 credentials -> 2).
ELEVENLABS_STT_MODEL = "scribe_v1"
# Shared vocabulary hint: boosts recognition of domain terms for both STT (as `keyterms`)
# and originally documented here for STT only; kept as one list, not duplicated.
STT_VOCAB_HINT = [
    "N+1 queries", "thread pool", "AsNoTracking", "EF Core", "IQueryable",
    "async/await", "indexes", "profiler", "APM", "Redis",
    "ConfigureAwait", "CancellationToken",
]

# --- LLM (Groq, via the OpenAI-compatible client) ---
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# openai/gpt-oss-120b and openai/gpt-oss-20b are the only two models on GroqCloud that
# support *strict* json_schema decoding (everything else is json_object at best, which
# guarantees valid JSON but not our schema). Every other candidate we compared is gone or
# going: kimi-k2*, qwen3-32b and deepseek-r1-distill-* are retired, llama-3.3-70b-versatile
# and llama-3.1-8b-instant shut down 2026-08-16, and qwen3.6-27b is Preview with json_object
# only. So the real choice is 120B vs 20B; see README "Inference configuration".
GROQ_MODEL = "openai/gpt-oss-120b"

# Greedy decoding. Sampling variance in summarize() otherwise perturbs claims_summary_en
# wording enough to flip razor-thin retrieval rankings between identical inputs — see
# data/docs/PROMPT_LOG.md Phase 3.
GROQ_TEMPERATURE = 0.0
# Groq's guidance is to alter temperature *or* top_p, never both. At temperature 0 the
# distribution is already collapsed, so top_p must stay at 1.0; set explicitly so the
# intent is visible rather than inherited.
GROQ_TOP_P = 1.0
# Groq documents seed as best-effort deterministic sampling for repeated identical
# requests. Run-to-run stability is a gate assertion here, so pinning it can only help.
GROQ_SEED = 20260801
# NOT SET, deliberately: frequency_penalty, presence_penalty, logprobs and logit_bias are
# documented by Groq as "not yet supported by any of our models". service_tier stays at the
# default on_demand — flex deprioritises requests and performance needs a paid tier.
# Streaming and tool use are incompatible with structured outputs; neither is used.

# Per-call-site inference profile. reasoning_effort and max_completion_tokens are not
# global properties of the model — they are properties of the *task*, and the four LLM
# stages in this pipeline have very different shapes: two are mechanical rewrites, two
# carry the actual judgement that decides a candidate's grade. Values are measured, not
# guessed; see artifacts/bench_reasoning.md for the run that chose each one.
#
# max_completion_tokens is squeezed from both sides here, which is why it is set per stage
# rather than once:
#   Floor — gpt-oss bills reasoning tokens inside completion_tokens, so the ceiling must
#   cover reasoning *plus* the JSON body. Under strict decoding a truncated response is a
#   hard failure (finish_reason="length" -> invalid JSON), so each value is sized against
#   the observed worst case, not the mean.
#   Ceiling — Groq reserves `prompt_tokens + max_completion_tokens` against the TPM bucket
#   *before* generating, and rejects the request with a non-retryable HTTP 413 if that sum
#   exceeds the limit. On this free-tier 8,000 TPM account an over-generous ceiling does not
#   waste budget, it makes the call impossible: 8192 here produced
#   "Requested 8658, Limit 8000" on the smallest stage in the pipeline.
# So the value has to be tight enough to fit beside the prompt and loose enough to never
# truncate. Measured per stage; see artifacts/bench_reasoning.md.
# Ceilings are 2x the longest completion observed across a full gate run on all three golden
# samples (summarize 158, analyze_coverage 559, write_followup 144, score 179 — reasoning
# included), rounded up to a multiple of 256. That 2x margin is the headroom against an input
# the golden set does not contain; provider.truncated fires loudly if it is ever wrong.
# Sizing these from measurement rather than leaving them generous cut the reserved cost of a
# gate run from 36,677 tokens to 22,341 (-39%), which is what stops the run tripping 429s
# against the 8K/minute bucket partway through.
LLM_STAGES: dict[str, dict] = {
    "summarize":        {"reasoning_effort": "low", "max_completion_tokens": 512},
    "analyze_coverage": {"reasoning_effort": "low", "max_completion_tokens": 1280},
    "write_followup":   {"reasoning_effort": "low", "max_completion_tokens": 512},
    "score":            {"reasoning_effort": "low", "max_completion_tokens": 512},
}
# Used when a caller does not name a stage (unit tests, ad-hoc probes). Never hit in the
# graph: all four judge.py call sites pass an explicit stage.
LLM_STAGE_FALLBACK = {"reasoning_effort": "low", "max_completion_tokens": 1280}

# --- TTS (ElevenLabs) ---
ELEVENLABS_TTS_MODEL = "eleven_multilingual_v2"
# Chosen in Phase 4's listening gate: intelligible Egyptian Arabic, "caching"/"async" stayed in
# English, accent acceptable. See data/docs/PROMPT_LOG.md for the listen notes.
ELEVENLABS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"

# --- Embeddings (fastembed, local, ONNX — no hosted embedding endpoint from either provider above) ---
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# --- Retrieval ---
# The five rubric chunks are mandatory context loaded directly by judge.py, never retrieved.
# Semantic search ranks only the five supplemental reference docs (three relevant, two planted
# distractors). One best supplemental hit is enough because retrieval no longer controls scope;
# historical golden rankings place a relevant reference first for all three samples.
RETRIEVAL_K = 1

# --- Provider timeouts and retries ---
# Measured over the golden samples: STT 3.4-7.3s, LLM 0.9-16.2s (the tail includes the SDK's
# own retries), TTS 1.2-2.3s. These ceilings sit well above the observed maxima and far below
# the SDK defaults (OpenAI 600s, ElevenLabs 240s): an unreachable provider should fail a turn,
# not pin a worker thread for ten minutes. The LLM ceiling carries extra headroom because a
# reasoning model's latency scales with reasoning_effort, and the tail matters more than the
# median when the alternative is a spurious 503.
LLM_TIMEOUT_S = 90.0
ELEVENLABS_TIMEOUT_S = 60.0
# The configured Groq account is free tier: 8,000 tokens/minute, and the bucket is shared
# across models (x-ratelimit-remaining-tokens reads identically for 120B and 20B, so the
# earlier assumption that 20B had an independent quota was wrong). A full drain needs ~60s
# to refill, which the SDK's default 2 retries cannot span. The SDK honours the Retry-After
# header on 429, so raising the count is enough — no hand-rolled backoff.
LLM_MAX_RETRIES = 5

# --- Uploads ---
MAX_AUDIO_BYTES = 25 * 1024 * 1024
