Provider configuration: `{"stt": {"provider": "elevenlabs", "model": "scribe_v1"}, "llm": {"provider": "groq", "model": "openai/gpt-oss-120b", "temperature": 0.0, "top_p": 1.0, "seed": 20260801, "stages": {"summarize": {"reasoning_effort": "low", "max_completion_tokens": 512}, "analyze_coverage": {"reasoning_effort": "low", "max_completion_tokens": 1280}, "write_followup": {"reasoning_effort": "low", "max_completion_tokens": 512}, "score": {"reasoning_effort": "low", "max_completion_tokens": 512}}}, "embeddings": {"provider": "fastembed-local", "model": "BAAI/bge-small-en-v1.5"}, "tts": {"provider": "elevenlabs", "model": "eleven_multilingual_v2"}}`

| Sample | Score | Golden Δ | Probed? | Competencies judged | Distractors clean | Status |
|---|---|---|---|---|---|---|
| Answer 1.mp3 | 3 | 0 | yes | 5/5 | yes | PASS |
| Answer 2.mp3 | 2 | 0 | no | 5/5 | yes | PASS |
| Answer 3.mp3 | 1 | 0 | no | 5/5 | yes | PASS |

**3/3 samples passed all assertions.**
