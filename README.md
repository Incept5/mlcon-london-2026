# MLCon London 2026

Demo scripts accompanying a two-day workshop on practical LLM application development at MLCon London 2026. The scripts walk through calling commercial and local model providers, building embeddings and RAG pipelines, function/tool calling, MCP, vision, OCR, scraping, and text-to-speech.

## Repository layout

```
day_1_scripts/         Day 1: model providers, prompting, embeddings, vision, audio
day_1_scripts/speech/  Day 1 TTS/STT demos (mlx-audio)
day_2_scripts/         Day 2: RAG, tool calling, MCP, vision, audio
day_2_scripts/data/    Sample texts, images, and audio used by the demos
day_2_scripts/MCP/     Self-contained MCP server + Ollama client demo
MLCon London 2026 - Hands-on GenAI Development Bootcamp Day 1.pdf   Day 1 slides
requirements.txt
```

### Day 1 — talking to models

- `basic_claude.py`, `basic_chatGPT.py`, `basic_groq.py`, `basic_mistral.py`, `basic_together.py`, `basic_fireworks.py`, `basic_grok.py` — minimal "hello world" against each provider.
- `getting_started_ollama.py`, `getting_started_ollama_params.py`, `getting_started_lm_studio.py`, `getting_started_groq.py` — first calls against local and hosted runtimes, including a variant that exposes sampling parameters.
- `three_local_backends.py`, `local_performance_demo.py` — comparing local inference backends.
- `embedding_demo.py`, `embedding_example.py`, `word_embeddings.py`, `3d_plot.html` — embeddings, similarity, and visualisation.
- `logit_probabilities.py`, `simple_token_test.py`, `icr_demo.py` — tokens, logits, and intermediate model state. `logit_probabilities.py` runs Qwen3 in no-think chat mode and visualises top-k token probabilities.
- `vision_demo_ollama.py` (with `image.jpg`) — image understanding via Ollama (`qwen3.5:4b`).
- `speech/tts.py`, `speech/tts-morgan.py`, `speech/stt.py` — text-to-speech and speech-to-text demos using `mlx-audio`, with a voice-cloning sample (`morgan-freeman-voice-sample.wav`).
- `ai_astrology.py`, `ai_astrology_groq.py`, `together_chat.py` — slightly larger end-to-end prompts.

### Day 2 — building things with models

- **RAG** — `rag_alice_simple.py`, `rag_alice_in_wonderland.py`, `rag_alice_in_wonderland_chromadb.py`, `rag_alice_in_wonderland_transformers.py`, `rag_grimm_fairy_tales.py`, `rag_grimm_fairy_tales_groq.py`, `grimm_fairy_tales_rag_demo.py`, `alice_in_one_go.py`.
- **Sentiment & data extraction** — `analyse_sentiment_01.py`, `analyse_sentiment_02.py`, `analyse_sentiment_kaggle.py`, `kaggle_summary_complete.py`, `data_extraction_ollama.py`, `data_extraction_groq.py`, `formatted_response_example.py`.
- **Tool / function calling** — `simple_tool_call.py`, `ollama_function_support.py`, `find_beer.py`, `payroll.py`, `payroll2.py`, `payroll2_groq.py`.
- **MCP** — see `day_2_scripts/MCP/README.md` for a self-contained server + agentic client demo.
- **Vision & OCR** — `visual_ollama.py`, `visual_ml_studio.py`, `test_lmstudio_vision.py`, `read_diagram.py`, `read_music.py`, `read_ocr_menu.py`.
- **Scraping** — `scrape.py`, `scrape_gdpr_article.py`.
- **Audio** — `chatterbox_tts.py` (output written to `day_2_scripts/output/`).

## Getting set up

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Several scripts read API keys from a `.env` file via `python-dotenv`. Create one in the repo root with whichever providers you intend to use:

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GROQ_API_KEY=...
MISTRAL_API_KEY=...
TOGETHER_API_KEY=...
FIREWORKS_API_KEY=...
XAI_API_KEY=...
```

The local-model demos expect either:

- [Ollama](https://ollama.ai) running on `http://localhost:11434` — pull models referenced in the script (e.g. `ollama pull qwen2.5:3b`, `ollama pull all-minilm`, `ollama pull embeddinggemma`).
- [LM Studio](https://lmstudio.ai) for the `lm_studio` / `visual_ml_studio` scripts.

## Running a demo

Each script is standalone. From the repo root:

```bash
python day_1_scripts/basic_claude.py
python day_2_scripts/rag_alice_simple.py
```

Demos that load sample data (texts, images, audio) read from `day_2_scripts/data/`, so run them from the repo root or `day_2_scripts/` so relative paths resolve.

## Notes

- `TOGETHER_KEY.txt` is gitignored; prefer `.env` for credentials.
- Generated artefacts (e.g. `day_1_scripts/embeddings.pkl`) are gitignored via `*.pkl` — they are produced on first run.
- The MCP demo has its own README with architecture diagram and troubleshooting tips: `day_2_scripts/MCP/README.md`.
- Slides for Day 1 are in the repo root as `MLCon London 2026 - Hands-on GenAI Development Bootcamp Day 1.pdf`.
