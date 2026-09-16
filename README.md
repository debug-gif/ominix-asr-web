# ominix-asr-web

Real-time speech recognition WebUI on Apple Silicon (M1/M2/M3/M4), built in pure Rust on Apple's [MLX](https://github.com/ml-explore/mlx) framework.

**Pipeline**: browser microphone → WebSocket → VAD segmentation → Qwen3-ASR-0.6B transcription → terminology correction → Qwen3-1.7B LLM polishing → fluent text.

## Features

- **Real-time ASR** — Qwen3-ASR 0.6B (8-bit), ~10-20x realtime on M4
- **Auto language detection** — 30+ languages, no need to pick (manual override available)
- **Audio source selection** — microphone (per-device picker) or system audio (screen/tab capture via `getDisplayMedia`)
- **AI summary** — on-demand transcript summarization (core topics, key points, conclusions, todos) via Qwen3-4B-4bit, lazily loaded and idle-unloaded
- **Subtitle mode** — detached floating subtitle window (desktop-lyrics style): large translation/polished text + small original text; adjustable font size, box size, background opacity, and position; Chrome supports always-on-top Picture-in-Picture window
- **Three-column UI** — Codex-style layout: left console (source/device/language/enhance/domain), middle live transcript, right AI summary & export
- **AI enhancement (same model, exclusive modes)** — Qwen3-1.7B-4bit powers both:
  - **润色 (polish)**: removes filler words & repetitions, fixes homophone/transcription errors by context
  - **实时翻译 (translate)**: translates transcriptions to a target language in real time (中/英/日/韩/德/法/西/俄)
  - modes are mutually exclusive: enabling translation disables polishing
- **Domain optimization** — `domain.txt` config: system prompt injection + terminology fuzzy correction (edit the file, hot-reloads within 5s)
- **Transcript saving** — manual save / auto-save to `transcripts/`, download in browser
- **Idle model unload** — both models are unloaded from memory after 10 min of silence (frees ~1.2 GB), auto-reloaded in ~1s when speech resumes

## Requirements

- macOS with Apple Silicon
- Rust toolchain (`rustup`): `rustup default stable`
- No Python, no Xcode required — prebuilt MLX static libraries are downloaded automatically by the build script

## Setup

```bash
# 1. Clone
git clone https://github.com/<you>/ominix-asr-web.git
cd ominix-asr-web

# 2. Download models (use hf-mirror.com if huggingface.co is unreachable)
mkdir -p ~/.OminiX/models/qwen3-asr-0.6b
curl -L -o config.json https://hf-mirror.com/mlx-community/Qwen3-ASR-0.6B-8bit/resolve/main/config.json ...
# (see models.sh)

# 3. Build
cargo build --release -p ominix-asr-web

# 4. Run
./target/release/ominix-asr-web
# WebUI auto-opens at http://localhost:8080 after 5s
```

## Domain configuration (`domain.txt`)

Created automatically on first run. Edit with any text editor; changes apply automatically within 5 seconds.

```
[system]
你正在转写计算机科学、机器学习领域的语音内容，优先采用专业术语。

[terms]
神经网络
梯度下降
神经网路|神经网络   # "常见错误|正确写法" pair
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OMINIX_ASR_MODEL` | `~/.OminiX/models/qwen3-asr-0.6b` | ASR model dir |
| `OMINIX_POLISH_MODEL` | `~/.OminiX/models/qwen3-1.7b-4bit` | Polish/translate LLM dir |
| `OMINIX_SUMMARY_MODEL` | `~/.OminiX/models/qwen3-4b-4bit` | Summary LLM dir |
| `OMINIX_SUMMARY` | `1` | `0` disables AI summary |
| `OMINIX_POLISH` | `1` | `0` disables AI polishing |
| `OMINIX_PORT` | `8080` | HTTP port |
| `OMINIX_IDLE_TIMEOUT` | `600` | Idle seconds before unloading models (0 = never) |
| `OMINIX_LANGUAGE` | `Chinese` | Default language |
| `OMINIX_TRANSCRIPTS_DIR` | `transcripts/` | Transcript save dir |
| `OMINIX_DOMAIN_FILE` | `domain.txt` | Domain config file |
| `OMINIX_MLX_MEMORY_LIMIT_MB` | `4096` | MLX active-memory soft limit (MB); exceeded → cache auto-freed |
| `OMINIX_MLX_PURGE_THRESHOLD_MB` | `768` | Purge MLX cache + compile cache when cached memory exceeds this (MB) |
| `OMINIX_AUTO_OPEN` | `1` | `0` disables auto-opening browser |
| `MLX_PREBUILT_PATH` | (auto-download) | Dir with `libmlx.a`, `libmlxc.a`, `mlx.metallib` |

## API

| Method | Path | Description |
|---|---|---|
| GET | `/` | WebUI |
| WS | `/ws` | Binary PCM f32 16kHz mono → transcription; text JSON control messages |
| GET | `/api/status` | Model/idle/segment status |
| GET | `/api/domain` | Current domain config |
| POST | `/api/save` | Save transcript to file |
| POST | `/api/unload` | Manually unload models |
| POST | `/api/clear` | Clear current transcript |
| GET | `/api/download` | Download last saved file |

## Architecture

```
Browser (mic capture + resample to 16k) 
  → WebSocket binary frames
  → VAD (adaptive energy threshold, 1s silence split, 12s cap)
  → ASR worker thread: Qwen3-ASR-0.6B (Metal GPU)
  → term correction (Levenshtein fuzzy match)
  → Polish worker thread: Qwen3-1.7B-4bit (rewrite to fluent text)
  → UI + transcript files
```

Models run on dedicated worker threads (MLX types are `!Send`); idle watchdog unloads both models and calls `mlx_clear_cache()` to release memory (~1.2 GB → ~100 MB).

## Credits

- [OminiX-MLX](https://github.com/OminiX-ai/OminiX-MLX) — ASR/LLM crates (Apache-2.0)
- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) & [Qwen3](https://github.com/QwenLM/Qwen3) — models
- [mlx-rs](https://github.com/oxideai/mlx-rs) — Rust bindings for MLX
- [mlx-community](https://huggingface.co/mlx-community) — quantized MLX model conversions
