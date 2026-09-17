# ominix-asr-web

Real-time speech recognition + AI enhancement WebUI on Apple Silicon (M1/M2/M3/M4), built in pure Rust on Apple's [MLX](https://github.com/ml-explore/mlx) framework.

**Pipeline**: browser audio → WebSocket → VAD segmentation → Qwen3-ASR-0.6B transcription → terminology correction → Qwen3-1.7B enhancement (polish/translate) → Qwen3-4B summary → UI / subtitle / files.

## Features

### 语音识别
- **Real-time ASR** — Qwen3-ASR 0.6B (8-bit), ~10-20x realtime on M4
- **Auto language detection** — 30+ languages, no need to pick (manual override available)
- **Audio source selection** — microphone (per-device picker) or system audio (screen/tab capture via `getDisplayMedia`)
- **Domain optimization** — `domain.txt` config: system prompt injection + terminology fuzzy correction (edit the file, hot-reloads within 5s)

### AI 增强（按需加载，空闲自动卸载）
- **润色 (polish)** — Qwen3-1.7B-4bit: removes filler words & repetitions, fixes homophone/transcription errors by context
- **实时翻译 (translate)** — same model, 8 target languages; mutually exclusive with polishing
- **AI 摘要 (summary)** — Qwen3-4B-4bit, 按需加载：
  - **类型**：会议摘要（结尾为**待办事项**）/ 内容摘要（结尾为**综合反思**）
  - **方式**：原文摘要 / 翻译摘要（8 种目标语言）/ 两者都生成
  - 结构：核心主题、关键要点、结论或决定、待办事项 / 综合反思
- **Token 速度统计** — generation speed (tok/s) in terminal log and UI badge

### 界面与输出
- **Three-column layout** (Codex style) — left console / middle live transcript / right summary & export; **draggable dividers** resize columns (persisted)
- **Subtitle mode** — detached semi-transparent subtitle window: large translated/polished text + small original text; adjustable font size, box width, opacity, position; Chrome supports always-on-top Picture-in-Picture
- **Newest-on-top** transcript & summary lists
- **Saving** — transcripts and summaries saved to `transcripts/` (manual / auto-save), downloadable in browser; **export mode**: 原文+翻译 / 仅原文 / 仅翻译

### 性能与稳定性
- **MLX memory management** — soft memory limit, automatic cache + graph-compile-cache purge on threshold, live memory badge
- **Idle model unload** — models are unloaded after 10 min of silence (frees ~1.2 GB with ASR only, more with all three), auto-reloaded when needed
- **Thread-safe MLX access** — global lock serializes MLX usage across worker threads (MLX C++ is not thread-safe); enhancement runs async so the audio pipeline never blocks

## Requirements

- macOS with Apple Silicon
- Rust toolchain (`rustup`): `rustup default stable`
- No Python, no Xcode required — prebuilt MLX static libraries are downloaded automatically by the build script

## Setup

```bash
# 1. Clone
git clone https://github.com/debug-gif/ominix-asr-web.git
cd ominix-asr-web

# 2. Download models (~4.2 GB total; hf-mirror.com by default)
./models.sh

# 3. Build
cargo build --release -p ominix-asr-web

# 4. Run
./target/release/ominix-asr-web
# WebUI auto-opens at http://localhost:8080 after a 5-4-3-2-1 countdown
```

Models: `Qwen3-ASR-0.6B-8bit` (ASR), `Qwen3-1.7B-4bit` (polish/translate), `Qwen3-4B-4bit` (summary). Any of them can be disabled (see env vars) — the corresponding feature is hidden automatically.

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
| `OMINIX_POLISH` | `1` | `0` disables polish/translate |
| `OMINIX_SUMMARY` | `1` | `0` disables AI summary |
| `OMINIX_PORT` | `8080` | HTTP port |
| `OMINIX_IDLE_TIMEOUT` | `600` | Idle seconds before unloading models (0 = never) |
| `OMINIX_LANGUAGE` | `Chinese` | Default language |
| `OMINIX_TRANSCRIPTS_DIR` | `transcripts/` | Transcript/summary save dir |
| `OMINIX_DOMAIN_FILE` | `domain.txt` | Domain config file |
| `OMINIX_MLX_MEMORY_LIMIT_MB` | `4096` | MLX active-memory soft limit (MB); exceeded → cache auto-freed |
| `OMINIX_MLX_PURGE_THRESHOLD_MB` | `768` | Purge MLX cache + compile cache when cached memory exceeds this (MB) |
| `OMINIX_DEREVERB` | `1` | `0` disables segment-level spectral enhancement (late-reverb suppression + loudness normalization) |
| `OMINIX_AUTO_OPEN` | `1` | `0` disables auto-opening browser |
| `MLX_PREBUILT_PATH` | (auto-download) | Dir with `libmlx.a`, `libmlxc.a`, `mlx.metallib` |

## API

| Method | Path | Description |
|---|---|---|
| GET | `/` | WebUI (three-column) |
| GET | `/subtitle` | Detached subtitle window |
| WS | `/ws` | Binary PCM f32 16kHz mono → transcription; JSON control messages |
| GET | `/api/status` | Models / idle / memory / segment status |
| GET | `/api/domain` | Current domain config |
| POST | `/api/save` | Save transcript; body `{"mode":"both\|original\|enhanced"}` |
| POST | `/api/save_summary` | Save summaries to a separate file |
| POST | `/api/unload` | Manually unload all models |
| POST | `/api/clear` | Clear current transcript |
| GET | `/api/download` | Download last saved file |

### WebSocket control messages

| Message | Description |
|---|---|
| `{"type":"lang","lang":"Auto"}` | Set recognition language |
| `{"type":"enhance","mode":"polish\|translate\|off","target":"English"}` | Enhancement mode |
| `{"type":"summarize","mode":"source\|translate\|both","target":"English","stype":"meeting\|content"}` | Generate summary (stype: meeting→action items / content→reflection) |
| `{"type":"clear_summary"}` | Clear summaries |
| `{"type":"reload_domain"}` | Reload `domain.txt` now |
| `{"type":"save"}` / `{"type":"unload"}` / `{"type":"clear"}` | Same as REST |

## Architecture

```
Browser (mic/system audio capture + resample to 16k)
  → WebSocket binary frames
  → VAD (adaptive energy threshold, 1s silence split, 12s cap, 2s partials)
  → ASR worker thread:      Qwen3-ASR-0.6B      (Metal GPU)
  → term correction (Levenshtein fuzzy match)
  → Enhancement worker:     Qwen3-1.7B-4bit     (polish / translate, async)
  → Summary worker:         Qwen3-4B-4bit       (on demand)
  → UI + subtitle window (BroadcastChannel) + transcript files
```

- Models live on dedicated worker threads (MLX types are `!Send`); a global `mlx_lock` serializes all MLX calls because MLX's C++ runtime is **not** thread-safe (concurrent use segfaults in `compile_fuse`)
- Idle watchdog unloads every model and calls `mlx_clear_cache()` + `mlx_detail_compile_clear_cache()` to actually return memory to the OS
- Enhancement/summary run asynchronously: raw transcription is pushed to the UI immediately, enhanced text follows when ready

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## Credits

- [OminiX-MLX](https://github.com/OminiX-ai/OminiX-MLX) — ASR/LLM crates (Apache-2.0)
- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) & [Qwen3](https://github.com/QwenLM/Qwen3) — models
- [mlx-rs](https://github.com/oxideai/mlx-rs) — Rust bindings for MLX
- [mlx-community](https://huggingface.co/mlx-community) — quantized MLX model conversions
