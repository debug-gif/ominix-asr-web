# ominix-asr-web

基于 Qwen3-ASR (MLX) 的实时语音识别 WebUI。浏览器采集麦克风音频，经 WebSocket 发送到 Rust 服务器实时转写，支持转写保存与模型空闲自动卸载。

## 功能

- **实时转写**：浏览器录音 → 服务器 VAD 分段 → Qwen3-ASR 转写 → WebSocket 实时推送（含部分结果）
- **转写保存**：手动保存 / 自动保存（前端每 60 秒），保存为 `transcripts/转写_YYYYMMDD_HHMMSS.txt`，支持下载
- **模型空闲卸载**：无音频活动超过指定时间（默认 600 秒）自动从内存卸载模型（约释放 1GB），下次说话自动重新加载（约 1 秒）
- **多语言**：中文、英语、日语、韩语、粤语等 30+ 语言（与 Qwen3-ASR 相同）
- **领域优化**：配置文件 `domain.txt`（启动时自动创建），`[system]` 段注入领域提示词引导专业词汇，`[terms]` 段术语表模糊纠错（支持「常见错误|正确写法」配对）；用文本编辑器修改后 **5 秒内自动热加载**，WebUI 中为只读预览
- **AI 润色纠错**：转写后用 Qwen3-1.7B-4bit 做二段推理——去口语词/填充词、合并重复表达、按上下文修正转写错误，输出流畅讲稿文本（可在界面开关，`OMINIX_POLISH=0` 禁用）

## 运行

```bash
cargo build --release -p ominix-asr-web
./target/release/ominix-asr-web
```

浏览器打开 http://localhost:8080（麦克风权限需要 localhost 或 https 访问）。

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `OMINIX_ASR_MODEL` | `~/.OminiX/models/qwen3-asr-0.6b` | 模型目录 |
| `OMINIX_PORT` | `8080` | HTTP 端口 |
| `OMINIX_IDLE_TIMEOUT` | `600` | 空闲多少秒后卸载模型（0 = 关闭） |
| `OMINIX_LANGUAGE` | `Chinese` | 默认语言 |
| `OMINIX_TRANSCRIPTS_DIR` | `transcripts`（绝对路径） | 转写保存目录 |
| `OMINIX_DOMAIN_FILE` | `domain.txt`（绝对路径） | 领域优化配置文件 |
| `OMINIX_POLISH_MODEL` | `~/.OminiX/models/qwen3-1.7b-4bit` | 润色模型目录 |
| `OMINIX_POLISH` | `1` | `0` 禁用 AI 润色 |

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | WebUI |
| WS | `/ws` | 二进制 PCM f32 16kHz 单声道 → 转写；文本 JSON 控制消息 |
| GET | `/api/status` | 模型/空闲/段落状态 |
| POST | `/api/save` | 保存转写到文件 |
| POST | `/api/unload` | 手动卸载模型 |
| POST | `/api/clear` | 清空当前会话转写 |
| GET | `/api/download` | 下载最近一次保存的文件 |

## 技术说明

- 模型运行在独立 ASR 工作线程（MLX 类型非 Send），与 HTTP/WS 完全解耦
- 卸载时调用 `mlx_clear_cache()` 释放 MLX 内存池，内存从约 1.1GB 降至约 100MB
- VAD：自适应能量阈值，1 秒静音断句，最长 12 秒强切（保留 0.5 秒尾部重叠），每 4 秒输出部分结果
