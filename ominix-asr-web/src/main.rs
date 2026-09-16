mod domain;
mod mem;
mod polish;
mod summary;
mod worker;

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use chrono::Local;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use domain::{file_mtime, load_domain_file, make_absolute, DomainConfig};
use polish::PolishWorker;
use summary::SummaryWorker;
use worker::{idle_secs, touch, AsrWorker};

// ── Settings ────────────────────────────────────────────────────

struct Settings {
    model_dir: PathBuf,
    polish_model_dir: PathBuf,
    summary_model_dir: PathBuf,
    domain_file: PathBuf,
    port: u16,
    idle_timeout_secs: u64,
    transcripts_dir: PathBuf,
    language: String,
    mlx_memory_limit_mb: usize,
    mlx_purge_threshold_mb: usize,
}

impl Settings {
    fn from_env() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        Settings {
            model_dir: PathBuf::from(
                std::env::var("OMINIX_ASR_MODEL")
                    .unwrap_or_else(|_| format!("{home}/.OminiX/models/qwen3-asr-0.6b")),
            ),
            polish_model_dir: PathBuf::from(
                std::env::var("OMINIX_POLISH_MODEL")
                    .unwrap_or_else(|_| format!("{home}/.OminiX/models/qwen3-1.7b-4bit")),
            ),
            summary_model_dir: PathBuf::from(
                std::env::var("OMINIX_SUMMARY_MODEL")
                    .unwrap_or_else(|_| format!("{home}/.OminiX/models/qwen3-4b-4bit")),
            ),
            domain_file: make_absolute(PathBuf::from(
                std::env::var("OMINIX_DOMAIN_FILE").unwrap_or_else(|_| "domain.txt".into()),
            )),
            port: std::env::var("OMINIX_PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(8080),
            idle_timeout_secs: std::env::var("OMINIX_IDLE_TIMEOUT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(600),
            transcripts_dir: make_absolute(PathBuf::from(
                std::env::var("OMINIX_TRANSCRIPTS_DIR").unwrap_or_else(|_| "transcripts".into()),
            )),
            language: std::env::var("OMINIX_LANGUAGE").unwrap_or_else(|_| "Chinese".into()),
            mlx_memory_limit_mb: std::env::var("OMINIX_MLX_MEMORY_LIMIT_MB")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(4096),
            mlx_purge_threshold_mb: std::env::var("OMINIX_MLX_PURGE_THRESHOLD_MB")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(768),
        }
    }

    fn polish_available(&self) -> bool {
        std::env::var("OMINIX_POLISH").map(|v| v != "0").unwrap_or(true)
            && self.polish_model_dir.join("config.json").exists()
            && self.polish_model_dir.join("model.safetensors").exists()
    }

    fn summary_available(&self) -> bool {
        std::env::var("OMINIX_SUMMARY").map(|v| v != "0").unwrap_or(true)
            && self.summary_model_dir.join("config.json").exists()
            && self.summary_model_dir.join("model.safetensors").exists()
    }
}

// ── App state ───────────────────────────────────────────────────

#[derive(Clone, Serialize)]
struct Segment {
    id: u64,
    text: String,
    polished: Option<String>,
    translated: Option<String>,
    ts: String,
}

#[derive(Clone)]
struct DomainState {
    path: PathBuf,
    mtime: Option<std::time::SystemTime>,
    config: DomainConfig,
}

struct AppState {
    worker: AsrWorker,
    polish: PolishWorker,
    summary: SummaryWorker,
    polish_available: bool,
    summary_available: bool,
    settings: Settings,
    domain: Mutex<DomainState>,
    segments: Mutex<Vec<Segment>>,
    summaries: Mutex<Vec<(String, String, String)>>,
    seg_id: AtomicU64,
    last_activity: Arc<Mutex<Instant>>,
    last_saved: Mutex<Option<String>>,
    events: broadcast::Sender<String>,
}

// ── VAD ─────────────────────────────────────────────────────────

const FRAME: usize = 320; // 20ms @ 16kHz

struct Vad {
    carry: Vec<f32>,
    prev_frame: Vec<f32>,
    noise_floor: f32,
    in_speech: bool,
    speech_frames: u32,
    silence_frames: u32,
    buffer: Vec<f32>,
    tail: Vec<f32>,
    partial_at: usize,
    utterance_samples: usize,
}

enum VadEvent {
    Final(Vec<f32>),
    Partial(Vec<f32>),
}

impl Vad {
    fn new() -> Self {
        Vad {
            carry: Vec::new(),
            prev_frame: Vec::new(),
            noise_floor: 0.005,
            in_speech: false,
            speech_frames: 0,
            silence_frames: 0,
            buffer: Vec::new(),
            tail: Vec::new(),
            partial_at: 0,
            utterance_samples: 0,
        }
    }

    fn push(&mut self, samples: &[f32]) -> Vec<VadEvent> {
        self.carry.extend_from_slice(samples);
        let mut events = Vec::new();
        while self.carry.len() >= FRAME {
            let chunk: Vec<f32> = self.carry.drain(..FRAME).collect();
            let rms = (chunk.iter().map(|x| x * x).sum::<f32>() / FRAME as f32).sqrt();
            let is_speech = rms > (self.noise_floor * 3.0).max(0.0025);

            if is_speech {
                self.speech_frames += 1;
                self.silence_frames = 0;
                if !self.in_speech && self.speech_frames >= 2 {
                    self.in_speech = true;
                    self.buffer.clear();
                    self.buffer.extend_from_slice(&self.tail);
                    self.tail.clear();
                    self.buffer.extend_from_slice(&self.prev_frame);
                    self.buffer.extend_from_slice(&chunk);
                    self.utterance_samples = 0;
                    self.partial_at = 0;
                } else if self.in_speech {
                    self.buffer.extend_from_slice(&chunk);
                }
            } else {
                self.speech_frames = 0;
                self.noise_floor =
                    0.9 * self.noise_floor + 0.1 * (rms.min(self.noise_floor * 2.0));
                if self.in_speech {
                    self.buffer.extend_from_slice(&chunk);
                    self.silence_frames += 1;
                }
            }
            self.prev_frame = chunk;

            if self.in_speech {
                self.utterance_samples += FRAME;
                let buffered = self.buffer.len();
                if buffered as f32 / 16000.0 >= 2.0 && buffered - self.partial_at >= 2 * 16000 {
                    self.partial_at = buffered;
                    events.push(VadEvent::Partial(self.buffer.clone()));
                }
                if self.utterance_samples >= 12 * 16000 {
                    let mut out = std::mem::take(&mut self.buffer);
                    let split = out.len().saturating_sub(8000);
                    self.tail = out.split_off(split);
                    self.in_speech = false;
                    self.utterance_samples = 0;
                    self.partial_at = 0;
                    events.push(VadEvent::Final(out));
                } else if self.silence_frames >= 40 {
                    let mut out = std::mem::take(&mut self.buffer);
                    let split = out.len().saturating_sub(8000);
                    self.tail = out.split_off(split);
                    self.in_speech = false;
                    self.utterance_samples = 0;
                    self.partial_at = 0;
                    if out.len() >= 20 * FRAME {
                        events.push(VadEvent::Final(out));
                    }
                }
            }
        }
        events
    }
}

// ── Transcript persistence ──────────────────────────────────────

/// 根据书写系统粗判语言 (用于 UI 显示"识别语言")
fn detect_lang(text: &str) -> &'static str {
    let mut cjk = 0usize;
    let mut kana = 0usize;
    let mut hangul = 0usize;
    let mut cyr = 0usize;
    for c in text.chars() {
        if ('\u{4e00}'..='\u{9fff}').contains(&c) {
            cjk += 1;
        } else if ('\u{3040}'..='\u{30ff}').contains(&c) {
            kana += 1;
        } else if ('\u{ac00}'..='\u{d7af}').contains(&c) {
            hangul += 1;
        } else if ('\u{0400}'..='\u{04ff}').contains(&c) {
            cyr += 1;
        }
    }
    if cjk > 0 {
        "中文"
    } else if kana > 0 {
        "日本語"
    } else if hangul > 0 {
        "한국어"
    } else if cyr > 0 {
        "Русский"
    } else {
        "English"
    }
}

fn save_transcript(state: &Arc<AppState>) -> Result<String, String> {
    let segments = state
        .segments
        .lock()
        .map_err(|_| "segments lock poisoned".to_string())?;
    if segments.is_empty() {
        return Err("暂无可保存的转写内容（请先录音转写）".into());
    }
    std::fs::create_dir_all(&state.settings.transcripts_dir)
        .map_err(|e| format!("创建目录失败: {e}"))?;
    let ts = Local::now().format("%Y%m%d_%H%M%S");
    let path = state
        .settings
        .transcripts_dir
        .join(format!("转写_{ts}.txt"));
    let mut content = String::new();
    content.push_str("# 实时语音转写记录\n");
    content.push_str(&format!("# 保存时间: {}\n", Local::now().format("%Y-%m-%d %H:%M:%S")));
    content.push_str(&format!("# 共 {} 段\n\n", segments.len()));
    for s in segments.iter() {
        match (&s.polished, &s.translated) {
            (Some(p), _) => {
                content.push_str(&format!("[{}] {}\n      润色: {}\n", s.ts, s.text, p));
            }
            (_, Some(t)) => {
                content.push_str(&format!("[{}] {}\n      翻译: {}\n", s.ts, s.text, t));
            }
            (None, None) => {
                content.push_str(&format!("[{}] {}\n", s.ts, s.text));
            }
        }
    }
    // 追加 AI 摘要
    let summaries = state.summaries.lock().map_err(|_| "summaries lock poisoned".to_string())?;
    if !summaries.is_empty() {
        content.push_str("\n\n## AI 摘要\n\n");
        for (ts, label, text) in summaries.iter() {
            content.push_str(&format!("### {label} [{}]\n{}\n\n", ts, text));
        }
    }
    drop(segments);
    drop(summaries);
    std::fs::write(&path, content).map_err(|e| format!("写入失败: {e}"))?;
    // 校验确实落盘
    let meta = std::fs::metadata(&path).map_err(|e| format!("写入后校验失败: {e}"))?;
    if meta.len() == 0 {
        return Err("写入后文件为空".into());
    }
    let p = path.to_string_lossy().to_string();
    if let Ok(mut last) = state.last_saved.lock() {
        *last = Some(p.clone());
    }
    Ok(p)
}

// ── HTTP handlers ───────────────────────────────────────────────

#[derive(Serialize)]
struct StatusResp {
    model_loaded: bool,
    polish_loaded: bool,
    polish_available: bool,
    summary_loaded: bool,
    summary_available: bool,
    idle_secs: u64,
    idle_timeout_secs: u64,
    segments: usize,
    last_saved: Option<String>,
    mem_active_mb: u64,
    mem_cache_mb: u64,
    mem_limit_mb: u64,
}

async fn api_status(State(state): State<Arc<AppState>>) -> Json<StatusResp> {
    Json(StatusResp {
        model_loaded: state.worker.model_loaded.load(Ordering::SeqCst),
        polish_loaded: state.polish.model_loaded.load(Ordering::SeqCst),
        polish_available: state.polish_available,
        summary_loaded: state.summary.model_loaded.load(Ordering::SeqCst),
        summary_available: state.summary_available,
        idle_secs: idle_secs(&state.last_activity),
        idle_timeout_secs: state.settings.idle_timeout_secs,
        segments: state.segments.lock().map(|s| s.len()).unwrap_or(0),
        last_saved: state
            .last_saved
            .lock()
            .ok()
            .and_then(|p| p.clone()),
        mem_active_mb: mem::active_mb(),
        mem_cache_mb: mem::cache_mb(),
        mem_limit_mb: state.settings.mlx_memory_limit_mb as u64,
    })
}

async fn api_save(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    match save_transcript(&state) {
        Ok(path) => Json(serde_json::json!({"ok": true, "path": path})),
        Err(e) => Json(serde_json::json!({"ok": false, "error": e})),
    }
}

fn save_summary(state: &Arc<AppState>) -> Result<String, String> {
    let summaries = state
        .summaries
        .lock()
        .map_err(|_| "summaries lock poisoned".to_string())?;
    if summaries.is_empty() {
        return Err("暂无可保存的摘要（请先生成摘要）".into());
    }
    std::fs::create_dir_all(&state.settings.transcripts_dir)
        .map_err(|e| format!("创建目录失败: {e}"))?;
    let ts = Local::now().format("%Y%m%d_%H%M%S");
    let path = state
        .settings
        .transcripts_dir
        .join(format!("摘要_{ts}.txt"));
    let mut content = String::new();
    content.push_str("# AI 摘要记录\n");
    content.push_str(&format!("# 保存时间: {}\n", Local::now().format("%Y-%m-%d %H:%M:%S")));
    content.push_str(&format!("# 共 {} 条\n\n", summaries.len()));
    for (ts, label, text) in summaries.iter() {
        content.push_str(&format!("## {label} [{}]\n{}\n\n", ts, text));
    }
    std::fs::write(&path, content).map_err(|e| format!("写入失败: {e}"))?;
    let meta = std::fs::metadata(&path).map_err(|e| format!("写入后校验失败: {e}"))?;
    if meta.len() == 0 {
        return Err("写入后文件为空".into());
    }
    let p = path.to_string_lossy().to_string();
    if let Ok(mut last) = state.last_saved.lock() {
        *last = Some(p.clone());
    }
    Ok(p)
}

async fn api_save_summary(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    match save_summary(&state) {
        Ok(path) => Json(serde_json::json!({"ok": true, "path": path})),
        Err(e) => Json(serde_json::json!({"ok": false, "error": e})),
    }
}

async fn api_unload(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    state.worker.unload();
    state.polish.unload();
    state.summary.unload();
    Json(serde_json::json!({"ok": true}))
}

async fn api_clear(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    if let Ok(mut s) = state.segments.lock() {
        s.clear();
    }
    Json(serde_json::json!({"ok": true}))
}

async fn api_download(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let path = state
        .last_saved
        .lock()
        .ok()
        .and_then(|p| p.clone());
    match path {
        Some(p) => match std::fs::read_to_string(&p) {
            Ok(content) => (
                [("content-type", "text/plain; charset=utf-8")],
                content,
            )
                .into_response(),
            Err(e) => (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("读取文件失败: {e}"),
            )
                .into_response(),
        },
        None => (axum::http::StatusCode::NOT_FOUND, "尚未保存过转写内容".to_string())
            .into_response(),
    }
}

async fn api_domain(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let d = state.domain.lock().ok().map(|g| g.clone());
    match d {
        Some(d) => Json(serde_json::json!({
            "system": d.config.system,
            "terms": d.config.terms,
            "path": d.path.to_string_lossy(),
        })),
        None => Json(serde_json::json!({"system": "", "terms": [], "path": ""})),
    }
}

async fn index() -> impl IntoResponse {
    (
        [
            ("content-type", "text/html; charset=utf-8"),
            ("cache-control", "no-store, no-cache, must-revalidate"),
        ],
        include_str!("web/index.html"),
    )
}

async fn subtitle_page() -> impl IntoResponse {
    (
        [
            ("content-type", "text/html; charset=utf-8"),
            ("cache-control", "no-store, no-cache, must-revalidate"),
        ],
        include_str!("web/subtitle.html"),
    )
}

// ── WebSocket ───────────────────────────────────────────────────

#[derive(Deserialize)]
struct WsCmd {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    lang: Option<String>,
    #[serde(default)]
    enabled: Option<bool>,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    target: Option<String>,
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<Arc<AppState>>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: Arc<AppState>) {
    let mut vad = Vad::new();
    let mut lang = state.settings.language.clone();
    let mut system_prompt = String::new();
    let mut terms: Vec<String> = Vec::new();
    let mut enhance_mode = String::from("polish");
    let mut target_lang = String::from("English");
    let mut event_rx = state.events.subscribe();

    // 连接时推送当前领域配置
    {
        let d = state.domain.lock().ok().map(|g| g.clone());
        if let Some(d) = d {
            system_prompt = d.config.system.clone();
            terms = d.config.terms.clone();
            let msg = serde_json::json!({
                "type": "domain",
                "system": system_prompt,
                "terms": terms,
                "path": d.path.to_string_lossy(),
            });
            let _ = socket.send(Message::Text(msg.to_string().into())).await;
        }
    }

    loop {
        tokio::select! {
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Binary(b))) => {
                        if b.len() % 4 != 0 {
                            continue;
                        }
                        let mut samples = Vec::with_capacity(b.len() / 4);
                        for c in b.chunks_exact(4) {
                            samples.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                        }
                        touch(&state.last_activity);
                        for ev in vad.push(&samples) {
                            match ev {
                                VadEvent::Final(s) => {
                                    let dur = s.len() as f32 / 16000.0;
                                    let recv = state.worker.transcribe(
                                        s,
                                        lang.clone(),
                                        system_prompt.clone(),
                                        terms.clone(),
                                    );
                                    let result = tokio::task::block_in_place(|| recv.recv().ok());
                                    match result {
                                        Some(Ok(text)) if !text.trim().is_empty() => {
                                            let ts = Local::now().format("%H:%M:%S").to_string();
                                            let id = state.seg_id.fetch_add(1, Ordering::SeqCst);
                                            let src_lang = detect_lang(&text).to_string();
                                            let seg = Segment {
                                                id,
                                                text: text.clone(),
                                                polished: None,
                                                translated: None,
                                                ts: ts.clone(),
                                            };
                                            // 先立即推送原文, 不阻塞音频管线
                                            let msg = serde_json::json!({
                                                "type":"final",
                                                "id":id,
                                                "text":text.clone(),
                                                "src_lang":src_lang,
                                                "ts":ts,
                                                "dur":dur,
                                            });
                                            let _ = socket.send(Message::Text(msg.to_string().into())).await;
                                            if let Ok(mut segs) = state.segments.lock() {
                                                segs.push(seg);
                                            }
                                            // 增强(润色/翻译)在后台异步执行, 完成后广播更新
                                            if enhance_mode == "polish" || enhance_mode == "translate" {
                                                let st = state.clone();
                                                let mode = enhance_mode.clone();
                                                let tgt = target_lang.clone();
                                                let src_lang_clone = src_lang.clone();
                                                tokio::spawn(async move {
                                                    let st2 = st.clone();
                                                    let mode2 = mode.clone();
                                                    let result = tokio::task::spawn_blocking(move || {
                                                        let rx = if mode2 == "translate" {
                                                            st2.polish.translate(text.clone(), tgt)
                                                        } else {
                                                            st2.polish.polish(text.clone(), "Auto".to_string())
                                                        };
                                                        rx.recv().ok().and_then(|r| r.ok())
                                                    })
                                                    .await
                                                    .unwrap_or(None);
                                                    match result {
                                                        Some(out) if !out.trim().is_empty() => {
                                                            if let Ok(mut segs) = st.segments.lock() {
                                                                for s in segs.iter_mut() {
                                                                    if s.id == id {
                                                                        if mode == "translate" {
                                                                            s.translated = Some(out.clone());
                                                                        } else {
                                                                            s.polished = Some(out.clone());
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                            let update = if mode == "translate" {
                                                                serde_json::json!({"type":"enhanced","id":id,"translated":out,"src_lang":src_lang_clone})
                                                            } else {
                                                                serde_json::json!({"type":"enhanced","id":id,"polished":out})
                                                            };
                                                            let _ = st.events.send(update.to_string());
                                                        }
                                                        _ => {
                                                            let err = serde_json::json!({
                                                                "type":"error",
                                                                "text": if mode == "translate" {"翻译失败，已保留原文"} else {"润色失败，已保留原文"}
                                                            });
                                                            let _ = st.events.send(err.to_string());
                                                        }
                                                    }
                                                });
                                            }
                                        }
                                        Some(Err(e)) => {
                                            let msg = serde_json::json!({"type":"error","text":e});
                                            let _ = socket.send(Message::Text(msg.to_string().into())).await;
                                        }
                                        _ => {}
                                    }
                                }
                                VadEvent::Partial(s) => {
                                    let recv = state.worker.transcribe(
                                        s,
                                        lang.clone(),
                                        system_prompt.clone(),
                                        terms.clone(),
                                    );
                                    let result = tokio::task::block_in_place(|| recv.recv().ok());
                                    if let Some(Ok(text)) = result {
                                        if !text.trim().is_empty() {
                                            let msg = serde_json::json!({"type":"partial","text":text});
                                            let _ = socket.send(Message::Text(msg.to_string().into())).await;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Some(Ok(Message::Text(t))) => {
                        if let Ok(cmd) = serde_json::from_str::<WsCmd>(&t) {
                            match cmd.kind.as_str() {
                                "lang" => {
                                    if let Some(l) = cmd.lang {
                                        lang = l;
                                    }
                                }
                                "reload_domain" => {
                                    match load_domain_file(&state.settings.domain_file) {
                                        Ok((cfg, mtime)) => {
                                            system_prompt = cfg.system.clone();
                                            terms = cfg.terms.clone();
                                            if let Ok(mut d) = state.domain.lock() {
                                                d.config = cfg;
                                                d.mtime = mtime;
                                            }
                                            let msg = serde_json::json!({
                                                "type": "domain",
                                                "system": system_prompt,
                                                "terms": terms,
                                                "path": state.settings.domain_file.to_string_lossy(),
                                            });
                                            let _ = socket.send(Message::Text(msg.to_string().into())).await;
                                            let ack = serde_json::json!({
                                                "type": "context_applied",
                                                "system": !system_prompt.trim().is_empty(),
                                                "terms": terms.len(),
                                            });
                                            let _ = socket.send(Message::Text(ack.to_string().into())).await;
                                        }
                                        Err(e) => {
                                            let msg = serde_json::json!({"type":"error","text":e});
                                            let _ = socket.send(Message::Text(msg.to_string().into())).await;
                                        }
                                    }
                                }
                                "save" => {
                                    let resp = match save_transcript(&state) {
                                        Ok(path) => serde_json::json!({"type":"saved","path":path}),
                                        Err(e) => serde_json::json!({"type":"error","text":e}),
                                    };
                                    let _ = socket.send(Message::Text(resp.to_string().into())).await;
                                }
                                 "unload" => {
                                     state.worker.unload();
                                     state.polish.unload();
                                     state.summary.unload();
                                 }
                                 "summarize" => {
                                     if !state.summary_available {
                                         let msg = serde_json::json!({
                                             "type": "error",
                                             "text": "总结模型不可用（未找到模型文件）"
                                         });
                                         let _ = socket.send(Message::Text(msg.to_string().into())).await;
                                     } else {
                                         // 汇总当前所有转写段(优先润色后文本)
                                         let text = {
                                             let segs = state.segments.lock().unwrap_or_else(|e| e.into_inner());
                                             let joined: Vec<String> = segs
                                                 .iter()
                                                 .map(|s| {
                                                     s.polished.clone().unwrap_or_else(|| s.text.clone())
                                                 })
                                                 .collect();
                                             let mut t = joined.join("\n");
                                             if t.chars().count() > 8000 {
                                                 t = t.chars().take(8000).collect();
                                             }
                                             t
                                         };
                                         if text.trim().is_empty() {
                                             let msg = serde_json::json!({
                                                 "type": "error",
                                                 "text": "暂无可总结的转写内容"
                                             });
                                             let _ = socket.send(Message::Text(msg.to_string().into())).await;
                                         } else {
                                             let mode = cmd.mode.clone().unwrap_or_else(|| "source".to_string());
                                             let target = cmd.target.clone().unwrap_or_else(|| "English".to_string());
                                             // 决定要生成哪些摘要: (输出语言, 标签)
                                             let mut jobs: Vec<(String, String)> = Vec::new();
                                             match mode.as_str() {
                                                 "translate" => jobs.push((target, "翻译摘要".to_string())),
                                                 "both" => {
                                                     jobs.push(("source".to_string(), "原文摘要".to_string()));
                                                     jobs.push((target, "翻译摘要".to_string()));
                                                 }
                                                 _ => jobs.push(("source".to_string(), "原文摘要".to_string())),
                                             }
                                             for (out_lang, label) in jobs {
                                                 let st = state.clone();
                                                 let text2 = text.clone();
                                                 let out_lang2 = out_lang.clone();
                                                 let label2 = label.clone();
                                                 tokio::spawn(async move {
                                                     let st2 = st.clone();
                                                     let text3 = text2.clone();
                                                     let out_lang3 = out_lang2.clone();
                                                     let result = tokio::task::spawn_blocking(move || {
                                                         let rx = st2.summary.summarize(text3, out_lang3);
                                                         rx.recv().ok().and_then(|r| r.ok())
                                                     })
                                                     .await
                                                     .unwrap_or(None);
                                                     match result {
                                                         Some(out) if !out.trim().is_empty() => {
                                                             let ts = Local::now().format("%H:%M:%S").to_string();
                                                             if let Ok(mut list) = st.summaries.lock() {
                                                                 list.push((ts.clone(), label2.clone(), out.clone()));
                                                             }
                                                             let _ = st.events.send(
                                                                 serde_json::json!({"type":"summary","text":out,"ts":ts,"label":label2})
                                                                     .to_string(),
                                                             );
                                                         }
                                                         _ => {
                                                             let _ = st.events.send(
                                                                 serde_json::json!({"type":"error","text":format!("{label2}生成失败")})
                                                                     .to_string(),
                                                             );
                                                         }
                                                     }
                                                 });
                                             }
                                         }
                                     }
                                 }
                                 "clear_summary" => {
                                     if let Ok(mut list) = state.summaries.lock() {
                                         list.clear();
                                     }
                                     let _ = state.events.send(
                                         serde_json::json!({"type":"summary_cleared"}).to_string(),
                                     );
                                 }
                                "polish" => {
                                    enhance_mode = if cmd.enabled.unwrap_or(true) {
                                        "polish".to_string()
                                    } else {
                                        "off".to_string()
                                    };
                                    let msg = serde_json::json!({
                                        "type": "enhance_applied",
                                        "mode": enhance_mode,
                                        "target": target_lang,
                                        "available": state.polish_available,
                                    });
                                    let _ = socket.send(Message::Text(msg.to_string().into())).await;
                                }
                                "enhance" => {
                                    let mode = cmd.mode.unwrap_or_else(|| "polish".to_string());
                                    enhance_mode = match mode.as_str() {
                                        "translate" => "translate".to_string(),
                                        "polish" => "polish".to_string(),
                                        _ => "off".to_string(),
                                    };
                                    if let Some(t) = cmd.target {
                                        target_lang = t;
                                    }
                                    let msg = serde_json::json!({
                                        "type": "enhance_applied",
                                        "mode": enhance_mode,
                                        "target": target_lang,
                                        "available": state.polish_available,
                                    });
                                    let _ = socket.send(Message::Text(msg.to_string().into())).await;
                                }
                                "clear" => {
                                    if let Ok(mut s) = state.segments.lock() {
                                        s.clear();
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    _ => {}
                }
            }
            ev = event_rx.recv() => {
                if let Ok(s) = ev {
                    // 领域配置热更新广播: 同步本地变量并转发给前端
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
                        if v["type"] == "domain" {
                            system_prompt = v["system"].as_str().unwrap_or("").to_string();
                            terms = v["terms"]
                                .as_array()
                                .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
                                .unwrap_or_default();
                        }
                    }
                    let _ = socket.send(Message::Text(s.into())).await;
                }
            }
        }
    }
}

// ── main ────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let settings = Settings::from_env();
    println!("模型路径: {}", settings.model_dir.display());
    println!("空闲卸载: {} 秒无活动后自动从内存卸载模型", settings.idle_timeout_secs);
    println!("转写保存目录: {}", settings.transcripts_dir.display());

    let (events, _) = broadcast::channel::<String>(64);
    // MLX C++ 运行时非线程安全, ASR 与增强线程共享此锁串行化
    let mlx_lock: Arc<Mutex<()>> = Arc::new(Mutex::new(()));
    // MLX 内存软上限: 超过时自动释放缓存池
    mem::set_memory_limit(settings.mlx_memory_limit_mb * 1048576);
    let purge_threshold = settings.mlx_purge_threshold_mb * 1048576;
    let worker = AsrWorker::start(
        settings.model_dir.clone(),
        events.clone(),
        mlx_lock.clone(),
        purge_threshold,
    );
    let polish = PolishWorker::start(
        settings.polish_model_dir.clone(),
        events.clone(),
        mlx_lock.clone(),
        purge_threshold,
    );
    let summary = SummaryWorker::start(
        settings.summary_model_dir.clone(),
        events.clone(),
        mlx_lock,
        purge_threshold,
    );
    let polish_available = settings.polish_available();
    let summary_available = settings.summary_available();

    if !polish_available {
        println!("提示: 未找到润色模型 ({}), AI 润色功能将不可用", settings.polish_model_dir.display());
    }
    if !summary_available {
        println!("提示: 未找到总结模型 ({}), AI 总结功能将不可用", settings.summary_model_dir.display());
    }

    // 加载领域配置文件 (不存在则创建默认)
    let settings_domain_file = settings.domain_file.clone();
    let (domain_config, domain_mtime) = load_domain_file(&settings.domain_file)
        .unwrap_or_else(|e| {
            eprintln!("警告: {e}, 使用空配置");
            (
                DomainConfig {
                    system: String::new(),
                    terms: Vec::new(),
                },
                None,
            )
        });
    println!(
        "领域配置文件: {} (提示词 {} 字 / {} 术语, 保存后自动热加载)",
        settings.domain_file.display(),
        domain_config.system.chars().count(),
        domain_config.terms.len()
    );
    std::fs::create_dir_all(&settings.transcripts_dir)
        .unwrap_or_else(|e| eprintln!("警告: 创建转写目录失败: {e}"));

    let state = Arc::new(AppState {
        worker,
        polish,
        summary,
        polish_available,
        summary_available,
        settings,
        domain: Mutex::new(DomainState {
            path: settings_domain_file.clone(),
            mtime: domain_mtime,
            config: domain_config,
        }),
        segments: Mutex::new(Vec::new()),
        summaries: Mutex::new(Vec::new()),
        seg_id: AtomicU64::new(1),
        last_activity: Arc::new(Mutex::new(Instant::now())),
        last_saved: Mutex::new(None),
        events,
    });

    // idle unload watchdog
    {
        let st = state.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(10)).await;
                let timeout = st.settings.idle_timeout_secs;
                if timeout == 0 {
                    continue;
                }
                if idle_secs(&st.last_activity) >= timeout {
                    if st.worker.model_loaded.load(Ordering::SeqCst) {
                        st.worker.unload();
                        let _ = st.events.send(
                            serde_json::json!({"type":"status","model_loaded":false}).to_string(),
                        );
                    }
                    if st.polish.model_loaded.load(Ordering::SeqCst) {
                        st.polish.unload();
                    }
                    if st.summary.model_loaded.load(Ordering::SeqCst) {
                        st.summary.unload();
                    }
                }
            }
        });
    }

    // domain.txt hot-reload watchdog
    {
        let st = state.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(5)).await;
                let path = st.settings.domain_file.clone();
                let current = st.domain.lock().map(|d| d.mtime).ok().flatten();
                match file_mtime(&path) {
                    Some(m) if current != Some(m) => match load_domain_file(&path) {
                        Ok((cfg, mtime)) => {
                            if let Ok(mut d) = st.domain.lock() {
                                d.config = cfg.clone();
                                d.mtime = mtime;
                            }
                            let _ = st.events.send(
                                serde_json::json!({
                                    "type": "domain",
                                    "system": cfg.system,
                                    "terms": cfg.terms,
                                    "path": path.to_string_lossy(),
                                })
                                .to_string(),
                            );
                            println!("[domain] 配置文件已热加载");
                        }
                        Err(e) => eprintln!("[domain] 热加载失败: {e}"),
                    },
                    _ => {}
                }
            }
        });
    }

    let transcripts_display = state.settings.transcripts_dir.clone();

    let app = Router::new()
        .route("/", get(index))
        .route("/subtitle", get(subtitle_page))
        .route("/api/status", get(api_status))
        .route("/api/save", post(api_save))
        .route("/api/save_summary", post(api_save_summary))
        .route("/api/unload", post(api_unload))
        .route("/api/clear", post(api_clear))
        .route("/api/download", get(api_download))
        .route("/api/domain", get(api_domain))
        .route("/ws", get(ws_handler))
        .with_state(state);

    let port = std::env::var("OMINIX_PORT")
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(8080);
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port))
        .await
        .expect("端口绑定失败");
    println!();
    println!("OminiX 实时语音识别已启动:");
    println!("  WebUI:  http://localhost:{port}");
    println!("  转写保存: {}", transcripts_display.display());
    println!("  提示: 浏览器需要麦克风权限, 请通过 localhost 访问");

    // auto-open browser after a short delay (disable with OMINIX_AUTO_OPEN=0)
    let auto_open = std::env::var("OMINIX_AUTO_OPEN").map(|v| v != "0").unwrap_or(true);
    if auto_open {
        std::thread::spawn(move || {
            for i in (1..=5).rev() {
                println!("  {i} 秒后自动打开浏览器...");
                std::thread::sleep(Duration::from_secs(1));
            }
            let url = format!("http://localhost:{port}");
            match std::process::Command::new("open").arg(&url).status() {
                Ok(s) if s.success() => println!("已在浏览器打开 {url}"),
                Ok(s) => eprintln!("打开浏览器失败 (exit {:?}), 请手动访问 {url}", s.code()),
                Err(e) => eprintln!("无法打开浏览器: {e}, 请手动访问 {url}"),
            }
        });
    }

    axum::serve(listener, app).await.unwrap();
}
