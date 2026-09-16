//! AI 总结 worker: 按需加载 Qwen3-4B-4bit, 转写记录 → 要点总结

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use mlx_lm_utils::tokenizer::{
    load_model_chat_template_from_file, ApplyChatTemplateArgs, Conversation, Role, Tokenizer,
};
use mlx_rs::ops::indexing::{IndexOp, NewAxis};
use mlx_rs::transforms::eval;
use mlx_rs::Array;
use qwen3_mlx::{load_model, Generate, KVCache, Model};

use crate::mem;

pub enum SummaryCmd {
    Summarize {
        text: String,
        lang: String,
        kind: String,
        reply: Sender<Result<String, String>>,
    },
    Unload,
}

pub struct SummaryWorker {
    tx: Sender<SummaryCmd>,
    pub model_loaded: Arc<AtomicBool>,
    _handle: JoinHandle<()>,
}

struct SummaryModel {
    model: Model,
    tokenizer: Tokenizer,
    chat_template: String,
}

const SYSTEM_MEETING_ZH: &str = "你是一位专业的会议记录整理助手。请对下面的语音转写记录进行总结，\
输出以下内容（用简洁的要点形式）：\n1. 核心主题\n2. 关键要点（3-6 条）\n3. 结论或决定\n4. 待办事项（如有）\n\
直接输出总结，不要任何解释或客套话。";

const SYSTEM_CONTENT_ZH: &str = "你是一位专业的内容整理助手。请对下面的语音转写内容进行总结，\
输出以下内容（用简洁的要点形式）：\n1. 核心主题\n2. 关键要点（3-6 条）\n3. 主要观点或论证\n4. 综合反思\n\
直接输出总结，不要任何解释或客套话。";

const SYSTEM_MEETING_EN: &str = "You are a professional meeting-minutes assistant. Summarize the \
following speech transcript and output (in concise bullet form):\n1. Core topic\n2. Key points (3-6)\n\
3. Conclusions or decisions\n4. Action items (if any)\nOutput only the summary in English, with no explanation.";

const SYSTEM_CONTENT_EN: &str = "You are a professional content summarization assistant. Summarize the \
following speech transcript and output (in concise bullet form):\n1. Core topic\n2. Key points (3-6)\n\
3. Main arguments\n4. Comprehensive reflection\nOutput only the summary in English, with no explanation.";

impl SummaryWorker {
    pub fn start(
        model_dir: PathBuf,
        events: tokio::sync::broadcast::Sender<String>,
        mlx_lock: Arc<Mutex<()>>,
        purge_threshold: usize,
    ) -> Self {
        let (tx, rx) = channel::<SummaryCmd>();
        let model_loaded = Arc::new(AtomicBool::new(false));
        let flag = model_loaded.clone();
        let handle = std::thread::Builder::new()
            .name("summary-worker".into())
            .spawn(move || worker_loop(rx, model_dir, flag, events, mlx_lock, purge_threshold))
            .expect("failed to spawn summary worker");
        SummaryWorker {
            tx,
            model_loaded,
            _handle: handle,
        }
    }

    pub fn summarize(&self, text: String, lang: String, kind: String) -> Receiver<Result<String, String>> {
        let (reply, recv) = channel();
        let _ = self.tx.send(SummaryCmd::Summarize { text, lang, kind, reply });
        recv
    }

    pub fn unload(&self) {
        let _ = self.tx.send(SummaryCmd::Unload);
    }
}

fn worker_loop(
    rx: Receiver<SummaryCmd>,
    model_dir: PathBuf,
    model_loaded: Arc<AtomicBool>,
    events: tokio::sync::broadcast::Sender<String>,
    mlx_lock: Arc<Mutex<()>>,
    purge_threshold: usize,
) {
    let mut state: Option<SummaryModel> = None;
    while let Ok(cmd) = rx.recv() {
        match cmd {
            SummaryCmd::Summarize { text, lang, kind, reply } => {
                let _mlx_guard = mlx_lock.lock().unwrap();
                if state.is_none() {
                    eprintln!("[summary] model not loaded, loading from {}...", model_dir.display());
                    let _ = events.send(
                        serde_json::json!({"type":"summary_status","loaded":false,"loading":true})
                            .to_string(),
                    );
                    match load(&model_dir) {
                        Ok(m) => {
                            state = Some(m);
                            model_loaded.store(true, Ordering::SeqCst);
                            eprintln!("[summary] model loaded");
                            let _ = events.send(
                                serde_json::json!({"type":"summary_status","loaded":true,"loading":false})
                                    .to_string(),
                            );
                        }
                        Err(e) => {
                            let _ = events.send(
                                serde_json::json!({"type":"summary_status","loaded":false,"loading":false})
                                    .to_string(),
                            );
                            let _ = reply.send(Err(format!("总结模型加载失败: {e}")));
                            continue;
                        }
                    }
                }
                let t0 = std::time::Instant::now();
                let result = summarize(&mut state.as_mut().unwrap(), &text, &lang, &kind);
                let elapsed = t0.elapsed().as_secs_f32();
                match &result {
                    Ok((out, tokens)) => {
                        let tok_s = *tokens as f32 / elapsed.max(0.001);
                        eprintln!(
                            "[summary] {} chars -> {} chars, {} tokens in {:.1}s ({:.1} tok/s)",
                            text.chars().count(),
                            out.chars().count(),
                            tokens,
                            elapsed,
                            tok_s
                        );
                        let _ = events.send(
                            serde_json::json!({"type":"tok_stat","mode":"summary","tokens":tokens,"secs":(elapsed*10.0).round()/10.0,"tok_s":(tok_s*10.0).round()/10.0})
                                .to_string(),
                        );
                    }
                    Err(e) => eprintln!("[summary] error: {e}"),
                }
                let _ = reply.send(result.map(|(s, _)| s));
                let (_, cache) = mem::snapshot();
                if cache > purge_threshold {
                    mem::purge_caches();
                }
            }
            SummaryCmd::Unload => {
                let _mlx_guard = mlx_lock.lock().unwrap();
                if state.take().is_some() {
                    model_loaded.store(false, Ordering::SeqCst);
                    eprintln!("[summary] model unloaded (idle timeout / manual)");
                    let _ = events.send(
                        serde_json::json!({"type":"summary_status","loaded":false,"loading":false})
                            .to_string(),
                    );
                    mem::purge_caches();
                }
            }
        }
    }
}

fn load(model_dir: &PathBuf) -> Result<SummaryModel, String> {
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| format!("tokenizer 加载失败: {e:?}"))?;
    let chat_template = load_model_chat_template_from_file(
        &model_dir.join("tokenizer_config.json"),
    )
    .map_err(|e| format!("chat template 读取失败: {e}"))?
    .ok_or_else(|| "tokenizer_config.json 中没有 chat template".to_string())?;
    let model = load_model(model_dir).map_err(|e| format!("模型加载失败: {e}"))?;
    Ok(SummaryModel {
        model,
        tokenizer,
        chat_template,
    })
}

fn summarize(
    state: &mut SummaryModel,
    text: &str,
    lang: &str,
    kind: &str,
) -> Result<(String, usize), String> {
    // 按输出语言 + 摘要类型选择提示词
    let has_cjk = text.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c));
    let use_en = if lang == "English" {
        true
    } else if lang == "source" {
        !has_cjk
    } else {
        false
    };
    let is_other = lang != "English" && lang != "Chinese" && lang != "source";
    let is_content = kind == "content";
    let system = if is_other {
        let tail = if is_content { "综合反思" } else { "待办事项" };
        format!(
            "你是一位专业的内容整理助手。请用{lang}输出总结，格式为：核心主题、关键要点、结论、{tail}。直接输出，不要解释。"
        )
    } else if use_en {
        if is_content { SYSTEM_CONTENT_EN.to_string() } else { SYSTEM_MEETING_EN.to_string() }
    } else {
        if is_content { SYSTEM_CONTENT_ZH.to_string() } else { SYSTEM_MEETING_ZH.to_string() }
    };
    let max_tokens = 1024;
    let user_msg = format!("{system}\n\n语音转写记录：\n{text}");
    let conversations = vec![Conversation {
        role: Role::User,
        content: user_msg.as_str(),
    }];
    let args = ApplyChatTemplateArgs {
        conversations: vec![conversations.into()],
        documents: None,
        model_id: "qwen3",
        chat_template_id: None,
        add_generation_prompt: None,
        continue_final_message: None,
    };
    let encodings = state
        .tokenizer
        .apply_chat_template_and_encode(state.chat_template.clone(), args)
        .map_err(|e| format!("prompt 编码失败: {e:?}"))?;
    let mut prompt: Vec<u32> = encodings
        .iter()
        .flat_map(|encoding| encoding.get_ids())
        .copied()
        .collect();
    if let Ok(suffix_enc) = state.tokenizer.encode("\n<think>\n\n</think>\n\n", false) {
        prompt.extend(suffix_enc.get_ids().iter().copied());
    }
    let prompt_tokens = Array::from(&prompt[..]).index(NewAxis);

    let mut cache: Vec<Option<KVCache>> = Vec::new();
    let generator = Generate::<KVCache>::new(&mut state.model, &mut cache, 0.3, &prompt_tokens);

    let mut tokens: Vec<Array> = Vec::new();
    let mut last_ids: Vec<u32> = Vec::with_capacity(10);
    for token in generator {
        let t = token.map_err(|e| format!("生成失败: {e}"))?;
        let token_id = t.item::<u32>();
        if token_id == 151643 || token_id == 151645 {
            break;
        }
        last_ids.push(token_id);
        if last_ids.len() > 10 {
            last_ids.remove(0);
        }
        if last_ids.len() == 10 && last_ids.iter().all(|&x| x == token_id) {
            break;
        }
        tokens.push(t);
        if tokens.len() >= max_tokens {
            break;
        }
        if tokens.len() % 16 == 0 {
            eval(&tokens).map_err(|e| format!("eval 失败: {e}"))?;
        }
    }
    eval(&tokens).map_err(|e| format!("eval 失败: {e}"))?;
    let ids: Vec<u32> = tokens.iter().map(|t| t.item::<u32>()).collect();
    let mut out = state
        .tokenizer
        .decode(&ids, true)
        .map_err(|e| format!("解码失败: {e:?}"))?;
    if let Some(idx) = out.rfind("</think>") {
        out = out[idx + "</think>".len()..].trim().to_string();
    }
    let out = out.trim().to_string();
    if out.is_empty() {
        Err("生成结果为空".to_string())
    } else {
        Ok((out, tokens.len()))
    }
}
