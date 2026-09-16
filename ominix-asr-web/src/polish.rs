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

pub enum PolishCmd {
    Polish {
        text: String,
        language: String,
        reply: Sender<Result<String, String>>,
    },
    Translate {
        text: String,
        target: String,
        reply: Sender<Result<String, String>>,
    },
    Unload,
}

pub struct PolishWorker {
    tx: Sender<PolishCmd>,
    pub model_loaded: Arc<AtomicBool>,
    _handle: JoinHandle<()>,
}

struct PolishModel {
    model: Model,
    tokenizer: Tokenizer,
    chat_template: String,
}

const ZH_SYSTEM: &str = "你是一位专业的文稿编辑。请将用户的语音转写草稿整理为流畅的书面讲稿：\
删除语气词、口头禅和填充词（如\"嗯\"\"啊\"\"那个\"\"然后\"），合并或删除重复啰嗦的表达，\
并根据上下文修正明显的同音字和转写错误。保持原意不变，保留专业术语与数字，\
不要补充原文没有的信息。直接输出整理后的文本，不要任何解释。";

const EN_SYSTEM: &str = "You are a professional transcript editor. Rewrite the user's raw \
speech-to-text draft into fluent written language: remove filler words and verbal tics, \
merge repeated expressions, and fix obvious homophone or transcription errors based on context. \
Keep the original meaning, preserve technical terms and numbers, and do not add new information. \
Output only the polished text, with no explanation.";

impl PolishWorker {
    pub fn start(
        model_dir: PathBuf,
        events: tokio::sync::broadcast::Sender<String>,
        mlx_lock: Arc<Mutex<()>>,
        purge_threshold: usize,
    ) -> Self {
        let (tx, rx) = channel::<PolishCmd>();
        let model_loaded = Arc::new(AtomicBool::new(false));
        let flag = model_loaded.clone();
        let handle = std::thread::Builder::new()
            .name("polish-worker".into())
            .spawn(move || worker_loop(rx, model_dir, flag, events, mlx_lock, purge_threshold))
            .expect("failed to spawn polish worker");
        PolishWorker {
            tx,
            model_loaded,
            _handle: handle,
        }
    }

    pub fn polish(
        &self,
        text: String,
        language: String,
    ) -> Receiver<Result<String, String>> {
        let (reply, recv) = channel();
        let _ = self.tx.send(PolishCmd::Polish {
            text,
            language,
            reply,
        });
        recv
    }

    pub fn translate(
        &self,
        text: String,
        target: String,
    ) -> Receiver<Result<String, String>> {
        let (reply, recv) = channel();
        let _ = self.tx.send(PolishCmd::Translate {
            text,
            target,
            reply,
        });
        recv
    }

    pub fn unload(&self) {
        let _ = self.tx.send(PolishCmd::Unload);
    }
}

fn worker_loop(
    rx: Receiver<PolishCmd>,
    model_dir: PathBuf,
    model_loaded: Arc<AtomicBool>,
    events: tokio::sync::broadcast::Sender<String>,
    mlx_lock: Arc<Mutex<()>>,
    purge_threshold: usize,
) {
    let mut state: Option<PolishModel> = None;
    while let Ok(cmd) = rx.recv() {
        match cmd {
            PolishCmd::Polish {
                text,
                language,
                reply,
            } => {
                // MLX 非线程安全: 与 ASR 线程串行化
                let _mlx_guard = mlx_lock.lock().unwrap();
                if state.is_none() {
                    eprintln!("[polish] model not loaded, loading from {}...", model_dir.display());
                    let _ = events.send(
                        serde_json::json!({"type":"polish_status","loaded":false,"loading":true})
                            .to_string(),
                    );
                    match load(&model_dir) {
                        Ok(m) => {
                            state = Some(m);
                            model_loaded.store(true, Ordering::SeqCst);
                            eprintln!("[polish] model loaded");
                            let _ = events.send(
                                serde_json::json!({"type":"polish_status","loaded":true,"loading":false})
                                    .to_string(),
                            );
                        }
                        Err(e) => {
                            let _ = events.send(
                                serde_json::json!({"type":"polish_status","loaded":false,"loading":false})
                                    .to_string(),
                            );
                            let _ = reply.send(Err(format!("润色模型加载失败: {e}")));
                            continue;
                        }
                    }
                }
                let t0 = std::time::Instant::now();
                let result = polish(&mut state.as_mut().unwrap(), &text, &language);
                let elapsed = t0.elapsed().as_secs_f32();
                match &result {
                    Ok((out, tokens)) => {
                        let tok_s = *tokens as f32 / elapsed.max(0.001);
                        eprintln!(
                            "[polish] {} chars -> {} chars, {} tokens in {:.1}s ({:.1} tok/s)",
                            text.chars().count(),
                            out.chars().count(),
                            tokens,
                            elapsed,
                            tok_s
                        );
                        let _ = events.send(
                            serde_json::json!({"type":"tok_stat","mode":"polish","tokens":tokens,"secs":(elapsed*10.0).round()/10.0,"tok_s":(tok_s*10.0).round()/10.0})
                                .to_string(),
                        );
                    }
                    Err(e) => eprintln!("[polish] error: {e}"),
                }
                let _ = reply.send(result.map(|(s, _)| s));
                // 内存监控: 缓存池超过阈值时清理 (含图编译缓存)
                let (active, cache) = mem::snapshot();
                if cache > purge_threshold {
                    mem::purge_caches();
                    eprintln!(
                        "[enhance] 内存缓存清理: active={}MB cache={}MB (阈值 {}MB)",
                        active / 1048576,
                        cache / 1048576,
                        purge_threshold / 1048576
                    );
                }
            }
            PolishCmd::Translate {
                text,
                target,
                reply,
            } => {
                // MLX 非线程安全: 与 ASR 线程串行化
                let _mlx_guard = mlx_lock.lock().unwrap();
                if state.is_none() {
                    eprintln!("[polish] model not loaded, loading from {}...", model_dir.display());
                    let _ = events.send(
                        serde_json::json!({"type":"polish_status","loaded":false,"loading":true})
                            .to_string(),
                    );
                    match load(&model_dir) {
                        Ok(m) => {
                            state = Some(m);
                            model_loaded.store(true, Ordering::SeqCst);
                            eprintln!("[polish] model loaded");
                            let _ = events.send(
                                serde_json::json!({"type":"polish_status","loaded":true,"loading":false})
                                    .to_string(),
                            );
                        }
                        Err(e) => {
                            let _ = events.send(
                                serde_json::json!({"type":"polish_status","loaded":false,"loading":false})
                                    .to_string(),
                            );
                            let _ = reply.send(Err(format!("增强模型加载失败: {e}")));
                            continue;
                        }
                    }
                }
                let t0 = std::time::Instant::now();
                let result = translate(&mut state.as_mut().unwrap(), &text, &target);
                let elapsed = t0.elapsed().as_secs_f32();
                match &result {
                    Ok((out, tokens)) => {
                        let tok_s = *tokens as f32 / elapsed.max(0.001);
                        eprintln!(
                            "[translate] {} chars -> {} chars, {} tokens in {:.1}s ({:.1} tok/s)",
                            text.chars().count(),
                            out.chars().count(),
                            tokens,
                            elapsed,
                            tok_s
                        );
                        let _ = events.send(
                            serde_json::json!({"type":"tok_stat","mode":"translate","tokens":tokens,"secs":(elapsed*10.0).round()/10.0,"tok_s":(tok_s*10.0).round()/10.0})
                                .to_string(),
                        );
                    }
                    Err(e) => eprintln!("[translate] error: {e}"),
                }
                let _ = reply.send(result.map(|(s, _)| s));
                // 内存监控: 缓存池超过阈值时清理 (含图编译缓存)
                let (active, cache) = mem::snapshot();
                if cache > purge_threshold {
                    mem::purge_caches();
                    eprintln!(
                        "[enhance] 内存缓存清理: active={}MB cache={}MB (阈值 {}MB)",
                        active / 1048576,
                        cache / 1048576,
                        purge_threshold / 1048576
                    );
                }
            }
            PolishCmd::Unload => {
                let _mlx_guard = mlx_lock.lock().unwrap();
                if state.take().is_some() {
                    model_loaded.store(false, Ordering::SeqCst);
                    eprintln!("[polish] model unloaded (idle timeout / manual)");
                    let _ = events.send(
                        serde_json::json!({"type":"polish_status","loaded":false,"loading":false}).to_string(),
                    );
                    mem::purge_caches();
                }
            }
        }
    }
}

fn load(model_dir: &PathBuf) -> Result<PolishModel, String> {
    let tokenizer_file = model_dir.join("tokenizer.json");
    let tokenizer_config_file = model_dir.join("tokenizer_config.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_file)
        .map_err(|e| format!("tokenizer 加载失败: {e:?}"))?;
    let chat_template = load_model_chat_template_from_file(&tokenizer_config_file)
        .map_err(|e| format!("chat template 读取失败: {e}"))?
        .ok_or_else(|| "tokenizer_config.json 中没有 chat template".to_string())?;
    let model = load_model(model_dir).map_err(|e| format!("模型加载失败: {e}"))?;
    Ok(PolishModel {
        model,
        tokenizer,
        chat_template,
    })
}

fn polish(
    state: &mut PolishModel,
    text: &str,
    language: &str,
) -> Result<(String, usize), String> {
    // "Auto" 语言识别: 按转写文本的书写系统选择润色提示词
    let use_zh = if language == "Chinese" {
        true
    } else if language == "Auto" {
        text.chars()
            .any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))
    } else {
        false
    };
    let system = if use_zh { ZH_SYSTEM } else { EN_SYSTEM };
    let max_tokens = (text.chars().count() * 2).clamp(512, 2048);
    let user_msg = format!("{system}\n\n待整理文本（引号内）：\n\"{text}\"\n\n请直接输出整理后的文本：");
    let (out, tokens) = run_llm(state, &user_msg, max_tokens)?;
    Ok((clean_output(&out), tokens))
}

fn translate(
    state: &mut PolishModel,
    text: &str,
    target: &str,
) -> Result<(String, usize), String> {
    let is_cjk_target = target.contains('中')
        || target == "Chinese"
        || target == "日本語"
        || target == "한국어";
    let system = if is_cjk_target {
        format!("你是一位专业翻译。把用户提供的文本翻译成{target}，保持原意，保留专业术语与数字，直接输出译文，不要任何解释。")
    } else {
        format!("You are a professional translator. Translate the user's text into {target}. Keep the original meaning, preserve technical terms and numbers, and output only the translation.")
    };
    let max_tokens = (text.chars().count() * 2).clamp(512, 2048);
    let user_msg = format!("{system}\n\n待翻译文本（引号内）：\n\"{text}\"\n\n请直接输出译文：");
    let (out, tokens) = run_llm(state, &user_msg, max_tokens)?;
    Ok((clean_output(&out), tokens))
}

/// 剥离常见的提示词泄漏片段（模型偶尔会回显指令前缀/客套话）
fn clean_output(out: &str) -> String {
    let mut s = out.trim().to_string();
    let prefixes: &[&str] = &[
        "好的，以下是翻译：",
        "好的，以下是译文：",
        "好的，以下是润色后的文本：",
        "好的，以下是整理后的文本：",
        "好的，以下是翻译结果：",
        "以下是翻译：",
        "以下是译文：",
        "以下是翻译结果：",
        "以下是润色后的文本：",
        "以下是整理后的文本：",
        "翻译如下：",
        "译文如下：",
        "翻译结果：",
        "译文：",
        "润色后的文本：",
        "整理后的文本：",
        "Sure, here is the translation:",
        "Sure, here's the translation:",
        "Here is the translation:",
        "Here's the translation:",
        "Translation:",
        "Polished text:",
    ];
    let suffixes: &[&str] = &[
        "希望对你有所帮助。",
        "希望对您有帮助。",
        "希望有帮助。",
        "Hope this helps!",
        "I hope this helps.",
        "如有需要请告诉我。",
        "如需要进一步调整请告诉我。",
    ];
    loop {
        let mut changed = false;
        for p in prefixes {
            if s.starts_with(p) {
                s = s[p.len()..].trim().to_string();
                changed = true;
            }
        }
        for suf in suffixes {
            if s.ends_with(suf) {
                s = s[..s.len() - suf.len()].trim().to_string();
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    s
}

fn run_llm(
    state: &mut PolishModel,
    user_msg: &str,
    max_tokens: usize,
) -> Result<(String, usize), String> {
    let conversations = vec![Conversation {
        role: Role::User,
        content: user_msg,
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

    // 关闭 Qwen3 思考模式: 与 chat template 中 enable_thinking=false 等价,
    // 在生成提示前注入空的 <think></think> 块, 避免长思考链拖慢/卡死翻译
    let no_think_suffix = "\n<think>\n\n</think>\n\n";
    if let Ok(suffix_enc) = state.tokenizer.encode(no_think_suffix, false) {
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
        // 重复退化保护: 连续 10 个相同 token 即停止
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_leaked_prefixes() {
        assert_eq!(
            clean_output("好的，以下是翻译：今天天气很好。"),
            "今天天气很好。"
        );
        assert_eq!(
            clean_output("Here is the translation: The weather is nice."),
            "The weather is nice."
        );
        assert_eq!(
            clean_output("译文：这是测试。希望对你有所帮助。"),
            "这是测试。"
        );
    }

    #[test]
    fn keeps_legit_content() {
        assert_eq!(clean_output("今天天气很好。"), "今天天气很好。");
        assert_eq!(
            clean_output("OK, let's go."),
            "OK, let's go."
        );
    }

    #[test]
    fn multiple_prefixes() {
        assert_eq!(
            clean_output("好的，以下是译文：译文：结果。"),
            "结果。"
        );
    }
}
