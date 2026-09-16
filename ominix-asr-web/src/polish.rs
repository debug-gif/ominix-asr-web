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

extern "C" {
    fn mlx_clear_cache() -> i32;
}

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
    ) -> Self {
        let (tx, rx) = channel::<PolishCmd>();
        let model_loaded = Arc::new(AtomicBool::new(false));
        let flag = model_loaded.clone();
        let handle = std::thread::Builder::new()
            .name("polish-worker".into())
            .spawn(move || worker_loop(rx, model_dir, flag, events, mlx_lock))
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
                match &result {
                    Ok(out) => eprintln!(
                        "[polish] {} chars -> {} chars ({}ms)",
                        text.chars().count(),
                        out.chars().count(),
                        t0.elapsed().as_millis()
                    ),
                    Err(e) => eprintln!("[polish] error: {e}"),
                }
                let _ = reply.send(result);
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
                match &result {
                    Ok(out) => eprintln!(
                        "[translate] {} chars -> {} ({}ms)",
                        text.chars().count(),
                        target,
                        t0.elapsed().as_millis()
                    ),
                    Err(e) => eprintln!("[translate] error: {e}"),
                }
                let _ = reply.send(result);
            }
            PolishCmd::Unload => {
                let _mlx_guard = mlx_lock.lock().unwrap();
                if state.take().is_some() {
                    model_loaded.store(false, Ordering::SeqCst);
                    eprintln!("[polish] model unloaded (idle timeout / manual)");
                    let _ = events.send(
                        serde_json::json!({"type":"polish_status","loaded":false,"loading":false}).to_string(),
                    );
                    unsafe {
                        mlx_clear_cache();
                    }
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

fn polish(state: &mut PolishModel, text: &str, language: &str) -> Result<String, String> {
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
    let user_msg = format!("{system}\n\n以下是需要整理的语音转写草稿：\n{text}");
    run_llm(state, &user_msg)
}

fn translate(state: &mut PolishModel, text: &str, target: &str) -> Result<String, String> {
    let is_cjk_target = target.contains('中')
        || target == "Chinese"
        || target == "日本語"
        || target == "한국어";
    let system = if is_cjk_target {
        format!("你是一位专业翻译。把用户提供的文本翻译成{target}，保持原意，保留专业术语与数字，直接输出译文，不要任何解释。")
    } else {
        format!("You are a professional translator. Translate the user's text into {target}. Keep the original meaning, preserve technical terms and numbers, and output only the translation.")
    };
    let user_msg = format!("{system}\n\n{text}");
    run_llm(state, &user_msg)
}

fn run_llm(state: &mut PolishModel, user_msg: &str) -> Result<String, String> {
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

    let max_tokens = 512;
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
        Ok(out)
    }
}
