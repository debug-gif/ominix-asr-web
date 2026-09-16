use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use qwen3_asr_mlx::Qwen3ASR;

extern "C" {
    fn mlx_clear_cache() -> i32;
}

pub enum Cmd {
    Transcribe {
        samples: Vec<f32>,
        language: String,
        system: String,
        terms: Vec<String>,
        reply: std::sync::mpsc::Sender<Result<String, String>>,
    },
    Unload,
    Status(std::sync::mpsc::Sender<WorkerStatus>),
}

#[derive(Clone)]
pub struct WorkerStatus {
    pub model_loaded: bool,
    pub loading: bool,
}

pub struct AsrWorker {
    tx: Sender<Cmd>,
    pub model_loaded: Arc<AtomicBool>,
    _handle: JoinHandle<()>,
}

impl AsrWorker {
    pub fn start(model_dir: PathBuf, events: tokio::sync::broadcast::Sender<String>) -> Self {
        let (tx, rx) = channel::<Cmd>();
        let model_loaded = Arc::new(AtomicBool::new(false));
        let flag = model_loaded.clone();
        let handle = std::thread::Builder::new()
            .name("asr-worker".into())
            .spawn(move || worker_loop(rx, model_dir, flag, events))
            .expect("failed to spawn asr worker");
        AsrWorker {
            tx,
            model_loaded,
            _handle: handle,
        }
    }

    pub fn transcribe(
        &self,
        samples: Vec<f32>,
        language: String,
        system: String,
        terms: Vec<String>,
    ) -> std::sync::mpsc::Receiver<Result<String, String>> {
        let (reply, recv) = channel();
        let _ = self.tx.send(Cmd::Transcribe {
            samples,
            language,
            system,
            terms,
            reply,
        });
        recv
    }

    pub fn unload(&self) {
        let _ = self.tx.send(Cmd::Unload);
    }

    pub fn status(&self) -> WorkerStatus {
        let (reply, recv) = channel();
        if self.tx.send(Cmd::Status(reply)).is_ok() {
            if let Ok(s) = recv.recv_timeout(std::time::Duration::from_secs(5)) {
                return s;
            }
        }
        WorkerStatus {
            model_loaded: false,
            loading: false,
        }
    }
}

fn worker_loop(
    rx: Receiver<Cmd>,
    model_dir: PathBuf,
    model_loaded: Arc<AtomicBool>,
    events: tokio::sync::broadcast::Sender<String>,
) {
    let mut model: Option<Qwen3ASR> = None;
    while let Ok(cmd) = rx.recv() {
        match cmd {
            Cmd::Transcribe {
                samples,
                language,
                system,
                terms,
                reply,
            } => {
                if model.is_none() {
                    eprintln!("[asr] model not loaded, loading from {}...", model_dir.display());
                    let _ = events.send(
                        serde_json::json!({"type":"status","model_loaded":false,"loading":true})
                            .to_string(),
                    );
                    match Qwen3ASR::load(&model_dir) {
                        Ok(m) => {
                            model = Some(m);
                            model_loaded.store(true, Ordering::SeqCst);
                            eprintln!("[asr] model loaded");
                            let _ = events.send(
                                serde_json::json!({"type":"status","model_loaded":true,"loading":false})
                                    .to_string(),
                            );
                        }
                        Err(e) => {
                            model_loaded.store(false, Ordering::SeqCst);
                            let _ = events.send(
                                serde_json::json!({"type":"status","model_loaded":false,"loading":false})
                                    .to_string(),
                            );
                            let _ = reply.send(Err(format!("模型加载失败: {e}")));
                            continue;
                        }
                    }
                }
                let t0 = std::time::Instant::now();
                let system_prompt = if system.trim().is_empty() {
                    None
                } else {
                    Some(system.trim())
                };
                let config = qwen3_asr_mlx::SamplingConfig::default();
                let result = model
                    .as_mut()
                    .unwrap()
                    .transcribe_samples_with_system(&samples, &language, system_prompt, &config)
                    .map(correct_terms_with(&terms))
                    .map_err(|e| format!("转写失败: {e}"));
                if let Ok(text) = &result {
                    eprintln!(
                        "[asr] {}s audio -> {} ({}ms, {:.1}x realtime)",
                        samples.len() as f32 / 16000.0,
                        text,
                        t0.elapsed().as_millis(),
                        samples.len() as f32 / 16000.0 / t0.elapsed().as_secs_f32()
                    );
                }
                let _ = reply.send(result);
            }
            Cmd::Unload => {
                if model.take().is_some() {
                    model_loaded.store(false, Ordering::SeqCst);
                    eprintln!("[asr] model unloaded (idle timeout / manual)");
                    let _ = events.send(
                        serde_json::json!({"type":"status","model_loaded":false,"loading":false})
                            .to_string(),
                    );
                    unsafe {
                        mlx_clear_cache();
                    }
                    eprintln!("[asr] mlx memory cache cleared");
                }
            }
            Cmd::Status(reply) => {
                let _ = reply.send(WorkerStatus {
                    model_loaded: model_loaded.load(Ordering::SeqCst),
                    loading: false,
                });
            }
        }
    }
}

pub fn touch(activity: &Arc<Mutex<std::time::Instant>>) {
    if let Ok(mut t) = activity.lock() {
        *t = std::time::Instant::now();
    }
}

pub fn idle_secs(activity: &Arc<Mutex<std::time::Instant>>) -> u64 {
    activity
        .lock()
        .map(|t| t.elapsed().as_secs())
        .unwrap_or(0)
}

// ── 术语纠错 ────────────────────────────────────────────────────

fn levenshtein(a: &[char], b: &[char]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            cur[j + 1] = (prev[j + 1] + 1)
                .min(cur[j] + 1)
                .min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// 术语纠错: 每行一个词, 或 "常见错误写法|正确写法" 成对。
/// 1) 精确替换错误写法; 2) 对术语做滑动窗口模糊匹配 (编辑距离阈值随词长放宽)。
fn correct_terms(text: &str, terms: &[String]) -> String {
    if terms.is_empty() {
        return text.to_string();
    }
    let mut pairs: Vec<(String, String)> = Vec::new();
    for raw in terms {
        let raw = raw.trim();
        if raw.is_empty() {
            continue;
        }
        if let Some((w, r)) = raw.split_once('|') {
            let w = w.trim().to_string();
            let r = r.trim().to_string();
            if !w.is_empty() && !r.is_empty() {
                pairs.push((w, r));
            }
        } else {
            pairs.push((raw.to_string(), raw.to_string()));
        }
    }

    let mut t = text.to_string();
    for (w, r) in &pairs {
        if w != r && t.contains(w.as_str()) {
            t = t.replace(w.as_str(), r.as_str());
        }
    }
    for (w, r) in &pairs {
        let wc: Vec<char> = w.chars().collect();
        let n = wc.len();
        if n < 3 || n > 24 {
            continue;
        }
        let threshold = if n >= 6 { 2 } else { 1 };
        let chars: Vec<char> = t.chars().collect();
        let mut out = String::new();
        let mut i = 0;
        while i < chars.len() {
            if i + n <= chars.len() {
                let window: Vec<char> = chars[i..i + n].to_vec();
                let d = levenshtein(&window, &wc);
                if d > 0 && d <= threshold {
                    out.push_str(r);
                    i += n;
                    continue;
                }
            }
            out.push(chars[i]);
            i += 1;
        }
        t = out;
    }
    t
}

fn correct_terms_with(terms: &[String]) -> impl Fn(String) -> String + '_ {
    move |text| correct_terms(&text, terms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_pair_replacement() {
        let terms = vec!["神剑网络|神经网络".to_string()];
        assert_eq!(
            correct_terms("这是神剑网络的测试", &terms),
            "这是神经网络的测试"
        );
    }

    #[test]
    fn fuzzy_canonical_snap() {
        let terms = vec![
            "神经网络".to_string(),
            "卷积".to_string(),
            "transformer".to_string(),
        ];
        // 单个错字 → 纠正
        assert_eq!(
            correct_terms("我们训练神经网路模型", &terms),
            "我们训练神经网络模型"
        );
        // 两字词不做模糊 (阈值0), 不误伤
        assert_eq!(correct_terms("卷积计算", &terms), "卷积计算");
    }

    #[test]
    fn english_fuzzy() {
        let terms = vec!["transformer".to_string()];
        assert_eq!(
            correct_terms("the transofrmer model", &terms),
            "the transformer model"
        );
    }

    #[test]
    fn empty_terms_passthrough() {
        assert_eq!(correct_terms("原样输出", &[]), "原样输出");
    }
}
