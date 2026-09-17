use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use qwen3_asr_mlx::Qwen3ASR;

use crate::enhance;
use crate::mem;

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
    pub fn start(
        model_dir: PathBuf,
        events: tokio::sync::broadcast::Sender<String>,
        mlx_lock: Arc<Mutex<()>>,
        purge_threshold: usize,
        dereverb: bool,
    ) -> Self {
        let (tx, rx) = channel::<Cmd>();
        let model_loaded = Arc::new(AtomicBool::new(false));
        let flag = model_loaded.clone();
        let handle = std::thread::Builder::new()
            .name("asr-worker".into())
            .spawn(move || worker_loop(rx, model_dir, flag, events, mlx_lock, purge_threshold, dereverb))
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
    mlx_lock: Arc<Mutex<()>>,
    purge_threshold: usize,
    dereverb: bool,
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
                // MLX 非线程安全: 与润色/翻译线程串行化
                let _mlx_guard = mlx_lock.lock().unwrap();
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
                // P1: 段级频谱增强(去晚期混响) + 响度归一
                let samples = if dereverb { enhance::enhance(&samples) } else { samples };
                let system_prompt = if system.trim().is_empty() {
                    None
                } else {
                    Some(system.trim())
                };
                // 限制最大输出 token, 防止极端情况 KV 缓存膨胀 (默认 8192 → 1024)
                let config = qwen3_asr_mlx::SamplingConfig {
                    temperature: 0.0,
                    max_tokens: 1024,
                };
                let result = model
                    .as_mut()
                    .unwrap()
                    .transcribe_samples_with_system(&samples, &language, system_prompt, &config)
                    .map(correct_terms_with(&terms))
                    .map_err(|e| format!("转写失败: {e}"));
                // 内存监控: 缓存池超过阈值时清理 (含图编译缓存)
                let (active, cache) = mem::snapshot();
                if cache > purge_threshold {
                    mem::purge_caches();
                    eprintln!(
                        "[asr] 内存缓存清理: active={}MB cache={}MB (阈值 {}MB)",
                        active / 1048576,
                        cache / 1048576,
                        purge_threshold / 1048576
                    );
                }
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
                    mem::purge_caches();
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
    lev(a, b)
}

fn lev<T: PartialEq>(a: &[T], b: &[T]) -> usize {
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

use pinyin::ToPinyin;

/// 拼音序列(不带声调); 含非汉字字符时返回 None
fn pinyin_plain_seq(s: &str) -> Option<Vec<String>> {
    let mut out = Vec::new();
    for c in s.chars() {
        out.push(c.to_pinyin()?.plain().to_string());
    }
    Some(out)
}

/// 拼音序列(带声调); 含非汉字字符时返回 None
fn pinyin_tone_seq(s: &str) -> Option<Vec<String>> {
    let mut out = Vec::new();
    for c in s.chars() {
        out.push(c.to_pinyin()?.with_tone().to_string());
    }
    Some(out)
}

/// L0: 拼音近音匹配 — 中文同音/近音错字比字符编辑距离更常见
fn pinyin_match(w: &str, window: &str, n: usize) -> bool {
    let (Some(wp), Some(up)) = (pinyin_plain_seq(w), pinyin_plain_seq(window)) else {
        return false;
    };
    if n == 2 {
        // 双字词要求声调完全一致, 防止"响亮→向量"这类误伤
        match (pinyin_tone_seq(w), pinyin_tone_seq(window)) {
            (Some(wt), Some(ut)) => wt == ut,
            _ => false,
        }
    } else {
        let d = lev(&wp, &up);
        d == 0 || d <= 1
    }
}

/// 术语纠错: 每行一个词, 或 "常见错误写法|正确写法" 成对。
/// 1) 精确替换错误写法;
/// 2) 滑动窗口模糊匹配: 字符编辑距离 + 拼音近音匹配 (L0)
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
        if n < 2 || n > 24 {
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
                let py = d > 0 && pinyin_match(w, &window.iter().collect::<String>(), n);
                if (d > 0 && d <= threshold) || py {
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

    // ── L0: 拼音近音匹配 ────────────────────────────────────────

    #[test]
    fn pinyin_homophone_two_chars() {
        // 相量(xiàng liàng) 与 向量(xiàng liàng) 声调一致 → 纠正
        let terms = vec!["向量".to_string()];
        assert_eq!(correct_terms("这是相量计算", &terms), "这是向量计算");
        // 响亮(xiǎng liàng) 声调不同 → 不误伤
        assert_eq!(correct_terms("声音很响亮", &terms), "声音很响亮");
    }

    #[test]
    fn pinyin_near_homophone_long_term() {
        // 神精网络(shén jīng wǎng luò) vs 神经网络(shén jīng wǎng luò) 拼音一致
        let terms = vec!["神经网络".to_string()];
        assert_eq!(correct_terms("使用神精网络模型", &terms), "使用神经网络模型");
    }

    #[test]
    fn pinyin_one_syllable_edit() {
        // 量子记算(jì suàn 相近) → 量子计算, 3字词允许 1 个音节差异
        let terms = vec!["量子计算".to_string()];
        assert_eq!(correct_terms("量子记算技术", &terms), "量子计算技术");
    }

    #[test]
    fn pinyin_pair_wrong_right() {
        let terms = vec!["相量|向量".to_string()];
        assert_eq!(correct_terms("相量分析", &terms), "向量分析");
    }
}
