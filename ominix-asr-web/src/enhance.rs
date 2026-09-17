//! 段级频谱增强 (P1): 维纳滤波抑制晚期混响尾 + 响度归一
//!
//! 晚期混响呈平稳噪声特性, 用"能量最安静的帧"估计噪声谱,
//! 逐帧维纳增益压制, 保留早期反射与直达声。

use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;

const FRAME: usize = 400; // 25ms @ 16kHz
const HOP: usize = 160; // 10ms
const FFT_N: usize = 512; // 零填充到 512

const OVERSUB: f32 = 1.3; // 过减因子 (温和, 防削掉弱音节)
const NOISE_BIAS: f32 = 1.2; // 噪声均值→瞬时值偏差修正 (Rayleigh)
const GAIN_FLOOR: f32 = 0.12; // 最大压制约 18dB (留有余地, 防"水声")
const TARGET_RMS: f32 = 0.08; // 响度归一目标 (-22 dBFS)
const GATE_RATIO: f32 = 0.12; // 噪声/信号幅度比低于此值 → 直通(干净音频不受损)

fn hann(i: usize, n: usize) -> f32 {
    0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos()
}

/// 对一段完整语音缓冲做离线增强: 频谱压制 + 响度归一
pub fn enhance(samples: &[f32]) -> Vec<f32> {
    let suppressed = suppress(samples);
    normalize(&suppressed)
}

/// 频谱压制 (维纳滤波, 抑制平稳噪声与晚期混响尾)
pub fn suppress(samples: &[f32]) -> Vec<f32> {
    if samples.len() < FRAME * 2 {
        return samples.to_vec();
    }
    let n_frames = 1 + (samples.len() - FRAME) / HOP;
    let n_bins = FFT_N / 2 + 1;

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_N);
    let ifft = planner.plan_fft_inverse(FFT_N);

    let win: Vec<f32> = (0..FRAME).map(|i| hann(i, FRAME)).collect();

    // ── 1. 分帧 FFT ────────────────────────────────────────────
    let mut frames: Vec<Vec<Complex32>> = Vec::with_capacity(n_frames);
    let mut frame_energy: Vec<f32> = Vec::with_capacity(n_frames);
    for k in 0..n_frames {
        let start = k * HOP;
        let mut buf = vec![Complex32::new(0.0, 0.0); FFT_N];
        for i in 0..FRAME {
            buf[i] = Complex32::new(samples[start + i] * win[i], 0.0);
        }
        fft.process(&mut buf);
        let e: f32 = buf.iter().map(|c| c.norm_sqr()).sum();
        frame_energy.push(e);
        frames.push(buf);
    }

    // ── 2. 噪声谱估计 ──────────────────────────────────────────
    // 首选段尾静音区(最后 25% 帧, VAD 挂起静音是天然噪声参考);
    // 若尾部能量高(无静音尾), 回退到最安静的 15% 帧
    let mean_frame_mag: f32 = {
        let mut s = 0.0f32;
        let mut c = 0u32;
        for f in &frames {
            for b in 0..n_bins {
                s += f[b].norm();
                c += 1;
            }
        }
        s / c as f32
    };

    let mut order: Vec<usize> = (0..n_frames).collect();
    order.sort_by(|&a, &b| frame_energy[a].partial_cmp(&frame_energy[b]).unwrap());
    let quiet_n = (n_frames as f32 * 0.15).max(1.0) as usize;

    let tail_start = n_frames.saturating_sub(n_frames / 4);
    let tail_energy: f32 = (tail_start..n_frames).map(|k| frame_energy[k]).sum::<f32>()
        / (n_frames - tail_start).max(1) as f32;
    let quiet_energy = order[..quiet_n]
        .iter()
        .map(|&k| frame_energy[k])
        .sum::<f32>()
        / quiet_n as f32;

    let use_tail = tail_energy < quiet_energy * 1.8;
    let noise_frames: Vec<usize> = if use_tail {
        (tail_start..n_frames).collect()
    } else {
        order[..quiet_n].to_vec()
    };

    let mut noise = vec![0.0f32; n_bins];
    for &k in &noise_frames {
        for b in 0..n_bins {
            noise[b] += frames[k][b].norm();
        }
    }
    for b in noise.iter_mut() {
        *b = *b / noise_frames.len() as f32 * NOISE_BIAS;
    }
    let mean_noise: f32 = noise.iter().sum::<f32>() / n_bins as f32;

    // ── 2.5 智能门控: 噪声地板过低 → 干净音频, 直通不处理 ──────
    if mean_noise < mean_frame_mag * GATE_RATIO {
        return samples.to_vec();
    }

    // ── 3. 维纳增益 (功率谱减) ─────────────────────────────────
    // 注意: 频谱必须保持共轭对称, 增益同时作用于正负频率
    for frame in frames.iter_mut() {
        for b in 0..n_bins {
            let p = frame[b].norm_sqr();
            let n2 = (OVERSUB * noise[b]) * (OVERSUB * noise[b]);
            let g = if p > n2 { (p - n2) / p.max(1e-12) } else { 0.0 };
            let g = g.max(GAIN_FLOOR);
            frame[b] *= g;
            if b > 0 && b < FFT_N / 2 {
                frame[FFT_N - b] *= g;
            }
        }
    }

    // ── 4. IFFT + 重叠相加 ─────────────────────────────────────
    let mut out = vec![0.0f32; samples.len()];
    let mut wsum = vec![0.0f32; samples.len()];
    for (k, frame) in frames.iter().enumerate() {
        let mut buf = frame.clone();
        ifft.process(&mut buf);
        let start = k * HOP;
        for i in 0..FRAME {
            let v = buf[i].re / FFT_N as f32 * win[i];
            out[start + i] += v;
            wsum[start + i] += win[i] * win[i];
        }
    }
    for i in 0..out.len() {
        if wsum[i] > 1e-6 {
            out[i] /= wsum[i];
        }
    }
    out
}

/// 响度归一 (增益限制, 防病态放大)
pub fn normalize(samples: &[f32]) -> Vec<f32> {
    let rms = (samples.iter().map(|x| x * x).sum::<f32>() / samples.len().max(1) as f32).sqrt();
    if rms <= 1e-6 {
        return samples.to_vec();
    }
    let scale = (TARGET_RMS / rms).clamp(0.3, 4.0);
    samples.iter().map(|x| x * scale).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
    }

    fn noise(seed: &mut u32, n: usize, amp: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            out.push(((*seed >> 8) as f32 / 16777216.0 - 0.5) * amp);
        }
        out
    }

    #[test]
    fn short_input_passthrough() {
        let x = vec![0.1f32; 100];
        assert_eq!(suppress(&x), x);
    }

    #[test]
    fn pure_noise_is_attenuated() {
        // 温和参数下的纯噪声段: 保证有实质压制即可(≈-5dB),
        // 过度追求压制会引入音乐噪声损害 ASR
        let mut seed = 42u32;
        let x = noise(&mut seed, 16000, 0.1);
        let y = suppress(&x);
        assert!(
            rms(&y) < rms(&x) * 0.75,
            "噪声无压制: {} vs {}",
            rms(&y),
            rms(&x)
        );
    }

    #[test]
    fn clean_speech_passthrough_via_gate() {
        // 干净语音(无噪声底)必须直通, 不得削掉弱音节 — 回归测试
        let fs = 16000;
        let mut clean = vec![0.0f32; fs];
        for k in 0..5 {
            let start = k * 2400;
            for i in 0..640 {
                let t = i as f32 / fs as f32;
                clean[start + i] = 0.25 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            }
        }
        let y = suppress(&clean);
        let corr = {
            let (mut a, mut b, mut c) = (0.0f32, 0.0f32, 0.0f32);
            for i in 0..fs {
                a += clean[i] * y[i];
                b += clean[i] * clean[i];
                c += y[i] * y[i];
            }
            a / (b.sqrt() * c.sqrt() + 1e-9)
        };
        assert!(corr > 0.99, "干净语音被改动, 相关度 {corr}");
    }

    #[test]
    fn speech_bursts_preserved_gap_noise_reduced() {
        // 合成"音节突发"语音: 5 个 40ms 正弦突发 + 全程平稳噪声底
        let fs = 16000;
        let mut clean = vec![0.0f32; fs];
        for k in 0..5 {
            let start = k * 2400;
            for i in 0..640 {
                let t = i as f32 / fs as f32;
                clean[start + i] = 0.25 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            }
        }
        let mut seed = 7u32;
        let nz = noise(&mut seed, fs, 0.03);
        let noisy: Vec<f32> = clean.iter().zip(&nz).map(|(a, b)| a + b).collect();
        let y = suppress(&noisy);

        // 信号保留: 输出与干净信号相关度高
        let corr = {
            let (mut a, mut b, mut c) = (0.0f32, 0.0f32, 0.0f32);
            for i in 0..fs {
                a += clean[i] * y[i];
                b += clean[i] * clean[i];
                c += y[i] * y[i];
            }
            a / (b.sqrt() * c.sqrt() + 1e-9)
        };
        assert!(corr > 0.7, "语音被破坏, 相关度 {corr}");

        // 静音间隙的噪声底被压制 (间隙区间: 640..2400)
        let gap_before: f32 = noisy[640..2400].iter().map(|x| x * x).sum();
        let gap_after: f32 = y[640..2400].iter().map(|x| x * x).sum();
        assert!(
            gap_after < gap_before * 0.5,
            "噪声底未抑制: {} vs {}",
            gap_after,
            gap_before
        );
    }

    #[test]
    fn normalize_targets_rms() {
        // 输入 RMS 0.05 → 增益 1.6 (在限幅内) → 输出 ≈ 0.08
        let x = vec![0.05f32; 8000];
        let y = normalize(&x);
        let r = rms(&y);
        assert!((r - TARGET_RMS).abs() < 0.005, "归一目标偏差: {r}");

        // 病态放大受限于 4x
        let tiny = vec![0.0001f32; 8000];
        let y2 = normalize(&tiny);
        assert!(rms(&y2) / rms(&tiny) <= 4.01);
    }
}
