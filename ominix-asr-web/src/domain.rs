use std::path::{Path, PathBuf};
use std::time::SystemTime;

#[derive(Clone, Debug)]
pub struct DomainConfig {
    pub system: String,
    pub terms: Vec<String>,
}

pub const DEFAULT_TERMS: &[&str] = &[
    "神经网络", "深度学习", "机器学习", "卷积神经网络", "循环神经网络", "transformer",
    "注意力机制", "反向传播", "梯度下降", "损失函数", "激活函数", "嵌入向量",
    "大语言模型", "自然语言处理", "计算机视觉", "语音识别", "图像分割",
    "芯片设计", "半导体", "集成电路", "中央处理器", "图形处理器", "神经网络处理单元",
    "量子计算", "云计算", "分布式系统", "数据库", "编译器", "操作系统",
    "算法复杂度", "数据结构", "哈希表", "二叉树", "动态规划", "排序算法",
    "傅里叶变换", "拉普拉斯变换", "微分方程", "线性代数", "矩阵运算", "特征值",
    "区块链", "密码学", "加密算法", "网络安全", "人工智能", "智能体",
];

/// 解析 domain.txt:
/// - `[system]` 段: 注入转写模型的领域提示词 (可多行)
/// - `[terms]` 段: 每行一个术语, 支持 "常见错误|正确写法" 配对
/// - `#` 开头为注释
pub fn parse_domain(content: &str) -> (String, Vec<String>) {
    let mut system = String::new();
    let mut terms = Vec::new();
    let mut section = "";
    for line in content.lines() {
        let l = line.trim();
        if l.is_empty() {
            continue;
        }
        if l.starts_with("[system]") {
            section = "system";
            continue;
        }
        if l.starts_with("[terms]") {
            section = "terms";
            continue;
        }
        if l.starts_with('#') {
            continue;
        }
        match section {
            "system" => {
                if !system.is_empty() {
                    system.push('\n');
                }
                system.push_str(l);
            }
            "terms" => terms.push(l.to_string()),
            _ => {}
        }
    }
    (system, terms)
}

pub fn serialize_domain(cfg: &DomainConfig) -> String {
    let mut out = String::new();
    out.push_str("# OminiX 领域优化配置\n");
    out.push_str("# [system] 段: 注入转写模型的领域提示词(可多行, 留空则不注入)\n");
    out.push_str("# [terms] 段: 每行一个术语; 可用 \"常见错误|正确写法\" 配对做纠错\n");
    out.push_str("# 以 # 开头的行是注释。保存文件后 5 秒内自动生效。\n\n");
    out.push_str("[system]\n");
    out.push_str(&cfg.system);
    if !cfg.system.ends_with('\n') {
        out.push('\n');
    }
    out.push_str("\n[terms]\n");
    for t in &cfg.terms {
        out.push_str(t);
        out.push('\n');
    }
    out
}

/// 读取配置文件; 不存在时创建带默认术语表的文件。
pub fn load_domain_file(path: &Path) -> Result<(DomainConfig, Option<SystemTime>), String> {
    if !path.exists() {
        let cfg = DomainConfig {
            system: String::new(),
            terms: DEFAULT_TERMS.iter().map(|s| s.to_string()).collect(),
        };
        std::fs::write(path, serialize_domain(&cfg))
            .map_err(|e| format!("创建领域配置文件失败: {e}"))?;
    }
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("读取领域配置文件失败: {e}"))?;
    let (system, terms) = parse_domain(&content);
    let mtime = std::fs::metadata(path).ok().and_then(|m| m.modified().ok());
    Ok((DomainConfig { system, terms }, mtime))
}

pub fn file_mtime(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).ok().and_then(|m| m.modified().ok())
}

pub fn make_absolute(p: PathBuf) -> PathBuf {
    if p.is_absolute() {
        p
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(&p))
            .unwrap_or(p)
    }
}
