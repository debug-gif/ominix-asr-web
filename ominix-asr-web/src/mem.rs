//! MLX 内存诊断与控制（全局原子统计, 任何线程可读）

use std::sync::atomic::{AtomicU64, Ordering};

extern "C" {
    fn mlx_get_active_memory(res: *mut usize) -> i32;
    fn mlx_get_cache_memory(res: *mut usize) -> i32;
    fn mlx_set_memory_limit(res: *mut usize, limit: usize) -> i32;
    fn mlx_clear_cache() -> i32;
    fn mlx_detail_compile_clear_cache() -> i32;
}

pub struct MemoryStats {
    pub active: AtomicU64,
    pub cache: AtomicU64,
}

pub static MEM: MemoryStats = MemoryStats {
    active: AtomicU64::new(0),
    cache: AtomicU64::new(0),
};

/// 读取当前 active/cache 内存 (字节)
pub fn snapshot() -> (usize, usize) {
    let (mut a, mut c) = (0usize, 0usize);
    unsafe {
        let _ = mlx_get_active_memory(&mut a);
        let _ = mlx_get_cache_memory(&mut c);
    }
    MEM.active.store(a as u64, Ordering::Relaxed);
    MEM.cache.store(c as u64, Ordering::Relaxed);
    (a, c)
}

/// 设置 active 内存软上限 (字节)。超过时 MLX 自动释放缓存池。
pub fn set_memory_limit(limit: usize) -> i32 {
    let mut old = 0usize;
    unsafe { mlx_set_memory_limit(&mut old, limit) }
}

/// 立即释放缓存池 + 图编译缓存。必须在 mlx_lock 保护下调用。
pub fn purge_caches() {
    unsafe {
        let _ = mlx_clear_cache();
        let _ = mlx_detail_compile_clear_cache();
    }
    snapshot();
}

pub fn active_mb() -> u64 {
    MEM.active.load(Ordering::Relaxed) / (1024 * 1024)
}

pub fn cache_mb() -> u64 {
    MEM.cache.load(Ordering::Relaxed) / (1024 * 1024)
}
