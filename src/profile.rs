use memory_stats::memory_stats;

fn get_current_process_memory_usage() -> usize {
    if let Some(stats) = memory_stats() {
        return stats.physical_mem;
    }
    0
}

// prints caller line and memory usage in MB.
pub fn debug_memory_usage(line: u32, file: &'static str) {
    let mem_usage = get_current_process_memory_usage();
    println!(
        "[Line {}] {}: RAM usage: {} MB",
        line,
        file,
        mem_usage / 1024 / 1024
    );
}

#[macro_export]
macro_rules! debug_memory {
    () => {
        $crate::profile::debug_memory_usage(line!(), file!());
    };
}
