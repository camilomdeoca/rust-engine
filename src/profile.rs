use std::time::Instant;

use memory_stats::memory_stats;

fn get_current_process_memory_usage() -> usize {
    if let Some(stats) = memory_stats() {
        return stats.physical_mem;
    }
    0
}

// prints caller line and memory usage in MB.
#[allow(dead_code)]
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

#[allow(dead_code)]
pub struct ProfileTimer<'a> {
    start_time: Instant,
    label: &'a str,
}

impl<'a> ProfileTimer<'a> {
    #[allow(dead_code)]
    pub fn start(label: &'a str) -> Self {
        let start_time = Instant::now();
        Self {
            start_time,
            label,
        }
    }
}

impl Drop for ProfileTimer<'_> {
    fn drop(&mut self) {
        println!("{:20}: {:>10} ns", self.label, self.start_time.elapsed().as_nanos());
    }
}
