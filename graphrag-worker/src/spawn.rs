//! Start `graphrag work` in the background after a write or a recall.

use crate::lock;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// `$GRAPHRAG_BIN`, then `graphrag` on PATH, then `~/.cargo/bin/graphrag`.
pub fn worker_binary() -> Option<PathBuf> {
    if let Some(bin) = std::env::var_os("GRAPHRAG_BIN") {
        return Some(PathBuf::from(bin));
    }
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            let candidate = dir.join("graphrag");
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    let cargo = dirs_home()?.join(".cargo/bin/graphrag");
    cargo.is_file().then_some(cargo)
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

/// Spawn a detached worker unless one already holds the lock. Returns
/// whether a worker was started. The child gets its own process group and
/// null stdio, so it outlives the caller and never blocks it.
pub fn spawn_detached(data_dir: &Path) -> Result<bool, String> {
    match lock::acquire(data_dir) {
        Ok(Some(guard)) => drop(guard),
        Ok(None) => return Ok(false),
        Err(e) => return Err(format!("worker lock: {e}")),
    }
    let bin = worker_binary().ok_or("no graphrag binary found (set GRAPHRAG_BIN)")?;
    Command::new(&bin)
        .arg("work")
        .env("GRAPHRAG_DATA_DIR", data_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .process_group(0)
        .spawn()
        .map_err(|e| format!("spawn {}: {e}", bin.display()))?;
    Ok(true)
}
