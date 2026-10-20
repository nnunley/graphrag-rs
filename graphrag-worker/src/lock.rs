//! Single-instance lock: an advisory `flock` on `<data_dir>/worker.lock`.

use std::fs::{File, OpenOptions, TryLockError};
use std::io;
use std::path::Path;

/// Held while a worker runs; dropping it releases the lock.
pub struct LockGuard(#[allow(dead_code)] File);

/// `Ok(None)` when another process (or handle) holds the lock.
pub fn acquire(data_dir: &Path) -> io::Result<Option<LockGuard>> {
    std::fs::create_dir_all(data_dir)?;
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(data_dir.join("worker.lock"))?;
    match file.try_lock() {
        Ok(()) => Ok(Some(LockGuard(file))),
        Err(TryLockError::WouldBlock) => Ok(None),
        Err(TryLockError::Error(e)) => Err(e),
    }
}
