//! Emit the linker arguments a Python extension module needs.
//!
//! On macOS a cdylib that will be loaded by the interpreter must defer its
//! Python symbols to load time (`-undefined dynamic_lookup`) rather than link
//! libpython directly. `pyo3-build-config` knows the right flags per target,
//! so a plain `cargo build --workspace` succeeds without every developer
//! needing a linkable libpython, and maturin builds an importable wheel.
fn main() {
    pyo3_build_config::add_extension_module_link_args();
}
