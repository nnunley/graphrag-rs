//! Code-aware chunking using tree-sitter for AST-based splitting
//!
//! Supports multiple programming languages with intelligent chunking that
//! respects code structure (functions, classes, modules).

use text_splitter::{CodeSplitter, Characters};

/// Supported programming languages for code chunking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeLanguage {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Go,
    C,
    Cpp,
}

impl CodeLanguage {
    /// Parse language from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "rs" => Some(Self::Rust),
            "py" => Some(Self::Python),
            "js" | "mjs" | "cjs" => Some(Self::JavaScript),
            "ts" | "tsx" => Some(Self::TypeScript),
            "go" => Some(Self::Go),
            "c" | "h" => Some(Self::C),
            "cpp" | "cc" | "cxx" | "hpp" | "hxx" => Some(Self::Cpp),
            _ => None,
        }
    }

    /// Parse language from name
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "rust" | "rs" => Some(Self::Rust),
            "python" | "py" => Some(Self::Python),
            "javascript" | "js" => Some(Self::JavaScript),
            "typescript" | "ts" => Some(Self::TypeScript),
            "go" | "golang" => Some(Self::Go),
            "c" => Some(Self::C),
            "cpp" | "c++" | "cxx" => Some(Self::Cpp),
            _ => None,
        }
    }

    /// Create a CodeSplitter for this language
    fn create_splitter(&self, chunk_size: usize) -> Result<CodeSplitter<Characters>, String> {
        let splitter = match self {
            Self::Rust => CodeSplitter::new(tree_sitter_rust::LANGUAGE, chunk_size),
            Self::Python => CodeSplitter::new(tree_sitter_python::LANGUAGE, chunk_size),
            Self::JavaScript => CodeSplitter::new(tree_sitter_javascript::LANGUAGE, chunk_size),
            Self::TypeScript => CodeSplitter::new(tree_sitter_typescript::LANGUAGE_TYPESCRIPT, chunk_size),
            Self::Go => CodeSplitter::new(tree_sitter_go::LANGUAGE, chunk_size),
            Self::C => CodeSplitter::new(tree_sitter_c::LANGUAGE, chunk_size),
            Self::Cpp => CodeSplitter::new(tree_sitter_cpp::LANGUAGE, chunk_size),
        };
        splitter.map_err(|e| format!("Failed to create code splitter: {}", e))
    }
}

/// Configuration for code chunking
#[derive(Debug, Clone)]
pub struct CodeChunkerConfig {
    /// Target chunk size in characters
    pub chunk_size: usize,
    /// Programming language
    pub language: CodeLanguage,
}

impl CodeChunkerConfig {
    pub fn new(language: CodeLanguage) -> Self {
        Self {
            chunk_size: 2000,
            language,
        }
    }

    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }
}

/// Chunk source code into AST-aware segments
///
/// Uses tree-sitter to parse the code and splits at semantically meaningful
/// boundaries like function definitions, class declarations, etc.
pub fn chunk_code(code: &str, config: &CodeChunkerConfig) -> Result<Vec<String>, String> {
    if code.is_empty() {
        return Ok(vec![]);
    }

    let splitter = config.language.create_splitter(config.chunk_size)?;
    Ok(splitter.chunks(code).map(String::from).collect())
}

/// Chunk code with automatic language detection from file extension
pub fn chunk_code_auto(code: &str, file_path: &str, chunk_size: usize) -> Result<Vec<String>, String> {
    let extension = file_path
        .rsplit('.')
        .next()
        .ok_or_else(|| "No file extension found".to_string())?;

    let language = CodeLanguage::from_extension(extension)
        .ok_or_else(|| format!("Unsupported file extension: {}", extension))?;

    let config = CodeChunkerConfig::new(language).with_chunk_size(chunk_size);
    chunk_code(code, &config)
}

/// Get list of supported languages
pub fn supported_languages() -> &'static [&'static str] {
    &["rust", "python", "javascript", "typescript", "go", "c", "cpp"]
}

/// Get supported file extensions
pub fn supported_extensions() -> &'static [&'static str] {
    &["rs", "py", "js", "mjs", "cjs", "ts", "tsx", "go", "c", "h", "cpp", "cc", "cxx", "hpp", "hxx"]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_from_extension() {
        assert_eq!(CodeLanguage::from_extension("rs"), Some(CodeLanguage::Rust));
        assert_eq!(CodeLanguage::from_extension("py"), Some(CodeLanguage::Python));
        assert_eq!(CodeLanguage::from_extension("js"), Some(CodeLanguage::JavaScript));
        assert_eq!(CodeLanguage::from_extension("ts"), Some(CodeLanguage::TypeScript));
        assert_eq!(CodeLanguage::from_extension("go"), Some(CodeLanguage::Go));
        assert_eq!(CodeLanguage::from_extension("c"), Some(CodeLanguage::C));
        assert_eq!(CodeLanguage::from_extension("cpp"), Some(CodeLanguage::Cpp));
        assert_eq!(CodeLanguage::from_extension("xyz"), None);
    }

    #[test]
    fn test_chunk_rust_code() {
        let code = r#"
fn main() {
    println!("Hello, world!");
}

fn add(a: i32, b: i32) -> i32 {
    a + b
}

struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}
"#;
        let config = CodeChunkerConfig::new(CodeLanguage::Rust).with_chunk_size(500);
        let chunks = chunk_code(code, &config).unwrap();
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(chunk.len() <= 600); // Allow some flexibility
        }
    }

    #[test]
    fn test_chunk_python_code() {
        let code = r#"
def hello():
    print("Hello, world!")

class Calculator:
    def __init__(self):
        self.value = 0

    def add(self, x):
        self.value += x
        return self

    def subtract(self, x):
        self.value -= x
        return self
"#;
        let config = CodeChunkerConfig::new(CodeLanguage::Python).with_chunk_size(300);
        let chunks = chunk_code(code, &config).unwrap();
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_chunk_empty_code() {
        let config = CodeChunkerConfig::new(CodeLanguage::Rust);
        let chunks = chunk_code("", &config).unwrap();
        assert!(chunks.is_empty());
    }
}
