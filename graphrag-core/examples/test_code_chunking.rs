//! Test code chunking with tree-sitter
//!
//! Run with: cargo run --example test_code_chunking --features code

use graphrag_core::{CodeChunkerConfig, CodeLanguage, chunk_code};

fn main() {
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

    fn distance(&self, other: &Point) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

trait Drawable {
    fn draw(&self);
}

impl Drawable for Point {
    fn draw(&self) {
        println!("Point({}, {})", self.x, self.y);
    }
}
"#;

    println!("=== Rust Code Chunking Test ===\n");
    println!("Original code: {} chars\n", code.len());

    let config = CodeChunkerConfig::new(CodeLanguage::Rust).with_chunk_size(300);
    let chunks = chunk_code(code, &config).unwrap();

    println!("Split into {} chunks (max 300 chars each):\n", chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        println!("--- Chunk {} ({} chars) ---", i + 1, chunk.len());
        println!("{}", chunk);
        println!();
    }

    // Test Python too
    let python_code = r#"
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

    def multiply(self, x):
        self.value *= x
        return self

def main():
    calc = Calculator()
    result = calc.add(5).multiply(3).subtract(2).value
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
"#;

    println!("\n=== Python Code Chunking Test ===\n");
    println!("Original code: {} chars\n", python_code.len());

    let config = CodeChunkerConfig::new(CodeLanguage::Python).with_chunk_size(250);
    let chunks = chunk_code(python_code, &config).unwrap();

    println!("Split into {} chunks (max 250 chars each):\n", chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        println!("--- Chunk {} ({} chars) ---", i + 1, chunk.len());
        println!("{}", chunk);
        println!();
    }
}
