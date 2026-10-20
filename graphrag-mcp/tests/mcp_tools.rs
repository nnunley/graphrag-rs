//! Task 17: MCP surfaces, driven over stdio JSON-RPC.

use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant};

struct Server {
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<String>,
    next_id: u64,
}

impl Server {
    fn start(data_dir: &Path) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_graphrag-mcp"))
            .env("GRAPHRAG_DATA_DIR", data_dir)
            .env("GRAPHRAG_BIN", "/usr/bin/true")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("server starts");
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let mut s = Self {
            child,
            stdin,
            lines: rx,
            next_id: 1,
        };
        s.request("initialize", json!({"protocolVersion": "2024-11-05", "capabilities": {"sampling": {}}, "clientInfo": {"name": "test", "version": "0"}}), Duration::from_secs(10));
        s
    }

    /// Send a request; return every line received until its response, and the response.
    fn request(&mut self, method: &str, params: Value, within: Duration) -> (Vec<String>, Value) {
        let id = self.next_id;
        self.next_id += 1;
        let msg = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        writeln!(self.stdin, "{msg}").unwrap();
        self.stdin.flush().unwrap();
        let deadline = Instant::now() + within;
        let mut seen = Vec::new();
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            let line = self.lines.recv_timeout(left).unwrap_or_else(|_| {
                panic!("no response to {method} within {within:?}; saw {seen:?}")
            });
            let v: Value = serde_json::from_str(&line).unwrap_or(Value::Null);
            seen.push(line);
            if v["id"] == json!(id) && (v.get("result").is_some() || v.get("error").is_some()) {
                return (seen, v);
            }
        }
    }

    fn call(&mut self, tool: &str, args: Value, within: Duration) -> (Vec<String>, String) {
        let (seen, v) = self.request(
            "tools/call",
            json!({"name": tool, "arguments": args}),
            within,
        );
        assert!(v.get("error").is_none(), "{tool} failed: {v}");
        let text = v["result"]["content"][0]["text"]
            .as_str()
            .unwrap_or("")
            .to_string();
        (seen, text)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

fn data_dir(label: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let d = std::env::temp_dir().join(format!(
        "graphrag-mcp-test-{label}-{}-{nanos}",
        std::process::id()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn add_document_does_not_sample() {
    let dir = data_dir("no-sample");
    let mut s = Server::start(&dir);
    let (seen, text) = s.call(
        "graphrag_add_document",
        json!({"content": "# Leiden\nLeiden refines partitions.\n", "source": "t", "groups": ["papers"]}),
        Duration::from_secs(2),
    );
    assert!(
        seen.iter().all(|l| !l.contains("sampling/createMessage")),
        "{seen:?}"
    );
    assert!(text.contains("document_id"), "{text}");
}

#[test]
fn add_document_creates_sections() {
    let dir = data_dir("sections");
    let mut s = Server::start(&dir);
    s.call(
        "graphrag_add_document",
        json!({"content": "# A\nalpha text\n## B\nbeta text\n", "groups": ["papers"]}),
        Duration::from_secs(2),
    );
    let db = graphrag_core::Database::open(&dir.join("graphrag.db")).unwrap();
    let docs = db.active_documents("memory").unwrap();
    assert_eq!(docs.len(), 1);
    assert_eq!(db.document_sections(docs[0]).unwrap().len(), 2);
    let node = db.document_node(docs[0]).unwrap();
    assert_eq!(db.groups_of(node).unwrap(), vec!["papers".to_string()]);
}

#[test]
fn add_code_uses_code_profile() {
    let dir = data_dir("code");
    let mut s = Server::start(&dir);
    let (_, text) = s.call(
        "graphrag_add_code",
        json!({"content": "fn main() {\n    println!(\"# not a heading\");\n}\n", "source": "main.rs"}),
        Duration::from_secs(2),
    );
    assert!(text.contains("document_id"), "{text}");
    let db = graphrag_core::Database::open(&dir.join("graphrag.db")).unwrap();
    let docs = db.active_documents("memory").unwrap();
    assert!(
        db.document_sections(docs[0]).unwrap().is_empty(),
        "code has no markdown sections"
    );
}

#[test]
fn recall_in_papers_scopes() {
    let dir = data_dir("recall-scope");
    let mut s = Server::start(&dir);
    s.call(
        "graphrag_add_document",
        json!({"content": "quokka habitats in papers", "groups": ["papers"]}),
        Duration::from_secs(2),
    );
    s.call(
        "graphrag_add_document",
        json!({"content": "quokka habitats in chat", "groups": ["conversations"]}),
        Duration::from_secs(2),
    );
    let (_, text) = s.call(
        "graphrag_recall",
        json!({"query": "quokka habitats", "in": ["papers"]}),
        Duration::from_secs(120),
    );
    assert!(text.contains("in papers"), "{text}");
    assert!(!text.contains("in chat"), "{text}");
}

#[test]
fn store_alias_accepted() {
    let dir = data_dir("store-alias");
    let mut s = Server::start(&dir);
    s.call(
        "graphrag_add_document",
        json!({"content": "wombat burrows in papers", "store": "papers"}),
        Duration::from_secs(2),
    );
    s.call(
        "graphrag_add_document",
        json!({"content": "wombat burrows in chat"}),
        Duration::from_secs(2),
    );
    let (_, text) = s.call(
        "graphrag_recall",
        json!({"query": "wombat burrows", "store": "papers"}),
        Duration::from_secs(120),
    );
    assert!(
        text.contains("in papers") && !text.contains("in chat"),
        "{text}"
    );
}

#[test]
fn status_tool_reports_groups() {
    let dir = data_dir("status");
    let mut s = Server::start(&dir);
    s.call(
        "graphrag_add_document",
        json!({"content": "a note", "groups": ["papers"]}),
        Duration::from_secs(2),
    );
    let (_, text) = s.call("graphrag_status", json!({}), Duration::from_secs(10));
    let v: Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{e}: {text}"));
    assert!(v["report"]["per_group"]["papers"].is_object(), "{v}");
}

#[test]
fn summarize_community_never_samples() {
    let dir = data_dir("summary");
    let mut s = Server::start(&dir);
    s.call(
        "graphrag_add_document",
        json!({"content": "a note"}),
        Duration::from_secs(2),
    );
    let (seen, v) = s.request(
        "tools/call",
        json!({"name": "graphrag_summarize_community", "arguments": {"community_id": 1}}),
        Duration::from_secs(5),
    );
    assert!(seen.iter().all(|l| !l.contains("sampling/createMessage")));
    let _ = v;
}
