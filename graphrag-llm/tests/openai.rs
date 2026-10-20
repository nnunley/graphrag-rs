//! Task 10: OpenAI-compatible chat client against a one-shot mock server.

use graphrag_llm::openai::{LlmErrorKind, OpenAiChatClient, classify_error};
use graphrag_llm::{ChatClient, ChatPrompt};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::mpsc;
use std::time::Duration;

/// Serve one request with `status` and `body`; returns the base URL and a
/// receiver for `(request line, request body)`.
fn mock(status: u16, body: &str) -> (String, mpsc::Receiver<(String, String)>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let body = body.to_string();
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut request_line = String::new();
        reader.read_line(&mut request_line).unwrap();
        let mut len = 0usize;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            if line == "\r\n" || line.is_empty() {
                break;
            }
            if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                len = v.trim().parse().unwrap();
            }
        }
        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf).unwrap();
        tx.send((
            request_line.trim().to_string(),
            String::from_utf8(buf).unwrap(),
        ))
        .unwrap();
        let reason = if status == 200 { "OK" } else { "ERR" };
        let mut stream = stream;
        write!(
            stream,
            "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
    });
    (format!("http://{addr}/v1"), rx)
}

fn prompt() -> ChatPrompt {
    ChatPrompt {
        system: "sys".into(),
        user: "hello".into(),
        max_tokens: 123,
    }
}

fn ok_body(content: &str) -> String {
    json!({"choices": [{"message": {"role": "assistant", "content": content}}]}).to_string()
}

#[test]
fn sends_json_schema_and_thinking_flags() {
    let (url, rx) = mock(200, &ok_body("{}"));
    let client = OpenAiChatClient::new(url, "qwen3.8-27b").with_thinking(true);
    let schema = json!({"type": "object"});
    client
        .complete_with_format(&prompt(), Some(&schema))
        .unwrap();
    let (line, body) = rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(line.starts_with("POST /v1/chat/completions"), "{line}");
    let body: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(body["model"], "qwen3.8-27b");
    assert_eq!(body["response_format"]["type"], "json_schema");
    assert_eq!(body["response_format"]["json_schema"]["name"], "graph");
    assert_eq!(body["response_format"]["json_schema"]["schema"], schema);
    assert_eq!(body["chat_template_kwargs"]["enable_thinking"], true);
    assert_eq!(body["temperature"], 0.0);
    assert_eq!(body["max_tokens"], 123);
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(body["messages"][1]["content"], "hello");
}

#[test]
fn no_schema_sends_no_response_format() {
    let (url, rx) = mock(200, &ok_body("x"));
    OpenAiChatClient::new(url, "m").complete(&prompt()).unwrap();
    let (_, body) = rx.recv_timeout(Duration::from_secs(5)).unwrap();
    let body: Value = serde_json::from_str(&body).unwrap();
    assert!(body.get("response_format").is_none());
    assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
}

#[test]
fn returns_message_content() {
    let (url, _rx) = mock(200, &ok_body("the answer"));
    let out = OpenAiChatClient::new(url, "m").complete(&prompt()).unwrap();
    assert_eq!(out, "the answer");
}

#[test]
fn connection_refused_is_transport() {
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let err = OpenAiChatClient::new(format!("http://127.0.0.1:{port}/v1"), "m")
        .complete(&prompt())
        .unwrap_err();
    assert_eq!(classify_error(&err), LlmErrorKind::Transport, "{err}");
}

#[test]
fn http_500_is_transport() {
    let (url, _rx) = mock(500, "{\"error\":\"boom\"}");
    let err = OpenAiChatClient::new(url, "m")
        .complete(&prompt())
        .unwrap_err();
    assert_eq!(classify_error(&err), LlmErrorKind::Transport, "{err}");
}

#[test]
fn http_400_is_http() {
    let (url, _rx) = mock(400, "{\"error\":\"bad schema\"}");
    let err = OpenAiChatClient::new(url, "m")
        .complete(&prompt())
        .unwrap_err();
    assert_eq!(classify_error(&err), LlmErrorKind::Http(400), "{err}");
    assert!(err.contains("bad schema"));
}

#[test]
fn missing_choices_is_malformed() {
    let (url, _rx) = mock(200, "{\"object\":\"chat.completion\"}");
    let err = OpenAiChatClient::new(url, "m")
        .complete(&prompt())
        .unwrap_err();
    assert_eq!(classify_error(&err), LlmErrorKind::Malformed, "{err}");
}

#[test]
fn served_models_reads_ids() {
    let (url, rx) = mock(
        200,
        &json!({"object": "list", "data": [{"id": "qwen3.8-27b"}, {"id": "other"}]}).to_string(),
    );
    let models = OpenAiChatClient::new(url, "m").served_models().unwrap();
    assert_eq!(models, vec!["qwen3.8-27b".to_string(), "other".to_string()]);
    let (line, _) = rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(line.starts_with("GET /v1/models"), "{line}");
}

#[test]
fn from_env_defaults() {
    // SAFETY: tests in this binary don't read these variables concurrently.
    unsafe {
        std::env::remove_var("GRAPHRAG_LLM_BASE_URL");
        std::env::remove_var("GRAPHRAG_LLM_MODEL");
    }
    let c = OpenAiChatClient::from_env();
    assert_eq!(c.base_url(), "http://ndn.lan:8080/v1");
    assert_eq!(c.model(), "qwen3.8-27b");
}
