//! OpenAI-compatible chat client (llama.cpp's `/v1/chat/completions`).
//!
//! No internal retry: the worker owns retry and backoff. Errors keep the
//! `ChatClient` `Result<String, String>` shape with a kind prefix
//! (`transport: …`, `http 400: …`, `malformed: …`) that
//! [`classify_error`] decodes.

use crate::{ChatClient, ChatPrompt};
use serde_json::{Value, json};
use std::time::Duration;

pub const DEFAULT_BASE_URL: &str = "http://ndn.lan:8080/v1";
pub const DEFAULT_MODEL: &str = "qwen3.8-27b";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlmError {
    /// Connect failure, timeout, or 5xx: retryable.
    Transport(String),
    /// 4xx: the request itself is wrong.
    Http(u16, String),
    /// A 2xx response without the expected shape.
    Malformed(String),
}

impl std::fmt::Display for LlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LlmError::Transport(m) => write!(f, "transport: {m}"),
            LlmError::Http(code, m) => write!(f, "http {code}: {m}"),
            LlmError::Malformed(m) => write!(f, "malformed: {m}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmErrorKind {
    Transport,
    Http(u16),
    Malformed,
    Other,
}

/// Decode the kind prefix of a `ChatClient` error string.
pub fn classify_error(msg: &str) -> LlmErrorKind {
    if msg.starts_with("transport: ") {
        LlmErrorKind::Transport
    } else if msg.starts_with("malformed: ") {
        LlmErrorKind::Malformed
    } else if let Some(rest) = msg.strip_prefix("http ") {
        rest.split(':')
            .next()
            .and_then(|c| c.parse().ok())
            .map_or(LlmErrorKind::Other, LlmErrorKind::Http)
    } else {
        LlmErrorKind::Other
    }
}

#[derive(Debug, Clone)]
pub struct OpenAiChatClient {
    base_url: String,
    model: String,
    timeout: Duration,
    enable_thinking: bool,
}

impl OpenAiChatClient {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            model: model.into(),
            timeout: Duration::from_secs(600),
            enable_thinking: false,
        }
    }

    /// `GRAPHRAG_LLM_BASE_URL` and `GRAPHRAG_LLM_MODEL`, with the llama.cpp
    /// host on ndn.lan and qwen3.8-27b as defaults.
    pub fn from_env() -> Self {
        let base =
            std::env::var("GRAPHRAG_LLM_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.into());
        let model = std::env::var("GRAPHRAG_LLM_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());
        Self::new(base, model)
    }

    pub fn with_thinking(mut self, on: bool) -> Self {
        self.enable_thinking = on;
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    fn agent(&self) -> ureq::Agent {
        ureq::AgentBuilder::new().timeout(self.timeout).build()
    }

    fn map_err(e: ureq::Error) -> LlmError {
        match e {
            ureq::Error::Status(code, resp) => {
                let body = resp.into_string().unwrap_or_default();
                if code >= 500 {
                    LlmError::Transport(format!("http {code}: {body}"))
                } else {
                    LlmError::Http(code, body)
                }
            }
            ureq::Error::Transport(t) => LlmError::Transport(t.to_string()),
        }
    }

    /// Model ids from `GET {base}/models`.
    pub fn served_models(&self) -> Result<Vec<String>, LlmError> {
        let v: Value = self
            .agent()
            .get(&format!("{}/models", self.base_url))
            .call()
            .map_err(Self::map_err)?
            .into_json()
            .map_err(|e| LlmError::Malformed(e.to_string()))?;
        let data = v
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| LlmError::Malformed("models response has no data[]".into()))?;
        Ok(data
            .iter()
            .filter_map(|m| m.get("id").and_then(Value::as_str).map(String::from))
            .collect())
    }

    fn request(&self, prompt: &ChatPrompt, schema: Option<&Value>) -> Value {
        let mut body = json!({
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": prompt.max_tokens,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
        });
        if let Some(schema) = schema {
            body["response_format"] =
                json!({"type": "json_schema", "json_schema": {"name": "graph", "schema": schema}});
        }
        body
    }

    fn chat(&self, prompt: &ChatPrompt, schema: Option<&Value>) -> Result<String, LlmError> {
        let v: Value = self
            .agent()
            .post(&format!("{}/chat/completions", self.base_url))
            .send_json(self.request(prompt, schema))
            .map_err(Self::map_err)?
            .into_json()
            .map_err(|e| LlmError::Malformed(e.to_string()))?;
        v.pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .map(String::from)
            .ok_or_else(|| LlmError::Malformed(format!("no choices[0].message.content in {v}")))
    }
}

impl ChatClient for OpenAiChatClient {
    fn complete(&self, prompt: &ChatPrompt) -> Result<String, String> {
        self.chat(prompt, None).map_err(|e| e.to_string())
    }

    fn complete_with_format(
        &self,
        prompt: &ChatPrompt,
        format: Option<&Value>,
    ) -> Result<String, String> {
        self.chat(prompt, format).map_err(|e| e.to_string())
    }
}
