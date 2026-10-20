//! Provider-independent token accounting contracts.
//!
//! Token counts are meaningful only together with a versioned tokenizer
//! identity. Core accepts an injected counter; model/provider adapters and
//! notebooks own the actual tokenizer implementation.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TokenizerId(String);

impl TokenizerId {
    pub fn new(value: impl Into<String>) -> Result<Self, TokenCountError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(TokenCountError::InvalidTokenizerId);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn validate(&self) -> Result<(), TokenCountError> {
        if self.0.trim().is_empty() {
            return Err(TokenCountError::InvalidTokenizerId);
        }
        Ok(())
    }
}

impl std::fmt::Display for TokenizerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenCount {
    pub tokenizer: TokenizerId,
    pub tokens: u32,
}

impl TokenCount {
    pub fn validate(&self) -> Result<(), TokenCountError> {
        self.tokenizer.validate()
    }
}

/// Object-safe boundary implemented by a provider/model adapter or notebook.
pub trait TokenCounter {
    fn tokenizer_id(&self) -> &TokenizerId;
    fn count(&self, text: &str) -> Result<u32, TokenCountError>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CountedTextV1 {
    pub schema_version: u32,
    pub text_fingerprint: String,
    pub token_count: TokenCount,
}

impl CountedTextV1 {
    pub fn measure(counter: &dyn TokenCounter, text: &str) -> Result<Self, TokenCountError> {
        counter.tokenizer_id().validate()?;
        let measured = Self {
            schema_version: 1,
            text_fingerprint: fingerprint_text(text),
            token_count: TokenCount {
                tokenizer: counter.tokenizer_id().clone(),
                tokens: counter.count(text)?,
            },
        };
        measured.validate_for_text(text)?;
        Ok(measured)
    }

    pub fn validate(&self) -> Result<(), TokenCountError> {
        if self.schema_version != 1 {
            return Err(TokenCountError::UnsupportedSchemaVersion {
                found: self.schema_version,
            });
        }
        self.token_count.validate()?;
        validate_sha256(&self.text_fingerprint)
    }

    pub fn validate_for_text(&self, text: &str) -> Result<(), TokenCountError> {
        self.validate()?;
        if self.text_fingerprint != fingerprint_text(text) {
            return Err(TokenCountError::FingerprintMismatch);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TokenCountError {
    #[error("tokenizer identity must be nonempty")]
    InvalidTokenizerId,
    #[error("token count does not fit the u32 wire contract")]
    Overflow,
    #[error("tokenizer failed: {0}")]
    Counter(String),
    #[error("unsupported counted-text schema version {found}; expected 1")]
    UnsupportedSchemaVersion { found: u32 },
    #[error("invalid sha256 fingerprint")]
    InvalidFingerprint,
    #[error("text fingerprint does not match text")]
    FingerprintMismatch,
}

pub fn fingerprint_text(text: &str) -> String {
    let digest = Sha256::digest(text.as_bytes());
    format!("sha256:{}", hex_bytes(&digest))
}

pub fn fingerprint_parts(parts: &[&[u8]]) -> String {
    let mut hash = Sha256::new();
    for part in parts {
        hash.update((part.len() as u64).to_be_bytes());
        hash.update(part);
    }
    format!("sha256:{}", hex_bytes(&hash.finalize()))
}

pub(crate) fn validate_sha256(value: &str) -> Result<(), TokenCountError> {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return Err(TokenCountError::InvalidFingerprint);
    };
    if hex.len() != 64
        || !hex
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(TokenCountError::InvalidFingerprint);
    }
    Ok(())
}

fn hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
