//! Project Capsule v1: typed, evidence-backed, resumable project state.
//!
//! A capsule is the synthesis-layer answer to "what is this project, what is
//! verified, what matters now, and where do I resume?". This module defines
//! the v1 schema, a total [`ProjectCapsuleV1::validate`] check, and a
//! deterministic [`ProjectCapsuleV1::reference`] that other systems (e.g. the
//! laneq Attention Steward's `resume_ref`) carry instead of large context.
//!
//! ## Guarantees
//!
//! - `validate()` is total: it returns a typed [`CapsuleError`] and never
//!   panics on any field content.
//! - `reference()` fingerprints the exact serde-JSON byte serialization of the
//!   capsule. All structs use fixed named fields and vectors (no maps), so
//!   serialization is deterministic for a given value. Vector order is
//!   semantically significant: reordering evidence or threads is a different
//!   capsule with a different fingerprint.
//!
//! ## Non-guarantees
//!
//! - No canonicalization across semantically-equal capsules (e.g. whitespace
//!   differences in prose produce different fingerprints). That is
//!   intentional: the fingerprint identifies bytes, not meaning.
//! - Timestamp validation checks RFC 3339 shape and field ranges; it does not
//!   validate leap seconds or calendar-day overflow beyond day 31.
//! - This module does not persist capsules or synthesize them from the graph;
//!   those are separate segments.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Maximum length for URI-safe identifiers.
pub const MAX_ID_LEN: usize = 128;

/// Capsule kinds. `project` is implemented; `segment` and `portfolio` are
/// reserved so `CapsuleRef` can carry them without a schema change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapsuleKind {
    Project,
    Segment,
    Portfolio,
}

impl CapsuleKind {
    /// The URI path segment for this kind.
    pub fn as_str(self) -> &'static str {
        match self {
            CapsuleKind::Project => "project",
            CapsuleKind::Segment => "segment",
            CapsuleKind::Portfolio => "portfolio",
        }
    }
}

/// A compact, validated pointer to a capsule. This is the cross-system wire
/// object; field order here is the canonical JSON field order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapsuleRef {
    pub schema_version: u32,
    pub uri: String,
    pub kind: CapsuleKind,
    pub capsule_id: String,
    pub project_id: String,
    /// `sha256:` + 64 lowercase hex over the capsule's serde-JSON bytes.
    pub content_fingerprint: String,
    /// RFC 3339 instant the referenced capsule was generated.
    pub generated_at: String,
}

/// Stable identity of the project a capsule describes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectIdentity {
    pub project_id: String,
    pub display_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repository: Option<String>,
    /// Known working locations (paths, hosts) in canonical order.
    pub locations: Vec<String>,
}

/// A fact whose truth was independently verified; must cite evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedFact {
    pub statement: String,
    /// Evidence IDs; must be nonempty and resolve in `evidence`.
    pub evidence: Vec<String>,
}

/// A recorded decision with rationale; must cite evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Decision {
    pub decision: String,
    pub rationale: String,
    pub evidence: Vec<String>,
}

/// An open line of work with explicit status/ownership, not prose-only.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenThread {
    pub thread_id: String,
    pub title: String,
    pub status: String,
    pub owner: String,
    pub next_action: String,
}

/// An externally-visible obligation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Commitment {
    pub commitment_id: String,
    pub title: String,
    pub status: String,
    pub owner: String,
    pub next_action: String,
    #[serde(default)]
    pub external_waiting: bool,
}

/// Where to resume: the next maker segment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NextSegment {
    pub title: String,
    pub entry_point: String,
    pub first_action: String,
}

/// A citable observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceItem {
    pub evidence_id: String,
    pub uri: String,
    pub observed_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fingerprint: Option<String>,
}

/// Freshness metadata for staleness detection and refresh triggers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Freshness {
    pub generated_at: String,
    /// Fingerprint of the synthesis inputs as a whole.
    pub source_fingerprint: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stale_after: Option<String>,
    /// Ordered fingerprints of individual inputs.
    pub input_fingerprints: Vec<String>,
}

/// The v1 project capsule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectCapsuleV1 {
    pub schema_version: u32,
    pub capsule_id: String,
    pub project: ProjectIdentity,
    pub purpose: String,
    pub verified_state: Vec<VerifiedFact>,
    pub decisions: Vec<Decision>,
    pub open_threads: Vec<OpenThread>,
    pub commitments: Vec<Commitment>,
    pub next_segment: NextSegment,
    pub evidence: Vec<EvidenceItem>,
    pub freshness: Freshness,
}

/// Typed validation failures. Every variant names the offending location.
#[derive(Debug, Error)]
pub enum CapsuleError {
    #[error("unsupported schema version {found}; expected {expected}")]
    UnsupportedSchemaVersion { found: u32, expected: u32 },
    #[error("invalid identifier in {field}: {value:?} (nonempty [A-Za-z0-9._-], max {MAX_ID_LEN})")]
    InvalidId { field: &'static str, value: String },
    #[error("required text field {field} is empty")]
    EmptyText { field: &'static str },
    #[error("duplicate evidence id {id:?}")]
    DuplicateEvidenceId { id: String },
    #[error("{field} cites unknown evidence id {id:?}")]
    DanglingEvidence { field: &'static str, id: String },
    #[error("{field} must cite at least one evidence id")]
    MissingEvidenceCitation { field: &'static str },
    #[error("invalid fingerprint in {field}: {value:?} (expected sha256: + 64 lowercase hex)")]
    InvalidFingerprint { field: &'static str, value: String },
    #[error("invalid RFC 3339 timestamp in {field}: {value:?}")]
    InvalidTimestamp { field: &'static str, value: String },
    #[error("uri {uri:?} does not match kind/capsule_id (expected {expected:?})")]
    UriMismatch { uri: String, expected: String },
    #[error("serialization failed: {0}")]
    Serialize(#[from] serde_json::Error),
}

/// True when `s` is a valid URI-safe identifier.
fn valid_id(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= MAX_ID_LEN
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'_' || b == b'-')
}

fn check_id(field: &'static str, value: &str) -> Result<(), CapsuleError> {
    if valid_id(value) {
        Ok(())
    } else {
        Err(CapsuleError::InvalidId {
            field,
            value: value.to_string(),
        })
    }
}

fn check_text(field: &'static str, value: &str) -> Result<(), CapsuleError> {
    if value.trim().is_empty() {
        Err(CapsuleError::EmptyText { field })
    } else {
        Ok(())
    }
}

/// True when `s` is `sha256:` followed by exactly 64 lowercase hex digits.
fn valid_fingerprint(s: &str) -> bool {
    match s.strip_prefix("sha256:") {
        Some(hex) => {
            hex.len() == 64
                && hex
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        }
        None => false,
    }
}

fn check_fingerprint(field: &'static str, value: &str) -> Result<(), CapsuleError> {
    if valid_fingerprint(value) {
        Ok(())
    } else {
        Err(CapsuleError::InvalidFingerprint {
            field,
            value: value.to_string(),
        })
    }
}

/// Minimal total RFC 3339 shape/range check (see module non-guarantees).
fn valid_rfc3339(s: &str) -> bool {
    let b = s.as_bytes();
    // Minimum: YYYY-MM-DDTHH:MM:SSZ (20 bytes).
    if b.len() < 20 {
        return false;
    }
    let digit = |i: usize| b[i].is_ascii_digit();
    let all_digits = |r: std::ops::Range<usize>| r.clone().all(digit);
    if !(all_digits(0..4) && b[4] == b'-' && all_digits(5..7) && b[7] == b'-' && all_digits(8..10))
    {
        return false;
    }
    if !(b[10] == b'T' || b[10] == b't') {
        return false;
    }
    if !(all_digits(11..13)
        && b[13] == b':'
        && all_digits(14..16)
        && b[16] == b':'
        && all_digits(17..19))
    {
        return false;
    }
    let num = |r: std::ops::Range<usize>| -> u32 { s[r].parse().unwrap_or(u32::MAX) };
    let (month, day) = (num(5..7), num(8..10));
    let (hour, minute, sec) = (num(11..13), num(14..16), num(17..19));
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return false;
    }
    if hour > 23 || minute > 59 || sec > 60 {
        return false;
    }
    // Optional fraction, then mandatory offset.
    let mut i = 19;
    if i < b.len() && b[i] == b'.' {
        i += 1;
        let frac_start = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if i == frac_start {
            return false;
        }
    }
    match b.get(i) {
        Some(b'Z') | Some(b'z') => i + 1 == b.len(),
        Some(b'+') | Some(b'-') => {
            let r = &s[i + 1..];
            let rb = r.as_bytes();
            r.len() == 5
                && rb[0].is_ascii_digit()
                && rb[1].is_ascii_digit()
                && rb[2] == b':'
                && rb[3].is_ascii_digit()
                && rb[4].is_ascii_digit()
                && r[0..2].parse::<u32>().is_ok_and(|h| h <= 23)
                && r[3..5].parse::<u32>().is_ok_and(|m| m <= 59)
        }
        _ => false,
    }
}

fn check_timestamp(field: &'static str, value: &str) -> Result<(), CapsuleError> {
    if valid_rfc3339(value) {
        Ok(())
    } else {
        Err(CapsuleError::InvalidTimestamp {
            field,
            value: value.to_string(),
        })
    }
}

/// Build the canonical capsule URI for a kind and id.
pub fn capsule_uri(kind: CapsuleKind, capsule_id: &str) -> String {
    format!("graphrag://capsules/v1/{}/{}", kind.as_str(), capsule_id)
}

impl CapsuleRef {
    /// Validate the reference in isolation (shape, IDs, URI agreement).
    pub fn validate(&self) -> Result<(), CapsuleError> {
        if self.schema_version != 1 {
            return Err(CapsuleError::UnsupportedSchemaVersion {
                found: self.schema_version,
                expected: 1,
            });
        }
        check_id("capsule_ref.capsule_id", &self.capsule_id)?;
        check_id("capsule_ref.project_id", &self.project_id)?;
        check_fingerprint("capsule_ref.content_fingerprint", &self.content_fingerprint)?;
        check_timestamp("capsule_ref.generated_at", &self.generated_at)?;
        let expected = capsule_uri(self.kind, &self.capsule_id);
        if self.uri != expected {
            return Err(CapsuleError::UriMismatch {
                uri: self.uri.clone(),
                expected,
            });
        }
        Ok(())
    }
}

impl ProjectCapsuleV1 {
    /// Total validation of the whole capsule.
    pub fn validate(&self) -> Result<(), CapsuleError> {
        if self.schema_version != 1 {
            return Err(CapsuleError::UnsupportedSchemaVersion {
                found: self.schema_version,
                expected: 1,
            });
        }
        check_id("capsule_id", &self.capsule_id)?;
        check_id("project.project_id", &self.project.project_id)?;
        check_text("project.display_name", &self.project.display_name)?;
        check_text("purpose", &self.purpose)?;
        check_text("next_segment.title", &self.next_segment.title)?;
        check_text("next_segment.entry_point", &self.next_segment.entry_point)?;
        check_text("next_segment.first_action", &self.next_segment.first_action)?;

        // Evidence table: unique IDs, valid shapes.
        let mut ids = std::collections::HashSet::new();
        for e in &self.evidence {
            check_id("evidence.evidence_id", &e.evidence_id)?;
            check_text("evidence.uri", &e.uri)?;
            check_timestamp("evidence.observed_at", &e.observed_at)?;
            if let Some(fp) = &e.fingerprint {
                check_fingerprint("evidence.fingerprint", fp)?;
            }
            if !ids.insert(e.evidence_id.as_str()) {
                return Err(CapsuleError::DuplicateEvidenceId {
                    id: e.evidence_id.clone(),
                });
            }
        }

        let cite = |field: &'static str, links: &[String]| -> Result<(), CapsuleError> {
            if links.is_empty() {
                return Err(CapsuleError::MissingEvidenceCitation { field });
            }
            for id in links {
                if !ids.contains(id.as_str()) {
                    return Err(CapsuleError::DanglingEvidence {
                        field,
                        id: id.clone(),
                    });
                }
            }
            Ok(())
        };
        for f in &self.verified_state {
            check_text("verified_state.statement", &f.statement)?;
            cite("verified_state", &f.evidence)?;
        }
        for d in &self.decisions {
            check_text("decisions.decision", &d.decision)?;
            check_text("decisions.rationale", &d.rationale)?;
            cite("decisions", &d.evidence)?;
        }
        for t in &self.open_threads {
            check_id("open_threads.thread_id", &t.thread_id)?;
            check_text("open_threads.title", &t.title)?;
            check_text("open_threads.status", &t.status)?;
            check_text("open_threads.owner", &t.owner)?;
            check_text("open_threads.next_action", &t.next_action)?;
        }
        for c in &self.commitments {
            check_id("commitments.commitment_id", &c.commitment_id)?;
            check_text("commitments.title", &c.title)?;
            check_text("commitments.status", &c.status)?;
            check_text("commitments.owner", &c.owner)?;
            check_text("commitments.next_action", &c.next_action)?;
        }

        check_timestamp("freshness.generated_at", &self.freshness.generated_at)?;
        check_fingerprint(
            "freshness.source_fingerprint",
            &self.freshness.source_fingerprint,
        )?;
        if let Some(sa) = &self.freshness.stale_after {
            check_timestamp("freshness.stale_after", sa)?;
        }
        for (i, fp) in self.freshness.input_fingerprints.iter().enumerate() {
            let _ = i;
            check_fingerprint("freshness.input_fingerprints", fp)?;
        }
        Ok(())
    }

    /// Validate, then produce the deterministic reference for this capsule.
    ///
    /// The fingerprint is SHA-256 over `serde_json::to_vec(self)`. Because all
    /// schema types serialize fixed named fields in declaration order and use
    /// vectors (never maps), equal values always produce equal bytes.
    pub fn reference(&self) -> Result<CapsuleRef, CapsuleError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)?;
        let digest = Sha256::digest(&bytes);
        let mut hex = String::with_capacity(64);
        for b in digest {
            use std::fmt::Write;
            let _ = write!(hex, "{b:02x}");
        }
        Ok(CapsuleRef {
            schema_version: 1,
            uri: capsule_uri(CapsuleKind::Project, &self.capsule_id),
            kind: CapsuleKind::Project,
            capsule_id: self.capsule_id.clone(),
            project_id: self.project.project_id.clone(),
            content_fingerprint: format!("sha256:{hex}"),
            generated_at: self.freshness.generated_at.clone(),
        })
    }
}
