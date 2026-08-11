//! Versioned, provider-independent contracts for the paper GraphRAG workflow.
//!
//! M0 defines wire truth, canonical context rendering, and total validators.
//! It deliberately does not add model execution or persistence tables. Later
//! milestones bind production stages to these contracts.

use crate::token::{
    CountedTextV1, TokenCount, TokenCountError, TokenCounter, TokenizerId, fingerprint_parts,
    fingerprint_text, validate_sha256,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

pub const PAPER_CONTRACT_SCHEMA_VERSION: u32 = 1;
pub const MAP_RESPONSE_SCHEMA_V1: &str = "paper-map-response-v1";
pub const REDUCE_RESPONSE_SCHEMA_V1: &str = "paper-reduce-response-v1";

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{code}: {detail}")]
pub struct PaperContractError {
    code: &'static str,
    detail: String,
}

impl PaperContractError {
    pub fn violation(code: &'static str, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }

    pub fn code(&self) -> &'static str {
        self.code
    }
}

fn token_error(error: TokenCountError) -> PaperContractError {
    PaperContractError::violation("token.invalid", error.to_string())
}

fn json_error(error: serde_json::Error) -> PaperContractError {
    PaperContractError::violation("wire.json", error.to_string())
}

fn require_v1(version: u32, field: &'static str) -> Result<(), PaperContractError> {
    if version != PAPER_CONTRACT_SCHEMA_VERSION {
        return Err(PaperContractError::violation(
            "schema.unsupported",
            format!("{field} has version {version}; expected 1"),
        ));
    }
    Ok(())
}

fn require_text(
    value: &str,
    code: &'static str,
    field: &'static str,
) -> Result<(), PaperContractError> {
    if value.trim().is_empty() {
        return Err(PaperContractError::violation(
            code,
            format!("{field} is empty"),
        ));
    }
    Ok(())
}

fn require_unique<T>(
    values: &[T],
    code: &'static str,
    field: &'static str,
) -> Result<(), PaperContractError>
where
    T: Eq + std::hash::Hash,
{
    if values.iter().collect::<HashSet<_>>().len() != values.len() {
        return Err(PaperContractError::violation(
            code,
            format!("{field} contains duplicates"),
        ));
    }
    Ok(())
}

fn require_fingerprint(value: &str, code: &'static str) -> Result<(), PaperContractError> {
    validate_sha256(value).map_err(|error| PaperContractError::violation(code, error.to_string()))
}

fn validate_counted_text(
    counter: &dyn TokenCounter,
    text: &str,
    fingerprint: &str,
    count: &TokenCount,
    code: &'static str,
) -> Result<(), PaperContractError> {
    if fingerprint != fingerprint_text(text) {
        return Err(PaperContractError::violation(
            code,
            "text fingerprint is stale",
        ));
    }
    count.validate().map_err(token_error)?;
    if count.tokenizer != *counter.tokenizer_id() {
        return Err(PaperContractError::violation(
            code,
            "tokenizer identity differs",
        ));
    }
    let actual = counter.count(text).map_err(token_error)?;
    if count.tokens != actual {
        return Err(PaperContractError::violation(
            code,
            format!("declared {} tokens, measured {actual}", count.tokens),
        ));
    }
    Ok(())
}

fn validate_token_count(
    counter: &dyn TokenCounter,
    text: &str,
    count: &TokenCount,
    code: &'static str,
) -> Result<(), PaperContractError> {
    count.validate().map_err(token_error)?;
    if count.tokenizer != *counter.tokenizer_id() {
        return Err(PaperContractError::violation(
            code,
            "tokenizer identity differs",
        ));
    }
    let actual = counter.count(text).map_err(token_error)?;
    if count.tokens != actual {
        return Err(PaperContractError::violation(
            code,
            format!("declared {} tokens, measured {actual}", count.tokens),
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceDocumentV1 {
    pub schema_version: u32,
    pub document_id: String,
    pub text: String,
}

impl SourceDocumentV1 {
    pub fn validate(&self) -> Result<(), PaperContractError> {
        require_v1(self.schema_version, "source document")?;
        require_text(&self.document_id, "source.id", "document_id")?;
        require_text(&self.text, "source.text", "text")
    }

    pub fn fingerprint(&self) -> Result<String, PaperContractError> {
        self.validate()?;
        Ok(fingerprint_parts(&[
            b"graphrag-paper-source-document-v1",
            self.document_id.as_bytes(),
            self.text.as_bytes(),
        ]))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PaperBenchmarkProfileV1 {
    pub schema_version: u32,
    pub profile_id: String,
    pub tokenizer: TokenizerId,
    pub map_prompt_version: String,
    pub map_response_schema_version: String,
    pub reduce_prompt_version: String,
    pub reduce_response_schema_version: String,
    pub chunk_tokens: u32,
    pub chunk_overlap_tokens: u32,
    /// Total context window. M1 must separately reserve prompt/output tokens.
    pub report_call_window_tokens: u32,
    /// Total context window. M1 must separately reserve prompt/output tokens.
    pub map_call_window_tokens: u32,
    /// Total context window. M1 must separately reserve prompt/output tokens.
    pub reduce_call_window_tokens: u32,
}

impl PaperBenchmarkProfileV1 {
    pub fn paper_v2(tokenizer: TokenizerId) -> Self {
        Self {
            schema_version: 1,
            profile_id: "paper-v2-600-100-8k".into(),
            tokenizer,
            map_prompt_version: "paper-map-prompt-v1".into(),
            map_response_schema_version: MAP_RESPONSE_SCHEMA_V1.into(),
            reduce_prompt_version: "paper-reduce-prompt-v1".into(),
            reduce_response_schema_version: REDUCE_RESPONSE_SCHEMA_V1.into(),
            chunk_tokens: 600,
            chunk_overlap_tokens: 100,
            report_call_window_tokens: 8192,
            map_call_window_tokens: 8192,
            reduce_call_window_tokens: 8192,
        }
    }

    pub fn validate(&self) -> Result<(), PaperContractError> {
        require_v1(self.schema_version, "paper benchmark profile")?;
        require_text(&self.profile_id, "profile.id", "profile_id")?;
        self.tokenizer.validate().map_err(token_error)?;
        for (field, value) in [
            ("map_prompt_version", self.map_prompt_version.as_str()),
            (
                "map_response_schema_version",
                self.map_response_schema_version.as_str(),
            ),
            ("reduce_prompt_version", self.reduce_prompt_version.as_str()),
            (
                "reduce_response_schema_version",
                self.reduce_response_schema_version.as_str(),
            ),
        ] {
            require_text(value, "profile.execution_version", field)?;
        }
        if self.map_response_schema_version != MAP_RESPONSE_SCHEMA_V1
            || self.reduce_response_schema_version != REDUCE_RESPONSE_SCHEMA_V1
        {
            return Err(PaperContractError::violation(
                "profile.response_schema",
                "paper contract v1 supports only the v1 map/reduce response schemas",
            ));
        }
        if self.chunk_tokens == 0 {
            return Err(PaperContractError::violation(
                "profile.chunk_tokens",
                "chunk_tokens must be nonzero",
            ));
        }
        if self.chunk_overlap_tokens >= self.chunk_tokens {
            return Err(PaperContractError::violation(
                "profile.overlap",
                "chunk overlap must be less than chunk size",
            ));
        }
        if [
            self.report_call_window_tokens,
            self.map_call_window_tokens,
            self.reduce_call_window_tokens,
        ]
        .contains(&0)
        {
            return Err(PaperContractError::violation(
                "profile.window",
                "report/map/reduce windows must be nonzero",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ExtractionAttemptIdentityParts<'a> {
    pub store: &'a str,
    pub chunk_id: i64,
    pub chunk_content: &'a str,
    pub model: &'a str,
    pub extraction_profile: &'a str,
    pub prompt_version: &'a str,
    pub response_schema_version: &'a str,
    pub glean_max_rounds: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractionAttemptIdentityV1 {
    pub schema_version: u32,
    pub fingerprint: String,
}

impl ExtractionAttemptIdentityV1 {
    pub fn derive(parts: &ExtractionAttemptIdentityParts<'_>) -> Result<Self, PaperContractError> {
        for (field, value) in [
            ("store", parts.store),
            ("model", parts.model),
            ("extraction_profile", parts.extraction_profile),
            ("prompt_version", parts.prompt_version),
            ("response_schema_version", parts.response_schema_version),
        ] {
            require_text(value, "attempt.field", field)?;
        }
        if parts.chunk_id <= 0 {
            return Err(PaperContractError::violation(
                "attempt.chunk_id",
                "chunk_id must be positive",
            ));
        }
        let chunk_id = parts.chunk_id.to_be_bytes();
        let glean = parts.glean_max_rounds.to_be_bytes();
        Ok(Self {
            schema_version: 1,
            fingerprint: fingerprint_parts(&[
                b"graphrag-paper-extraction-attempt-v1",
                parts.store.as_bytes(),
                &chunk_id,
                parts.chunk_content.as_bytes(),
                parts.model.as_bytes(),
                parts.extraction_profile.as_bytes(),
                parts.prompt_version.as_bytes(),
                parts.response_schema_version.as_bytes(),
                &glean,
            ]),
        })
    }

    pub fn validate(&self) -> Result<(), PaperContractError> {
        require_v1(self.schema_version, "extraction attempt identity")?;
        require_fingerprint(&self.fingerprint, "attempt.fingerprint")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordRefV1 {
    pub kind: String,
    pub id: String,
    pub fingerprint: String,
}

impl RecordRefV1 {
    pub fn validate(&self) -> Result<(), PaperContractError> {
        require_text(&self.kind, "record_ref.kind", "kind")?;
        require_text(&self.id, "record_ref.id", "id")?;
        require_fingerprint(&self.fingerprint, "record_ref.fingerprint")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReportContextRowV1 {
    pub schema_version: u32,
    pub report_id: String,
    pub community_id: i64,
    pub hierarchy_run_id: String,
    pub cut: u32,
    pub text: String,
    pub text_fingerprint: String,
    pub token_count: TokenCount,
    pub references: Vec<RecordRefV1>,
}

impl ReportContextRowV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        report_id: impl Into<String>,
        community_id: i64,
        hierarchy_run_id: impl Into<String>,
        cut: u32,
        text: impl Into<String>,
        references: Vec<RecordRefV1>,
        counter: &dyn TokenCounter,
    ) -> Result<Self, PaperContractError> {
        let text = text.into();
        let counted = CountedTextV1::measure(counter, &text).map_err(token_error)?;
        let row = Self {
            schema_version: 1,
            report_id: report_id.into(),
            community_id,
            hierarchy_run_id: hierarchy_run_id.into(),
            cut,
            text,
            text_fingerprint: counted.text_fingerprint,
            token_count: counted.token_count,
            references,
        };
        row.validate_with_counter(counter)?;
        Ok(row)
    }

    pub fn validate(&self) -> Result<(), PaperContractError> {
        require_v1(self.schema_version, "report context row")?;
        require_text(&self.report_id, "report.id", "report_id")?;
        require_text(
            &self.hierarchy_run_id,
            "report.hierarchy_run",
            "hierarchy_run_id",
        )?;
        if self.community_id <= 0 {
            return Err(PaperContractError::violation(
                "report.community_id",
                "community_id must be positive",
            ));
        }
        require_text(&self.text, "report.text", "text")?;
        if self.text_fingerprint != fingerprint_text(&self.text) {
            return Err(PaperContractError::violation(
                "report.text_fingerprint",
                "text fingerprint is stale",
            ));
        }
        self.token_count.validate().map_err(token_error)?;
        for reference in &self.references {
            reference.validate()?;
        }
        if !self
            .references
            .iter()
            .any(|reference| reference.kind == "source")
        {
            return Err(PaperContractError::violation(
                "report.source_references",
                "report must carry an exact source reference",
            ));
        }
        let keys = self
            .references
            .iter()
            .map(|reference| (reference.kind.as_str(), reference.id.as_str()))
            .collect::<Vec<_>>();
        require_unique(&keys, "report.references", "references")
    }

    pub fn validate_with_counter(
        &self,
        counter: &dyn TokenCounter,
    ) -> Result<(), PaperContractError> {
        self.validate()?;
        validate_counted_text(
            counter,
            &self.text,
            &self.text_fingerprint,
            &self.token_count,
            "report.token_count",
        )
    }
}

pub fn canonical_report_order(seed: u64, reports: &[ReportContextRowV1]) -> Vec<String> {
    let seed_bytes = seed.to_be_bytes();
    let mut keyed = reports
        .iter()
        .map(|report| {
            (
                fingerprint_parts(&[
                    b"graphrag-paper-seeded-report-order-v1",
                    &seed_bytes,
                    report.report_id.as_bytes(),
                ]),
                report.report_id.clone(),
            )
        })
        .collect::<Vec<_>>();
    keyed.sort();
    keyed.into_iter().map(|(_, report_id)| report_id).collect()
}

#[derive(Serialize)]
struct MapContextWire<'a> {
    schema_version: u32,
    query: &'a str,
    reports: &'a [&'a ReportContextRowV1],
}

pub fn render_map_context(
    query: &str,
    reports: &[&ReportContextRowV1],
) -> Result<String, PaperContractError> {
    require_text(query, "map_context.query", "query")?;
    if reports.is_empty() {
        return Err(PaperContractError::violation(
            "map_context.reports",
            "map context has no reports",
        ));
    }
    serde_json::to_string(&MapContextWire {
        schema_version: 1,
        query,
        reports,
    })
    .map_err(json_error)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MapBatchV1 {
    pub schema_version: u32,
    pub batch_id: String,
    pub query_fingerprint: String,
    pub prompt_version: String,
    pub response_schema_version: String,
    pub report_ids: Vec<String>,
    pub input_context: String,
    pub input_fingerprint: String,
    pub input_token_count: TokenCount,
}

impl MapBatchV1 {
    pub fn new(
        batch_id: impl Into<String>,
        query: &str,
        reports: &[&ReportContextRowV1],
        profile: &PaperBenchmarkProfileV1,
        counter: &dyn TokenCounter,
    ) -> Result<Self, PaperContractError> {
        let context = render_map_context(query, reports)?;
        let counted = CountedTextV1::measure(counter, &context).map_err(token_error)?;
        let input_fingerprint = fingerprint_parts(&[
            b"graphrag-paper-map-input-v1",
            profile.map_prompt_version.as_bytes(),
            profile.map_response_schema_version.as_bytes(),
            context.as_bytes(),
        ]);
        Ok(Self {
            schema_version: 1,
            batch_id: batch_id.into(),
            query_fingerprint: fingerprint_text(query),
            prompt_version: profile.map_prompt_version.clone(),
            response_schema_version: profile.map_response_schema_version.clone(),
            report_ids: reports
                .iter()
                .map(|report| report.report_id.clone())
                .collect(),
            input_context: context,
            input_fingerprint,
            input_token_count: counted.token_count,
        })
    }

    fn validate_against(
        &self,
        query: &str,
        reports: &[&ReportContextRowV1],
        profile: &PaperBenchmarkProfileV1,
        counter: &dyn TokenCounter,
    ) -> Result<(), PaperContractError> {
        require_v1(self.schema_version, "map batch")?;
        require_text(&self.batch_id, "batch.id", "batch_id")?;
        if self.query_fingerprint != fingerprint_text(query) {
            return Err(PaperContractError::violation(
                "batch.query",
                "batch query fingerprint is stale",
            ));
        }
        let expected_ids = reports
            .iter()
            .map(|report| report.report_id.clone())
            .collect::<Vec<_>>();
        if self.report_ids != expected_ids {
            return Err(PaperContractError::violation(
                "batch.reports",
                "batch report IDs differ from selected reports",
            ));
        }
        let expected_context = render_map_context(query, reports)?;
        if self.input_context != expected_context {
            return Err(PaperContractError::violation(
                "batch.context",
                "batch context differs from canonical report input",
            ));
        }
        if self.prompt_version != profile.map_prompt_version
            || self.response_schema_version != profile.map_response_schema_version
        {
            return Err(PaperContractError::violation(
                "batch.execution_version",
                "map prompt/response schema differs from profile",
            ));
        }
        let expected_input_fingerprint = fingerprint_parts(&[
            b"graphrag-paper-map-input-v1",
            self.prompt_version.as_bytes(),
            self.response_schema_version.as_bytes(),
            self.input_context.as_bytes(),
        ]);
        if self.input_fingerprint != expected_input_fingerprint {
            return Err(PaperContractError::violation(
                "batch.input_fingerprint",
                "map input fingerprint is stale",
            ));
        }
        validate_token_count(
            counter,
            &self.input_context,
            &self.input_token_count,
            "batch.token_count",
        )?;
        if self.input_token_count.tokenizer != profile.tokenizer {
            return Err(PaperContractError::violation(
                "batch.tokenizer",
                "batch tokenizer differs from profile",
            ));
        }
        if self.input_token_count.tokens > profile.map_call_window_tokens {
            return Err(PaperContractError::violation(
                "batch.window",
                "batch exceeds map call window",
            ));
        }
        Ok(())
    }
}

pub fn plan_map_batches(
    query: &str,
    seed: u64,
    reports: &[ReportContextRowV1],
    profile: &PaperBenchmarkProfileV1,
    counter: &dyn TokenCounter,
) -> Result<Vec<MapBatchV1>, PaperContractError> {
    profile.validate()?;
    let report_by_id = reports
        .iter()
        .map(|report| (report.report_id.as_str(), report))
        .collect::<HashMap<_, _>>();
    if report_by_id.len() != reports.len() || reports.is_empty() {
        return Err(PaperContractError::violation(
            "batch.report_inventory",
            "report inventory must be nonempty with unique IDs",
        ));
    }
    let order = canonical_report_order(seed, reports);
    let mut batches = Vec::new();
    let mut current = Vec::new();
    for report_id in order {
        let report = report_by_id[report_id.as_str()];
        let mut candidate = current.clone();
        candidate.push(report);
        let candidate_batch = MapBatchV1::new(
            format!("map:{seed}:batch:{}", batches.len()),
            query,
            &candidate,
            profile,
            counter,
        )?;
        if candidate_batch.input_token_count.tokens <= profile.map_call_window_tokens {
            current = candidate;
            continue;
        }
        if current.is_empty() {
            return Err(PaperContractError::violation(
                "batch.oversized_report",
                format!("report {report_id} cannot fit map call window"),
            ));
        }
        batches.push(MapBatchV1::new(
            format!("map:{seed}:batch:{}", batches.len()),
            query,
            &current,
            profile,
            counter,
        )?);
        current = vec![report];
        let single = MapBatchV1::new(
            format!("map:{seed}:batch:{}", batches.len()),
            query,
            &current,
            profile,
            counter,
        )?;
        if single.input_token_count.tokens > profile.map_call_window_tokens {
            return Err(PaperContractError::violation(
                "batch.oversized_report",
                format!("report {report_id} cannot fit map call window"),
            ));
        }
    }
    if !current.is_empty() {
        batches.push(MapBatchV1::new(
            format!("map:{seed}:batch:{}", batches.len()),
            query,
            &current,
            profile,
            counter,
        )?);
    }
    Ok(batches)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MapResponsePayloadV1 {
    pub schema_version: u32,
    pub answer: String,
    pub helpfulness: u8,
    pub referenced_report_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MapResultV1 {
    pub schema_version: u32,
    pub query_fingerprint: String,
    pub batch_id: String,
    pub input_fingerprint: String,
    pub prompt_version: String,
    pub response_schema_version: String,
    pub report_ids: Vec<String>,
    pub answer: String,
    pub helpfulness: u8,
    pub model: String,
    pub raw_response: String,
    pub response_sha256: String,
    pub answer_token_count: TokenCount,
    pub referenced_report_ids: Vec<String>,
}

impl MapResultV1 {
    pub fn new(
        batch: &MapBatchV1,
        answer: impl Into<String>,
        helpfulness: u8,
        model: impl Into<String>,
        referenced_report_ids: Vec<String>,
        counter: &dyn TokenCounter,
    ) -> Result<Self, PaperContractError> {
        let payload = MapResponsePayloadV1 {
            schema_version: 1,
            answer: answer.into(),
            helpfulness,
            referenced_report_ids,
        };
        let raw_response = serde_json::to_string(&payload).map_err(json_error)?;
        Self::from_raw(batch, model, raw_response, counter)
    }

    pub fn from_raw(
        batch: &MapBatchV1,
        model: impl Into<String>,
        raw_response: impl Into<String>,
        counter: &dyn TokenCounter,
    ) -> Result<Self, PaperContractError> {
        let raw_response = raw_response.into();
        let payload: MapResponsePayloadV1 =
            serde_json::from_str(&raw_response).map_err(|error| {
                PaperContractError::violation("map.response_schema", error.to_string())
            })?;
        require_v1(payload.schema_version, "map response payload")?;
        let answer_counted =
            CountedTextV1::measure(counter, &payload.answer).map_err(token_error)?;
        Ok(Self {
            schema_version: 1,
            query_fingerprint: batch.query_fingerprint.clone(),
            batch_id: batch.batch_id.clone(),
            input_fingerprint: batch.input_fingerprint.clone(),
            prompt_version: batch.prompt_version.clone(),
            response_schema_version: batch.response_schema_version.clone(),
            report_ids: batch.report_ids.clone(),
            answer: payload.answer,
            helpfulness: payload.helpfulness,
            model: model.into(),
            response_sha256: fingerprint_text(&raw_response),
            raw_response,
            answer_token_count: answer_counted.token_count,
            referenced_report_ids: payload.referenced_report_ids,
        })
    }

    fn validate_against(
        &self,
        batch: &MapBatchV1,
        profile: &PaperBenchmarkProfileV1,
        counter: &dyn TokenCounter,
    ) -> Result<(), PaperContractError> {
        require_v1(self.schema_version, "map result")?;
        if self.query_fingerprint != batch.query_fingerprint
            || self.batch_id != batch.batch_id
            || self.input_fingerprint != batch.input_fingerprint
            || self.prompt_version != batch.prompt_version
            || self.response_schema_version != batch.response_schema_version
            || self.report_ids != batch.report_ids
        {
            return Err(PaperContractError::violation(
                "map.input",
                "map result does not bind the exact versioned batch input",
            ));
        }
        if self.prompt_version != profile.map_prompt_version
            || self.response_schema_version != profile.map_response_schema_version
        {
            return Err(PaperContractError::violation(
                "map.execution_version",
                "map execution version differs from profile",
            ));
        }
        if self.helpfulness > 100 {
            return Err(PaperContractError::violation(
                "map.helpfulness",
                "helpfulness must be in 0..=100",
            ));
        }
        if self.helpfulness == 0 {
            if !self.answer.is_empty() || !self.referenced_report_ids.is_empty() {
                return Err(PaperContractError::violation(
                    "map.zero_answer",
                    "score-zero map result must have empty answer and references",
                ));
            }
        } else {
            require_text(&self.answer, "map.answer", "answer")?;
            if self.referenced_report_ids.is_empty() {
                return Err(PaperContractError::violation(
                    "map.references",
                    "positive map answer must cite a report",
                ));
            }
        }
        require_text(&self.model, "map.model", "model")?;
        if self.response_sha256 != fingerprint_text(&self.raw_response) {
            return Err(PaperContractError::violation(
                "map.response_fingerprint",
                "raw response fingerprint is stale",
            ));
        }
        let payload: MapResponsePayloadV1 =
            serde_json::from_str(&self.raw_response).map_err(|error| {
                PaperContractError::violation("map.response_schema", error.to_string())
            })?;
        if payload.schema_version != 1
            || payload.answer != self.answer
            || payload.helpfulness != self.helpfulness
            || payload.referenced_report_ids != self.referenced_report_ids
        {
            return Err(PaperContractError::violation(
                "map.response_binding",
                "parsed map fields differ from raw response payload",
            ));
        }
        validate_token_count(
            counter,
            &self.answer,
            &self.answer_token_count,
            "map.answer_token_count",
        )?;
        if self.answer_token_count.tokenizer != profile.tokenizer {
            return Err(PaperContractError::violation(
                "map.tokenizer",
                "map answer tokenizer differs from profile",
            ));
        }
        require_unique(
            &self.referenced_report_ids,
            "map.references",
            "referenced_report_ids",
        )?;
        let reports = batch.report_ids.iter().collect::<HashSet<_>>();
        if !self
            .referenced_report_ids
            .iter()
            .all(|report_id| reports.contains(report_id))
        {
            return Err(PaperContractError::violation(
                "map.references",
                "map answer references a report outside its batch",
            ));
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct ReduceMapWire<'a> {
    batch_id: &'a str,
    input_fingerprint: &'a str,
    helpfulness: u8,
    answer: &'a str,
    response_sha256: &'a str,
    referenced_report_ids: &'a [String],
}

#[derive(Serialize)]
struct ReduceContextWire<'a> {
    schema_version: u32,
    query: &'a str,
    maps: Vec<ReduceMapWire<'a>>,
}

pub fn render_reduce_context(
    query: &str,
    maps: &[&MapResultV1],
) -> Result<String, PaperContractError> {
    require_text(query, "reduce_context.query", "query")?;
    if maps.is_empty() {
        return Err(PaperContractError::violation(
            "reduce_context.maps",
            "reduce context has no positive map answers",
        ));
    }
    let maps = maps
        .iter()
        .map(|map| ReduceMapWire {
            batch_id: &map.batch_id,
            input_fingerprint: &map.input_fingerprint,
            helpfulness: map.helpfulness,
            answer: &map.answer,
            response_sha256: &map.response_sha256,
            referenced_report_ids: &map.referenced_report_ids,
        })
        .collect();
    serde_json::to_string(&ReduceContextWire {
        schema_version: 1,
        query,
        maps,
    })
    .map_err(json_error)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReduceInputV1 {
    pub schema_version: u32,
    pub reduce_id: String,
    pub query_fingerprint: String,
    pub prompt_version: String,
    pub response_schema_version: String,
    pub input_batch_ids: Vec<String>,
    pub input_context: String,
    pub input_fingerprint: String,
    pub input_token_count: TokenCount,
}

impl ReduceInputV1 {
    pub fn new(
        reduce_id: impl Into<String>,
        query: &str,
        maps: &[&MapResultV1],
        profile: &PaperBenchmarkProfileV1,
        counter: &dyn TokenCounter,
    ) -> Result<Self, PaperContractError> {
        let context = render_reduce_context(query, maps)?;
        let counted = CountedTextV1::measure(counter, &context).map_err(token_error)?;
        let input_fingerprint = fingerprint_parts(&[
            b"graphrag-paper-reduce-input-v1",
            profile.reduce_prompt_version.as_bytes(),
            profile.reduce_response_schema_version.as_bytes(),
            context.as_bytes(),
        ]);
        Ok(Self {
            schema_version: 1,
            reduce_id: reduce_id.into(),
            query_fingerprint: fingerprint_text(query),
            prompt_version: profile.reduce_prompt_version.clone(),
            response_schema_version: profile.reduce_response_schema_version.clone(),
            input_batch_ids: maps.iter().map(|map| map.batch_id.clone()).collect(),
            input_context: context,
            input_fingerprint,
            input_token_count: counted.token_count,
        })
    }

    fn validate_against(
        &self,
        query: &str,
        maps: &[&MapResultV1],
        profile: &PaperBenchmarkProfileV1,
        counter: &dyn TokenCounter,
    ) -> Result<(), PaperContractError> {
        require_v1(self.schema_version, "reduce input")?;
        require_text(&self.reduce_id, "reduce_input.id", "reduce_id")?;
        if self.query_fingerprint != fingerprint_text(query) {
            return Err(PaperContractError::violation(
                "reduce_input.query",
                "reduce input query fingerprint is stale",
            ));
        }
        let expected_ids = maps
            .iter()
            .map(|map| map.batch_id.clone())
            .collect::<Vec<_>>();
        if self.input_batch_ids != expected_ids {
            return Err(PaperContractError::violation(
                "reduce.inputs",
                "reduce input is not the maximal ranked prefix",
            ));
        }
        let expected_context = render_reduce_context(query, maps)?;
        if self.input_context != expected_context {
            return Err(PaperContractError::violation(
                "reduce.context",
                "reduce context differs from canonical map-answer input",
            ));
        }
        if self.prompt_version != profile.reduce_prompt_version
            || self.response_schema_version != profile.reduce_response_schema_version
        {
            return Err(PaperContractError::violation(
                "reduce.execution_version",
                "reduce prompt/response schema differs from profile",
            ));
        }
        let expected_input_fingerprint = fingerprint_parts(&[
            b"graphrag-paper-reduce-input-v1",
            self.prompt_version.as_bytes(),
            self.response_schema_version.as_bytes(),
            self.input_context.as_bytes(),
        ]);
        if self.input_fingerprint != expected_input_fingerprint {
            return Err(PaperContractError::violation(
                "reduce.input_fingerprint",
                "reduce input fingerprint is stale",
            ));
        }
        validate_token_count(
            counter,
            &self.input_context,
            &self.input_token_count,
            "reduce.token_count",
        )?;
        if self.input_token_count.tokenizer != profile.tokenizer {
            return Err(PaperContractError::violation(
                "reduce.tokenizer",
                "reduce tokenizer differs from profile",
            ));
        }
        if self.input_token_count.tokens > profile.reduce_call_window_tokens {
            return Err(PaperContractError::violation(
                "reduce.window",
                "reduce context exceeds reduce call window",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReduceResponsePayloadV1 {
    pub schema_version: u32,
    pub answer: String,
    pub referenced_report_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReduceResultV1 {
    pub schema_version: u32,
    pub query_fingerprint: String,
    pub reduce_id: String,
    pub input_fingerprint: String,
    pub prompt_version: String,
    pub response_schema_version: String,
    pub answer: String,
    pub model: String,
    pub raw_response: String,
    pub response_sha256: String,
    pub answer_token_count: TokenCount,
    pub referenced_report_ids: Vec<String>,
}

impl ReduceResultV1 {
    pub fn new(
        input: &ReduceInputV1,
        answer: impl Into<String>,
        model: impl Into<String>,
        referenced_report_ids: Vec<String>,
        counter: &dyn TokenCounter,
    ) -> Result<Self, PaperContractError> {
        let payload = ReduceResponsePayloadV1 {
            schema_version: 1,
            answer: answer.into(),
            referenced_report_ids,
        };
        let raw_response = serde_json::to_string(&payload).map_err(json_error)?;
        Self::from_raw(input, model, raw_response, counter)
    }

    pub fn from_raw(
        input: &ReduceInputV1,
        model: impl Into<String>,
        raw_response: impl Into<String>,
        counter: &dyn TokenCounter,
    ) -> Result<Self, PaperContractError> {
        let raw_response = raw_response.into();
        let payload: ReduceResponsePayloadV1 =
            serde_json::from_str(&raw_response).map_err(|error| {
                PaperContractError::violation("reduce.response_schema", error.to_string())
            })?;
        require_v1(payload.schema_version, "reduce response payload")?;
        let answer_counted =
            CountedTextV1::measure(counter, &payload.answer).map_err(token_error)?;
        Ok(Self {
            schema_version: 1,
            query_fingerprint: input.query_fingerprint.clone(),
            reduce_id: input.reduce_id.clone(),
            input_fingerprint: input.input_fingerprint.clone(),
            prompt_version: input.prompt_version.clone(),
            response_schema_version: input.response_schema_version.clone(),
            answer: payload.answer,
            model: model.into(),
            response_sha256: fingerprint_text(&raw_response),
            raw_response,
            answer_token_count: answer_counted.token_count,
            referenced_report_ids: payload.referenced_report_ids,
        })
    }

    fn validate_against(
        &self,
        input: &ReduceInputV1,
        allowed_reports: &HashSet<&String>,
        profile: &PaperBenchmarkProfileV1,
        counter: &dyn TokenCounter,
    ) -> Result<(), PaperContractError> {
        require_v1(self.schema_version, "reduce result")?;
        if self.query_fingerprint != input.query_fingerprint
            || self.reduce_id != input.reduce_id
            || self.input_fingerprint != input.input_fingerprint
            || self.prompt_version != input.prompt_version
            || self.response_schema_version != input.response_schema_version
        {
            return Err(PaperContractError::violation(
                "reduce.input",
                "reduce result does not bind the exact versioned reduce input",
            ));
        }
        if self.prompt_version != profile.reduce_prompt_version
            || self.response_schema_version != profile.reduce_response_schema_version
        {
            return Err(PaperContractError::violation(
                "reduce.execution_version",
                "reduce execution version differs from profile",
            ));
        }
        require_text(&self.answer, "reduce.answer", "answer")?;
        require_text(&self.model, "reduce.model", "model")?;
        if self.referenced_report_ids.is_empty() {
            return Err(PaperContractError::violation(
                "reduce.references",
                "reduce answer must cite a positive map answer",
            ));
        }
        if self.response_sha256 != fingerprint_text(&self.raw_response) {
            return Err(PaperContractError::violation(
                "reduce.response_fingerprint",
                "raw response fingerprint is stale",
            ));
        }
        let payload: ReduceResponsePayloadV1 =
            serde_json::from_str(&self.raw_response).map_err(|error| {
                PaperContractError::violation("reduce.response_schema", error.to_string())
            })?;
        if payload.schema_version != 1
            || payload.answer != self.answer
            || payload.referenced_report_ids != self.referenced_report_ids
        {
            return Err(PaperContractError::violation(
                "reduce.response_binding",
                "parsed reduce fields differ from raw response payload",
            ));
        }
        validate_token_count(
            counter,
            &self.answer,
            &self.answer_token_count,
            "reduce.answer_token_count",
        )?;
        if self.answer_token_count.tokenizer != profile.tokenizer {
            return Err(PaperContractError::violation(
                "reduce.answer_tokenizer",
                "reduce answer tokenizer differs from profile",
            ));
        }
        require_unique(
            &self.referenced_report_ids,
            "reduce.references",
            "referenced_report_ids",
        )?;
        if !self
            .referenced_report_ids
            .iter()
            .all(|report_id| allowed_reports.contains(report_id))
        {
            return Err(PaperContractError::violation(
                "reduce.references",
                "reduce answer references evidence not cited by selected maps",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryTraceV1 {
    pub schema_version: u32,
    pub query: String,
    pub query_fingerprint: String,
    pub hierarchy_run_id: String,
    pub cut: u32,
    pub seed: u64,
    pub profile: PaperBenchmarkProfileV1,
    pub reports: Vec<ReportContextRowV1>,
    pub ordered_report_ids: Vec<String>,
    pub batches: Vec<MapBatchV1>,
    pub map_results: Vec<MapResultV1>,
    pub retained_batch_ids: Vec<String>,
    pub reduce_inputs: Vec<ReduceInputV1>,
    pub reduce_results: Vec<ReduceResultV1>,
}

impl QueryTraceV1 {
    /// Full M0 validation. `expected_cut` is authoritative inventory supplied
    /// by the hierarchy/report snapshot, so deleting a report everywhere in a
    /// trace cannot redefine “exhaustive”.
    pub fn validate_against_cut(
        &self,
        expected_cut: &[ReportContextRowV1],
        counter: &dyn TokenCounter,
    ) -> Result<(), PaperContractError> {
        require_v1(self.schema_version, "query trace")?;
        require_text(&self.query, "trace.query", "query")?;
        if self.query_fingerprint != fingerprint_text(&self.query) {
            return Err(PaperContractError::violation(
                "trace.query_fingerprint",
                "query fingerprint is stale",
            ));
        }
        require_text(
            &self.hierarchy_run_id,
            "trace.hierarchy_run",
            "hierarchy_run_id",
        )?;
        self.profile.validate()?;
        if self.profile.tokenizer != *counter.tokenizer_id() {
            return Err(PaperContractError::violation(
                "trace.tokenizer",
                "trace profile tokenizer differs from validator",
            ));
        }
        if expected_cut.is_empty() {
            return Err(PaperContractError::violation(
                "trace.expected_cut",
                "authoritative cut is empty",
            ));
        }

        let mut expected_by_id = HashMap::new();
        for report in expected_cut {
            report.validate_with_counter(counter)?;
            if report.hierarchy_run_id != self.hierarchy_run_id || report.cut != self.cut {
                return Err(PaperContractError::violation(
                    "trace.expected_cut",
                    "authoritative report belongs to another run/cut",
                ));
            }
            if expected_by_id.insert(&report.report_id, report).is_some() {
                return Err(PaperContractError::violation(
                    "trace.expected_cut",
                    "authoritative cut has duplicate report IDs",
                ));
            }
        }

        let mut report_by_id = HashMap::new();
        for report in &self.reports {
            report.validate_with_counter(counter)?;
            if report.hierarchy_run_id != self.hierarchy_run_id || report.cut != self.cut {
                return Err(PaperContractError::violation(
                    "trace.report_cut",
                    "trace report belongs to another run/cut",
                ));
            }
            if report_by_id.insert(&report.report_id, report).is_some() {
                return Err(PaperContractError::violation(
                    "trace.report_coverage",
                    "duplicate trace report ID",
                ));
            }
        }
        if report_by_id.len() != expected_by_id.len()
            || expected_by_id
                .iter()
                .any(|(id, expected)| report_by_id.get(id) != Some(expected))
        {
            return Err(PaperContractError::violation(
                "trace.report_coverage",
                "trace reports do not equal the authoritative cut",
            ));
        }
        require_unique(
            &self.ordered_report_ids,
            "trace.report_coverage",
            "ordered_report_ids",
        )?;
        let expected_ids = expected_by_id.keys().copied().collect::<HashSet<_>>();
        let ordered_ids = self.ordered_report_ids.iter().collect::<HashSet<_>>();
        if expected_ids != ordered_ids {
            return Err(PaperContractError::violation(
                "trace.report_coverage",
                "ordered reports are not exhaustive",
            ));
        }
        let canonical_order = canonical_report_order(self.seed, expected_cut);
        if self.ordered_report_ids != canonical_order {
            return Err(PaperContractError::violation(
                "trace.seeded_order",
                "report order is not the canonical seeded shuffle",
            ));
        }
        let canonical_batches =
            plan_map_batches(&self.query, self.seed, expected_cut, &self.profile, counter)?;
        if self.batches != canonical_batches {
            return Err(PaperContractError::violation(
                "trace.batch_plan",
                "map batches are not canonical greedy/maximal packing",
            ));
        }

        let mut batch_by_id = HashMap::new();
        let mut flattened = Vec::new();
        for batch in &self.batches {
            let mut reports = Vec::new();
            for report_id in &batch.report_ids {
                let Some(report) = expected_by_id.get(report_id) else {
                    return Err(PaperContractError::violation(
                        "trace.report_coverage",
                        "batch names report outside authoritative cut",
                    ));
                };
                reports.push(*report);
            }
            batch.validate_against(&self.query, &reports, &self.profile, counter)?;
            if batch_by_id.insert(&batch.batch_id, batch).is_some() {
                return Err(PaperContractError::violation(
                    "trace.batch_ids",
                    "duplicate batch ID",
                ));
            }
            flattened.extend(batch.report_ids.iter().cloned());
        }
        if flattened != self.ordered_report_ids {
            return Err(PaperContractError::violation(
                "trace.report_coverage",
                "batches drop, duplicate, or reorder reports",
            ));
        }

        let mut map_by_batch = HashMap::new();
        for result in &self.map_results {
            let Some(batch) = batch_by_id.get(&result.batch_id) else {
                return Err(PaperContractError::violation(
                    "trace.map_coverage",
                    "map result names unknown batch",
                ));
            };
            result.validate_against(batch, &self.profile, counter)?;
            if map_by_batch.insert(&result.batch_id, result).is_some() {
                return Err(PaperContractError::violation(
                    "trace.map_coverage",
                    "duplicate map result",
                ));
            }
        }
        if map_by_batch.len() != self.batches.len()
            || self
                .batches
                .iter()
                .any(|batch| !map_by_batch.contains_key(&batch.batch_id))
        {
            return Err(PaperContractError::violation(
                "trace.map_coverage",
                "every batch needs exactly one map result",
            ));
        }

        let batch_position = self
            .batches
            .iter()
            .enumerate()
            .map(|(position, batch)| (&batch.batch_id, position))
            .collect::<HashMap<_, _>>();
        let mut positive = self
            .map_results
            .iter()
            .filter(|result| result.helpfulness > 0)
            .collect::<Vec<_>>();
        positive.sort_by(|left, right| {
            right
                .helpfulness
                .cmp(&left.helpfulness)
                .then_with(|| batch_position[&left.batch_id].cmp(&batch_position[&right.batch_id]))
        });
        if positive.is_empty() {
            return Err(PaperContractError::violation(
                "trace.no_evidence",
                "all map answers have score zero; do not fabricate a reduce answer",
            ));
        }
        let expected_retained = positive
            .iter()
            .map(|result| result.batch_id.clone())
            .collect::<Vec<_>>();
        if self.retained_batch_ids != expected_retained {
            return Err(PaperContractError::violation(
                "trace.retained_batches",
                "retained batches must be positive-score and ranked",
            ));
        }

        let mut maximal_prefix = 0usize;
        for length in 1..=positive.len() {
            let context = render_reduce_context(&self.query, &positive[..length])?;
            let tokens = counter.count(&context).map_err(token_error)?;
            if tokens <= self.profile.reduce_call_window_tokens {
                maximal_prefix = length;
            } else {
                break;
            }
        }
        if maximal_prefix == 0 {
            return Err(PaperContractError::violation(
                "reduce.oversized_first_answer",
                "highest-ranked map answer cannot fit reduce window",
            ));
        }
        let selected = &positive[..maximal_prefix];
        if self.reduce_inputs.len() != 1 {
            return Err(PaperContractError::violation(
                "trace.reduce_input_count",
                "paper global query has exactly one reduce input",
            ));
        }
        self.reduce_inputs[0].validate_against(&self.query, selected, &self.profile, counter)?;
        if self.reduce_results.len() != 1 {
            return Err(PaperContractError::violation(
                "trace.reduce_count",
                "paper global query has exactly one reduce result",
            ));
        }

        let mut allowed_reports = HashSet::new();
        for batch_id in &self.reduce_inputs[0].input_batch_ids {
            let Some(map) = map_by_batch.get(batch_id) else {
                return Err(PaperContractError::violation(
                    "reduce.inputs",
                    "reduce input names unknown map batch",
                ));
            };
            allowed_reports.extend(map.referenced_report_ids.iter());
        }
        self.reduce_results[0].validate_against(
            &self.reduce_inputs[0],
            &allowed_reports,
            &self.profile,
            counter,
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaperCapability {
    M0Contract,
    M1ThinGlobalQuery,
    M2ElementInstances,
    M3RecursiveHierarchy,
    M4CommunityReports,
    M5PaperGlobalQuery,
    M6Evaluation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaperCapabilityState {
    Implemented {
        binding_test: &'static str,
    },
    Gated {
        milestone: &'static str,
        reason: &'static str,
    },
}

impl PaperCapabilityState {
    pub fn is_implemented(self) -> bool {
        matches!(self, Self::Implemented { .. })
    }

    pub fn reason(self) -> &'static str {
        match self {
            Self::Implemented { binding_test } => binding_test,
            Self::Gated { reason, .. } => reason,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PaperCapabilityRecord {
    pub capability: PaperCapability,
    pub state: PaperCapabilityState,
}

pub fn paper_capabilities() -> Vec<PaperCapabilityRecord> {
    use PaperCapability::*;
    use PaperCapabilityState::*;
    vec![
        PaperCapabilityRecord {
            capability: M0Contract,
            state: Implemented {
                binding_test: "paper_contract::fake_executor_traverses_the_green_contract_oracle",
            },
        },
        PaperCapabilityRecord {
            capability: M1ThinGlobalQuery,
            state: Gated {
                milestone: "M1",
                reason: "production exhaustive single-cut map/score/reduce is absent",
            },
        },
        PaperCapabilityRecord {
            capability: M2ElementInstances,
            state: Gated {
                milestone: "M2",
                reason: "versioned accepted element instances and support weights are absent",
            },
        },
        PaperCapabilityRecord {
            capability: M3RecursiveHierarchy,
            state: Gated {
                milestone: "M3",
                reason: "recursive parent-induced Leiden cuts are absent",
            },
        },
        PaperCapabilityRecord {
            capability: M4CommunityReports,
            state: Gated {
                milestone: "M4",
                reason: "current citation-resolvable community reports are absent",
            },
        },
        PaperCapabilityRecord {
            capability: M5PaperGlobalQuery,
            state: Gated {
                milestone: "M5",
                reason: "report-backed paper global answer binding is absent",
            },
        },
        PaperCapabilityRecord {
            capability: M6Evaluation,
            state: Gated {
                milestone: "M6",
                reason: "SS/TS/C-cut evaluation protocol is absent",
            },
        },
    ]
}
