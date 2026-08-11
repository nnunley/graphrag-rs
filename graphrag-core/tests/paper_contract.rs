//! Executable M0 contract for arXiv:2404.16130v2 §3.1.

use graphrag_core::paper::{
    ExtractionAttemptIdentityParts, ExtractionAttemptIdentityV1, MapResultV1,
    PaperBenchmarkProfileV1, PaperCapability, PaperCapabilityState, PaperContractError,
    QueryTraceV1, RecordRefV1, ReduceInputV1, ReduceResultV1, ReportContextRowV1, SourceDocumentV1,
    paper_capabilities, plan_map_batches,
};
use graphrag_core::token::{TokenCountError, TokenCounter, TokenizerId, fingerprint_text};
use hegel::TestCase;
use hegel::generators as gs;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Deserialize)]
struct Oracle {
    schema_version: u32,
    fixture_id: String,
    profile: PaperBenchmarkProfileV1,
    source_fingerprints: HashMap<String, String>,
    fake_extractions: Vec<FakeExtraction>,
    repeated_entity: RepeatedEntity,
    repeated_relationship: RepeatedRelationship,
    isolated_entity: String,
    cuts: Vec<Cut>,
    oversized_community: OversizedCommunity,
    questions: Vec<QuestionSpec>,
}

#[derive(Debug, Clone, Deserialize)]
struct FakeExtraction {
    source_id: String,
    entities: Vec<String>,
    relationships: Vec<Relationship>,
}

#[derive(Debug, Clone, Deserialize)]
struct Relationship {
    source: String,
    target: String,
    kind: String,
}

#[derive(Debug, Clone, Deserialize)]
struct RepeatedEntity {
    name: String,
    source_ids: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct RepeatedRelationship {
    source: String,
    target: String,
    kind: String,
    source_ids: Vec<String>,
    support_count: u32,
}

#[derive(Debug, Clone, Deserialize)]
struct Cut {
    cut: u32,
    communities: Vec<ExpectedCommunity>,
}

#[derive(Debug, Clone, Deserialize)]
struct ExpectedCommunity {
    community_id: i64,
    members: Vec<String>,
    parent_id: Option<i64>,
    projected_from: Option<i64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OversizedCommunity {
    community_id: i64,
    element_context: String,
}

#[derive(Debug, Clone, Deserialize)]
struct QuestionSpec {
    question_id: String,
    text: String,
    seed: u64,
    map_answers: Vec<FakeMapAnswer>,
    reduce_answer: String,
    reduce_references: Vec<String>,
    required_aspects: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct FakeMapAnswer {
    report_id: String,
    helpfulness: u8,
    answer: String,
    referenced_report_ids: Vec<String>,
}

#[derive(Debug)]
struct WordCounter {
    id: TokenizerId,
}

impl WordCounter {
    fn new() -> Self {
        Self {
            id: TokenizerId::new("test:unicode-whitespace:v1").unwrap(),
        }
    }
}

impl TokenCounter for WordCounter {
    fn tokenizer_id(&self) -> &TokenizerId {
        &self.id
    }

    fn count(&self, text: &str) -> Result<u32, TokenCountError> {
        u32::try_from(text.split_whitespace().count()).map_err(|_| TokenCountError::Overflow)
    }
}

fn fixture() -> (Vec<SourceDocumentV1>, Oracle) {
    let sources = include_str!("fixtures/paper_m0/sources.jsonl")
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let oracle = serde_json::from_str(include_str!("fixtures/paper_m0/oracle.json")).unwrap();
    (sources, oracle)
}

fn fixture_error(code: &'static str, detail: impl Into<String>) -> PaperContractError {
    PaperContractError::violation(code, detail)
}

fn extracted_entities(oracle: &Oracle) -> HashSet<&str> {
    oracle
        .fake_extractions
        .iter()
        .flat_map(|extraction| extraction.entities.iter().map(String::as_str))
        .collect()
}

fn validate_fixture(
    sources: &[SourceDocumentV1],
    oracle: &Oracle,
) -> Result<(), PaperContractError> {
    if oracle.schema_version != 1 || oracle.fixture_id.trim().is_empty() {
        return Err(fixture_error("fixture.schema", "bad oracle header"));
    }
    oracle.profile.validate()?;
    let source_by_id = sources
        .iter()
        .map(|source| (source.document_id.as_str(), source))
        .collect::<HashMap<_, _>>();
    if sources.iter().any(|source| source.validate().is_err())
        || source_by_id.len() != sources.len()
        || oracle.source_fingerprints.len() != sources.len()
        || sources.iter().any(|source| {
            oracle.source_fingerprints.get(&source.document_id)
                != source.fingerprint().ok().as_ref()
        })
    {
        return Err(fixture_error(
            "fixture.sources",
            "source inventory/fingerprint mismatch",
        ));
    }

    let extraction_ids = oracle
        .fake_extractions
        .iter()
        .map(|extraction| extraction.source_id.as_str())
        .collect::<HashSet<_>>();
    if extraction_ids.len() != oracle.fake_extractions.len()
        || extraction_ids != source_by_id.keys().copied().collect()
    {
        return Err(fixture_error(
            "fixture.extractions",
            "one extraction per source required",
        ));
    }
    for extraction in &oracle.fake_extractions {
        let source = source_by_id[extraction.source_id.as_str()];
        let entities = extraction
            .entities
            .iter()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        if entities.len() != extraction.entities.len()
            || entities.iter().any(|entity| entity.trim().is_empty())
            || extraction
                .entities
                .iter()
                .any(|entity| !source.text.contains(entity))
        {
            return Err(fixture_error(
                "fixture.extractions",
                "entities must be nonempty, unique, and present in source text",
            ));
        }
        if extraction.relationships.iter().any(|relationship| {
            relationship.source.trim().is_empty()
                || relationship.target.trim().is_empty()
                || relationship.kind.trim().is_empty()
                || !entities.contains(relationship.source.as_str())
                || !entities.contains(relationship.target.as_str())
                || !source.text.contains(&relationship.source)
                || !source.text.contains(&relationship.target)
        }) {
            return Err(fixture_error(
                "fixture.extractions",
                "edge must have typed endpoints in the extracted entity set",
            ));
        }
    }

    let repeated_sources = oracle
        .repeated_entity
        .source_ids
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    if repeated_sources.len() != oracle.repeated_entity.source_ids.len()
        || repeated_sources.len() < 2
        || !repeated_sources.iter().all(|source_id| {
            oracle.fake_extractions.iter().any(|extraction| {
                extraction.source_id == *source_id
                    && extraction.entities.contains(&oracle.repeated_entity.name)
            })
        })
    {
        return Err(fixture_error(
            "fixture.repeated_entity",
            "bad repeated entity evidence",
        ));
    }

    let repeated = &oracle.repeated_relationship;
    let repeated_source_ids = repeated
        .source_ids
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    let actual_sources = oracle
        .fake_extractions
        .iter()
        .filter(|extraction| {
            extraction.relationships.iter().any(|relationship| {
                relationship.source == repeated.source
                    && relationship.target == repeated.target
                    && relationship.kind == repeated.kind
            })
        })
        .map(|extraction| extraction.source_id.as_str())
        .collect::<HashSet<_>>();
    if repeated_source_ids.len() != repeated.source_ids.len()
        || actual_sources != repeated_source_ids
        || actual_sources.len() as u32 != repeated.support_count
        || repeated.support_count < 2
        || actual_sources.iter().any(|source_id| {
            let text = &source_by_id[source_id].text;
            !text.contains(&repeated.source)
                || !text.contains(&repeated.target)
                || !text.to_lowercase().contains(&repeated.kind.to_lowercase())
        })
    {
        return Err(fixture_error(
            "fixture.edge_support",
            "support is not distinct source evidence",
        ));
    }

    let entities = extracted_entities(oracle);
    if !entities.contains(oracle.isolated_entity.as_str()) {
        return Err(fixture_error("fixture.entities", "isolate not extracted"));
    }
    let isolate = oracle
        .fake_extractions
        .iter()
        .find(|extraction| extraction.entities.contains(&oracle.isolated_entity))
        .ok_or_else(|| fixture_error("fixture.entities", "missing isolate extraction"))?;
    if !isolate.relationships.is_empty() {
        return Err(fixture_error(
            "fixture.entities",
            "isolate has a relationship",
        ));
    }

    if oracle.cuts.len() < 2
        || oracle
            .cuts
            .iter()
            .enumerate()
            .any(|(index, cut)| cut.cut as usize != index)
    {
        return Err(fixture_error(
            "fixture.cut_count",
            "cuts must be sequential from zero",
        ));
    }
    let mut all_community_ids = HashSet::new();
    let mut prior = HashMap::<i64, HashSet<&str>>::new();
    for cut in &oracle.cuts {
        let mut seen = HashSet::new();
        for community in &cut.communities {
            if community.members.is_empty() {
                return Err(fixture_error("fixture.cut_mece", "empty community"));
            }
            if !all_community_ids.insert(community.community_id) {
                return Err(fixture_error(
                    "fixture.community_ids",
                    "duplicate community ID",
                ));
            }
            for member in &community.members {
                if !seen.insert(member.as_str()) {
                    return Err(fixture_error("fixture.cut_mece", "duplicate member in cut"));
                }
            }
            if cut.cut == 0 {
                if community.parent_id.is_some() || community.projected_from.is_some() {
                    return Err(fixture_error(
                        "fixture.cut_parent",
                        "coarsest cut has parent",
                    ));
                }
            } else {
                let parent_id = community
                    .parent_id
                    .ok_or_else(|| fixture_error("fixture.cut_parent", "missing parent"))?;
                let parent = prior
                    .get(&parent_id)
                    .ok_or_else(|| fixture_error("fixture.cut_parent", "unknown parent"))?;
                if !community
                    .members
                    .iter()
                    .all(|member| parent.contains(member.as_str()))
                {
                    return Err(fixture_error("fixture.cut_parent", "child outside parent"));
                }
                if let Some(projected) = community.projected_from
                    && (projected != parent_id
                        || community
                            .members
                            .iter()
                            .map(String::as_str)
                            .collect::<HashSet<_>>()
                            != *parent)
                {
                    return Err(fixture_error(
                        "fixture.terminal_projection",
                        "projection does not preserve terminal parent",
                    ));
                }
            }
        }
        if seen != entities {
            return Err(fixture_error(
                "fixture.cut_mece",
                "cut does not equal extracted entities",
            ));
        }
        prior = cut
            .communities
            .iter()
            .map(|community| {
                (
                    community.community_id,
                    community.members.iter().map(String::as_str).collect(),
                )
            })
            .collect();
    }
    if !oracle.cuts.iter().skip(1).any(|cut| {
        cut.communities.iter().any(|community| {
            community.members == [oracle.isolated_entity.clone()]
                && community.projected_from.is_some()
        })
    }) {
        return Err(fixture_error(
            "fixture.terminal_projection",
            "isolate not projected",
        ));
    }

    let counter = WordCounter::new();
    let reports = build_reports(sources, oracle, &counter)?;
    let cut_zero = oracle.cuts.iter().find(|cut| cut.cut == 0).unwrap();
    let community_by_id = cut_zero
        .communities
        .iter()
        .map(|community| (community.community_id, community))
        .collect::<HashMap<_, _>>();
    let report_communities = reports
        .iter()
        .map(|report| report.community_id)
        .collect::<HashSet<_>>();
    if reports.len() != cut_zero.communities.len()
        || report_communities.len() != reports.len()
        || report_communities != community_by_id.keys().copied().collect()
    {
        return Err(fixture_error(
            "fixture.reports",
            "cut zero requires exactly one report per community",
        ));
    }
    let report_ids = reports
        .iter()
        .map(|report| report.report_id.as_str())
        .collect::<HashSet<_>>();
    for report in &reports {
        let community = community_by_id[&report.community_id];
        let entity_refs = report
            .references
            .iter()
            .filter(|reference| reference.kind == "entity")
            .map(|reference| reference.id.as_str())
            .collect::<HashSet<_>>();
        let expected_entities = community
            .members
            .iter()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        let source_refs = report
            .references
            .iter()
            .filter(|reference| reference.kind == "source")
            .collect::<Vec<_>>();
        if entity_refs != expected_entities || source_refs.is_empty() {
            return Err(fixture_error(
                "fixture.report_references",
                "report lineage does not cover its community and sources",
            ));
        }
        for reference in &report.references {
            let valid = match reference.kind.as_str() {
                "entity" => {
                    expected_entities.contains(reference.id.as_str())
                        && reference.fingerprint == fingerprint_text(&reference.id)
                }
                "source" => {
                    source_by_id.contains_key(reference.id.as_str())
                        && oracle.source_fingerprints.get(&reference.id)
                            == Some(&reference.fingerprint)
                }
                _ => false,
            };
            if !valid {
                return Err(fixture_error(
                    "fixture.report_references",
                    "unresolved report reference",
                ));
            }
        }
    }

    if counter
        .count(&oracle.oversized_community.element_context)
        .map_err(|error| fixture_error("fixture.token_count", error.to_string()))?
        <= oracle.profile.report_call_window_tokens
        || !community_by_id.contains_key(&oracle.oversized_community.community_id)
    {
        return Err(fixture_error(
            "fixture.oversized_context",
            "context is not forced oversized",
        ));
    }
    if oracle.questions.len() != 2 {
        return Err(fixture_error(
            "fixture.questions",
            "need exactly two global questions",
        ));
    }
    for question in &oracle.questions {
        let aspects = question
            .required_aspects
            .iter()
            .map(|aspect| aspect.trim())
            .collect::<HashSet<_>>();
        if question.question_id.trim().is_empty()
            || question.text.trim().is_empty()
            || question.required_aspects.len() < 3
            || aspects.len() != question.required_aspects.len()
            || aspects.contains("")
        {
            return Err(fixture_error("fixture.questions", "bad question inventory"));
        }
        let answer_ids = question
            .map_answers
            .iter()
            .map(|answer| answer.report_id.as_str())
            .collect::<HashSet<_>>();
        if answer_ids != report_ids || question.map_answers.len() != reports.len() {
            return Err(fixture_error(
                "fixture.questions",
                "map-answer coverage mismatch",
            ));
        }
        for answer in &question.map_answers {
            let expected_refs = if answer.helpfulness == 0 {
                Vec::new()
            } else {
                vec![answer.report_id.clone()]
            };
            if answer.referenced_report_ids != expected_refs {
                return Err(fixture_error(
                    "fixture.map_references",
                    "fake map answer lacks exact report evidence",
                ));
            }
        }
        let reduce_refs = question.reduce_references.iter().collect::<HashSet<_>>();
        if reduce_refs.is_empty()
            || reduce_refs.len() != question.reduce_references.len()
            || !question
                .reduce_references
                .iter()
                .all(|report_id| report_ids.contains(report_id.as_str()))
        {
            return Err(fixture_error(
                "fixture.reduce_references",
                "fake reduce answer lacks report evidence",
            ));
        }
        if question
            .required_aspects
            .iter()
            .any(|aspect| !question.reduce_answer.contains(aspect))
        {
            return Err(fixture_error(
                "fixture.answer_aspects",
                "reduce answer omits required aspect",
            ));
        }
    }

    Ok(())
}

fn build_reports(
    sources: &[SourceDocumentV1],
    oracle: &Oracle,
    counter: &dyn TokenCounter,
) -> Result<Vec<ReportContextRowV1>, PaperContractError> {
    let source_by_id = sources
        .iter()
        .map(|source| (source.document_id.as_str(), source))
        .collect::<HashMap<_, _>>();
    let cut = oracle
        .cuts
        .iter()
        .find(|cut| cut.cut == 0)
        .ok_or_else(|| fixture_error("fake.reports", "missing cut zero"))?;
    let mut reports = Vec::new();
    for community in &cut.communities {
        let members = community
            .members
            .iter()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        let mut source_ids = oracle
            .fake_extractions
            .iter()
            .filter(|extraction| {
                extraction
                    .entities
                    .iter()
                    .any(|entity| members.contains(entity.as_str()))
            })
            .map(|extraction| extraction.source_id.clone())
            .collect::<Vec<_>>();
        source_ids.sort();
        source_ids.dedup();
        let mut member_ids = community.members.clone();
        member_ids.sort();
        let mut relationships = oracle
            .fake_extractions
            .iter()
            .flat_map(|extraction| extraction.relationships.iter())
            .filter(|relationship| {
                members.contains(relationship.source.as_str())
                    && members.contains(relationship.target.as_str())
            })
            .map(|relationship| {
                format!(
                    "{} {} {}",
                    relationship.source, relationship.kind, relationship.target
                )
            })
            .collect::<Vec<_>>();
        relationships.sort();
        relationships.dedup();
        let relationship_text = if relationships.is_empty() {
            "none".into()
        } else {
            relationships.join("; ")
        };
        let text = format!(
            "Community {}; members: {}; relationships: {}; sources: {}.",
            community.community_id,
            member_ids.join(", "),
            relationship_text,
            source_ids.join(", ")
        );
        let mut references = member_ids
            .iter()
            .map(|entity| RecordRefV1 {
                kind: "entity".into(),
                id: entity.clone(),
                fingerprint: fingerprint_text(entity),
            })
            .collect::<Vec<_>>();
        for source_id in source_ids {
            let source = source_by_id
                .get(source_id.as_str())
                .ok_or_else(|| fixture_error("fake.reports", "unknown source"))?;
            references.push(RecordRefV1 {
                kind: "source".into(),
                id: source_id,
                fingerprint: source.fingerprint()?,
            });
        }
        reports.push(ReportContextRowV1::new(
            format!("report:community:{}:v1", community.community_id),
            community.community_id,
            "hierarchy:fixture:v1",
            0,
            text,
            references,
            counter,
        )?);
    }
    Ok(reports)
}

fn execute_fake_query(
    question: &QuestionSpec,
    reports: &[ReportContextRowV1],
    profile: &PaperBenchmarkProfileV1,
    counter: &dyn TokenCounter,
) -> Result<QueryTraceV1, PaperContractError> {
    let answer_by_report = question
        .map_answers
        .iter()
        .map(|answer| (answer.report_id.as_str(), answer))
        .collect::<HashMap<_, _>>();
    let batches = plan_map_batches(&question.text, question.seed, reports, profile, counter)?;
    let mut maps = Vec::new();
    for batch in &batches {
        let answers = batch
            .report_ids
            .iter()
            .map(|report_id| answer_by_report[report_id.as_str()])
            .collect::<Vec<_>>();
        let helpfulness = answers
            .iter()
            .map(|answer| answer.helpfulness)
            .max()
            .unwrap();
        let answer = answers
            .iter()
            .filter(|answer| answer.helpfulness > 0)
            .map(|answer| answer.answer.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let mut references = answers
            .iter()
            .flat_map(|answer| answer.referenced_report_ids.iter().cloned())
            .collect::<Vec<_>>();
        references.sort();
        references.dedup();
        maps.push(MapResultV1::new(
            batch,
            answer,
            helpfulness,
            "fake-map-v1",
            references,
            counter,
        )?);
    }

    let batch_position = batches
        .iter()
        .enumerate()
        .map(|(position, batch)| (batch.batch_id.as_str(), position))
        .collect::<HashMap<_, _>>();
    let mut positive = maps
        .iter()
        .filter(|map| map.helpfulness > 0)
        .collect::<Vec<_>>();
    positive.sort_by(|left, right| {
        right.helpfulness.cmp(&left.helpfulness).then_with(|| {
            batch_position[&left.batch_id.as_str()].cmp(&batch_position[&right.batch_id.as_str()])
        })
    });
    let retained_batch_ids = positive.iter().map(|map| map.batch_id.clone()).collect();
    let mut selected = Vec::new();
    for map in &positive {
        let mut candidate = selected.clone();
        candidate.push(*map);
        let input = ReduceInputV1::new(
            format!("{}:reduce:0", question.question_id),
            &question.text,
            &candidate,
            profile,
            counter,
        )?;
        if input.input_token_count.tokens > profile.reduce_call_window_tokens {
            break;
        }
        selected = candidate;
    }
    if selected.is_empty() {
        return Err(fixture_error("fake.reduce_window", "no map answer fits"));
    }
    let reduce_input = ReduceInputV1::new(
        format!("{}:reduce:0", question.question_id),
        &question.text,
        &selected,
        profile,
        counter,
    )?;
    let reduce_result = ReduceResultV1::new(
        &reduce_input,
        question.reduce_answer.clone(),
        "fake-reduce-v1",
        question.reduce_references.clone(),
        counter,
    )?;
    let ordered_report_ids = batches
        .iter()
        .flat_map(|batch| batch.report_ids.iter().cloned())
        .collect();
    Ok(QueryTraceV1 {
        schema_version: 1,
        query: question.text.clone(),
        query_fingerprint: fingerprint_text(&question.text),
        hierarchy_run_id: "hierarchy:fixture:v1".into(),
        cut: 0,
        seed: question.seed,
        profile: profile.clone(),
        reports: reports.to_vec(),
        ordered_report_ids,
        batches,
        map_results: maps,
        retained_batch_ids,
        reduce_inputs: vec![reduce_input],
        reduce_results: vec![reduce_result],
    })
}

fn paper_profile() -> PaperBenchmarkProfileV1 {
    PaperBenchmarkProfileV1::paper_v2(TokenizerId::new("test:unicode-whitespace:v1").unwrap())
}

fn attempt_parts<'a>(content: &'a str) -> ExtractionAttemptIdentityParts<'a> {
    ExtractionAttemptIdentityParts {
        store: "fixture",
        chunk_id: 7,
        chunk_content: content,
        model: "executor-model",
        extraction_profile: "paper-elements-v1",
        prompt_version: "prompt-v1",
        response_schema_version: "entity-relationship-v1",
        glean_max_rounds: 2,
    }
}

#[test]
fn fake_executor_traverses_the_green_contract_oracle() {
    let (sources, oracle) = fixture();
    validate_fixture(&sources, &oracle).unwrap();
    let counter = WordCounter::new();
    let reports = build_reports(&sources, &oracle, &counter).unwrap();
    for question in &oracle.questions {
        let trace = execute_fake_query(question, &reports, &oracle.profile, &counter).unwrap();
        trace.validate_against_cut(&reports, &counter).unwrap();
        assert!(trace.batches.len() >= 2);
        assert!(trace.reports.iter().all(|report| {
            report
                .references
                .iter()
                .any(|reference| reference.kind == "source")
        }));
        assert!(
            trace
                .map_results
                .iter()
                .filter(|result| result.helpfulness > 0)
                .all(|result| !result.referenced_report_ids.is_empty())
        );
        assert!(!trace.reduce_results[0].referenced_report_ids.is_empty());
        assert!(
            question
                .required_aspects
                .iter()
                .all(|aspect| trace.reduce_results[0].answer.contains(aspect))
        );
        let json = serde_json::to_string(&trace).unwrap();
        let decoded: QueryTraceV1 = serde_json::from_str(&json).unwrap();
        decoded.validate_against_cut(&reports, &counter).unwrap();
    }
}

#[test]
fn fixture_mutants_have_specific_diagnostics() {
    let (sources, oracle) = fixture();
    validate_fixture(&sources, &oracle).unwrap();

    let mut duplicate_edge_source = oracle.clone();
    duplicate_edge_source.repeated_relationship.source_ids[1] =
        duplicate_edge_source.repeated_relationship.source_ids[0].clone();
    assert_eq!(
        validate_fixture(&sources, &duplicate_edge_source)
            .unwrap_err()
            .code(),
        "fixture.edge_support"
    );
    let mut missing_entity_everywhere = oracle.clone();
    let removed = missing_entity_everywhere.fake_extractions[0]
        .entities
        .remove(0);
    for cut in &mut missing_entity_everywhere.cuts {
        for community in &mut cut.communities {
            community.members.retain(|member| member != &removed);
        }
    }
    assert!(validate_fixture(&sources, &missing_entity_everywhere).is_err());
    let mut bad_projection = oracle.clone();
    bad_projection.cuts[1]
        .communities
        .last_mut()
        .unwrap()
        .projected_from = Some(999);
    assert_eq!(
        validate_fixture(&sources, &bad_projection)
            .unwrap_err()
            .code(),
        "fixture.terminal_projection"
    );
    let mut empty_entity = oracle.clone();
    empty_entity.fake_extractions[0].entities.push(" ".into());
    assert_eq!(
        validate_fixture(&sources, &empty_entity)
            .unwrap_err()
            .code(),
        "fixture.extractions"
    );
    let mut empty_community = oracle.clone();
    empty_community.cuts[0].communities.push(ExpectedCommunity {
        community_id: 999,
        members: Vec::new(),
        parent_id: None,
        projected_from: None,
    });
    assert_eq!(
        validate_fixture(&sources, &empty_community)
            .unwrap_err()
            .code(),
        "fixture.cut_mece"
    );
    let mut empty_aspects = oracle.clone();
    empty_aspects.questions[0].required_aspects = vec![String::new(); 3];
    assert_eq!(
        validate_fixture(&sources, &empty_aspects)
            .unwrap_err()
            .code(),
        "fixture.questions"
    );

    let mut missing_aspect = oracle;
    missing_aspect.questions[0].reduce_answer = "Only one theme.".into();
    assert_eq!(
        validate_fixture(&sources, &missing_aspect)
            .unwrap_err()
            .code(),
        "fixture.answer_aspects"
    );
}

#[test]
fn paper_profile_records_published_values_and_configurable_fixture_values() {
    let profile = paper_profile();
    assert_eq!(profile.profile_id, "paper-v2-600-100-8k");
    assert_eq!(
        (profile.chunk_tokens, profile.chunk_overlap_tokens),
        (600, 100)
    );
    assert_eq!(profile.report_call_window_tokens, 8192);
    assert_eq!(profile.map_call_window_tokens, 8192);
    assert_eq!(profile.reduce_call_window_tokens, 8192);
    profile.validate().unwrap();
    assert_eq!(
        serde_json::from_str::<PaperBenchmarkProfileV1>(&serde_json::to_string(&profile).unwrap())
            .unwrap(),
        profile
    );
    let (_, oracle) = fixture();
    assert_ne!(oracle.profile.profile_id, profile.profile_id);
    oracle.profile.validate().unwrap();
}

#[test]
fn invalid_profiles_have_stable_diagnostic_codes() {
    let mut profile = paper_profile();
    profile.chunk_overlap_tokens = profile.chunk_tokens;
    assert_eq!(profile.validate().unwrap_err().code(), "profile.overlap");
    let mut profile = paper_profile();
    profile.map_call_window_tokens = 0;
    assert_eq!(profile.validate().unwrap_err().code(), "profile.window");
    let mut profile = paper_profile();
    profile.profile_id.clear();
    assert_eq!(profile.validate().unwrap_err().code(), "profile.id");
    let mut profile = paper_profile();
    profile.map_response_schema_version = "bogus-v99".into();
    assert_eq!(
        profile.validate().unwrap_err().code(),
        "profile.response_schema"
    );
}

#[test]
fn token_boundary_recomputes_counts_instead_of_trusting_json() {
    let counter = WordCounter::new();
    let (sources, oracle) = fixture();
    let mut reports = build_reports(&sources, &oracle, &counter).unwrap();
    let mut missing_source = reports[0].clone();
    missing_source
        .references
        .retain(|reference| reference.kind != "source");
    assert_eq!(
        missing_source.validate().unwrap_err().code(),
        "report.source_references"
    );
    reports[0].token_count.tokens = 0;
    assert_eq!(
        reports[0]
            .validate_with_counter(&counter)
            .unwrap_err()
            .code(),
        "report.token_count"
    );
    assert!(TokenizerId::new(" ").is_err());
}

#[test]
fn extraction_attempt_identity_binds_every_planned_dimension() {
    let base_parts = attempt_parts("Northstar acquired Ember.");
    let base = ExtractionAttemptIdentityV1::derive(&base_parts).unwrap();
    assert_eq!(
        base,
        ExtractionAttemptIdentityV1::derive(&base_parts).unwrap()
    );
    macro_rules! changed {
        ($parts:expr, $name:literal) => {
            assert_ne!(
                base,
                ExtractionAttemptIdentityV1::derive(&$parts).unwrap(),
                $name
            )
        };
    }
    let mut p = base_parts.clone();
    p.store = "other";
    changed!(p, "store");
    let mut p = base_parts.clone();
    p.chunk_id = 8;
    changed!(p, "chunk_id");
    let mut p = base_parts.clone();
    p.chunk_content = "changed";
    changed!(p, "content");
    let mut p = base_parts.clone();
    p.model = "other";
    changed!(p, "model");
    let mut p = base_parts.clone();
    p.extraction_profile = "other";
    changed!(p, "profile");
    let mut p = base_parts.clone();
    p.prompt_version = "v2";
    changed!(p, "prompt");
    let mut p = base_parts.clone();
    p.response_schema_version = "v2";
    changed!(p, "schema");
    let mut p = base_parts;
    p.glean_max_rounds = 3;
    changed!(p, "glean");
}

#[test]
fn trace_mutants_reject_dropped_or_stale_expensive_inputs() {
    let (sources, oracle) = fixture();
    let counter = WordCounter::new();
    let reports = build_reports(&sources, &oracle, &counter).unwrap();
    execute_fake_query(&oracle.questions[0], &reports, &oracle.profile, &counter)
        .unwrap()
        .validate_against_cut(&reports, &counter)
        .unwrap();

    // Delete one report everywhere inside a now self-consistent replanned trace:
    // only the separately supplied authoritative cut can catch the omission.
    let removed = reports.last().unwrap().report_id.clone();
    let reduced_reports = reports
        .iter()
        .filter(|report| report.report_id != removed)
        .cloned()
        .collect::<Vec<_>>();
    let mut reduced_question = oracle.questions[0].clone();
    reduced_question
        .map_answers
        .retain(|answer| answer.report_id != removed);
    reduced_question
        .reduce_references
        .retain(|report_id| report_id != &removed);
    reduced_question.reduce_answer = "The remaining corpus still has cited themes.".into();
    let reduced_trace = execute_fake_query(
        &reduced_question,
        &reduced_reports,
        &oracle.profile,
        &counter,
    )
    .unwrap();
    reduced_trace
        .validate_against_cut(&reduced_reports, &counter)
        .unwrap();
    assert_eq!(
        reduced_trace
            .validate_against_cut(&reports, &counter)
            .unwrap_err()
            .code(),
        "trace.report_coverage"
    );

    let mut stale =
        execute_fake_query(&oracle.questions[0], &reports, &oracle.profile, &counter).unwrap();
    stale.reports[0].text.push_str(" changed");
    stale.reports[0].text_fingerprint = fingerprint_text(&stale.reports[0].text);
    stale.reports[0].token_count.tokens = counter.count(&stale.reports[0].text).unwrap();
    let mut changed_cut = reports.clone();
    changed_cut[0] = stale.reports[0].clone();
    assert_eq!(
        stale
            .validate_against_cut(&changed_cut, &counter)
            .unwrap_err()
            .code(),
        "trace.batch_plan"
    );

    let mut parsed_only =
        execute_fake_query(&oracle.questions[0], &reports, &oracle.profile, &counter).unwrap();
    parsed_only.map_results[0].answer.push_str(" changed");
    parsed_only.map_results[0].answer_token_count.tokens =
        counter.count(&parsed_only.map_results[0].answer).unwrap();
    assert_eq!(
        parsed_only
            .validate_against_cut(&reports, &counter)
            .unwrap_err()
            .code(),
        "map.response_binding"
    );

    let mut changed_map =
        execute_fake_query(&oracle.questions[0], &reports, &oracle.profile, &counter).unwrap();
    let original = changed_map.map_results[0].clone();
    let batch = changed_map
        .batches
        .iter()
        .find(|batch| batch.batch_id == original.batch_id)
        .unwrap();
    changed_map.map_results[0] = MapResultV1::new(
        batch,
        format!("{} changed", original.answer),
        original.helpfulness,
        original.model,
        original.referenced_report_ids,
        &counter,
    )
    .unwrap();
    assert_eq!(
        changed_map
            .validate_against_cut(&reports, &counter)
            .unwrap_err()
            .code(),
        "reduce.context"
    );
}

#[test]
fn execution_versions_parsed_payloads_and_citations_are_binding() {
    let (sources, oracle) = fixture();
    let counter = WordCounter::new();
    let reports = build_reports(&sources, &oracle, &counter).unwrap();
    let base =
        execute_fake_query(&oracle.questions[0], &reports, &oracle.profile, &counter).unwrap();

    let mut changed_prompt = base.clone();
    changed_prompt
        .profile
        .map_prompt_version
        .push_str("-changed");
    assert_eq!(
        changed_prompt
            .validate_against_cut(&reports, &counter)
            .unwrap_err()
            .code(),
        "trace.batch_plan"
    );

    let mut no_map_citation = base.clone();
    let original = no_map_citation.map_results[0].clone();
    let batch = no_map_citation
        .batches
        .iter()
        .find(|batch| batch.batch_id == original.batch_id)
        .unwrap();
    no_map_citation.map_results[0] = MapResultV1::new(
        batch,
        original.answer,
        original.helpfulness,
        original.model,
        vec![],
        &counter,
    )
    .unwrap();
    assert_eq!(
        no_map_citation
            .validate_against_cut(&reports, &counter)
            .unwrap_err()
            .code(),
        "map.references"
    );

    let mut no_reduce_citation = base.clone();
    let input = &no_reduce_citation.reduce_inputs[0];
    no_reduce_citation.reduce_results[0] = ReduceResultV1::new(
        input,
        no_reduce_citation.reduce_results[0].answer.clone(),
        "fake-reduce-v1",
        vec![],
        &counter,
    )
    .unwrap();
    assert_eq!(
        no_reduce_citation
            .validate_against_cut(&reports, &counter)
            .unwrap_err()
            .code(),
        "reduce.references"
    );

    let mut parsed_only = base;
    parsed_only.reduce_results[0].answer.push_str(" changed");
    parsed_only.reduce_results[0].answer_token_count.tokens = counter
        .count(&parsed_only.reduce_results[0].answer)
        .unwrap();
    assert_eq!(
        parsed_only
            .validate_against_cut(&reports, &counter)
            .unwrap_err()
            .code(),
        "reduce.response_binding"
    );
}

#[test]
fn canonical_seeded_shuffle_and_maximal_map_packing_are_enforced() {
    let (sources, oracle) = fixture();
    let counter = WordCounter::new();
    let reports = build_reports(&sources, &oracle, &counter).unwrap();
    let base =
        execute_fake_query(&oracle.questions[0], &reports, &oracle.profile, &counter).unwrap();

    let mut changed_seed = base.clone();
    let original_order = changed_seed.ordered_report_ids.clone();
    changed_seed.seed = (changed_seed.seed + 1..)
        .find(|seed| {
            graphrag_core::paper::canonical_report_order(*seed, &reports) != original_order
        })
        .unwrap();
    assert_eq!(
        changed_seed
            .validate_against_cut(&reports, &counter)
            .unwrap_err()
            .code(),
        "trace.seeded_order"
    );

    let mut underpacked = base;
    underpacked.profile.map_call_window_tokens = 100;
    assert_eq!(
        underpacked
            .validate_against_cut(&reports, &counter)
            .unwrap_err()
            .code(),
        "trace.batch_plan"
    );
}

#[test]
fn trace_validator_is_total_and_enforces_zero_filter_maximal_reduce_and_one_reduce() {
    let (sources, oracle) = fixture();
    let counter = WordCounter::new();
    let reports = build_reports(&sources, &oracle, &counter).unwrap();
    let base =
        execute_fake_query(&oracle.questions[1], &reports, &oracle.profile, &counter).unwrap();

    let mut unknown = base.clone();
    unknown.reduce_inputs[0].input_batch_ids = vec!["unknown".into()];
    let result = std::panic::catch_unwind(|| unknown.validate_against_cut(&reports, &counter));
    assert!(result.is_ok(), "untrusted JSON must never panic validation");
    assert_eq!(result.unwrap().unwrap_err().code(), "reduce.inputs");

    let mut retained_zero = base.clone();
    let zero = retained_zero
        .map_results
        .iter()
        .find(|map| map.helpfulness == 0)
        .unwrap()
        .batch_id
        .clone();
    retained_zero.retained_batch_ids.push(zero);
    assert_eq!(
        retained_zero
            .validate_against_cut(&reports, &counter)
            .unwrap_err()
            .code(),
        "trace.retained_batches"
    );

    let mut all_zero = base.clone();
    all_zero.map_results = all_zero
        .batches
        .iter()
        .map(|batch| MapResultV1::new(batch, "", 0, "fake-map-v1", vec![], &counter).unwrap())
        .collect();
    all_zero.retained_batch_ids.clear();
    assert_eq!(
        all_zero
            .validate_against_cut(&reports, &counter)
            .unwrap_err()
            .code(),
        "trace.no_evidence"
    );

    let mut short_reduce = base.clone();
    short_reduce.reduce_inputs[0].input_batch_ids.pop();
    assert_eq!(
        short_reduce
            .validate_against_cut(&reports, &counter)
            .unwrap_err()
            .code(),
        "reduce.inputs"
    );

    let mut extra_reduce = base;
    extra_reduce
        .reduce_results
        .push(extra_reduce.reduce_results[0].clone());
    assert_eq!(
        extra_reduce
            .validate_against_cut(&reports, &counter)
            .unwrap_err()
            .code(),
        "trace.reduce_count"
    );
}

#[test]
fn capability_registry_names_every_binding_and_milestone() {
    let capabilities = paper_capabilities();
    assert_eq!(capabilities.len(), 7);
    assert_eq!(capabilities[0].capability, PaperCapability::M0Contract);
    assert_eq!(
        capabilities[0].state.reason(),
        "paper_contract::fake_executor_traverses_the_green_contract_oracle"
    );
    assert!(matches!(
        capabilities[0].state,
        PaperCapabilityState::Implemented { .. }
    ));
    for (index, capability) in capabilities[1..].iter().enumerate() {
        let PaperCapabilityState::Gated { milestone, reason } = capability.state else {
            panic!("future capability reported implemented without binding test")
        };
        assert_eq!(milestone, format!("M{}", index + 1));
        assert!(!reason.trim().is_empty());
    }
}

#[hegel::test]
fn text_fingerprint_is_deterministic_and_byte_sensitive(tc: TestCase) {
    let text = tc.draw(gs::text().max_size(1024));
    assert_eq!(fingerprint_text(&text), fingerprint_text(&text));
    assert_ne!(fingerprint_text(&text), fingerprint_text(&(text + "\0")));
}

#[test]
#[ignore = "M1: remove only with production exhaustive-query binding test"]
fn m1_production_global_query_binding() {
    assert!(paper_capabilities()[1].state.is_implemented());
}
#[test]
#[ignore = "M2: remove only with accepted-instance binding tests"]
fn m2_production_element_instance_binding() {
    assert!(paper_capabilities()[2].state.is_implemented());
}
#[test]
#[ignore = "M3: remove only with recursive-Leiden binding tests"]
fn m3_production_hierarchy_binding() {
    assert!(paper_capabilities()[3].state.is_implemented());
}
#[test]
#[ignore = "M4: remove only with current-report coverage binding tests"]
fn m4_production_report_binding() {
    assert!(paper_capabilities()[4].state.is_implemented());
}
#[test]
#[ignore = "M5: remove only with report-backed global-answer binding tests"]
fn m5_production_global_answer_binding() {
    assert!(paper_capabilities()[5].state.is_implemented());
}
#[test]
#[ignore = "M6: remove only with SS/TS/C-cut evaluation binding tests"]
fn m6_production_evaluation_binding() {
    assert!(paper_capabilities()[6].state.is_implemented());
}
