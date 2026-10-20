//! Task 11: the paper-json-v1.1 extraction profile.

use graphrag_core::attempts::AttemptOutcome;
use graphrag_llm::paper_json::{
    PROFILE, PROMPT_VERSION, SCHEMA_VERSION, build_prompt, is_fragment, parse, response_schema,
};

fn prompt_text(classes: &[String]) -> String {
    let p = build_prompt(
        "Some chunk text.",
        &["Doc".into(), "Section".into()],
        classes,
    );
    format!("{}\n{}", p.system, p.user)
}

#[test]
fn constants_are_pinned() {
    assert_eq!(PROFILE, "paper-json");
    assert_eq!(PROMPT_VERSION, "paper-json-v1.2");
    assert_eq!(SCHEMA_VERSION, "graph-v1");
    assert_eq!(build_prompt("x", &[], &[]).max_tokens, 12000);
}

#[test]
fn prompt_lists_canonical_vocabulary() {
    let t = prompt_text(&[]);
    for word in [
        "software", "person", "concept", "uses", "part_of", "subject", "isa", "genls",
    ] {
        assert!(t.contains(word), "prompt lacks {word}");
    }
    assert!(t.contains("Some chunk text."));
    assert!(t.contains("Doc › Section"), "heading path shown");
}

#[test]
fn prompt_contains_software_and_example_rules() {
    let t = prompt_text(&[]);
    assert!(t.contains(
        "tools, version-control systems, libraries and AI models are Software, never Person"
    ));
    assert!(t.contains(
        "ignore illustrative example names (e.g. alice, bob, foo) and bibliographic citations"
    ));
    assert!(
        t.contains("prefer a specific relation; use associated_with only when nothing else fits")
    );
    // v1.2: the re-spike still typed Opus as Person and took #alice from a code sample.
    assert!(t.contains(
        "AI models (Claude, Opus, Sonnet, GPT, Codex, qwen) are Software even when they act"
    ));
    assert!(t.contains(
        "names that appear only inside code samples or examples (such as #alice) are not entities"
    ));
}

#[test]
fn prompt_distinguishes_isa_from_genls() {
    let t = prompt_text(&[]);
    assert!(t.contains(
        "use isa when something is an instance of a kind, genls when one kind is a more specific kind of another"
    ));
}

#[test]
fn prompt_lists_known_classes() {
    let t = prompt_text(&["version control system".into(), "proxy service".into()]);
    assert!(t.contains("version control system"));
    assert!(t.contains("proxy service"));
    assert!(!prompt_text(&[]).contains("Known classes"));
}

#[test]
fn schema_requires_kind_and_is_class() {
    let s = response_schema();
    let req = s["properties"]["entities"]["items"]["required"]
        .as_array()
        .unwrap();
    let req: Vec<&str> = req.iter().map(|v| v.as_str().unwrap()).collect();
    assert!(req.contains(&"kind") && req.contains(&"is_class"));
    assert!(!req.contains(&"broader"));
    assert!(s["properties"]["entities"]["items"]["properties"]["broader"].is_object());
}

#[test]
fn parses_kind_and_broader() {
    let raw = r#"{"entities":[{"name":"jj","type":"Software","description":"a VCS","kind":"version control system","is_class":false,"broader":"software tool"}],
                  "relationships":[]}"#;
    let (out, rejected) = parse(raw).unwrap();
    assert_eq!(rejected, 0);
    let AttemptOutcome::Ok { entities, .. } = out else {
        panic!()
    };
    assert_eq!(entities[0].kind.as_deref(), Some("version control system"));
    assert_eq!(entities[0].broader.as_deref(), Some("software tool"));
    assert!(!entities[0].is_class);
}

#[test]
fn parses_spike_arm_c_output() {
    let raw = include_str!("fixtures/arm_c_1212.json");
    let (out, _) = parse(raw).unwrap();
    let AttemptOutcome::Ok {
        entities,
        relations,
    } = out
    else {
        panic!()
    };
    assert!(entities.len() >= 10, "{}", entities.len());
    assert!(relations.len() >= 5);
    assert!(entities.iter().any(|e| e.name == "GraphRAG"));
    assert!(
        entities.iter().all(|e| e.kind.is_none()),
        "arm C predates kind"
    );
}

#[test]
fn empty_lists_are_empty_outcome() {
    let (out, _) = parse(r#"{"entities":[],"relationships":[]}"#).unwrap();
    assert_eq!(out, AttemptOutcome::Empty);
}

#[test]
fn broken_json_is_err() {
    assert!(parse("{\"entities\": [").is_err());
    assert!(
        parse("{\"entities\": []}").is_err(),
        "relationships is required"
    );
    assert!(parse("not json").is_err());
}

#[test]
fn fragments_rejected_and_counted() {
    assert!(is_fragment("use JWT tokens for session management"));
    assert!(is_fragment("It works."));
    assert!(is_fragment(""));
    assert!(is_fragment("12345"));
    assert!(is_fragment(&"x".repeat(61)));
    assert!(!is_fragment("Leiden algorithm"));
    assert!(!is_fragment("graphrag-rs"));
    let raw = r#"{"entities":[
        {"name":"use JWT tokens for session management","type":"Concept","description":"d","kind":"x","is_class":false},
        {"name":"JWT","type":"Concept","description":"d","kind":"token format","is_class":false}],
      "relationships":[{"source":"use JWT tokens for session management","target":"JWT","relation":"uses","description":"d","strength":3}]}"#;
    let (out, rejected) = parse(raw).unwrap();
    assert_eq!(rejected, 1);
    let AttemptOutcome::Ok {
        entities,
        relations,
    } = out
    else {
        panic!()
    };
    assert_eq!(entities.len(), 1);
    assert!(
        relations.is_empty(),
        "a relation to a rejected fragment is dropped"
    );
}

#[test]
fn relation_to_undeclared_entity_is_kept() {
    let raw = r#"{"entities":[{"name":"Alice","type":"Person","description":"d","kind":"person","is_class":false}],
      "relationships":[{"source":"Alice","target":"Ghost","relation":"uses","description":"d","strength":12}]}"#;
    let (out, _) = parse(raw).unwrap();
    let AttemptOutcome::Ok { relations, .. } = out else {
        panic!()
    };
    assert_eq!(relations.len(), 1);
    assert_eq!(relations[0].target, "Ghost");
    assert_eq!(relations[0].strength, Some(10), "strength clamped to 1-10");
}
