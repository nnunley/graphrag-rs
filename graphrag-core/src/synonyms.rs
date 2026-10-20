//! Standard relation synonyms based on Dublin Core and Schema.org vocabularies
//!
//! This module provides canonical relation types and their common synonyms
//! for knowledge graph regularization.

/// Standard canonical relations and their synonyms
/// Based on Dublin Core (https://www.dublincore.org/specifications/dublin-core/dcmi-terms/)
/// and Schema.org (https://schema.org) vocabularies
pub const STANDARD_SYNONYMS: &[(&str, &str)] = &[
    // Usage relations -> "uses"
    ("utilizes", "uses"),
    ("employs", "uses"),
    ("leverages", "uses"),
    ("makes_use_of", "uses"),
    ("applies", "uses"),
    // Usage passive -> "used_by"
    ("utilized_by", "used_by"),
    ("employed_by", "used_by"),
    ("leveraged_by", "used_by"),
    // Usage purpose -> "used_for"
    ("utilized_for", "used_for"),
    ("employed_for", "used_for"),
    ("applied_to", "used_for"),
    // Implementation relations -> "implements"
    ("implemented_by", "implements"), // Note: direction flip - may want separate canonical
    ("realizes", "implements"),
    ("instantiates", "implements"),
    // Part-whole (Dublin Core) -> "has_part" / "part_of"
    ("contains", "has_part"),
    ("includes", "has_part"),
    ("comprises", "has_part"),
    ("is_part_of", "part_of"),
    ("belongs_to", "part_of"),
    ("contained_in", "part_of"),
    ("included_in", "part_of"),
    ("member_of", "part_of"),
    // Dependency (Dublin Core) -> "requires" / "required_by"
    ("depends_on", "requires"),
    ("needs", "requires"),
    ("is_required_by", "required_by"),
    ("needed_by", "required_by"),
    // References (Dublin Core) -> "references" / "referenced_by"
    ("cites", "references"),
    ("mentions", "references"),
    ("refers_to", "references"),
    ("cited_by", "referenced_by"),
    ("mentioned_in", "referenced_by"),
    ("referred_to_by", "referenced_by"),
    // Authorship (Schema.org) -> "author_of" / "authored_by"
    ("wrote", "author_of"),
    ("created", "author_of"),
    ("authored", "author_of"),
    ("written_by", "authored_by"),
    ("created_by", "authored_by"),
    // Description -> "describes" / "described_by"
    ("explains", "describes"),
    ("documents", "describes"),
    ("explained_by", "described_by"),
    ("documented_by", "described_by"),
    ("described_in", "described_by"),
    // Location -> "located_in" / "location_of"
    ("based_in", "located_in"),
    ("situated_in", "located_in"),
    ("found_in", "located_in"),
    // Versioning (Dublin Core) -> "version_of" / "has_version"
    ("is_version_of", "version_of"),
    ("variant_of", "version_of"),
    ("edition_of", "version_of"),
    // Replacement (Dublin Core) -> "replaces" / "replaced_by"
    ("supersedes", "replaces"),
    ("obsoletes", "replaces"),
    ("is_replaced_by", "replaced_by"),
    ("superseded_by", "replaced_by"),
    ("obsoleted_by", "replaced_by"),
    // Employment (Schema.org) -> "works_at" / "employs"
    ("works_for", "works_at"),
    ("employed_at", "works_at"),
    ("hires", "employs"),
    // Association -> "associated_with"
    ("related_to", "associated_with"),
    ("connected_to", "associated_with"),
    ("linked_to", "associated_with"),
    // Classification (Cyc): an individual is an instance of a collection
    // ("isa"); one collection specializes another ("genls").
    ("is_a", "isa"),
    ("instance_of", "isa"),
    ("type_of", "isa"),
    ("kind_of", "genls"),
    ("subclass_of", "genls"),
    ("subtype_of", "genls"),
    // Aboutness (Dublin Core subject)
    ("about", "subject"),
    ("topic_of", "subject"),
    // Result/causation -> "produces" / "produced_by"
    ("results_in", "produces"),
    ("yields", "produces"),
    ("generates", "produces"),
    ("result_of", "produced_by"),
    ("caused_by", "produced_by"),
    ("generated_by", "produced_by"),
];

/// Canonical relations with no synonyms. `disjoint_with`, `links_to` and
/// `same_as` are reserved: recognized now, enforced later.
pub const EXTRA_CANONICAL_RELATIONS: &[&str] = &[
    "subject",
    "isa",
    "genls",
    "disjoint_with",
    "links_to",
    "same_as",
    "mentions",
];

/// Get all canonical relation types (synonym targets plus the extras)
pub fn canonical_relations() -> Vec<&'static str> {
    let mut canonicals: Vec<&str> = STANDARD_SYNONYMS
        .iter()
        .map(|(_, c)| *c)
        .chain(EXTRA_CANONICAL_RELATIONS.iter().copied())
        .collect();
    canonicals.sort();
    canonicals.dedup();
    canonicals
}

/// How a raw relation name from an extractor maps to the vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelationMapping {
    Canonical(&'static str),
    /// Kept under its normalized raw name and counted, so the vocabulary
    /// can grow from evidence.
    Unmapped(String),
}

fn snake(raw: &str) -> String {
    raw.trim()
        .to_lowercase()
        .split(|c: char| c.is_whitespace() || c == '-')
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("_")
}

/// Lowercase, spaces and hyphens to `_`, then synonym lookup. Canonical
/// names map to themselves.
pub fn normalize_relation(raw: &str) -> RelationMapping {
    let key = snake(raw);
    if let Some(c) = canonical_relations().into_iter().find(|c| *c == key) {
        return RelationMapping::Canonical(c);
    }
    match STANDARD_SYNONYMS.iter().find(|(s, _)| *s == key) {
        Some((_, c)) => RelationMapping::Canonical(c),
        None => RelationMapping::Unmapped(key),
    }
}

/// Canonical entity type for a raw type; `"other"` when nothing matches.
pub fn normalize_entity_type(raw: &str) -> &'static str {
    let key = snake(raw);
    let types = crate::entity_types::CANONICAL_ENTITY_TYPES;
    if let Some(t) = types.iter().find(|t| **t == key) {
        return t;
    }
    crate::entity_types::STANDARD_TYPE_SYNONYMS
        .iter()
        .find(|(s, _)| *s == key)
        .map_or("other", |(_, c)| c)
}

/// The match key for an entity name: trimmed, internal whitespace
/// collapsed, case-folded. The display form is kept separately.
pub fn normalize_entity_name(raw: &str) -> String {
    display_name(raw).to_lowercase()
}

/// Trimmed with internal whitespace collapsed; case kept.
pub fn display_name(raw: &str) -> String {
    raw.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Load standard synonyms into a database
pub fn load_standard_synonyms(db: &crate::Database) -> Result<usize, crate::GraphRagError> {
    db.add_relation_synonyms(STANDARD_SYNONYMS)?;
    Ok(STANDARD_SYNONYMS.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_relations() {
        let canonicals = canonical_relations();
        assert!(canonicals.contains(&"uses"));
        assert!(canonicals.contains(&"part_of"));
        assert!(canonicals.contains(&"references"));
        // Should not contain raw synonyms
        assert!(!canonicals.contains(&"utilizes"));
        assert!(!canonicals.contains(&"leverages"));
    }
}
