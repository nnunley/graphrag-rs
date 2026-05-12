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
    // Type/classification -> "type_of" / "has_type"
    ("is_a", "type_of"),
    ("instance_of", "type_of"),
    ("kind_of", "type_of"),
    // Result/causation -> "produces" / "produced_by"
    ("results_in", "produces"),
    ("yields", "produces"),
    ("generates", "produces"),
    ("result_of", "produced_by"),
    ("caused_by", "produced_by"),
    ("generated_by", "produced_by"),
];

/// Get all canonical relation types (unique values from STANDARD_SYNONYMS)
pub fn canonical_relations() -> Vec<&'static str> {
    let mut canonicals: Vec<&str> = STANDARD_SYNONYMS.iter().map(|(_, c)| *c).collect();
    canonicals.sort();
    canonicals.dedup();
    canonicals
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
