//! Standard entity types based on Schema.org and Dublin Core vocabularies
//!
//! This module provides canonical entity types and their synonyms
//! for knowledge graph regularization.

/// Standard canonical entity types and their synonyms
/// Based on Schema.org (https://schema.org) and Dublin Core DCMI Types
/// (https://www.dublincore.org/specifications/dublin-core/dcmi-type-vocabulary/)
pub const STANDARD_TYPE_SYNONYMS: &[(&str, &str)] = &[
    // Person (Schema.org)
    ("human", "person"),
    ("individual", "person"),
    ("author", "person"),
    ("researcher", "person"),
    ("developer", "person"),
    ("user", "person"),
    ("contributor", "person"),
    ("maintainer", "person"),
    // Organization (Schema.org)
    ("company", "organization"),
    ("corporation", "organization"),
    ("institution", "organization"),
    ("university", "organization"),
    ("team", "organization"),
    ("group", "organization"),
    ("consortium", "organization"),
    // Place (Schema.org)
    ("location", "place"),
    ("city", "place"),
    ("country", "place"),
    ("region", "place"),
    ("address", "place"),
    ("venue", "place"),
    // Event (Schema.org / Dublin Core)
    ("conference", "event"),
    ("meeting", "event"),
    ("workshop", "event"),
    ("release", "event"),
    ("publication", "event"),
    // CreativeWork (Schema.org) -> "document"
    ("article", "document"),
    ("paper", "document"),
    ("report", "document"),
    ("publication", "document"),
    ("book", "document"),
    ("chapter", "document"),
    ("specification", "document"),
    // Software (Dublin Core)
    ("program", "software"),
    ("application", "software"),
    ("tool", "software"),
    ("library", "software"),
    ("framework", "software"),
    ("package", "software"),
    ("module", "software"),
    ("plugin", "software"),
    ("crate", "software"), // Rust-specific
    // Dataset (Dublin Core)
    ("data", "dataset"),
    ("corpus", "dataset"),
    ("benchmark", "dataset"),
    ("collection", "dataset"),
    // Code entities
    ("class", "code_entity"),
    ("function", "code_entity"),
    ("method", "code_entity"),
    ("struct", "code_entity"),
    ("enum", "code_entity"),
    ("interface", "code_entity"),
    ("trait", "code_entity"),
    ("type", "code_entity"),
    ("variable", "code_entity"),
    ("constant", "code_entity"),
    // Concept/Topic (SKOS-like)
    ("topic", "concept"),
    ("idea", "concept"),
    ("theme", "concept"),
    ("subject", "concept"),
    ("category", "concept"),
    // Technology/Method
    ("algorithm", "method"),
    ("technique", "method"),
    ("approach", "method"),
    ("process", "method"),
    ("procedure", "method"),
    ("protocol", "method"),
    ("pattern", "method"),
    // Technology stack
    ("technology", "technology"),
    ("language", "technology"),
    ("platform", "technology"),
    ("runtime", "technology"),
    ("database", "technology"),
    ("api", "technology"),
    ("service", "technology"),
    // Project/Repository
    ("repository", "project"),
    ("repo", "project"),
    ("codebase", "project"),
    // Version/Release
    ("version", "version"),
    ("release", "version"),
    ("edition", "version"),
    ("revision", "version"),
    // File/Resource
    ("file", "resource"),
    ("image", "resource"),
    ("video", "resource"),
    ("audio", "resource"),
    ("url", "resource"),
    ("link", "resource"),
];

/// Canonical entity types (the target types in STANDARD_TYPE_SYNONYMS)
pub const CANONICAL_ENTITY_TYPES: &[&str] = &[
    "person",
    "organization",
    "place",
    "event",
    "document",
    "software",
    "dataset",
    "code_entity",
    "concept",
    "method",
    "technology",
    "project",
    "version",
    "resource",
];

/// Get all canonical entity types
pub fn canonical_entity_types() -> &'static [&'static str] {
    CANONICAL_ENTITY_TYPES
}

/// Load standard entity type synonyms into a database
/// Note: This uses the same synonym table mechanism as relations
pub fn load_standard_type_synonyms(db: &crate::Database) -> Result<usize, crate::GraphRagError> {
    db.add_entity_type_synonyms(STANDARD_TYPE_SYNONYMS)?;
    Ok(STANDARD_TYPE_SYNONYMS.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_types() {
        let types = canonical_entity_types();
        assert!(types.contains(&"person"));
        assert!(types.contains(&"software"));
        assert!(types.contains(&"document"));
    }

    #[test]
    fn test_all_synonyms_map_to_canonical() {
        let canonical_set: std::collections::HashSet<_> =
            CANONICAL_ENTITY_TYPES.iter().copied().collect();

        for (_, canonical) in STANDARD_TYPE_SYNONYMS {
            assert!(
                canonical_set.contains(canonical),
                "Synonym maps to non-canonical type: {}",
                canonical
            );
        }
    }
}
