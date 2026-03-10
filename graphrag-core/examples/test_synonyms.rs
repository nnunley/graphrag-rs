use graphrag_core::{Database, load_standard_synonyms, canonical_relations};
use std::path::PathBuf;

fn main() {
    // Hardcode path for testing
    let db_path = PathBuf::from(std::env::var("HOME").unwrap())
        .join("Library/Application Support/graphrag/graphrag.db");

    println!("Opening database at {:?}", db_path);
    let db = Database::open(&db_path).expect("Failed to open db");

    // Check stores
    let stores = db.list_stores().expect("Failed to list stores");
    println!("Stores: {:?}", stores.iter().map(|s| &s.name).collect::<Vec<_>>());

    // Check relation types before
    if let Some(store) = stores.first() {
        match db.list_relation_types(&store.name) {
            Ok(types) => {
                println!("\nRelation types BEFORE loading synonyms in '{}':", store.name);
                for (rel, canonical, count) in types.iter().take(10) {
                    println!("  {:20} -> {:?} ({} instances)", rel, canonical, count);
                }
            }
            Err(e) => println!("Error listing relation types: {}", e),
        }
    }

    // Load standard synonyms (Dublin Core + Schema.org based)
    println!("\nLoading standard synonyms...");
    match load_standard_synonyms(&db) {
        Ok(count) => println!("Loaded {} standard synonym mappings", count),
        Err(e) => println!("Error loading synonyms: {}", e),
    }

    // Show canonical relations available
    let canonicals = canonical_relations();
    println!("\nCanonical relation types ({}):", canonicals.len());
    for chunk in canonicals.chunks(5) {
        println!("  {}", chunk.join(", "));
    }

    // Refresh existing relations with new synonyms
    println!("\nRefreshing existing relations...");
    match db.refresh_canonical_relations() {
        Ok(count) => println!("Updated {} existing relations with canonical forms", count),
        Err(e) => println!("Error refreshing relations: {}", e),
    }

    // Show updated relation types
    if let Some(store) = stores.first() {
        match db.list_relation_types(&store.name) {
            Ok(types) => {
                println!("\nRelation types AFTER loading synonyms:");
                for (rel, canonical, count) in types.iter().take(15) {
                    println!("  {:20} -> {:20} ({} instances)",
                        rel,
                        canonical.as_deref().unwrap_or("-"),
                        count
                    );
                }
            }
            Err(e) => println!("Error listing relation types: {}", e),
        }
    }

    // Test querying by canonical type
    if let Some(store) = stores.first() {
        println!("\nQuerying relations with canonical type 'references':");
        match db.get_relations_by_canonical(&store.name, "referenced_by") {
            Ok(rels) => {
                println!("Found {} relations:", rels.len());
                for rel in rels.iter().take(5) {
                    println!("  {} -[{}]-> {} (canonical: {:?})",
                        rel.head_id, rel.relation, rel.tail_id, rel.canonical_relation);
                }
            }
            Err(e) => println!("Error: {}", e),
        }
    }
}
