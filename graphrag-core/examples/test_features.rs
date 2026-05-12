//! Test the new GraphRAG features: entity merging, global query support
//!
//! Run with: cargo run --example test_features

use graphrag_core::{Database, Embedder, EmbedderConfig, EmbedderModel};
use std::path::PathBuf;

fn main() {
    println!("=== GraphRAG Feature Test ===\n");

    // Use a test database
    let db_path = PathBuf::from("/tmp/graphrag_test.db");
    if db_path.exists() {
        std::fs::remove_file(&db_path).ok();
    }

    let db = Database::open(&db_path).expect("Failed to open database");
    println!("✓ Database opened at {:?}", db_path);

    // Create a test store
    let store_name = "test_store";
    let dim = 768;
    db.create_store(store_name, dim)
        .expect("Failed to create store");
    println!("✓ Created store '{}' with dim={}", store_name, dim);

    // Add some test entities that should be merge candidates
    let entities = [
        ("Microsoft", "organization"),
        ("Microsoft Corporation", "organization"),
        ("MSFT", "organization"),
        ("GraphRAG", "software"),
        ("Graph RAG", "software"),
        ("Rust", "language"),
        ("Rust programming language", "language"),
        ("Python", "language"),
        ("SQLite", "software"),
    ];

    for (name, etype) in &entities {
        db.get_or_create_entity(store_name, name, Some(etype), None)
            .expect("Failed to create entity");
    }
    println!("✓ Created {} test entities", entities.len());

    // Initialize embedder
    println!("\nInitializing embedder (may download model on first use)...");
    let config = EmbedderConfig {
        model: EmbedderModel::NomicEmbedText,
        show_download_progress: true,
        cache_dir: None,
    };
    let embedder = match Embedder::new(config) {
        Ok(e) => e,
        Err(e) => {
            println!("⚠ Embedder initialization failed: {}", e);
            println!("  Skipping embedding tests");
            return;
        }
    };
    println!("✓ Embedder initialized (dim={})", embedder.dimension());

    // Embed all entities
    println!("\nEmbedding entities...");
    let db_entities = db
        .list_entities(store_name)
        .expect("Failed to list entities");
    for entity in &db_entities {
        let embedding = embedder.embed(&entity.name).expect("Failed to embed");
        db.set_entity_embedding(entity.id, &embedding)
            .expect("Failed to store embedding");
    }
    println!("✓ Embedded {} entities", db_entities.len());

    // Find merge candidates
    println!("\n=== Finding Merge Candidates ===");
    let thresholds = [0.95, 0.90, 0.85, 0.80];

    for threshold in thresholds {
        let candidates = db
            .find_merge_candidates(store_name, threshold)
            .expect("Failed to find candidates");

        println!("\nThreshold >= {:.2}:", threshold);
        if candidates.is_empty() {
            println!("  No candidates found");
        } else {
            for (id1, name1, id2, name2, sim) in &candidates {
                println!(
                    "  [{:.3}] '{}' (id:{}) <-> '{}' (id:{})",
                    sim, name1, id1, name2, id2
                );
            }
        }
    }

    // Test merging
    println!("\n=== Testing Entity Merge ===");

    // Find "Microsoft Corporation" and "MSFT" to merge into "Microsoft"
    let ms = db
        .get_entity_by_name(store_name, "Microsoft")
        .expect("Microsoft not found");
    let ms_corp = db
        .get_entity_by_name(store_name, "Microsoft Corporation")
        .ok();
    let msft = db.get_entity_by_name(store_name, "MSFT").ok();

    if let Some(corp) = ms_corp {
        println!(
            "Merging 'Microsoft Corporation' (id:{}) into 'Microsoft' (id:{})",
            corp.id, ms.id
        );
        db.merge_entities(ms.id, corp.id).expect("Merge failed");
        println!("✓ Merge successful");
    }

    if let Some(msft_entity) = msft {
        println!(
            "Merging 'MSFT' (id:{}) into 'Microsoft' (id:{})",
            msft_entity.id, ms.id
        );
        db.merge_entities(ms.id, msft_entity.id)
            .expect("Merge failed");
        println!("✓ Merge successful");
    }

    // Check merge history
    let history = db
        .get_entity_merge_history(ms.id)
        .expect("Failed to get history");
    println!("\nMerge history for 'Microsoft':");
    for (name, date) in &history {
        println!("  - {} (merged at {})", name, date);
    }

    // Verify entities after merge
    let final_entities = db
        .list_entities(store_name)
        .expect("Failed to list entities");
    println!(
        "\n✓ Final entity count: {} (was {})",
        final_entities.len(),
        entities.len()
    );

    // Clean up
    std::fs::remove_file(&db_path).ok();
    println!("\n=== Test Complete ===");
}
