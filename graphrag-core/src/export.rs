//! Export functionality for GraphRAG stores.

use crate::{Database, GraphRagError};
use serde_json::json;
use std::io::Write;

/// Exports the contents of a store to a JSON Lines format.
pub fn export_store(db: &Database, store: &str, out: &mut impl Write) -> Result<(), GraphRagError> {
    let s = db.get_store(store)?;
    writeln!(
        out,
        "{}",
        json!({"record": "store", "name": s.name, "dim": s.dim})
    )?;

    for c in db.list_chunks(store)? {
        writeln!(
            out,
            "{}",
            json!({
                "record": "chunk",
                "id": c.id,
                "content": c.content,
                "source": c.source,
                "metadata": c.metadata
            })
        )?;
    }

    let mut entities = db.list_entities(store)?;
    entities.sort_by_key(|e| e.id);
    for e in entities {
        writeln!(
            out,
            "{}",
            json!({
                "record": "entity",
                "id": e.id,
                "name": e.name,
                "entity_type": e.entity_type,
                "properties": e.properties
            })
        )?;
    }

    let mut relations = db.list_relations(store)?;
    relations.sort_by_key(|r| r.id);
    for r in relations {
        writeln!(
            out,
            "{}",
            json!({
                "record": "relation",
                "id": r.id,
                "head_id": r.head_id,
                "tail_id": r.tail_id,
                "relation": r.relation,
                "canonical_relation": r.canonical_relation,
                "properties": r.properties
            })
        )?;
    }

    let mut communities = db.list_communities(store)?;
    communities.sort_by_key(|c| c.id);
    for c in communities {
        writeln!(
            out,
            "{}",
            json!({
                "record": "community",
                "id": c.id,
                "level": c.level,
                "parent_id": c.parent_id,
                "summary": c.summary,
                "modularity": c.modularity
            })
        )?;
    }

    Ok(())
}
