//! Generalization: classes named by extraction, and (Task 12A) the relation
//! hierarchy and lifted class-level edges.

use crate::db::Database;
use crate::error::GraphRagError;
use crate::synonyms::normalize_entity_name;
use rusqlite::{OptionalExtension, params};

impl Database {
    /// The class node for `kind`: an alias match, else an entity with the
    /// same name key, else a new class (`entity_type = 'collection'`).
    pub fn resolve_class(&self, store: &str, kind: &str) -> Result<i64, GraphRagError> {
        let key = normalize_entity_name(kind);
        if let Some(id) = self
            .conn()
            .query_row(
                "SELECT a.class_id FROM class_aliases a JOIN entities e ON e.id = a.class_id
                 WHERE a.alias = ?1 AND e.store = ?2",
                params![key, store],
                |r| r.get(0),
            )
            .optional()?
        {
            return Ok(id);
        }
        let id = self.get_or_create_entity(store, kind, None, None)?;
        self.conn().execute(
            "UPDATE entities SET entity_type = 'collection' WHERE id = ?1 AND entity_type IS NULL",
            params![id],
        )?;
        Ok(id)
    }
}
