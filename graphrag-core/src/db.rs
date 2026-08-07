use crate::error::GraphRagError;
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Store metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Store {
    pub name: String,
    pub dim: usize,
    pub created_at: String,
}

/// A chunk of content with its embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: i64,
    pub store: String,
    pub content: String,
    pub source: Option<String>,
    pub metadata: Option<String>,
    pub created_at: String,
}

/// An entity in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: i64,
    pub store: String,
    pub name: String,
    pub entity_type: Option<String>,
    pub properties: Option<String>,
}

/// A relation between entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    pub id: i64,
    pub store: String,
    pub head_id: i64,
    pub tail_id: i64,
    pub relation: String,
    pub canonical_relation: Option<String>,
    pub properties: Option<String>,
}

/// Input format for adding entities (from pipeline)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct EntityInput {
    pub head: String,
    pub head_type: Option<String>,
    pub relation: String,
    pub tail: String,
    pub tail_type: Option<String>,
    pub properties: Option<serde_json::Value>,
}

/// A candidate pair for entity merging:
/// `(entity1_id, entity1_name, entity2_id, entity2_name, cosine_similarity)`.
pub type MergeCandidate = (i64, String, i64, String, f32);

/// A detected community of entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityRecord {
    pub id: i64,
    pub store: String,
    pub level: i32,
    pub parent_id: Option<i64>,
    pub summary: Option<String>,
    pub modularity: Option<f64>,
    pub created_at: String,
}

pub struct Database {
    conn: Connection,
}

impl Database {
    pub fn open(path: &Path) -> Result<Self, GraphRagError> {
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(path)?;
        let db = Self { conn };
        db.init_schema()?;
        db.migrate_schema()?;
        Ok(db)
    }

    /// Run schema migrations for existing databases
    fn migrate_schema(&self) -> Result<(), GraphRagError> {
        // Check if communities table has parent_id column
        let has_parent_id: bool = self.conn.query_row(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('communities') WHERE name = 'parent_id'",
            [],
            |row| row.get(0),
        ).unwrap_or(false);

        if !has_parent_id {
            // Add parent_id column to existing communities table
            self.conn.execute(
                "ALTER TABLE communities ADD COLUMN parent_id INTEGER REFERENCES communities(id) ON DELETE CASCADE",
                [],
            )?;
            // Create index
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_communities_parent ON communities(parent_id)",
                [],
            )?;
        }

        // Check if relations table has canonical_relation column
        let has_canonical: bool = self.conn.query_row(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('relations') WHERE name = 'canonical_relation'",
            [],
            |row| row.get(0),
        ).unwrap_or(false);

        if !has_canonical {
            // Add canonical_relation column to existing relations table
            self.conn.execute(
                "ALTER TABLE relations ADD COLUMN canonical_relation TEXT",
                [],
            )?;
            // Create index
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_relations_canonical ON relations(canonical_relation)",
                [],
            )?;
        }

        // Create relation_synonyms table if it doesn't exist (for older DBs)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS relation_synonyms (
                raw_relation TEXT PRIMARY KEY,
                canonical_relation TEXT NOT NULL
            )",
            [],
        )?;

        // Create indexes (safe for both new and existing DBs)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_relations_canonical ON relations(canonical_relation)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_synonyms_canonical ON relation_synonyms(canonical_relation)",
            [],
        )?;

        // Create entity_type_synonyms table if it doesn't exist (for older DBs)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS entity_type_synonyms (
                raw_type TEXT PRIMARY KEY,
                canonical_type TEXT NOT NULL
            )",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_type_synonyms_canonical ON entity_type_synonyms(canonical_type)",
            [],
        )?;

        // Create entity_embeddings table if it doesn't exist
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS entity_embeddings (
                entity_id INTEGER PRIMARY KEY REFERENCES entities(id) ON DELETE CASCADE,
                embedding BLOB NOT NULL
            )",
            [],
        )?;

        // Create entity_merges table if it doesn't exist
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS entity_merges (
                id INTEGER PRIMARY KEY,
                target_entity_id INTEGER NOT NULL REFERENCES entities(id),
                merged_entity_name TEXT NOT NULL,
                merged_at TEXT DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        Ok(())
    }

    /// Crate-internal connection access for sibling storage providers.
    pub(crate) fn conn(&self) -> &Connection {
        &self.conn
    }

    fn init_schema(&self) -> Result<(), GraphRagError> {
        self.conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS stores (
                name TEXT PRIMARY KEY,
                dim INTEGER NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );

            -- Append-only capsule bodies with full fingerprint history.
            CREATE TABLE IF NOT EXISTS capsules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                capsule_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                content_fingerprint TEXT NOT NULL,
                generated_at TEXT NOT NULL,
                body TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(capsule_id, content_fingerprint)
            );
            CREATE INDEX IF NOT EXISTS idx_capsules_project ON capsules(project_id);

            -- Canonical chunk vectors; index files are disposable caches.
            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                embedding BLOB NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                store TEXT NOT NULL REFERENCES stores(name) ON DELETE CASCADE,
                content TEXT NOT NULL,
                source TEXT,
                metadata TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_store ON chunks(store);

            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY,
                store TEXT NOT NULL REFERENCES stores(name) ON DELETE CASCADE,
                name TEXT NOT NULL,
                entity_type TEXT,
                properties TEXT,
                UNIQUE(store, name)
            );
            CREATE INDEX IF NOT EXISTS idx_entities_store ON entities(store);
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(store, name);

            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY,
                store TEXT NOT NULL REFERENCES stores(name) ON DELETE CASCADE,
                head_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                tail_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                relation TEXT NOT NULL,
                canonical_relation TEXT,
                properties TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_relations_store ON relations(store);
            CREATE INDEX IF NOT EXISTS idx_relations_head ON relations(head_id);
            CREATE INDEX IF NOT EXISTS idx_relations_tail ON relations(tail_id);
            -- Note: idx_relations_canonical created in migrate_schema for existing DBs

            -- Synonym mapping: raw relation -> canonical relation
            CREATE TABLE IF NOT EXISTS relation_synonyms (
                raw_relation TEXT PRIMARY KEY,
                canonical_relation TEXT NOT NULL
            );

            -- Synonym mapping: raw entity type -> canonical entity type
            CREATE TABLE IF NOT EXISTS entity_type_synonyms (
                raw_type TEXT PRIMARY KEY,
                canonical_type TEXT NOT NULL
            );

            -- Entity embeddings for similarity-based merging
            CREATE TABLE IF NOT EXISTS entity_embeddings (
                entity_id INTEGER PRIMARY KEY REFERENCES entities(id) ON DELETE CASCADE,
                embedding BLOB NOT NULL
            );

            -- Entity merge history (for audit trail)
            CREATE TABLE IF NOT EXISTS entity_merges (
                id INTEGER PRIMARY KEY,
                target_entity_id INTEGER NOT NULL REFERENCES entities(id),
                merged_entity_name TEXT NOT NULL,
                merged_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS chunk_entities (
                chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                PRIMARY KEY (chunk_id, entity_id)
            );

            CREATE TABLE IF NOT EXISTS communities (
                id INTEGER PRIMARY KEY,
                store TEXT NOT NULL REFERENCES stores(name) ON DELETE CASCADE,
                level INTEGER NOT NULL DEFAULT 0,
                summary TEXT,
                modularity REAL,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_communities_store ON communities(store);

            CREATE TABLE IF NOT EXISTS entity_communities (
                entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                community_id INTEGER NOT NULL REFERENCES communities(id) ON DELETE CASCADE,
                PRIMARY KEY (entity_id, community_id)
            );
            CREATE INDEX IF NOT EXISTS idx_entity_communities_community ON entity_communities(community_id);

            PRAGMA foreign_keys = ON;
            "#,
        )?;
        Ok(())
    }

    // Store operations

    pub fn create_store(&self, name: &str, dim: usize) -> Result<Store, GraphRagError> {
        // Check if exists
        let exists: bool = self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM stores WHERE name = ?1)",
            params![name],
            |row| row.get(0),
        )?;

        if exists {
            return Err(GraphRagError::StoreExists(name.to_string()));
        }

        self.conn.execute(
            "INSERT INTO stores (name, dim) VALUES (?1, ?2)",
            params![name, dim as i64],
        )?;

        self.get_store(name)
    }

    pub fn get_store(&self, name: &str) -> Result<Store, GraphRagError> {
        self.conn
            .query_row(
                "SELECT name, dim, created_at FROM stores WHERE name = ?1",
                params![name],
                |row| {
                    Ok(Store {
                        name: row.get(0)?,
                        dim: row.get::<_, i64>(1)? as usize,
                        created_at: row.get(2)?,
                    })
                },
            )
            .map_err(|_| GraphRagError::StoreNotFound(name.to_string()))
    }

    pub fn list_stores(&self) -> Result<Vec<Store>, GraphRagError> {
        let mut stmt = self
            .conn
            .prepare("SELECT name, dim, created_at FROM stores ORDER BY name")?;

        let stores = stmt
            .query_map([], |row| {
                Ok(Store {
                    name: row.get(0)?,
                    dim: row.get::<_, i64>(1)? as usize,
                    created_at: row.get(2)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(stores)
    }

    pub fn delete_store(&self, name: &str) -> Result<(), GraphRagError> {
        let affected = self
            .conn
            .execute("DELETE FROM stores WHERE name = ?1", params![name])?;

        if affected == 0 {
            return Err(GraphRagError::StoreNotFound(name.to_string()));
        }

        Ok(())
    }

    pub fn store_stats(&self, name: &str) -> Result<(i64, i64, i64), GraphRagError> {
        // Verify store exists
        let _ = self.get_store(name)?;

        let chunk_count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM chunks WHERE store = ?1",
            params![name],
            |row| row.get(0),
        )?;

        let entity_count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM entities WHERE store = ?1",
            params![name],
            |row| row.get(0),
        )?;

        let relation_count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM relations WHERE store = ?1",
            params![name],
            |row| row.get(0),
        )?;

        Ok((chunk_count, entity_count, relation_count))
    }

    // Chunk operations

    pub fn add_chunk(
        &self,
        store: &str,
        content: &str,
        source: Option<&str>,
        metadata: Option<&str>,
    ) -> Result<i64, GraphRagError> {
        // Verify store exists
        let _ = self.get_store(store)?;

        self.conn.execute(
            "INSERT INTO chunks (store, content, source, metadata) VALUES (?1, ?2, ?3, ?4)",
            params![store, content, source, metadata],
        )?;

        Ok(self.conn.last_insert_rowid())
    }

    /// List the most-recent chunks in a store, newest first, capped at `limit`.
    pub fn list_recent_chunks(
        &self,
        store: &str,
        limit: usize,
    ) -> Result<Vec<Chunk>, GraphRagError> {
        let _ = self.get_store(store)?;
        let mut stmt = self.conn.prepare(
            "SELECT id, store, content, source, metadata, created_at
             FROM chunks
             WHERE store = ?1
             ORDER BY id DESC
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![store, limit as i64], |row| {
            Ok(Chunk {
                id: row.get(0)?,
                store: row.get(1)?,
                content: row.get(2)?,
                source: row.get(3)?,
                metadata: row.get(4)?,
                created_at: row.get(5)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// List all chunks in a store, oldest first.
    pub fn list_chunks(&self, store: &str) -> Result<Vec<Chunk>, GraphRagError> {
        let _ = self.get_store(store)?;
        let mut stmt = self.conn.prepare(
            "SELECT id, store, content, source, metadata, created_at
             FROM chunks
             WHERE store = ?1
             ORDER BY id ASC",
        )?;
        let rows = stmt.query_map(params![store], |row| {
            Ok(Chunk {
                id: row.get(0)?,
                store: row.get(1)?,
                content: row.get(2)?,
                source: row.get(3)?,
                metadata: row.get(4)?,
                created_at: row.get(5)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn get_chunks_by_ids(&self, ids: &[i64]) -> Result<Vec<Chunk>, GraphRagError> {
        if ids.is_empty() {
            return Ok(vec![]);
        }

        let placeholders: Vec<String> = ids.iter().map(|_| "?".to_string()).collect();
        let sql = format!(
            "SELECT id, store, content, source, metadata, created_at FROM chunks WHERE id IN ({})",
            placeholders.join(", ")
        );

        let mut stmt = self.conn.prepare(&sql)?;
        let params: Vec<&dyn rusqlite::ToSql> =
            ids.iter().map(|id| id as &dyn rusqlite::ToSql).collect();

        let chunks = stmt
            .query_map(params.as_slice(), |row| {
                Ok(Chunk {
                    id: row.get(0)?,
                    store: row.get(1)?,
                    content: row.get(2)?,
                    source: row.get(3)?,
                    metadata: row.get(4)?,
                    created_at: row.get(5)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(chunks)
    }

    // Entity operations

    pub fn get_or_create_entity(
        &self,
        store: &str,
        name: &str,
        entity_type: Option<&str>,
        properties: Option<&str>,
    ) -> Result<i64, GraphRagError> {
        // Try to get existing
        let existing: Option<i64> = self
            .conn
            .query_row(
                "SELECT id FROM entities WHERE store = ?1 AND name = ?2",
                params![store, name],
                |row| row.get(0),
            )
            .ok();

        if let Some(id) = existing {
            // Update type/properties if provided
            if entity_type.is_some() || properties.is_some() {
                self.conn.execute(
                    "UPDATE entities SET entity_type = COALESCE(?3, entity_type), properties = COALESCE(?4, properties) WHERE id = ?1",
                    params![id, store, entity_type, properties],
                )?;
            }
            return Ok(id);
        }

        // Create new
        self.conn.execute(
            "INSERT INTO entities (store, name, entity_type, properties) VALUES (?1, ?2, ?3, ?4)",
            params![store, name, entity_type, properties],
        )?;

        Ok(self.conn.last_insert_rowid())
    }

    pub fn list_entities(&self, store: &str) -> Result<Vec<Entity>, GraphRagError> {
        let _ = self.get_store(store)?;

        let mut stmt = self.conn.prepare(
            "SELECT id, store, name, entity_type, properties FROM entities WHERE store = ?1 ORDER BY name",
        )?;

        let entities = stmt
            .query_map(params![store], |row| {
                Ok(Entity {
                    id: row.get(0)?,
                    store: row.get(1)?,
                    name: row.get(2)?,
                    entity_type: row.get(3)?,
                    properties: row.get(4)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(entities)
    }

    // Relation operations

    pub fn add_relation(
        &self,
        store: &str,
        head_id: i64,
        tail_id: i64,
        relation: &str,
        properties: Option<&str>,
    ) -> Result<i64, GraphRagError> {
        // Look up canonical form from synonyms table
        let canonical: Option<String> = self.get_canonical_relation(relation);

        self.conn.execute(
            "INSERT INTO relations (store, head_id, tail_id, relation, canonical_relation, properties) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![store, head_id, tail_id, relation, canonical, properties],
        )?;

        Ok(self.conn.last_insert_rowid())
    }

    pub fn get_relations_for_entity(&self, entity_id: i64) -> Result<Vec<Relation>, GraphRagError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, store, head_id, tail_id, relation, canonical_relation, properties FROM relations WHERE head_id = ?1 OR tail_id = ?1",
        )?;

        let relations = stmt
            .query_map(params![entity_id], |row| {
                Ok(Relation {
                    id: row.get(0)?,
                    store: row.get(1)?,
                    head_id: row.get(2)?,
                    tail_id: row.get(3)?,
                    relation: row.get(4)?,
                    canonical_relation: row.get(5)?,
                    properties: row.get(6)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(relations)
    }

    pub fn get_entity_by_name(&self, store: &str, name: &str) -> Result<Entity, GraphRagError> {
        self.conn
            .query_row(
                "SELECT id, store, name, entity_type, properties FROM entities WHERE store = ?1 AND name = ?2",
                params![store, name],
                |row| {
                    Ok(Entity {
                        id: row.get(0)?,
                        store: row.get(1)?,
                        name: row.get(2)?,
                        entity_type: row.get(3)?,
                        properties: row.get(4)?,
                    })
                },
            )
            .map_err(|_| GraphRagError::EntityNotFound(name.to_string()))
    }

    // Chunk-Entity linking

    pub fn link_chunk_entity(&self, chunk_id: i64, entity_id: i64) -> Result<(), GraphRagError> {
        self.conn.execute(
            "INSERT OR IGNORE INTO chunk_entities (chunk_id, entity_id) VALUES (?1, ?2)",
            params![chunk_id, entity_id],
        )?;
        Ok(())
    }

    pub fn get_chunks_for_entity(&self, entity_id: i64) -> Result<Vec<i64>, GraphRagError> {
        let mut stmt = self
            .conn
            .prepare("SELECT chunk_id FROM chunk_entities WHERE entity_id = ?1")?;

        let ids = stmt
            .query_map(params![entity_id], |row| row.get(0))?
            .collect::<Result<Vec<i64>, _>>()?;

        Ok(ids)
    }

    pub fn get_entities_for_chunk(&self, chunk_id: i64) -> Result<Vec<Entity>, GraphRagError> {
        let mut stmt = self.conn.prepare(
            "SELECT e.id, e.store, e.name, e.entity_type, e.properties
             FROM entities e
             JOIN chunk_entities ce ON e.id = ce.entity_id
             WHERE ce.chunk_id = ?1",
        )?;

        let entities = stmt
            .query_map(params![chunk_id], |row| {
                Ok(Entity {
                    id: row.get(0)?,
                    store: row.get(1)?,
                    name: row.get(2)?,
                    entity_type: row.get(3)?,
                    properties: row.get(4)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(entities)
    }

    /// Get all relations in a store (for community detection)
    pub fn list_relations(&self, store: &str) -> Result<Vec<Relation>, GraphRagError> {
        let _ = self.get_store(store)?;

        let mut stmt = self.conn.prepare(
            "SELECT id, store, head_id, tail_id, relation, canonical_relation, properties FROM relations WHERE store = ?1",
        )?;

        let relations = stmt
            .query_map(params![store], |row| {
                Ok(Relation {
                    id: row.get(0)?,
                    store: row.get(1)?,
                    head_id: row.get(2)?,
                    tail_id: row.get(3)?,
                    relation: row.get(4)?,
                    canonical_relation: row.get(5)?,
                    properties: row.get(6)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(relations)
    }

    /// Get entity by ID
    pub fn get_entity_by_id(&self, entity_id: i64) -> Result<Entity, GraphRagError> {
        self.conn
            .query_row(
                "SELECT id, store, name, entity_type, properties FROM entities WHERE id = ?1",
                params![entity_id],
                |row| {
                    Ok(Entity {
                        id: row.get(0)?,
                        store: row.get(1)?,
                        name: row.get(2)?,
                        entity_type: row.get(3)?,
                        properties: row.get(4)?,
                    })
                },
            )
            .map_err(|_| GraphRagError::EntityNotFound(format!("id={}", entity_id)))
    }

    // Community operations

    /// Clear existing communities for a store
    pub fn clear_communities(&self, store: &str) -> Result<(), GraphRagError> {
        self.conn
            .execute("DELETE FROM communities WHERE store = ?1", params![store])?;
        Ok(())
    }

    /// Create a new community
    pub fn create_community(
        &self,
        store: &str,
        level: i32,
        modularity: f64,
        parent_id: Option<i64>,
    ) -> Result<i64, GraphRagError> {
        self.conn.execute(
            "INSERT INTO communities (store, level, modularity, parent_id) VALUES (?1, ?2, ?3, ?4)",
            params![store, level, modularity, parent_id],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Link an entity to a community
    pub fn link_entity_community(
        &self,
        entity_id: i64,
        community_id: i64,
    ) -> Result<(), GraphRagError> {
        self.conn.execute(
            "INSERT OR IGNORE INTO entity_communities (entity_id, community_id) VALUES (?1, ?2)",
            params![entity_id, community_id],
        )?;
        Ok(())
    }

    /// Update community summary
    pub fn update_community_summary(
        &self,
        community_id: i64,
        summary: &str,
    ) -> Result<(), GraphRagError> {
        self.conn.execute(
            "UPDATE communities SET summary = ?2 WHERE id = ?1",
            params![community_id, summary],
        )?;
        Ok(())
    }

    /// Get all communities for a store
    pub fn list_communities(&self, store: &str) -> Result<Vec<CommunityRecord>, GraphRagError> {
        let _ = self.get_store(store)?;

        let mut stmt = self.conn.prepare(
            "SELECT id, store, level, parent_id, summary, modularity, created_at FROM communities WHERE store = ?1 ORDER BY level, id",
        )?;

        let communities = stmt
            .query_map(params![store], |row| {
                Ok(CommunityRecord {
                    id: row.get(0)?,
                    store: row.get(1)?,
                    level: row.get(2)?,
                    parent_id: row.get(3)?,
                    summary: row.get(4)?,
                    modularity: row.get(5)?,
                    created_at: row.get(6)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(communities)
    }

    /// Get child communities of a parent
    pub fn get_child_communities(
        &self,
        parent_id: i64,
    ) -> Result<Vec<CommunityRecord>, GraphRagError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, store, level, parent_id, summary, modularity, created_at FROM communities WHERE parent_id = ?1 ORDER BY id",
        )?;

        let communities = stmt
            .query_map(params![parent_id], |row| {
                Ok(CommunityRecord {
                    id: row.get(0)?,
                    store: row.get(1)?,
                    level: row.get(2)?,
                    parent_id: row.get(3)?,
                    summary: row.get(4)?,
                    modularity: row.get(5)?,
                    created_at: row.get(6)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(communities)
    }

    /// Get entities in a community
    pub fn get_community_entities(&self, community_id: i64) -> Result<Vec<Entity>, GraphRagError> {
        let mut stmt = self.conn.prepare(
            "SELECT e.id, e.store, e.name, e.entity_type, e.properties
             FROM entities e
             JOIN entity_communities ec ON e.id = ec.entity_id
             WHERE ec.community_id = ?1
             ORDER BY e.name",
        )?;

        let entities = stmt
            .query_map(params![community_id], |row| {
                Ok(Entity {
                    id: row.get(0)?,
                    store: row.get(1)?,
                    name: row.get(2)?,
                    entity_type: row.get(3)?,
                    properties: row.get(4)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(entities)
    }

    /// Get communities for an entity (can be multiple due to hierarchy)
    pub fn get_entity_communities(
        &self,
        entity_id: i64,
    ) -> Result<Vec<CommunityRecord>, GraphRagError> {
        let mut stmt = self.conn.prepare(
            "SELECT c.id, c.store, c.level, c.parent_id, c.summary, c.modularity, c.created_at
             FROM communities c
             JOIN entity_communities ec ON c.id = ec.community_id
             WHERE ec.entity_id = ?1
             ORDER BY c.level",
        )?;

        let communities = stmt
            .query_map(params![entity_id], |row| {
                Ok(CommunityRecord {
                    id: row.get(0)?,
                    store: row.get(1)?,
                    level: row.get(2)?,
                    parent_id: row.get(3)?,
                    summary: row.get(4)?,
                    modularity: row.get(5)?,
                    created_at: row.get(6)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(communities)
    }

    // Relation synonym operations

    /// Get the canonical form of a relation (or None if no synonym defined)
    pub fn get_canonical_relation(&self, relation: &str) -> Option<String> {
        self.conn
            .query_row(
                "SELECT canonical_relation FROM relation_synonyms WHERE raw_relation = ?1",
                params![relation],
                |row| row.get(0),
            )
            .ok()
    }

    /// Add or update a relation synonym mapping
    pub fn add_relation_synonym(&self, raw: &str, canonical: &str) -> Result<(), GraphRagError> {
        self.conn.execute(
            "INSERT OR REPLACE INTO relation_synonyms (raw_relation, canonical_relation) VALUES (?1, ?2)",
            params![raw, canonical],
        )?;
        Ok(())
    }

    /// Add multiple synonyms at once (batch)
    pub fn add_relation_synonyms(&self, mappings: &[(&str, &str)]) -> Result<(), GraphRagError> {
        for (raw, canonical) in mappings {
            self.add_relation_synonym(raw, canonical)?;
        }
        Ok(())
    }

    /// Remove a synonym mapping
    pub fn remove_relation_synonym(&self, raw: &str) -> Result<(), GraphRagError> {
        self.conn.execute(
            "DELETE FROM relation_synonyms WHERE raw_relation = ?1",
            params![raw],
        )?;
        Ok(())
    }

    /// List all synonym mappings
    pub fn list_relation_synonyms(&self) -> Result<Vec<(String, String)>, GraphRagError> {
        let mut stmt = self.conn.prepare(
            "SELECT raw_relation, canonical_relation FROM relation_synonyms ORDER BY canonical_relation, raw_relation",
        )?;

        let mappings = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(mappings)
    }

    /// Get all raw relations that map to a canonical form
    pub fn get_synonyms_for_canonical(
        &self,
        canonical: &str,
    ) -> Result<Vec<String>, GraphRagError> {
        let mut stmt = self.conn.prepare(
            "SELECT raw_relation FROM relation_synonyms WHERE canonical_relation = ?1 ORDER BY raw_relation",
        )?;

        let synonyms = stmt
            .query_map(params![canonical], |row| row.get(0))?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(synonyms)
    }

    /// Re-canonicalize all existing relations using current synonym mappings
    /// Call this after adding new synonyms to update existing data
    pub fn refresh_canonical_relations(&self) -> Result<usize, GraphRagError> {
        let affected = self.conn.execute(
            "UPDATE relations
             SET canonical_relation = (
                 SELECT canonical_relation FROM relation_synonyms
                 WHERE relation_synonyms.raw_relation = relations.relation
             )
             WHERE EXISTS (
                 SELECT 1 FROM relation_synonyms
                 WHERE relation_synonyms.raw_relation = relations.relation
             )",
            [],
        )?;
        Ok(affected)
    }

    /// Get distinct relation types in a store (optionally with counts)
    pub fn list_relation_types(
        &self,
        store: &str,
    ) -> Result<Vec<(String, Option<String>, i64)>, GraphRagError> {
        let _ = self.get_store(store)?;

        let mut stmt = self.conn.prepare(
            "SELECT relation, canonical_relation, COUNT(*) as cnt
             FROM relations WHERE store = ?1
             GROUP BY relation ORDER BY cnt DESC",
        )?;

        let types = stmt
            .query_map(params![store], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(types)
    }

    /// Query relations by canonical type (finds all raw variants)
    pub fn get_relations_by_canonical(
        &self,
        store: &str,
        canonical: &str,
    ) -> Result<Vec<Relation>, GraphRagError> {
        let _ = self.get_store(store)?;

        let mut stmt = self.conn.prepare(
            "SELECT id, store, head_id, tail_id, relation, canonical_relation, properties
             FROM relations
             WHERE store = ?1 AND (canonical_relation = ?2 OR relation = ?2)",
        )?;

        let relations = stmt
            .query_map(params![store, canonical], |row| {
                Ok(Relation {
                    id: row.get(0)?,
                    store: row.get(1)?,
                    head_id: row.get(2)?,
                    tail_id: row.get(3)?,
                    relation: row.get(4)?,
                    canonical_relation: row.get(5)?,
                    properties: row.get(6)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(relations)
    }

    // Entity type synonym operations

    /// Get the canonical form of an entity type (or None if no synonym defined)
    pub fn get_canonical_entity_type(&self, entity_type: &str) -> Option<String> {
        self.conn
            .query_row(
                "SELECT canonical_type FROM entity_type_synonyms WHERE raw_type = ?1",
                params![entity_type.to_lowercase()],
                |row| row.get(0),
            )
            .ok()
    }

    /// Add or update an entity type synonym mapping
    pub fn add_entity_type_synonym(&self, raw: &str, canonical: &str) -> Result<(), GraphRagError> {
        self.conn.execute(
            "INSERT OR REPLACE INTO entity_type_synonyms (raw_type, canonical_type) VALUES (?1, ?2)",
            params![raw.to_lowercase(), canonical.to_lowercase()],
        )?;
        Ok(())
    }

    /// Add multiple entity type synonyms at once (batch)
    pub fn add_entity_type_synonyms(&self, mappings: &[(&str, &str)]) -> Result<(), GraphRagError> {
        for (raw, canonical) in mappings {
            self.add_entity_type_synonym(raw, canonical)?;
        }
        Ok(())
    }

    /// List all entity type synonym mappings
    pub fn list_entity_type_synonyms(&self) -> Result<Vec<(String, String)>, GraphRagError> {
        let mut stmt = self.conn.prepare(
            "SELECT raw_type, canonical_type FROM entity_type_synonyms ORDER BY canonical_type, raw_type",
        )?;

        let mappings = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(mappings)
    }

    /// Normalize an entity type using synonyms (returns canonical or original)
    pub fn normalize_entity_type(&self, entity_type: &str) -> String {
        self.get_canonical_entity_type(entity_type)
            .unwrap_or_else(|| entity_type.to_lowercase())
    }

    // Entity embedding and merging operations

    /// Store an embedding for an entity (for similarity search)
    pub fn set_entity_embedding(
        &self,
        entity_id: i64,
        embedding: &[f32],
    ) -> Result<(), GraphRagError> {
        let blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.conn.execute(
            "INSERT OR REPLACE INTO entity_embeddings (entity_id, embedding) VALUES (?1, ?2)",
            params![entity_id, blob],
        )?;
        Ok(())
    }

    /// Get embedding for an entity
    pub fn get_entity_embedding(&self, entity_id: i64) -> Result<Option<Vec<f32>>, GraphRagError> {
        let blob: Option<Vec<u8>> = self
            .conn
            .query_row(
                "SELECT embedding FROM entity_embeddings WHERE entity_id = ?1",
                params![entity_id],
                |row| row.get(0),
            )
            .ok();

        Ok(blob.map(|b| {
            b.chunks(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()
        }))
    }

    /// Get all entity embeddings in a store (for bulk similarity search)
    pub fn get_all_entity_embeddings(
        &self,
        store: &str,
    ) -> Result<Vec<(i64, String, Vec<f32>)>, GraphRagError> {
        let _ = self.get_store(store)?;

        let mut stmt = self.conn.prepare(
            "SELECT e.id, e.name, ee.embedding
             FROM entities e
             JOIN entity_embeddings ee ON e.id = ee.entity_id
             WHERE e.store = ?1",
        )?;

        let results = stmt
            .query_map(params![store], |row| {
                let id: i64 = row.get(0)?;
                let name: String = row.get(1)?;
                let blob: Vec<u8> = row.get(2)?;
                let embedding: Vec<f32> = blob
                    .chunks(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Ok((id, name, embedding))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(results)
    }

    /// Find entities with embeddings similar to the given one
    /// Returns (entity_id, name, similarity_score) sorted by similarity descending
    pub fn find_similar_entities(
        &self,
        store: &str,
        embedding: &[f32],
        threshold: f32,
        limit: usize,
    ) -> Result<Vec<(i64, String, f32)>, GraphRagError> {
        let all_embeddings = self.get_all_entity_embeddings(store)?;

        let mut similarities: Vec<(i64, String, f32)> = all_embeddings
            .into_iter()
            .filter_map(|(id, name, emb)| {
                let sim = cosine_similarity(embedding, &emb);
                if sim >= threshold {
                    Some((id, name, sim))
                } else {
                    None
                }
            })
            .collect();

        // Sort by similarity descending
        similarities.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(limit);

        Ok(similarities)
    }

    /// Merge one entity into another, transferring all relations and chunk links
    /// The source entity is deleted after merging
    pub fn merge_entities(&self, target_id: i64, source_id: i64) -> Result<(), GraphRagError> {
        // Get source entity name for audit
        let source_name: String = self.conn.query_row(
            "SELECT name FROM entities WHERE id = ?1",
            params![source_id],
            |row| row.get(0),
        )?;

        // Update relations: change source_id to target_id in both head and tail
        self.conn.execute(
            "UPDATE relations SET head_id = ?1 WHERE head_id = ?2",
            params![target_id, source_id],
        )?;
        self.conn.execute(
            "UPDATE relations SET tail_id = ?1 WHERE tail_id = ?2",
            params![target_id, source_id],
        )?;

        // Update chunk_entities links
        // First, delete links that would become duplicates
        self.conn.execute(
            "DELETE FROM chunk_entities
             WHERE entity_id = ?1
             AND chunk_id IN (SELECT chunk_id FROM chunk_entities WHERE entity_id = ?2)",
            params![source_id, target_id],
        )?;
        // Then update remaining links
        self.conn.execute(
            "UPDATE chunk_entities SET entity_id = ?1 WHERE entity_id = ?2",
            params![target_id, source_id],
        )?;

        // Update community membership
        self.conn.execute(
            "DELETE FROM entity_communities
             WHERE entity_id = ?1
             AND community_id IN (SELECT community_id FROM entity_communities WHERE entity_id = ?2)",
            params![source_id, target_id],
        )?;
        self.conn.execute(
            "UPDATE entity_communities SET entity_id = ?1 WHERE entity_id = ?2",
            params![target_id, source_id],
        )?;

        // Record the merge for audit
        self.conn.execute(
            "INSERT INTO entity_merges (target_entity_id, merged_entity_name) VALUES (?1, ?2)",
            params![target_id, source_name],
        )?;

        // Delete source entity (cascades to embedding)
        self.conn
            .execute("DELETE FROM entities WHERE id = ?1", params![source_id])?;

        Ok(())
    }

    /// Get merge history for an entity (what was merged into it)
    pub fn get_entity_merge_history(
        &self,
        entity_id: i64,
    ) -> Result<Vec<(String, String)>, GraphRagError> {
        let mut stmt = self.conn.prepare(
            "SELECT merged_entity_name, merged_at FROM entity_merges WHERE target_entity_id = ?1 ORDER BY merged_at",
        )?;

        let history = stmt
            .query_map(params![entity_id], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(history)
    }

    /// Find candidate entity pairs for merging based on embedding similarity
    /// Returns pairs of (entity1_id, entity1_name, entity2_id, entity2_name, similarity)
    pub fn find_merge_candidates(
        &self,
        store: &str,
        threshold: f32,
    ) -> Result<Vec<MergeCandidate>, GraphRagError> {
        let all_embeddings = self.get_all_entity_embeddings(store)?;
        let mut candidates = Vec::new();

        for i in 0..all_embeddings.len() {
            for j in (i + 1)..all_embeddings.len() {
                let (id1, name1, emb1) = &all_embeddings[i];
                let (id2, name2, emb2) = &all_embeddings[j];

                let sim = cosine_similarity(emb1, emb2);
                if sim >= threshold {
                    candidates.push((*id1, name1.clone(), *id2, name2.clone(), sim));
                }
            }
        }

        // Sort by similarity descending
        candidates.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));

        Ok(candidates)
    }
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db() -> (Database, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("temp dir");
        let db = Database::open(&dir.path().join("test.db")).expect("db opens");
        (db, dir)
    }

    #[test]
    fn list_recent_chunks_returns_most_recent_first() {
        let (db, _dir) = temp_db();
        db.create_store("s", 8).expect("create store");
        let id1 = db.add_chunk("s", "first note", None, None).expect("add 1");
        let id2 = db.add_chunk("s", "second note", None, None).expect("add 2");
        let id3 = db.add_chunk("s", "third note", None, None).expect("add 3");

        let recent = db.list_recent_chunks("s", 10).expect("list");

        assert_eq!(recent.len(), 3, "expected 3 chunks");
        // Most-recent first
        assert_eq!(recent[0].id, id3);
        assert_eq!(recent[1].id, id2);
        assert_eq!(recent[2].id, id1);
        assert_eq!(recent[0].content, "third note");
    }

    #[test]
    fn list_recent_chunks_respects_limit() {
        let (db, _dir) = temp_db();
        db.create_store("s", 8).expect("create store");
        for i in 0..5 {
            db.add_chunk("s", &format!("note {i}"), None, None)
                .expect("add");
        }
        let recent = db.list_recent_chunks("s", 2).expect("list");
        assert_eq!(recent.len(), 2);
    }
}
