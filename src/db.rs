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

        Ok(())
    }

    fn init_schema(&self) -> Result<(), GraphRagError> {
        self.conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS stores (
                name TEXT PRIMARY KEY,
                dim INTEGER NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
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
                properties TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_relations_store ON relations(store);
            CREATE INDEX IF NOT EXISTS idx_relations_head ON relations(head_id);
            CREATE INDEX IF NOT EXISTS idx_relations_tail ON relations(tail_id);

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
        self.conn.execute(
            "INSERT INTO relations (store, head_id, tail_id, relation, properties) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![store, head_id, tail_id, relation, properties],
        )?;

        Ok(self.conn.last_insert_rowid())
    }

    pub fn get_relations_for_entity(&self, entity_id: i64) -> Result<Vec<Relation>, GraphRagError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, store, head_id, tail_id, relation, properties FROM relations WHERE head_id = ?1 OR tail_id = ?1",
        )?;

        let relations = stmt
            .query_map(params![entity_id], |row| {
                Ok(Relation {
                    id: row.get(0)?,
                    store: row.get(1)?,
                    head_id: row.get(2)?,
                    tail_id: row.get(3)?,
                    relation: row.get(4)?,
                    properties: row.get(5)?,
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
            "SELECT id, store, head_id, tail_id, relation, properties FROM relations WHERE store = ?1",
        )?;

        let relations = stmt
            .query_map(params![store], |row| {
                Ok(Relation {
                    id: row.get(0)?,
                    store: row.get(1)?,
                    head_id: row.get(2)?,
                    tail_id: row.get(3)?,
                    relation: row.get(4)?,
                    properties: row.get(5)?,
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
        self.conn.execute(
            "DELETE FROM communities WHERE store = ?1",
            params![store],
        )?;
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
    pub fn link_entity_community(&self, entity_id: i64, community_id: i64) -> Result<(), GraphRagError> {
        self.conn.execute(
            "INSERT OR IGNORE INTO entity_communities (entity_id, community_id) VALUES (?1, ?2)",
            params![entity_id, community_id],
        )?;
        Ok(())
    }

    /// Update community summary
    pub fn update_community_summary(&self, community_id: i64, summary: &str) -> Result<(), GraphRagError> {
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
    pub fn get_child_communities(&self, parent_id: i64) -> Result<Vec<CommunityRecord>, GraphRagError> {
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
    pub fn get_entity_communities(&self, entity_id: i64) -> Result<Vec<CommunityRecord>, GraphRagError> {
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

}
