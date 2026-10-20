//! Graph node kinds. Documents, sections and groups are rows in `entities`
//! (with `kind`), so every edge connects two entity rows.

use crate::db::Database;
use crate::error::GraphRagError;
use rusqlite::{OptionalExtension, params};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeKind {
    Entity,
    Document,
    Section,
    Group,
}

impl NodeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            NodeKind::Entity => "entity",
            NodeKind::Document => "document",
            NodeKind::Section => "section",
            NodeKind::Group => "group",
        }
    }

    pub fn parse(s: &str) -> Result<Self, GraphRagError> {
        Ok(match s {
            "entity" => NodeKind::Entity,
            "document" => NodeKind::Document,
            "section" => NodeKind::Section,
            "group" => NodeKind::Group,
            other => {
                return Err(GraphRagError::InvalidInput(format!(
                    "unknown node kind {other:?}"
                )));
            }
        })
    }
}

/// Group nodes are named `group:<name>` so a group never collides with an
/// extracted entity of the same name.
pub(crate) const GROUP_PREFIX: &str = "group:";

impl Database {
    /// Create a node row. Non-entity nodes get `entity_type` = their kind.
    pub fn create_node(
        &self,
        store: &str,
        kind: NodeKind,
        name: &str,
    ) -> Result<i64, GraphRagError> {
        let entity_type = (kind != NodeKind::Entity).then_some(kind.as_str());
        self.conn().execute(
            "INSERT INTO entities (store, name, entity_type, kind, name_key) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                store,
                name,
                entity_type,
                kind.as_str(),
                crate::synonyms::normalize_entity_name(name)
            ],
        )?;
        Ok(self.conn().last_insert_rowid())
    }

    pub fn node_kind(&self, id: i64) -> Result<NodeKind, GraphRagError> {
        let kind: String = self.conn().query_row(
            "SELECT kind FROM entities WHERE id = ?1",
            params![id],
            |r| r.get(0),
        )?;
        NodeKind::parse(&kind)
    }

    /// The group node for `name`, created on first use.
    pub fn get_or_create_group(&self, store: &str, name: &str) -> Result<i64, GraphRagError> {
        let name = name.trim();
        if name.is_empty() {
            return Err(GraphRagError::InvalidInput("empty group name".into()));
        }
        let node = format!("{GROUP_PREFIX}{name}");
        if let Some(id) = self
            .conn()
            .query_row(
                "SELECT id FROM entities WHERE store = ?1 AND name = ?2 AND kind = 'group'",
                params![store, node],
                |r| r.get(0),
            )
            .optional()?
        {
            return Ok(id);
        }
        self.create_node(store, NodeKind::Group, &node)
    }

    /// Names of every group `node` is `part_of`, directly or through
    /// subgroups. Computed by a recursive query; no closure rows exist.
    pub fn groups_of(&self, node: i64) -> Result<Vec<String>, GraphRagError> {
        let mut stmt = self.conn().prepare(
            "WITH RECURSIVE up(id) AS (
                 SELECT tail_id FROM assertions
                 WHERE head_id = ?1 AND relation = 'part_of' AND retracted_at IS NULL
                 UNION
                 SELECT a.tail_id FROM assertions a JOIN up ON a.head_id = up.id
                 WHERE a.relation = 'part_of' AND a.retracted_at IS NULL
             )
             SELECT substr(e.name, ?2) FROM up JOIN entities e ON e.id = up.id
             WHERE e.kind = 'group' ORDER BY e.name",
        )?;
        let rows = stmt.query_map(params![node, GROUP_PREFIX.len() as i64 + 1], |r| r.get(0))?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// Document nodes `part_of` the group, directly or through subgroups.
    pub fn members_of_group(&self, store: &str, group: &str) -> Result<Vec<i64>, GraphRagError> {
        let mut stmt = self.conn().prepare(
            "WITH RECURSIVE down(id) AS (
                 SELECT id FROM entities WHERE store = ?1 AND name = ?2 AND kind = 'group'
                 UNION
                 SELECT a.head_id FROM assertions a JOIN down ON a.tail_id = down.id
                 WHERE a.relation = 'part_of' AND a.retracted_at IS NULL
             )
             SELECT e.id FROM down JOIN entities e ON e.id = down.id
             WHERE e.kind = 'document' ORDER BY e.id",
        )?;
        let rows = stmt.query_map(params![store, format!("{GROUP_PREFIX}{group}")], |r| {
            r.get(0)
        })?;
        Ok(rows.collect::<Result<_, _>>()?)
    }
}
