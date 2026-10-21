//! Salience and dormancy. Every use of an entity or relation (a new
//! evidence unit stating it, a recall hit, an assertion) goes in a use log;
//! salience is ACT-R base-level activation over it, `B = ln Σ w·a^-d` with
//! ages in days. An item goes dormant only when `B < τ` *and* its gist
//! survives one level up (every relation it takes part in is covered by an
//! admitted lifted class edge). Dormant items leave recall expansion,
//! clustering and summaries but are never deleted, and the next use
//! revives them. Documents, sections and chunks are never touched.

use crate::db::Database;
use crate::error::GraphRagError;
use rusqlite::params;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UseKind {
    Mention,
    Recall,
    Assert,
}

impl UseKind {
    fn as_str(self) -> &'static str {
        match self {
            UseKind::Mention => "mention",
            UseKind::Recall => "recall",
            UseKind::Assert => "assert",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SalienceConfig {
    pub d: f64,
    pub tau: f64,
}

impl Default for SalienceConfig {
    fn default() -> Self {
        Self { d: 0.5, tau: -3.0 }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SalienceReport {
    pub refreshed: usize,
    pub went_dormant: usize,
    pub revived: usize,
}

/// `ln Σ w·max(age, 1/24)^-d`; `-∞` with no uses.
pub fn activation(uses: &[(f64, f64)], d: f64) -> f64 {
    if uses.is_empty() {
        return f64::NEG_INFINITY;
    }
    uses.iter()
        .map(|(age, w)| w * age.max(1.0 / 24.0).powf(-d))
        .sum::<f64>()
        .ln()
}

impl Database {
    /// Record a use. A recall counts at most once per item per UTC day.
    pub fn log_use(
        &self,
        item_kind: &str,
        item_id: i64,
        kind: UseKind,
        source_id: Option<i64>,
        at: &str,
    ) -> Result<(), GraphRagError> {
        if kind == UseKind::Recall {
            let seen: bool = self.conn().query_row(
                "SELECT EXISTS(SELECT 1 FROM uses WHERE item_kind = ?1 AND item_id = ?2
                     AND kind = 'recall' AND substr(at, 1, 10) = substr(?3, 1, 10))",
                params![item_kind, item_id, at],
                |r| r.get(0),
            )?;
            if seen {
                return Ok(());
            }
        }
        self.conn().execute(
            "INSERT INTO uses (item_kind, item_id, at, kind, source_id, weight) VALUES (?1, ?2, ?3, ?4, ?5, 1.0)",
            params![item_kind, item_id, at, kind.as_str(), source_id],
        )?;
        Ok(())
    }

    /// Relations covered by an admitted lifted class edge.
    fn covered_relations(&self) -> Result<HashSet<i64>, GraphRagError> {
        let mut stmt = self.conn().prepare(
            "SELECT lf.relation_id FROM lifted_from lf JOIN lifted_edges l ON l.id = lf.lifted_id WHERE l.admitted = 1",
        )?;
        Ok(stmt
            .query_map([], |r| r.get(0))?
            .collect::<Result<_, _>>()?)
    }

    /// True when the entity has an `isa` class and every relation it takes
    /// part in (besides isa/genls) is covered by an admitted lifted edge.
    pub fn gist_covered(&self, entity_id: i64) -> Result<bool, GraphRagError> {
        let covered = self.covered_relations()?;
        let has_class: bool = self.conn().query_row(
            "SELECT EXISTS(SELECT 1 FROM relations WHERE head_id = ?1
                 AND COALESCE(canonical_relation, relation) = 'isa')",
            params![entity_id],
            |r| r.get(0),
        )?;
        if !has_class {
            return Ok(false);
        }
        let mut stmt = self.conn().prepare(
            "SELECT id FROM relations WHERE (head_id = ?1 OR tail_id = ?1)
               AND COALESCE(canonical_relation, relation) NOT IN ('isa', 'genls') AND edge_status != 'implied'",
        )?;
        let rels: Vec<i64> = stmt
            .query_map(params![entity_id], |r| r.get(0))?
            .collect::<Result<_, _>>()?;
        Ok(!rels.is_empty() && rels.iter().all(|r| covered.contains(r)))
    }

    fn ages(
        &self,
        kinds: &[&str],
        now: &str,
    ) -> Result<HashMap<i64, Vec<(f64, f64)>>, GraphRagError> {
        let list = kinds
            .iter()
            .map(|k| format!("'{k}'"))
            .collect::<Vec<_>>()
            .join(",");
        let mut stmt = self.conn().prepare(&format!(
            "SELECT item_id, julianday(?1) - julianday(at), weight FROM uses WHERE item_kind IN ({list})"
        ))?;
        let mut out: HashMap<i64, Vec<(f64, f64)>> = HashMap::new();
        for row in stmt.query_map(params![now], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, f64>(1)?,
                r.get::<_, f64>(2)?,
            ))
        })? {
            let (id, age, w) = row?;
            out.entry(id).or_default().push((age, w));
        }
        Ok(out)
    }

    pub fn refresh_salience(
        &self,
        store: &str,
        cfg: &SalienceConfig,
        now: &str,
    ) -> Result<SalienceReport, GraphRagError> {
        let tx = self.write_tx()?;
        let covered = self.covered_relations()?;
        let mut report = SalienceReport::default();

        // Relations first: an entity's gist depends on its relations, not their status.
        let rel_ages = self.ages(&["relation"], now)?;
        let mut by_entity: HashMap<i64, Vec<i64>> = HashMap::new();
        let mut classed: HashSet<i64> = HashSet::new();
        let relations: Vec<(i64, i64, i64, String, String)> = {
            let mut stmt = tx.prepare(
                "SELECT id, head_id, tail_id, COALESCE(canonical_relation, relation), edge_status
                 FROM relations WHERE store = ?1",
            )?;
            stmt.query_map(params![store], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))
            })?
            .collect::<Result<_, _>>()?
        };
        for (id, h, t, rel, status) in &relations {
            if rel == "isa" {
                classed.insert(*h);
            }
            if rel == "isa" || rel == "genls" || status == "implied" {
                continue;
            }
            by_entity.entry(*h).or_default().push(*id);
            by_entity.entry(*t).or_default().push(*id);
            let b = activation(rel_ages.get(id).map_or(&[][..], Vec::as_slice), cfg.d);
            let next = if b < cfg.tau && covered.contains(id) {
                "dormant"
            } else {
                "active"
            };
            self.apply_status(&tx, "relations", *id, b, status, next, &mut report)?;
        }

        let ent_ages = self.ages(&["entity", "class"], now)?;
        let entities: Vec<(i64, String)> = {
            let mut stmt = tx.prepare(
                "SELECT id, edge_status FROM entities WHERE store = ?1 AND kind = 'entity'",
            )?;
            stmt.query_map(params![store], |r| Ok((r.get(0)?, r.get(1)?)))?
                .collect::<Result<_, _>>()?
        };
        for (id, status) in &entities {
            if status == "implied" {
                continue;
            }
            let b = activation(ent_ages.get(id).map_or(&[][..], Vec::as_slice), cfg.d);
            let gist = classed.contains(id)
                && by_entity
                    .get(id)
                    .is_some_and(|rs| !rs.is_empty() && rs.iter().all(|r| covered.contains(r)));
            let next = if b < cfg.tau && gist {
                "dormant"
            } else {
                "active"
            };
            self.apply_status(&tx, "entities", *id, b, status, next, &mut report)?;
        }
        tx.commit()?;
        Ok(report)
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_status(
        &self,
        tx: &rusqlite::Transaction<'_>,
        table: &str,
        id: i64,
        b: f64,
        from: &str,
        to: &str,
        report: &mut SalienceReport,
    ) -> Result<(), GraphRagError> {
        let salience = if b.is_finite() { Some(b) } else { None };
        tx.execute(
            &format!("UPDATE {table} SET salience = ?2, edge_status = ?3 WHERE id = ?1"),
            params![id, salience, to],
        )?;
        report.refreshed += 1;
        match (from, to) {
            ("active", "dormant") => report.went_dormant += 1,
            ("dormant", "active") => report.revived += 1,
            _ => {}
        }
        Ok(())
    }
}
