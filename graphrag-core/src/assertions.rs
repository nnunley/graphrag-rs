//! Asserted edges: who said it, when, why, in what context. Retraction
//! keeps the record. Entity merges are `same_as` assertions, so un-merging
//! is a retraction and loses nothing.

use crate::db::Database;
use crate::error::GraphRagError;
use rusqlite::{OptionalExtension, params};
use std::collections::HashSet;

/// An actor-prefixed asserter: `human:<name>` or `agent:<name>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Asserter(pub String);

impl Asserter {
    pub fn new(s: &str) -> Result<Self, GraphRagError> {
        let ok = ["human:", "agent:"].iter().any(|p| {
            s.strip_prefix(p)
                .is_some_and(|rest| !rest.trim().is_empty())
        });
        if ok {
            Ok(Self(s.to_string()))
        } else {
            Err(GraphRagError::InvalidInput(format!(
                "asserter must be human:<name> or agent:<name>, got {s:?}"
            )))
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssertionRecord {
    pub id: i64,
    pub head_id: i64,
    pub relation: String,
    pub tail_id: i64,
    pub asserter: String,
    pub reason: Option<String>,
    pub context: Option<String>,
    pub asserted_at: String,
    pub retracted_at: Option<String>,
    pub retracted_by: Option<String>,
}

const COLUMNS: &str = "id, head_id, relation, tail_id, asserter, reason, context, asserted_at, retracted_at, retracted_by";

fn record(r: &rusqlite::Row<'_>) -> rusqlite::Result<AssertionRecord> {
    Ok(AssertionRecord {
        id: r.get(0)?,
        head_id: r.get(1)?,
        relation: r.get(2)?,
        tail_id: r.get(3)?,
        asserter: r.get(4)?,
        reason: r.get(5)?,
        context: r.get(6)?,
        asserted_at: r.get(7)?,
        retracted_at: r.get(8)?,
        retracted_by: r.get(9)?,
    })
}

impl Database {
    /// Assert `head -[relation]-> tail`. An identical active assertion (same
    /// endpoints, relation and context) is returned instead of duplicated.
    pub fn assert_edge(
        &self,
        head: i64,
        relation: &str,
        tail: i64,
        by: &Asserter,
        reason: Option<&str>,
        context: Option<&str>,
    ) -> Result<i64, GraphRagError> {
        if let Some(id) = self
            .conn()
            .query_row(
                "SELECT id FROM assertions
                 WHERE head_id = ?1 AND relation = ?2 AND tail_id = ?3 AND context IS ?4
                   AND retracted_at IS NULL",
                params![head, relation, tail, context],
                |r| r.get(0),
            )
            .optional()?
        {
            return Ok(id);
        }
        self.conn().execute(
            "INSERT INTO assertions (head_id, relation, tail_id, asserter, reason, context)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![head, relation, tail, by.0, reason, context],
        )?;
        Ok(self.conn().last_insert_rowid())
    }

    /// Retract an assertion. The record stays, with who retracted it and when.
    pub fn retract_edge(&self, assertion_id: i64, by: &Asserter) -> Result<(), GraphRagError> {
        let n = self.conn().execute(
            "UPDATE assertions
             SET retracted_at = strftime('%Y-%m-%dT%H:%M:%fZ','now'), retracted_by = ?2
             WHERE id = ?1 AND retracted_at IS NULL",
            params![assertion_id, by.0],
        )?;
        if n == 0 {
            return Err(GraphRagError::InvalidInput(format!(
                "assertion {assertion_id} is not active"
            )));
        }
        Ok(())
    }

    pub fn assertion(&self, id: i64) -> Result<AssertionRecord, GraphRagError> {
        Ok(self.conn().query_row(
            &format!("SELECT {COLUMNS} FROM assertions WHERE id = ?1"),
            params![id],
            record,
        )?)
    }

    /// Active assertions with `node` at either end.
    pub fn active_assertions(&self, node: i64) -> Result<Vec<AssertionRecord>, GraphRagError> {
        let mut stmt = self.conn().prepare(&format!(
            "SELECT {COLUMNS} FROM assertions
             WHERE (head_id = ?1 OR tail_id = ?1) AND retracted_at IS NULL ORDER BY id"
        ))?;
        let rows = stmt.query_map(params![node], record)?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// Merge `from` into `into` by asserting `from -[same_as]-> into`.
    /// Rows are untouched; reads resolve through [`Database::resolve_entity`].
    pub fn merge_as_same(&self, from: i64, into: i64, by: &Asserter) -> Result<i64, GraphRagError> {
        if from == into || self.resolve_entity(into)? == from {
            return Err(GraphRagError::InvalidInput(format!(
                "same_as {from} -> {into} would form a cycle"
            )));
        }
        self.assert_edge(from, "same_as", into, by, None, None)
    }

    /// Follow active `same_as` assertions to the end of the chain.
    pub fn resolve_entity(&self, id: i64) -> Result<i64, GraphRagError> {
        let mut seen = HashSet::from([id]);
        let mut cur = id;
        loop {
            let next: Option<i64> = self
                .conn()
                .query_row(
                    "SELECT tail_id FROM assertions
                     WHERE head_id = ?1 AND relation = 'same_as' AND retracted_at IS NULL
                     ORDER BY id DESC LIMIT 1",
                    params![cur],
                    |r| r.get(0),
                )
                .optional()?;
            match next {
                Some(n) if seen.insert(n) => cur = n,
                Some(_) => {
                    return Err(GraphRagError::Other(format!("same_as cycle through {id}")));
                }
                None => return Ok(cur),
            }
        }
    }
}
