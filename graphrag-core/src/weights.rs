//! The weighted graph Leiden clusters. An extracted edge weighs its support
//! (distinct evidence units with a current attempt). Asserted, derived
//! `mentions`, structural `has_part` and class edges contribute only through
//! their config weights, all 0 by default, so by default clusters come from
//! text alone. `same_as` merges contract to one node; dormant items and
//! retracted assertions are left out.

use crate::db::Database;
use crate::error::GraphRagError;
use crate::fingerprints::CommunityConfig;
use crate::leiden::CommunityGraph;
use rusqlite::params;
use std::collections::{BTreeMap, BTreeSet, HashMap};

pub struct Projection {
    pub graph: CommunityGraph,
    /// Undirected `(low id, high id, weight)`, sorted.
    pub edges: Vec<(i64, i64, f64)>,
    /// Entity nodes with no projected edge, sorted.
    pub isolates: Vec<i64>,
}

impl Database {
    /// Active `same_as` map, resolved to chain ends in memory.
    fn same_as_map(&self) -> Result<HashMap<i64, i64>, GraphRagError> {
        let mut stmt = self.conn().prepare(
            "SELECT head_id, tail_id FROM assertions
             WHERE relation = 'same_as' AND retracted_at IS NULL ORDER BY id",
        )?;
        let direct: HashMap<i64, i64> = stmt
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<Result<_, _>>()?;
        let mut out = HashMap::new();
        for &start in direct.keys() {
            let mut cur = start;
            let mut seen = BTreeSet::from([start]);
            while let Some(&n) = direct.get(&cur) {
                if !seen.insert(n) {
                    break;
                }
                cur = n;
            }
            out.insert(start, cur);
        }
        Ok(out)
    }

    pub fn project_graph(
        &self,
        store: &str,
        cfg: &CommunityConfig,
    ) -> Result<Projection, GraphRagError> {
        let same = self.same_as_map()?;
        let rep = |id: i64| *same.get(&id).unwrap_or(&id);

        // Node kinds and dormancy for every row in the store.
        let mut kind: HashMap<i64, (String, bool)> = HashMap::new();
        {
            let mut stmt = self.conn().prepare(
                "SELECT id, kind, edge_status = 'dormant' FROM entities WHERE store = ?1",
            )?;
            for row in stmt.query_map(params![store], |r| {
                Ok((r.get::<_, i64>(0)?, r.get(1)?, r.get(2)?))
            })? {
                let (id, k, dormant) = row?;
                kind.insert(id, (k, dormant));
            }
        }
        let structural_on =
            cfg.asserted_weight > 0.0 || cfg.mention_weight > 0.0 || cfg.structure_weight > 0.0;
        let usable = |id: i64| -> Option<i64> {
            let r = rep(id);
            let (k, dormant) = kind.get(&r)?;
            if *dormant {
                return None;
            }
            (k == "entity" || structural_on).then_some(r)
        };

        let mut acc: BTreeMap<(i64, i64), f64> = BTreeMap::new();
        let mut add = |a: i64, b: i64, w: f64| {
            if a != b && w > 0.0 {
                *acc.entry((a.min(b), a.max(b))).or_default() += w;
            }
        };

        // Extracted relations, weighted by support.
        {
            let mut stmt = self.conn().prepare(
                "SELECT r.head_id, r.tail_id, COALESCE(r.canonical_relation, r.relation),
                        COUNT(DISTINCT ar.evidence_kind || ':' || ar.evidence_id)
                 FROM relations r JOIN attempt_relations ar ON ar.relation_id = r.id
                 WHERE r.store = ?1 AND r.edge_status = 'active'
                 GROUP BY r.id",
            )?;
            let rows = stmt.query_map(params![store], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, i64>(3)?,
                ))
            })?;
            for row in rows {
                let (h, t, rel, support) = row?;
                let w = if rel == "isa" || rel == "genls" {
                    cfg.class_weight * support as f64
                } else {
                    support as f64
                };
                if let (Some(h), Some(t)) = (usable(h), usable(t)) {
                    add(h, t, w);
                }
            }
        }

        // Admitted lifted class edges.
        if cfg.class_weight > 0.0 {
            let mut stmt = self.conn().prepare(
                "SELECT l.head_class, l.tail_class, l.pairs FROM lifted_edges l
                 JOIN entities e ON e.id = l.head_class WHERE e.store = ?1 AND l.admitted = 1",
            )?;
            for row in stmt.query_map(params![store], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, i64>(2)?,
                ))
            })? {
                let (h, t, pairs) = row?;
                if let (Some(h), Some(t)) = (usable(h), usable(t)) {
                    add(h, t, cfg.class_weight * pairs as f64);
                }
            }
        }

        if structural_on {
            // Assertions: structural has_part vs everything else (same_as is a merge, not an edge).
            let mut stmt = self.conn().prepare(
                "SELECT a.head_id, a.tail_id, a.relation, a.asserter FROM assertions a
                 JOIN entities e ON e.id = a.head_id
                 WHERE e.store = ?1 AND a.retracted_at IS NULL AND a.relation != 'same_as'",
            )?;
            let rows = stmt.query_map(params![store], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                ))
            })?;
            for row in rows {
                let (h, t, rel, by) = row?;
                let w = if rel == "has_part" && by == "agent:structure" {
                    cfg.structure_weight
                } else {
                    cfg.asserted_weight
                };
                if let (Some(h), Some(t)) = (usable(h), usable(t)) {
                    add(h, t, w);
                }
            }
            // Derived mentions: evidence node -> entity, weighted by the
            // number of that unit's chunks that mention it.
            if cfg.mention_weight > 0.0 {
                let mut stmt = self.conn().prepare(
                    "SELECT COALESCE(s.node_id, d.node_id), ae.entity_id, COUNT(DISTINCT a.chunk_id)
                     FROM attempt_entities ae
                     JOIN attempts a ON a.attempt_fp = ae.attempt_fp
                     JOIN chunks c ON c.id = a.chunk_id
                     JOIN documents d ON d.id = c.document_id
                     LEFT JOIN sections s ON s.id = c.section_id
                     WHERE c.store = ?1
                     GROUP BY 1, 2",
                )?;
                let rows = stmt.query_map(params![store], |r| {
                    Ok((
                        r.get::<_, Option<i64>>(0)?,
                        r.get::<_, i64>(1)?,
                        r.get::<_, i64>(2)?,
                    ))
                })?;
                for row in rows {
                    let (ev, e, n) = row?;
                    if let (Some(ev), Some(e)) = (ev.and_then(usable), usable(e)) {
                        add(ev, e, cfg.mention_weight * n as f64);
                    }
                }
            }
        }

        let edges: Vec<(i64, i64, f64)> = acc.into_iter().map(|((a, b), w)| (a, b, w)).collect();
        let connected: BTreeSet<i64> = edges.iter().flat_map(|e| [e.0, e.1]).collect();
        let mut isolates: Vec<i64> = kind
            .iter()
            .filter(|(id, (k, dormant))| {
                k == "entity" && !dormant && rep(**id) == **id && !connected.contains(id)
            })
            .map(|(id, _)| *id)
            .collect();
        isolates.sort_unstable();
        let graph =
            CommunityGraph::from_edges_and_nodes(edges.iter().copied(), isolates.iter().copied());
        Ok(Projection {
            graph,
            edges,
            isolates,
        })
    }
}
