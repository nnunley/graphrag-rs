//! Community runs. One path for every surface: weighted hierarchical Leiden
//! over the current projection, identified by the projection's fingerprint.
//! An unchanged graph reuses the current run; a changed one replaces it in
//! one transaction, so there are never left-over runs. Membership never
//! comes from an LLM.

use crate::db::Database;
use crate::error::GraphRagError;
use crate::fingerprints::{CommunityConfig, graph_fp};
use crate::status::{Step, StepState, SubjectKind};
use crate::token::fingerprint_text;
use rusqlite::{OptionalExtension, params};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq)]
pub struct RunReport {
    pub run_id: i64,
    pub reused: bool,
    pub communities: usize,
    pub levels: i32,
    pub modularity: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CommunityRunRecord {
    pub run_id: i64,
    pub store: String,
    pub graph_fp: String,
    pub config_json: String,
    pub seed: u64,
    pub modularity: Option<f64>,
    pub created_at: String,
}

impl Database {
    pub fn current_run(&self, store: &str) -> Result<Option<CommunityRunRecord>, GraphRagError> {
        Ok(self
            .conn()
            .query_row(
                "SELECT run_id, store, graph_fp, config_json, seed, modularity, created_at
                 FROM community_runs WHERE store = ?1 AND current = 1",
                params![store],
                |r| {
                    Ok(CommunityRunRecord {
                        run_id: r.get(0)?,
                        store: r.get(1)?,
                        graph_fp: r.get(2)?,
                        config_json: r.get(3)?,
                        seed: r.get::<_, i64>(4)? as u64,
                        modularity: r.get(5)?,
                        created_at: r.get(6)?,
                    })
                },
            )
            .optional()?)
    }

    fn run_shape(&self, run_id: i64) -> Result<(usize, i32), GraphRagError> {
        Ok(self.conn().query_row(
            "SELECT COUNT(*), COALESCE(MAX(level) + 1, 0) FROM communities WHERE run_id = ?1",
            params![run_id],
            |r| Ok((r.get::<_, i64>(0)? as usize, r.get(1)?)),
        )?)
    }

    pub fn build_community_run(
        &self,
        store: &str,
        cfg: &CommunityConfig,
    ) -> Result<RunReport, GraphRagError> {
        let projection = self.project_graph(store, cfg)?;
        let fp = graph_fp(&projection.edges, &projection.isolates, cfg, cfg.seed);
        if let Some(cur) = self.current_run(store)?.filter(|c| c.graph_fp == fp) {
            self.record_step(
                SubjectKind::Graph,
                0,
                Step::Communities,
                &fp,
                StepState::Current,
                None,
            )?;
            let (communities, levels) = self.run_shape(cur.run_id)?;
            return Ok(RunReport {
                run_id: cur.run_id,
                reused: true,
                communities,
                levels,
                modularity: cur.modularity.unwrap_or(0.0),
            });
        }

        // `leiden_hierarchical` counts levels from 0; cap the total at max_levels.
        let result = projection.graph.leiden_hierarchical(
            None,
            1e-4,
            cfg.min_size_to_split,
            Some((cfg.max_levels - 1).max(0)),
        );
        let modularity = result
            .communities
            .iter()
            .find(|c| c.level == 0)
            .map_or(0.0, |c| c.modularity);

        let tx = self.conn().unchecked_transaction()?;
        tx.execute(
            "INSERT INTO community_runs (store, graph_fp, config_json, seed, modularity, current)
             VALUES (?1, ?2, ?3, ?4, ?5, 0)",
            params![
                store,
                fp,
                serde_json::to_string(cfg)?,
                cfg.seed as i64,
                modularity
            ],
        )?;
        let run_id = tx.last_insert_rowid();
        let mut ids: Vec<i64> = Vec::with_capacity(result.communities.len());
        let mut levels = 0;
        for c in &result.communities {
            levels = levels.max(c.level + 1);
            let members: BTreeSet<i64> = c.nodes.iter().copied().collect();
            let internal: Vec<String> = projection
                .edges
                .iter()
                .filter(|(a, b, _)| members.contains(a) && members.contains(b))
                .map(|(a, b, w)| format!("{a}-{b}:{w}"))
                .collect();
            let member_list: Vec<String> = members.iter().map(i64::to_string).collect();
            let cfp =
                fingerprint_text(&format!("{}|{}", member_list.join(","), internal.join(",")));
            tx.execute(
                "INSERT INTO communities (store, level, parent_id, modularity, run_id, fp)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    store,
                    c.level,
                    c.parent_index.map(|p| ids[p]),
                    c.modularity,
                    run_id,
                    cfp
                ],
            )?;
            let id = tx.last_insert_rowid();
            for n in &members {
                tx.execute(
                    "INSERT INTO entity_communities (entity_id, community_id) VALUES (?1, ?2)",
                    params![n, id],
                )?;
            }
            ids.push(id);
        }
        // Replace whatever came before, including ad hoc pre-v2 communities.
        tx.execute(
            "DELETE FROM step_status WHERE subject_kind = 'community' AND subject_id IN
                 (SELECT id FROM communities WHERE store = ?1 AND (run_id IS NULL OR run_id != ?2))",
            params![store, run_id],
        )?;
        tx.execute(
            "DELETE FROM communities WHERE store = ?1 AND (run_id IS NULL OR run_id != ?2)",
            params![store, run_id],
        )?;
        tx.execute(
            "DELETE FROM community_runs WHERE store = ?1 AND run_id != ?2",
            params![store, run_id],
        )?;
        tx.execute(
            "UPDATE community_runs SET current = 1 WHERE run_id = ?1",
            params![run_id],
        )?;
        self.record_step(
            SubjectKind::Graph,
            0,
            Step::Communities,
            &fp,
            StepState::Current,
            None,
        )?;
        tx.commit()?;
        Ok(RunReport {
            run_id,
            reused: false,
            communities: ids.len(),
            levels,
            modularity,
        })
    }
}
