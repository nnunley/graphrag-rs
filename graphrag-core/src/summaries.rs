//! Community summaries, bottom-up. A leaf is summarized from its members,
//! the weighted edges between them, a few grounding excerpts and the groups
//! those excerpts come from; a parent from its children's summaries. Each
//! summary is keyed by the fingerprint of those inputs, so an unchanged
//! community in a new run reuses its summary without an LLM call.
//! Communities with fewer than 3 active members are `skipped_small`.

use crate::db::Database;
use crate::error::GraphRagError;
use crate::fingerprints::{SummaryInput, summary_fp};
use crate::prompt::ChatPrompt;
use crate::status::{Step, StepState, SubjectKind};
use rusqlite::{OptionalExtension, params};
use std::collections::{BTreeMap, BTreeSet};

pub const SUMMARY_PROMPT_VERSION: &str = "summary-v1";
const MIN_MEMBERS: usize = 3;
const DESCRIPTION_BUDGET: usize = 300;
const EXCERPT_CHARS: usize = 300;
const EXCERPTS: usize = 3;

#[derive(Debug, Clone, PartialEq)]
pub enum SummaryPlan {
    /// Already complete: `skipped_small` or `reused` (recorded by planning).
    Skip {
        community_id: i64,
        reason: &'static str,
    },
    Call {
        community_id: i64,
        fp: String,
        prompt: ChatPrompt,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum SummaryGateError {
    #[error("summary is empty")]
    Empty,
    #[error("summary is too short")]
    TooShort,
    #[error("summary names no member entity")]
    Ungrounded,
    #[error("summary inputs changed since planning")]
    Stale,
    #[error(transparent)]
    Db(#[from] GraphRagError),
}

impl From<rusqlite::Error> for SummaryGateError {
    fn from(e: rusqlite::Error) -> Self {
        SummaryGateError::Db(e.into())
    }
}

/// Everything one community's summary is written from.
struct Gathered {
    input: SummaryInput,
    /// Group → number of member-mentioning chunks from it.
    group_counts: BTreeMap<String, usize>,
    excerpts: Vec<String>,
    /// `(child id, child summary)` for a parent.
    children: Vec<(i64, String)>,
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(max).collect::<String>())
    }
}

const SYSTEM: &str = "You summarize one community of a knowledge graph for later retrieval.
Write 2-4 sentences of plain prose. State the community's theme and name its key member entities.
Ground every statement only in the material provided. When the evidence is thin, prefer a shorter, honest summary.
No preamble, no headings, no lists.";

impl Database {
    fn community_children(&self, id: i64) -> Result<Vec<i64>, GraphRagError> {
        let mut stmt = self
            .conn()
            .prepare("SELECT id FROM communities WHERE parent_id = ?1 ORDER BY id")?;
        Ok(stmt
            .query_map(params![id], |r| r.get(0))?
            .collect::<Result<_, _>>()?)
    }

    fn summary_status_fp(&self, id: i64) -> Result<Option<String>, GraphRagError> {
        Ok(self
            .conn()
            .query_row(
                "SELECT fp FROM step_status WHERE subject_kind = 'community' AND subject_id = ?1
                   AND step = 'summary' AND state = 'current'",
                params![id],
                |r| r.get(0),
            )
            .optional()?)
    }

    fn gather(&self, community_id: i64) -> Result<Gathered, GraphRagError> {
        let members: Vec<(i64, String, String)> = {
            let mut stmt = self.conn().prepare(
                "SELECT e.id, e.name, COALESCE(e.entity_type, 'other') FROM entity_communities ec
                 JOIN entities e ON e.id = ec.entity_id
                 WHERE ec.community_id = ?1 AND e.kind = 'entity' AND e.edge_status != 'dormant'
                 ORDER BY e.name, e.id",
            )?;
            stmt.query_map(params![community_id], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })?
            .collect::<Result<_, _>>()?
        };
        let ids: BTreeSet<i64> = members.iter().map(|m| m.0).collect();
        let names: BTreeMap<i64, String> = members.iter().map(|m| (m.0, m.1.clone())).collect();

        let mut input_members = Vec::new();
        for (id, name, ty) in &members {
            let mut stmt = self.conn().prepare(
                "SELECT DISTINCT ae.description FROM attempt_entities ae JOIN attempts a ON a.attempt_fp = ae.attempt_fp
                 WHERE ae.entity_id = ?1 AND ae.description != '' ORDER BY a.created_at DESC",
            )?;
            let descs: Vec<String> = stmt
                .query_map(params![id], |r| r.get(0))?
                .collect::<Result<_, _>>()?;
            input_members.push((
                name.clone(),
                ty.clone(),
                truncate(&descs.join("; "), DESCRIPTION_BUDGET),
            ));
        }

        let mut edges = Vec::new();
        {
            let mut stmt = self.conn().prepare(
                "SELECT r.id, r.head_id, r.tail_id, COALESCE(r.canonical_relation, r.relation),
                        COUNT(DISTINCT ar.evidence_kind || ':' || ar.evidence_id)
                 FROM relations r JOIN attempt_relations ar ON ar.relation_id = r.id
                 JOIN entity_communities h ON h.entity_id = r.head_id AND h.community_id = ?1
                 JOIN entity_communities t ON t.entity_id = r.tail_id AND t.community_id = ?1
                 WHERE r.edge_status != 'dormant' GROUP BY r.id",
            )?;
            let rows: Vec<(i64, i64, i64, String, i64)> = stmt
                .query_map(params![community_id], |r| {
                    Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))
                })?
                .collect::<Result<_, _>>()?;
            for (rid, h, t, rel, support) in rows {
                let (Some(hn), Some(tn)) = (names.get(&h), names.get(&t)) else {
                    continue;
                };
                let desc: Option<String> = self
                    .conn()
                    .query_row(
                        "SELECT ar.description FROM attempt_relations ar JOIN attempts a ON a.attempt_fp = ar.attempt_fp
                         WHERE ar.relation_id = ?1 AND ar.description != '' ORDER BY a.created_at DESC LIMIT 1",
                        params![rid],
                        |r| r.get(0),
                    )
                    .optional()?;
                edges.push((
                    hn.clone(),
                    tn.clone(),
                    rel,
                    support as f64,
                    truncate(&desc.unwrap_or_default(), 200),
                ));
            }
            edges.sort_by(|a, b| (&a.0, &a.1, &a.2).cmp(&(&b.0, &b.1, &b.2)));
        }

        // Chunks that mention members, most members first.
        let mut chunk_hits: BTreeMap<i64, usize> = BTreeMap::new();
        for id in &ids {
            let mut stmt = self
                .conn()
                .prepare("SELECT chunk_id FROM chunk_entities WHERE entity_id = ?1")?;
            for c in stmt.query_map(params![id], |r| r.get::<_, i64>(0))? {
                *chunk_hits.entry(c?).or_default() += 1;
            }
        }
        let mut ranked: Vec<(i64, usize)> = chunk_hits.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut group_counts: BTreeMap<String, usize> = BTreeMap::new();
        let mut excerpts = Vec::new();
        for (chunk, _) in &ranked {
            let node: Option<i64> = self.conn().query_row(
                "SELECT d.node_id FROM chunks c JOIN documents d ON d.id = c.document_id WHERE c.id = ?1",
                params![chunk],
                |r| r.get(0),
            )?;
            if let Some(n) = node {
                for g in self.groups_of(n)? {
                    *group_counts.entry(g).or_default() += 1;
                }
            }
            if excerpts.len() < EXCERPTS {
                excerpts.push(truncate(self.chunk_text(*chunk)?.trim(), EXCERPT_CHARS));
            }
        }

        let mut children = Vec::new();
        let mut child_fps = Vec::new();
        for child in self.community_children(community_id)? {
            child_fps.push(self.summary_status_fp(child)?.unwrap_or_default());
            let s: Option<String> = self.conn().query_row(
                "SELECT summary FROM communities WHERE id = ?1",
                params![child],
                |r| r.get(0),
            )?;
            if let Some(s) = s.filter(|s| !s.is_empty()) {
                children.push((child, s));
            }
        }
        let groups: Vec<String> = group_counts.keys().cloned().collect();
        Ok(Gathered {
            input: SummaryInput {
                members: input_members,
                edges,
                child_fps,
                groups,
            },
            group_counts,
            excerpts,
            children,
        })
    }

    /// The fingerprint a community's summary must carry to be current.
    pub fn summary_expected_fp(
        &self,
        community_id: i64,
        model: &str,
    ) -> Result<String, GraphRagError> {
        Ok(summary_fp(
            &self.gather(community_id)?.input,
            model,
            SUMMARY_PROMPT_VERSION,
        ))
    }

    fn summary_prompt(&self, g: &Gathered) -> ChatPrompt {
        let mut user = String::new();
        if g.children.is_empty() {
            user.push_str("Members:\n");
            for (name, ty, desc) in &g.input.members {
                user.push_str(&format!("- {name} ({ty}): {desc}\n"));
            }
            if !g.input.edges.is_empty() {
                user.push_str("\nRelationships (weight = independent sources stating it):\n");
                for (h, t, r, w, d) in &g.input.edges {
                    user.push_str(&format!("- {h} -[{r}]-> {t} (weight {w}): {d}\n"));
                }
            }
            if !g.excerpts.is_empty() {
                user.push_str("\nExcerpts:\n");
                for e in &g.excerpts {
                    user.push_str(&format!("> {e}\n"));
                }
            }
        } else {
            user.push_str("Sub-community summaries:\n");
            for (id, s) in &g.children {
                user.push_str(&format!("- [{id}] {s}\n"));
            }
            let names: Vec<&str> = g
                .input
                .members
                .iter()
                .map(|m| m.0.as_str())
                .take(20)
                .collect();
            user.push_str(&format!("\nMembers include: {}\n", names.join(", ")));
        }
        let mut system = SYSTEM.to_string();
        if g.group_counts.len() > 1 {
            let groups: Vec<String> = g
                .group_counts
                .iter()
                .map(|(n, c)| format!("{n} ({c} excerpts)"))
                .collect();
            user.push_str(&format!(
                "\nThe evidence comes from these groups: {}\n",
                groups.join(", ")
            ));
            system.push_str(
                "\nThe evidence comes from more than one group; say which groups it mostly comes from (for example \"mostly from papers, with three notes\").",
            );
        } else if let Some(g) = g.group_counts.keys().next() {
            user.push_str(&format!("\nThe evidence comes from the group: {g}\n"));
        }
        ChatPrompt {
            system,
            user,
            max_tokens: 2000,
        }
    }

    /// Plan summaries for the current run, bottom-up. Skips (small or
    /// reused) are recorded here, so they count as complete immediately; a
    /// parent is planned only when every child is complete.
    pub fn plan_summaries(
        &self,
        store: &str,
        model: &str,
        limit: usize,
    ) -> Result<Vec<SummaryPlan>, GraphRagError> {
        let communities: Vec<i64> = {
            let mut stmt = self.conn().prepare(
                "SELECT c.id FROM communities c JOIN community_runs r ON r.run_id = c.run_id
                 WHERE r.store = ?1 AND r.current = 1 ORDER BY c.level DESC, c.id",
            )?;
            stmt.query_map(params![store], |r| r.get(0))?
                .collect::<Result<_, _>>()?
        };
        let mut out = Vec::new();
        let mut calls = 0;
        for id in communities {
            let g = self.gather(id)?;
            let fp = summary_fp(&g.input, model, SUMMARY_PROMPT_VERSION);
            if self.summary_status_fp(id)?.as_deref() == Some(fp.as_str()) {
                continue;
            }
            let children = self.community_children(id)?;
            let mut ready = true;
            for c in &children {
                if self
                    .summary_status_fp(*c)?
                    .is_none_or(|f| f != self.summary_expected_fp(*c, model).unwrap_or_default())
                {
                    ready = false;
                    break;
                }
            }
            if !ready {
                continue;
            }
            if g.input.members.len() < MIN_MEMBERS {
                self.record_step(
                    SubjectKind::Community,
                    id,
                    Step::Summary,
                    &fp,
                    StepState::Current,
                    Some("skipped_small"),
                )?;
                out.push(SummaryPlan::Skip {
                    community_id: id,
                    reason: "skipped_small",
                });
                continue;
            }
            let existing: Option<String> = self
                .conn()
                .query_row(
                    "SELECT summary FROM summaries WHERE community_fp = ?1",
                    params![fp],
                    |r| r.get(0),
                )
                .optional()?;
            if let Some(text) = existing {
                self.conn().execute(
                    "UPDATE communities SET summary = ?2 WHERE id = ?1",
                    params![id, text],
                )?;
                self.record_step(
                    SubjectKind::Community,
                    id,
                    Step::Summary,
                    &fp,
                    StepState::Current,
                    None,
                )?;
                out.push(SummaryPlan::Skip {
                    community_id: id,
                    reason: "reused",
                });
                continue;
            }
            if calls < limit {
                calls += 1;
                out.push(SummaryPlan::Call {
                    community_id: id,
                    fp,
                    prompt: self.summary_prompt(&g),
                });
            }
        }
        Ok(out)
    }

    /// Store a generated summary after the mechanical gates.
    pub fn apply_summary(
        &self,
        community_id: i64,
        fp: &str,
        text: &str,
        model: &str,
    ) -> Result<(), SummaryGateError> {
        let text = text.trim();
        if text.is_empty() {
            return Err(SummaryGateError::Empty);
        }
        if text.chars().count() <= 40 {
            return Err(SummaryGateError::TooShort);
        }
        let g = self.gather(community_id)?;
        if summary_fp(&g.input, model, SUMMARY_PROMPT_VERSION) != fp {
            return Err(SummaryGateError::Stale);
        }
        let lower = text.to_lowercase();
        if !g
            .input
            .members
            .iter()
            .any(|(name, _, _)| lower.contains(&name.to_lowercase()))
        {
            return Err(SummaryGateError::Ungrounded);
        }
        let tx = self.write_tx()?;
        tx.execute(
            "INSERT OR REPLACE INTO summaries (community_fp, summary, model, prompt_version) VALUES (?1, ?2, ?3, ?4)",
            params![fp, text, model, SUMMARY_PROMPT_VERSION],
        )?;
        tx.execute(
            "UPDATE communities SET summary = ?2 WHERE id = ?1",
            params![community_id, text],
        )?;
        self.record_step(
            SubjectKind::Community,
            community_id,
            Step::Summary,
            fp,
            StepState::Current,
            None,
        )?;
        tx.commit()?;
        Ok(())
    }
}
