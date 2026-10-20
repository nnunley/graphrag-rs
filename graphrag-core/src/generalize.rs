//! Generalization: entities and relations work their way up to classes.
//!
//! - Names match on base forms (`llms` ≡ `llm`).
//! - Classes are canonicalized as they arrive: exact, then by embedding
//!   similarity (recorded as an alias), else a new live class.
//! - Relations have a hierarchy (`relation_genls`, Cyc's genlPreds).
//! - Lifted class edges: every instance relation adds its distinct pair to
//!   `class(a) -[r']-> class(b)` for r and each broader r', at each class
//!   level above the information-content floor. A candidate becomes an
//!   edge only when it beats chance (G² on a 2×2 table of distinct pairs).
//!   Closures (genls chains, co-membership) are never stored.

use crate::db::Database;
use crate::error::GraphRagError;
use crate::status::{Step, StepState, SubjectKind};
use crate::synonyms::normalize_entity_name;
use crate::token::{fingerprint_parts, fingerprint_text};
use rusqlite::{OptionalExtension, params};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// Words whose trailing `s` is not a plural.
const SINGULAR_EXCEPTIONS: &[&str] = &[
    "kubernetes",
    "nixos",
    "redis",
    "analysis",
    "status",
    "postgres",
    "series",
    "news",
    "species",
    "basis",
    "thesis",
    "axis",
    "bus",
    "gas",
    "ios",
    "macos",
    "jenkins",
    "pandas",
    "sass",
    "less",
    "class",
    "process",
    "access",
    "address",
    "business",
    "canvas",
    "alias",
    "corpus",
    "virus",
];

fn singular(word: &str, in_phrase: bool) -> Option<String> {
    if word.chars().count() <= 3 || !word.chars().all(char::is_alphabetic) {
        return None;
    }
    let lower = word.to_lowercase();
    if SINGULAR_EXCEPTIONS.contains(&lower.as_str()) {
        return None;
    }
    let (stem, last) = word.split_at(word.len() - 1);
    // Acronym plural: LLMs, APIs.
    if last == "s" && stem.chars().all(char::is_uppercase) {
        return Some(stem.to_string());
    }
    // A capitalized single word is usually a name (Pandas, Postgres): leave
    // it. As the head of a phrase ("Knowledge Graphs") it is a common noun.
    let title_case = word.chars().skip(1).eq(lower.chars().skip(1));
    if word != lower && !(in_phrase && title_case) {
        return None;
    }
    let strip = |n: usize| word[..word.len() - n].to_string();
    if lower.ends_with("ies") && lower.len() > 4 {
        return Some(format!("{}y", strip(3)));
    }
    if lower.ends_with("sses") || lower.ends_with("xes") || lower.ends_with("shes") {
        return Some(strip(2));
    }
    if lower.ends_with("ss") || lower.ends_with("us") || lower.ends_with("is") {
        return None;
    }
    lower.ends_with('s').then(|| strip(1))
}

/// Singularize the head (last) noun: `knowledge graphs` → `knowledge graph`.
pub fn base_form(name: &str) -> String {
    let name = name.trim();
    match name.rsplit_once(' ') {
        Some((head, last)) => match singular(last, true) {
            Some(s) => format!("{head} {s}"),
            None => name.to_string(),
        },
        None => singular(name, false).unwrap_or_else(|| name.to_string()),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LiftConfig {
    pub min_pairs: usize,
    pub g2_min: f64,
    /// Classes with information content below this are never lift targets.
    pub ic_floor: f64,
}

impl Default for LiftConfig {
    fn default() -> Self {
        Self {
            min_pairs: 3,
            g2_min: 10.83,
            ic_floor: 0.5,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LiftReport {
    pub candidates: usize,
    pub admitted: usize,
    pub newly_admitted: usize,
    pub retired: usize,
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// G² for a 2×2 table `[[k11, k12], [k21, k22]]`, plus the expected k11.
fn g_squared(k11: f64, k12: f64, k21: f64, k22: f64) -> (f64, f64) {
    let n = k11 + k12 + k21 + k22;
    if n == 0.0 {
        return (0.0, 0.0);
    }
    let (r1, r2) = (k11 + k12, k21 + k22);
    let (c1, c2) = (k11 + k21, k12 + k22);
    let cells = [
        (k11, r1 * c1 / n),
        (k12, r1 * c2 / n),
        (k21, r2 * c1 / n),
        (k22, r2 * c2 / n),
    ];
    let g2 = 2.0
        * cells
            .iter()
            .filter(|(k, e)| *k > 0.0 && *e > 0.0)
            .map(|(k, e)| k * (k / e).ln())
            .sum::<f64>();
    (g2, r1 * c1 / n)
}

/// A lifted-edge candidate: `(relation, head class, tail class)`.
type Key = (String, i64, i64);
/// Its distinct instance pairs and the instance relations behind them.
type Candidate = (BTreeSet<(i64, i64)>, BTreeSet<i64>);

/// The class graph: direct classes per instance and parents per class.
struct Classes {
    isa: HashMap<i64, BTreeSet<i64>>,
    genls: HashMap<i64, BTreeSet<i64>>,
}

impl Classes {
    fn ancestors(&self, class: i64) -> BTreeSet<i64> {
        let mut out = BTreeSet::from([class]);
        let mut stack = vec![class];
        while let Some(c) = stack.pop() {
            for p in self.genls.get(&c).into_iter().flatten() {
                if out.insert(*p) {
                    stack.push(*p);
                }
            }
        }
        out
    }

    /// Every class an instance is under, through isa then genls.
    fn classes_of(&self, entity: i64) -> BTreeSet<i64> {
        self.isa
            .get(&entity)
            .into_iter()
            .flatten()
            .flat_map(|c| self.ancestors(*c))
            .collect()
    }

    /// `-ln(instances under c / all instances)` for every class.
    fn information_content(&self) -> HashMap<i64, f64> {
        let total = self.isa.len() as f64;
        let mut under: HashMap<i64, usize> = HashMap::new();
        for e in self.isa.keys() {
            for c in self.classes_of(*e) {
                *under.entry(c).or_default() += 1;
            }
        }
        under
            .into_iter()
            .map(|(c, n)| (c, -(n as f64 / total).ln()))
            .collect()
    }
}

impl Database {
    /// The class node for `kind`: an alias match, else an entity with the
    /// same name key, else a new class (`entity_type = 'collection'`).
    pub fn resolve_class(&self, store: &str, kind: &str) -> Result<i64, GraphRagError> {
        let key = normalize_entity_name(kind);
        if let Some(id) = self.class_alias(store, &key)? {
            return Ok(id);
        }
        let id = self.get_or_create_entity(store, kind, None, None)?;
        self.conn().execute(
            "UPDATE entities SET entity_type = 'collection' WHERE id = ?1 AND entity_type IS NULL",
            params![id],
        )?;
        Ok(id)
    }

    fn class_alias(&self, store: &str, key: &str) -> Result<Option<i64>, GraphRagError> {
        Ok(self
            .conn()
            .query_row(
                "SELECT a.class_id FROM class_aliases a JOIN entities e ON e.id = a.class_id
                 WHERE a.alias = ?1 AND e.store = ?2",
                params![key, store],
                |r| r.get(0),
            )
            .optional()?)
    }

    /// Like [`Database::resolve_class`], but a new kind whose embedding is
    /// within `min_cos` of an existing class maps to it (recorded as an
    /// alias, so exact resolution finds it next time).
    pub fn resolve_class_similar(
        &self,
        store: &str,
        kind: &str,
        embed: &dyn Fn(&str) -> Vec<f32>,
        min_cos: f32,
    ) -> Result<i64, GraphRagError> {
        let key = normalize_entity_name(kind);
        if let Some(id) = self.class_alias(store, &key)? {
            return Ok(id);
        }
        let exact: Option<i64> = self
            .conn()
            .query_row(
                "SELECT id FROM entities WHERE store = ?1 AND kind = 'entity' AND name_key = ?2",
                params![store, key],
                |r| r.get(0),
            )
            .optional()?;
        if exact.is_some() {
            return self.resolve_class(store, kind);
        }
        let classes: Vec<(i64, String)> = {
            let mut stmt = self.conn().prepare(
                "SELECT id, name FROM entities WHERE store = ?1 AND kind = 'entity' AND entity_type = 'collection'",
            )?;
            stmt.query_map(params![store], |r| Ok((r.get(0)?, r.get(1)?)))?
                .collect::<Result<_, _>>()?
        };
        let query = embed(kind);
        let mut best: Option<(i64, f32)> = None;
        for (id, name) in classes {
            let v = match self.get_entity_embedding(id)? {
                Some(v) => v,
                None => {
                    let v = embed(&name);
                    self.set_entity_embedding(id, &v)?;
                    v
                }
            };
            let s = cosine(&query, &v);
            if best.is_none_or(|(_, b)| s > b) {
                best = Some((id, s));
            }
        }
        if let Some((id, _)) = best.filter(|(_, s)| *s >= min_cos) {
            self.conn().execute(
                "INSERT OR IGNORE INTO class_aliases (alias, class_id) VALUES (?1, ?2)",
                params![key, id],
            )?;
            return Ok(id);
        }
        let id = self.resolve_class(store, kind)?;
        self.set_entity_embedding(id, &query)?;
        Ok(id)
    }

    /// Class names to show the extractor: the top `n` by member count, plus
    /// the classes of entities already extracted from `document_id`.
    pub fn known_classes(
        &self,
        store: &str,
        document_id: Option<i64>,
        n: usize,
    ) -> Result<Vec<String>, GraphRagError> {
        let mut out: Vec<String> = {
            let mut stmt = self.conn().prepare(
                "SELECT c.name FROM relations r JOIN entities c ON c.id = r.tail_id
                 WHERE c.store = ?1 AND COALESCE(r.canonical_relation, r.relation) = 'isa'
                 GROUP BY c.id ORDER BY COUNT(DISTINCT r.head_id) DESC, c.name LIMIT ?2",
            )?;
            stmt.query_map(params![store, n as i64], |r| r.get(0))?
                .collect::<Result<_, _>>()?
        };
        if let Some(d) = document_id {
            let mut stmt = self.conn().prepare(
                "SELECT DISTINCT c.name FROM attempt_entities ae
                 JOIN attempts a ON a.attempt_fp = ae.attempt_fp
                 JOIN chunks ch ON ch.id = a.chunk_id
                 JOIN relations r ON r.head_id = ae.entity_id AND COALESCE(r.canonical_relation, r.relation) = 'isa'
                 JOIN entities c ON c.id = r.tail_id
                 WHERE ch.document_id = ?1 ORDER BY c.name",
            )?;
            for name in stmt.query_map(params![d], |r| r.get::<_, String>(0))? {
                let name = name?;
                if !out.contains(&name) {
                    out.push(name);
                }
            }
        }
        Ok(out)
    }

    /// Extracted relation names that map to no canonical relation and have
    /// no broader relation yet: what the generalize step asks the LLM about.
    pub fn unmapped_relation_names(&self, store: &str) -> Result<Vec<String>, GraphRagError> {
        let mut stmt = self.conn().prepare(
            "SELECT DISTINCT r.relation FROM relations r
             WHERE r.store = ?1 AND r.canonical_relation IS NULL
               AND NOT EXISTS (SELECT 1 FROM relation_genls g WHERE g.relation = r.relation)
             ORDER BY r.relation",
        )?;
        Ok(stmt
            .query_map(params![store], |r| r.get(0))?
            .collect::<Result<_, _>>()?)
    }

    pub fn relation_broader(&self, relation: &str) -> Result<Option<String>, GraphRagError> {
        Ok(self
            .conn()
            .query_row(
                "SELECT broader FROM relation_genls WHERE relation = ?1",
                params![relation],
                |r| r.get(0),
            )
            .optional()?)
    }

    pub fn set_relation_broader(
        &self,
        relation: &str,
        broader: &str,
        origin: &str,
    ) -> Result<(), GraphRagError> {
        self.conn().execute(
            "INSERT INTO relation_genls (relation, broader, origin) VALUES (?1, ?2, ?3)
             ON CONFLICT(relation) DO UPDATE SET broader = excluded.broader, origin = excluded.origin",
            params![relation, broader, origin],
        )?;
        Ok(())
    }

    fn relation_levels(&self, relation: &str, genls: &HashMap<String, String>) -> Vec<String> {
        let mut out = vec![relation.to_string()];
        let mut cur = relation;
        while let Some(b) = genls.get(cur) {
            if out.contains(b) {
                break;
            }
            out.push(b.clone());
            cur = b;
        }
        out
    }

    fn load_classes(&self, store: &str) -> Result<Classes, GraphRagError> {
        let same = self.same_as_resolved()?;
        let rep = |id: i64| *same.get(&id).unwrap_or(&id);
        let mut isa: HashMap<i64, BTreeSet<i64>> = HashMap::new();
        let mut genls: HashMap<i64, BTreeSet<i64>> = HashMap::new();
        let mut stmt = self.conn().prepare(
            "SELECT r.head_id, r.tail_id, COALESCE(r.canonical_relation, r.relation) FROM relations r
             WHERE r.store = ?1 AND COALESCE(r.canonical_relation, r.relation) IN ('isa', 'genls')
               AND EXISTS (SELECT 1 FROM attempt_relations ar WHERE ar.relation_id = r.id)",
        )?;
        for row in stmt.query_map(params![store], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, String>(2)?,
            ))
        })? {
            let (h, t, rel) = row?;
            let (h, t) = (rep(h), rep(t));
            if h == t {
                continue;
            }
            let map = if rel == "isa" { &mut isa } else { &mut genls };
            map.entry(h).or_default().insert(t);
        }
        Ok(Classes { isa, genls })
    }

    fn same_as_resolved(&self) -> Result<HashMap<i64, i64>, GraphRagError> {
        let mut stmt = self.conn().prepare(
            "SELECT head_id FROM assertions WHERE relation = 'same_as' AND retracted_at IS NULL",
        )?;
        let heads: Vec<i64> = stmt
            .query_map([], |r| r.get(0))?
            .collect::<Result<_, _>>()?;
        let mut out = HashMap::new();
        for h in heads {
            out.insert(h, self.resolve_entity(h)?);
        }
        Ok(out)
    }

    /// `-ln(p(c))`, p = share of class instances under `class_id`.
    pub fn information_content(&self, class_id: i64) -> Result<f64, GraphRagError> {
        let store: String = self.conn().query_row(
            "SELECT store FROM entities WHERE id = ?1",
            params![class_id],
            |r| r.get(0),
        )?;
        let ic = self.load_classes(&store)?.information_content();
        Ok(*ic.get(&class_id).unwrap_or(&f64::INFINITY))
    }

    /// `(relation id, head, tail, relation key)` for supported, non-class
    /// instance relations (dormant ones included: their evidence still counts).
    fn instance_relations(
        &self,
        store: &str,
    ) -> Result<Vec<(i64, i64, i64, String)>, GraphRagError> {
        let same = self.same_as_resolved()?;
        let rep = |id: i64| *same.get(&id).unwrap_or(&id);
        let mut stmt = self.conn().prepare(
            "SELECT r.id, r.head_id, r.tail_id, COALESCE(r.canonical_relation, r.relation) FROM relations r
             WHERE r.store = ?1 AND COALESCE(r.canonical_relation, r.relation) NOT IN ('isa', 'genls')
               AND EXISTS (SELECT 1 FROM attempt_relations ar WHERE ar.relation_id = r.id)
             ORDER BY r.id",
        )?;
        let rows = stmt.query_map(params![store], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, i64>(2)?,
                r.get::<_, String>(3)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (id, h, t, rel) = row?;
            out.push((id, rep(h), rep(t), rel));
        }
        Ok(out)
    }

    /// Fingerprint of everything the generalize step reads.
    pub fn generalize_input_fp(
        &self,
        store: &str,
        cfg: &LiftConfig,
    ) -> Result<String, GraphRagError> {
        let mut parts: Vec<Vec<u8>> = vec![b"graphrag-step-generalize-v1".to_vec()];
        let mut stmt = self.conn().prepare(
            "SELECT r.id, r.head_id, r.tail_id, COALESCE(r.canonical_relation, r.relation) FROM relations r
             WHERE r.store = ?1 AND EXISTS (SELECT 1 FROM attempt_relations ar WHERE ar.relation_id = r.id)
             ORDER BY r.id",
        )?;
        for row in stmt.query_map(params![store], |r| {
            Ok(format!(
                "{}:{}:{}:{}",
                r.get::<_, i64>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, i64>(2)?,
                r.get::<_, String>(3)?
            ))
        })? {
            parts.push(row?.into_bytes());
        }
        parts.push(b"|genls".to_vec());
        let mut stmt = self
            .conn()
            .prepare("SELECT relation || '>' || broader FROM relation_genls ORDER BY relation")?;
        for row in stmt.query_map([], |r| r.get::<_, String>(0))? {
            parts.push(row?.into_bytes());
        }
        parts.push(b"|aliases".to_vec());
        let mut stmt = self
            .conn()
            .prepare("SELECT alias || '>' || class_id FROM class_aliases ORDER BY alias")?;
        for row in stmt.query_map([], |r| r.get::<_, String>(0))? {
            parts.push(row?.into_bytes());
        }
        parts.push(b"|same_as".to_vec());
        let mut stmt = self.conn().prepare(
            "SELECT head_id || '>' || tail_id FROM assertions WHERE relation = 'same_as' AND retracted_at IS NULL ORDER BY id",
        )?;
        for row in stmt.query_map([], |r| r.get::<_, String>(0))? {
            parts.push(row?.into_bytes());
        }
        parts.push(
            format!(
                "{}|{}|{}",
                cfg.min_pairs,
                cfg.g2_min.to_bits(),
                cfg.ic_floor.to_bits()
            )
            .into_bytes(),
        );
        let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
        Ok(fingerprint_parts(&refs))
    }

    /// Recompute lifted class edges. A row is rewritten only when its
    /// distinct-pair set or its admission decision changes.
    pub fn lift(&self, store: &str, cfg: &LiftConfig) -> Result<LiftReport, GraphRagError> {
        let input_fp = self.generalize_input_fp(store, cfg)?;
        let classes = self.load_classes(store)?;
        let ic = classes.information_content();
        let eligible = |e: i64| -> BTreeSet<i64> {
            classes
                .classes_of(e)
                .into_iter()
                .filter(|c| ic.get(c).is_some_and(|v| *v >= cfg.ic_floor))
                .collect()
        };
        let genls: HashMap<String, String> = {
            let mut stmt = self
                .conn()
                .prepare("SELECT relation, broader FROM relation_genls")?;
            stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
                .collect::<Result<_, _>>()?
        };

        let instances = self.instance_relations(store)?;
        let mut all_pairs: HashSet<(i64, i64)> = HashSet::new();
        let mut pairs_by_rel: HashMap<String, HashSet<(i64, i64)>> = HashMap::new();
        let mut cands: BTreeMap<Key, Candidate> = BTreeMap::new();
        let mut anc_cache: HashMap<i64, BTreeSet<i64>> = HashMap::new();
        for (rid, h, t, rel) in &instances {
            all_pairs.insert((*h, *t));
            let levels = self.relation_levels(rel, &genls);
            for r in &levels {
                pairs_by_rel.entry(r.clone()).or_default().insert((*h, *t));
            }
            let ha = anc_cache.entry(*h).or_insert_with(|| eligible(*h)).clone();
            let ta = anc_cache.entry(*t).or_insert_with(|| eligible(*t)).clone();
            for r in &levels {
                for c1 in &ha {
                    for c2 in &ta {
                        let e = cands.entry((r.clone(), *c1, *c2)).or_default();
                        e.0.insert((*h, *t));
                        e.1.insert(*rid);
                    }
                }
            }
        }
        // Distinct pairs (any relation) between each class pair.
        let mut class_pairs: HashMap<(i64, i64), usize> = HashMap::new();
        for (h, t) in &all_pairs {
            let ha = anc_cache.entry(*h).or_insert_with(|| eligible(*h)).clone();
            let ta = anc_cache.entry(*t).or_insert_with(|| eligible(*t)).clone();
            for c1 in &ha {
                for c2 in &ta {
                    *class_pairs.entry((*c1, *c2)).or_default() += 1;
                }
            }
        }
        let n_total = all_pairs.len() as f64;

        let tx = self.conn().unchecked_transaction()?;
        let existing: HashMap<Key, (i64, String, bool)> = {
            let mut stmt = tx.prepare(
                "SELECT l.relation, l.head_class, l.tail_class, l.id, l.fp, l.admitted FROM lifted_edges l
                 JOIN entities e ON e.id = l.head_class WHERE e.store = ?1",
            )?;
            stmt.query_map(params![store], |r| {
                Ok((
                    (r.get(0)?, r.get(1)?, r.get(2)?),
                    (r.get(3)?, r.get(4)?, r.get(5)?),
                ))
            })?
            .collect::<Result<_, _>>()?
        };
        let mut report = LiftReport {
            candidates: cands.len(),
            ..LiftReport::default()
        };
        for (key, (pairs, rel_ids)) in &cands {
            let n = pairs.len() as f64;
            let r_total = pairs_by_rel.get(&key.0).map_or(0, HashSet::len) as f64;
            let c_total = *class_pairs.get(&(key.1, key.2)).unwrap_or(&0) as f64;
            let (g2, expected) =
                g_squared(n, r_total - n, c_total - n, n_total - r_total - c_total + n);
            let admitted = pairs.len() >= cfg.min_pairs && g2 >= cfg.g2_min && n > expected;
            if admitted {
                report.admitted += 1;
            }
            let fp = fingerprint_text(
                &pairs
                    .iter()
                    .map(|(a, b)| format!("{a}>{b}"))
                    .collect::<Vec<_>>()
                    .join(","),
            );
            let id = match existing.get(key) {
                Some((id, old_fp, old_admitted)) if *old_fp == fp && *old_admitted == admitted => {
                    continue;
                }
                Some((id, _, old_admitted)) => {
                    if admitted && !old_admitted {
                        report.newly_admitted += 1;
                    }
                    tx.execute(
                        "UPDATE lifted_edges SET pairs = ?2, g2 = ?3, admitted = ?4, fp = ?5,
                             updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id = ?1",
                        params![id, pairs.len() as i64, g2, admitted, fp],
                    )?;
                    tx.execute("DELETE FROM lifted_from WHERE lifted_id = ?1", params![id])?;
                    *id
                }
                None => {
                    if admitted {
                        report.newly_admitted += 1;
                    }
                    tx.execute(
                        "INSERT INTO lifted_edges (relation, head_class, tail_class, pairs, g2, admitted, fp)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                        params![key.0, key.1, key.2, pairs.len() as i64, g2, admitted, fp],
                    )?;
                    tx.last_insert_rowid()
                }
            };
            for rid in rel_ids {
                tx.execute(
                    "INSERT OR IGNORE INTO lifted_from (lifted_id, relation_id) VALUES (?1, ?2)",
                    params![id, rid],
                )?;
            }
        }
        for (key, (id, _, _)) in &existing {
            if !cands.contains_key(key) {
                report.retired +=
                    tx.execute("DELETE FROM lifted_edges WHERE id = ?1", params![id])?;
            }
        }
        self.record_step(
            SubjectKind::Graph,
            0,
            Step::Generalize,
            &input_fp,
            StepState::Current,
            None,
        )?;
        tx.commit()?;
        Ok(report)
    }
}
