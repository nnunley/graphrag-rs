//! Python bindings for the graphrag top-level surface.
//!
//! Exposes the operations an orchestrating notebook needs — search, graph
//! reads, and the map-reduce plan/apply loop — as typed Python objects.
//!
//! The point is not speed (IPC is ~1.5% of a work unit); it is that results
//! arrive as named fields instead of positionally-parsed CLI text. The former
//! `vec=` column was a bare float whose meaning silently inverted when the
//! vector backend changed from cosine distance to similarity; here it is
//! `Hit.similarity`, and it means one thing.

use graphrag_core::db::EntityInput;
use graphrag_core::mr::{
    ExtractPrompt, ExtractResult, ExtractionStrategy, SummaryResult, apply_extract,
    apply_summarize, plan_extract, plan_summarize,
};
use graphrag_core::{
    BruteForceVectorSource, Database, Embedder, LexicalIndex, VectorCandidateSource,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Mutex;

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Adapter from graphrag-llm's model-resolved strategy to the core trait.
struct LlmStrategy {
    inner: Box<dyn graphrag_llm::ExtractionStrategy>,
}

impl LlmStrategy {
    fn for_model(model: &str) -> Self {
        Self {
            inner: graphrag_llm::strategy_for_model(model),
        }
    }
}

impl ExtractionStrategy for LlmStrategy {
    fn prompt(&self, chunk: &str, idx: usize, total: usize) -> ExtractPrompt {
        let p = self.inner.build_prompt(chunk, idx, total);
        ExtractPrompt {
            system: p.system,
            user: p.user,
            format: self.inner.response_format(),
        }
    }
    fn parse(&self, response: &str) -> Vec<EntityInput> {
        self.inner
            .parse(response)
            .triples
            .into_iter()
            .map(|t| EntityInput {
                head: t.head,
                head_type: t.head_type,
                relation: t.relation,
                tail: t.tail,
                tail_type: t.tail_type,
                properties: None,
            })
            .collect()
    }
}

/// One hybrid-search hit. `similarity` is cosine similarity (higher = closer)
/// and is `None` for lexical-only matches that carry no vector evidence.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct Hit {
    pub chunk_id: i64,
    pub text: String,
    pub source: Option<String>,
    pub rrf: f64,
    pub similarity: Option<f32>,
    pub lexical: Option<f32>,
}

#[pymethods]
impl Hit {
    fn __repr__(&self) -> String {
        let sim = self
            .similarity
            .map(|s| format!("{s:.3}"))
            .unwrap_or_else(|| "-".into());
        format!(
            "Hit(chunk_id={}, similarity={}, rrf={:.4}, text={:?})",
            self.chunk_id,
            sim,
            self.rrf,
            self.text.chars().take(48).collect::<String>()
        )
    }
}

/// A graphrag store: the top-level handle.
#[pyclass]
pub struct Store {
    /// SQLite connections are `Send` but not `Sync`; the mutex makes the
    /// handle safe to hold from a Python thread pool, which is how the
    /// orchestrating notebook drives map-reduce stages.
    db: Mutex<Database>,
    name: String,
}

#[pymethods]
impl Store {
    /// Open (or create) the database under `data_dir`.
    #[new]
    #[pyo3(signature = (data_dir, store = "conversations"))]
    fn new(data_dir: &str, store: &str) -> PyResult<Self> {
        let dir = std::path::PathBuf::from(shellexpand(data_dir));
        std::fs::create_dir_all(&dir).map_err(err)?;
        let db = Database::open(&dir.join("graphrag.db")).map_err(err)?;
        Ok(Self {
            db: Mutex::new(db),
            name: store.to_string(),
        })
    }

    fn __repr__(&self) -> String {
        format!("Store(store={:?})", self.name)
    }

    /// Row counts for the active store.
    fn stats<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyDict>> {
        let db = self.db.lock().map_err(|e| err(e.to_string()))?;
        let d = PyDict::new(py);
        d.set_item("store", &self.name)?;
        d.set_item("chunks", db.list_chunks(&self.name).map_err(err)?.len())?;
        d.set_item("entities", db.list_entities(&self.name).map_err(err)?.len())?;
        d.set_item(
            "relations",
            db.list_relations(&self.name).map_err(err)?.len(),
        )?;
        d.set_item(
            "communities",
            db.list_communities(&self.name).map_err(err)?.len(),
        )?;
        d.set_item(
            "pending_extract",
            db.pending_extraction_chunks(&self.name, None)
                .map_err(err)?
                .len(),
        )?;
        d.set_item(
            "pending_summarize",
            db.pending_summary_communities(&self.name, None)
                .map_err(err)?
                .len(),
        )?;
        Ok(d)
    }

    /// Hybrid search: exact cosine + BM25, fused with RRF.
    #[pyo3(signature = (query, top = 5))]
    fn search(&self, query: &str, top: usize) -> PyResult<Vec<Hit>> {
        let db = self.db.lock().map_err(|e| err(e.to_string()))?;
        let rec = db.get_store(&self.name).map_err(err)?;
        let embedder = Embedder::new(Default::default()).map_err(err)?;
        let embedding = embedder.embed(query).map_err(err)?;
        if embedding.len() != rec.dim {
            return Err(PyValueError::new_err(format!(
                "dimension mismatch: embedder {} vs store {}",
                embedding.len(),
                rec.dim
            )));
        }
        let k = top.max(5) * 4;
        let vsrc = BruteForceVectorSource::for_store(&db, &self.name).map_err(err)?;
        let vhits = vsrc.top_candidates(&embedding, k).map_err(err)?;
        let chunks = db.list_chunks(&self.name).map_err(err)?;
        let lhits = match LexicalIndex::build_from_chunks(&chunks).map_err(err)? {
            Some(ix) => ix.search(query, k).map_err(err)?,
            None => Vec::new(),
        };

        use graphrag_core::leit_fusion::{RankedResult, fuse_default};
        let vr: Vec<RankedResult> = vhits
            .iter()
            .enumerate()
            .map(|(i, h)| RankedResult::new(h.chunk_id.to_string(), i + 1))
            .collect();
        let lr: Vec<RankedResult> = lhits
            .iter()
            .enumerate()
            .map(|(i, (id, _))| RankedResult::new(id.to_string(), i + 1))
            .collect();

        let mut out = Vec::new();
        for f in fuse_default(&[vr, lr]).into_iter().take(top) {
            let id: i64 = match f.id.parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let Some(c) = chunks.iter().find(|c| c.id == id) else {
                continue;
            };
            out.push(Hit {
                chunk_id: id,
                text: c.content.clone(),
                source: c.source.clone(),
                rrf: f.score,
                similarity: vhits.iter().find(|h| h.chunk_id == id).map(|h| h.score),
                lexical: lhits.iter().find(|(lid, _)| *lid == id).map(|(_, s)| *s),
            });
        }
        Ok(out)
    }

    /// Pending map-reduce units for `stage` ("extract" | "summarize").
    #[pyo3(signature = (stage, model, limit = None))]
    fn plan<'p>(
        &self,
        py: Python<'p>,
        stage: &str,
        model: &str,
        limit: Option<usize>,
    ) -> PyResult<Vec<Bound<'p, PyDict>>> {
        let db = self.db.lock().map_err(|e| err(e.to_string()))?;
        let units = match stage {
            "extract" => {
                let s = LlmStrategy::for_model(model);
                plan_extract(&db, &self.name, model, &s, limit).map_err(err)?
            }
            "summarize" => plan_summarize(&db, &self.name, model, limit).map_err(err)?,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown stage {other:?} (supported: extract, summarize)"
                )));
            }
        };
        units
            .into_iter()
            .map(|u| {
                let d = PyDict::new(py);
                d.set_item("unit_id", u.unit_id)?;
                d.set_item("kind", u.kind)?;
                d.set_item("chunk_id", u.chunk_id)?;
                d.set_item("community_id", u.community_id)?;
                d.set_item("model", u.model)?;
                d.set_item("system", u.system)?;
                d.set_item("user", u.user)?;
                d.set_item("format", u.format.map(|f| f.to_string()))?;
                Ok(d)
            })
            .collect()
    }

    /// Ingest executor results. Accepts them partially and out of order.
    fn apply(&self, stage: &str, results: Vec<Bound<'_, PyDict>>) -> PyResult<usize> {
        let get = |d: &Bound<'_, PyDict>, k: &str| -> PyResult<String> {
            Ok(d.get_item(k)?
                .map(|v| v.extract::<String>())
                .transpose()?
                .unwrap_or_default())
        };
        let db = self.db.lock().map_err(|e| err(e.to_string()))?;
        match stage {
            "extract" => {
                let mut rs = Vec::new();
                for d in &results {
                    let chunk_id: i64 = d
                        .get_item("chunk_id")?
                        .ok_or_else(|| PyValueError::new_err("result missing chunk_id"))?
                        .extract()?;
                    rs.push(ExtractResult {
                        chunk_id,
                        model: get(d, "model")?,
                        response: get(d, "response")?,
                    });
                }
                let model = rs.first().map(|r| r.model.clone()).unwrap_or_default();
                let s = LlmStrategy::for_model(&model);
                apply_extract(&db, &self.name, &s, &rs).map_err(err)
            }
            "summarize" => {
                let mut rs = Vec::new();
                for d in &results {
                    let community_id: i64 = d
                        .get_item("community_id")?
                        .ok_or_else(|| PyValueError::new_err("result missing community_id"))?
                        .extract()?;
                    rs.push(SummaryResult {
                        community_id,
                        model: get(d, "model")?,
                        response: get(d, "response")?,
                    });
                }
                apply_summarize(&db, &self.name, &rs).map_err(err)
            }
            other => Err(PyValueError::new_err(format!(
                "unknown stage {other:?} (supported: extract, summarize)"
            ))),
        }
    }

    /// Communities with their summaries.
    fn communities<'p>(&self, py: Python<'p>) -> PyResult<Vec<Bound<'p, PyDict>>> {
        let db = self.db.lock().map_err(|e| err(e.to_string()))?;
        db.list_communities(&self.name)
            .map_err(err)?
            .into_iter()
            .map(|c| {
                let d = PyDict::new(py);
                d.set_item("id", c.id)?;
                d.set_item("level", c.level)?;
                d.set_item("summary", c.summary)?;
                d.set_item("modularity", c.modularity)?;
                Ok(d)
            })
            .collect()
    }

    /// Relations as (head, relation, tail) triples.
    fn triples(&self) -> PyResult<Vec<(String, String, String)>> {
        let db = self.db.lock().map_err(|e| err(e.to_string()))?;
        let ents = db.list_entities(&self.name).map_err(err)?;
        let name_of = |id: i64| {
            ents.iter()
                .find(|e| e.id == id)
                .map(|e| e.name.clone())
                .unwrap_or_default()
        };
        Ok(db
            .list_relations(&self.name)
            .map_err(err)?
            .into_iter()
            .map(|r| (name_of(r.head_id), r.relation, name_of(r.tail_id)))
            .collect())
    }

    /// Export the store as JSON Lines.
    fn export(&self) -> PyResult<String> {
        let db = self.db.lock().map_err(|e| err(e.to_string()))?;
        let mut buf: Vec<u8> = Vec::new();
        graphrag_core::export::export_store(&db, &self.name, &mut buf).map_err(err)?;
        String::from_utf8(buf).map_err(err)
    }
}

fn shellexpand(p: &str) -> String {
    if let Some(rest) = p.strip_prefix("~/")
        && let Some(home) = std::env::var_os("HOME")
    {
        return std::path::Path::new(&home)
            .join(rest)
            .to_string_lossy()
            .into_owned();
    }
    p.to_string()
}

#[pymodule]
fn graphrag(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Store>()?;
    m.add_class::<Hit>()?;
    m.add(
        "__doc__",
        "Python bindings for the graphrag top-level surface.",
    )?;
    Ok(())
}
