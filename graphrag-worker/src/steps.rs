//! The drain loop and one handler per step.

use crate::{ExitStatus, WorkerConfig, WorkerLlm};
use graphrag_core::attempts::{AttemptOutcome, AttemptRecord};
use graphrag_core::status::{StaticConfig, Step, StepState, SubjectKind, WorkItem};
use graphrag_core::summaries::{SummaryGateError, SummaryPlan};
use graphrag_core::synonyms::canonical_relations;
use graphrag_core::text_embedder::TextEmbedder;
use graphrag_core::{Database, GraphRagError};
use graphrag_llm::openai::{LlmErrorKind, classify_error};
use graphrag_llm::{ChatPrompt, paper_json};
use serde_json::{Value, json};

/// Malformed responses (or 4xx) tolerated per item before it is `failed`.
const MAX_MALFORMED: u32 = 3;
/// Longest backoff after transport failures.
const MAX_BACKOFF_SECS: u64 = 3600;

pub(crate) struct Driver<'a> {
    cfg: &'a WorkerConfig,
    llm: &'a dyn WorkerLlm,
    emb: &'a dyn TextEmbedder,
    db: Database,
    sc: StaticConfig,
    processed: usize,
    consecutive_transport: u32,
}

enum Outcome {
    Continue,
    Stop(ExitStatus),
}

impl<'a> Driver<'a> {
    pub(crate) fn open(
        cfg: &'a WorkerConfig,
        llm: &'a dyn WorkerLlm,
        emb: &'a dyn TextEmbedder,
    ) -> Result<Self, GraphRagError> {
        let db = Database::open(&cfg.db_path())?;
        Ok(Self {
            sc: cfg.static_config(emb),
            cfg,
            llm,
            emb,
            db,
            processed: 0,
            consecutive_transport: 0,
        })
    }

    pub(crate) fn run(&mut self) -> Result<ExitStatus, GraphRagError> {
        let now = self.db.now_utc()?;
        self.db.set_worker_state("last_run_started_at", &now)?;
        let status = self.run_inner()?;
        let now = self.db.now_utc()?;
        self.db.set_worker_state("last_run_finished_at", &now)?;
        Ok(status)
    }

    fn run_inner(&mut self) -> Result<ExitStatus, GraphRagError> {
        // Model pin: nothing runs against a model the endpoint doesn't serve.
        // Local steps (embed, re-chunk) still run, so documents stay searchable.
        let pin_error = match self.llm.served_models() {
            Ok(models) if models.iter().any(|m| m == &self.cfg.model) => None,
            Ok(models) => Some(format!(
                "model {} not served (served: {})",
                self.cfg.model,
                models.join(", ")
            )),
            Err(e) => Some(format!("LLM endpoint unavailable: {e}")),
        };
        if let Some(msg) = pin_error {
            eprintln!("{msg}; running local steps only");
            self.db.set_worker_state("last_llm_error", &msg)?;
            self.local_only()?;
            return Ok(ExitStatus::Unavailable);
        }
        if self.cfg.retry_failed {
            self.db.clear_failed_steps()?;
        }
        // Re-extract guard: a config change that supersedes >10% of extractions
        // has to be acknowledged.
        let report = self.db.status_report(&self.sc)?;
        let stale = *report.total.stale.get(&Step::Extract).unwrap_or(&0);
        let total: usize = [
            &report.total.current,
            &report.total.stale,
            &report.total.missing,
            &report.total.failed,
        ]
        .iter()
        .map(|m| *m.get(&Step::Extract).unwrap_or(&0))
        .sum();
        if stale * 10 > total && !self.cfg.accept_reextract {
            eprintln!(
                "{stale} of {total} chunk extractions would be superseded; rerun with --accept-reextract"
            );
            return Ok(ExitStatus::Usage);
        }

        loop {
            if self.budget_left() == 0 {
                return Ok(ExitStatus::Done);
            }
            let now = self.db.now_utc()?;
            let batch = self.cfg.concurrency.max(1) * 4;
            let items = self.db.next_work(&self.sc, &now, batch)?;
            if items.is_empty() {
                return Ok(ExitStatus::Done);
            }
            let outcome = self.dispatch(items)?;
            if let Outcome::Stop(s) = outcome {
                return Ok(s);
            }
        }
    }

    /// Drain only the steps that need no LLM: re-chunk and embed.
    fn local_only(&mut self) -> Result<(), GraphRagError> {
        loop {
            if self.budget_left() == 0 {
                return Ok(());
            }
            let now = self.db.now_utc()?;
            let items: Vec<WorkItem> = self
                .db
                .next_work(&self.sc, &now, self.cfg.concurrency.max(1) * 8)?
                .into_iter()
                .filter(|w| w.kind == SubjectKind::Document || w.step == Step::Embed)
                .collect();
            if items.is_empty() {
                return Ok(());
            }
            self.dispatch(items)?;
        }
    }

    fn budget_left(&self) -> usize {
        self.cfg
            .max_items
            .map_or(usize::MAX, |m| m.saturating_sub(self.processed))
    }

    fn dispatch(&mut self, items: Vec<WorkItem>) -> Result<Outcome, GraphRagError> {
        // Re-chunking can delete chunk ids that later items reference: do it
        // alone, then re-query.
        let docs: Vec<&WorkItem> = items
            .iter()
            .filter(|w| w.kind == SubjectKind::Document)
            .collect();
        if !docs.is_empty() {
            for w in docs.into_iter().take(self.budget_left()) {
                let profile = self.db.document_chunker(w.id, &self.cfg.chunker)?;
                self.db.rechunk_document(w.id, &profile)?;
                self.processed += 1;
            }
            return Ok(Outcome::Continue);
        }
        let first = &items[0];
        match (first.kind, first.step) {
            (SubjectKind::Chunk, _) => self.chunk_items(&items),
            (SubjectKind::Graph, Step::Generalize) => {
                self.generalize()?;
                self.processed += 1;
                Ok(Outcome::Continue)
            }
            (SubjectKind::Graph, Step::Communities) => {
                self.db
                    .build_community_run(&self.cfg.store, &self.cfg.community)?;
                self.processed += 1;
                Ok(Outcome::Continue)
            }
            (SubjectKind::Community, Step::Summary) => self.summaries(),
            (kind, step) => Err(GraphRagError::Other(format!(
                "no handler for {kind:?}/{step:?}"
            ))),
        }
    }

    fn chunk_items(&mut self, items: &[WorkItem]) -> Result<Outcome, GraphRagError> {
        let mut extracts: Vec<&WorkItem> = Vec::new();
        for w in items.iter().filter(|w| w.kind == SubjectKind::Chunk) {
            if self.budget_left() == 0 {
                break;
            }
            match w.step {
                Step::Embed => {
                    let text = self.db.chunk_text(w.id)?;
                    let v = self.emb.embed(&text)?;
                    self.db.set_chunk_embedding(w.id, &v)?;
                    self.db.record_step(
                        SubjectKind::Chunk,
                        w.id,
                        Step::Embed,
                        &w.expected_fp,
                        StepState::Current,
                        None,
                    )?;
                    self.processed += 1;
                }
                Step::Extract if extracts.len() < self.cfg.concurrency.max(1) => {
                    extracts.push(w);
                    self.processed += 1;
                }
                _ => {}
            }
        }
        if extracts.is_empty() {
            return Ok(Outcome::Continue);
        }
        let mut prompts = Vec::with_capacity(extracts.len());
        for w in &extracts {
            let ctx = self.db.chunk_context(w.id)?;
            let classes = self.db.known_classes(
                &self.cfg.store,
                Some(ctx.document_id),
                self.cfg.known_classes,
            )?;
            prompts.push(paper_json::build_prompt(
                &ctx.text,
                &ctx.heading_path,
                &classes,
            ));
        }
        let schema = paper_json::response_schema();
        let responses = self.parallel(&prompts, Some(&schema));
        for (w, resp) in extracts.into_iter().zip(responses) {
            if let Outcome::Stop(s) = self.handle_extraction(w, resp)? {
                return Ok(Outcome::Stop(s));
            }
        }
        Ok(Outcome::Continue)
    }

    /// Run prompts concurrently; results come back in prompt order.
    fn parallel(
        &self,
        prompts: &[ChatPrompt],
        schema: Option<&Value>,
    ) -> Vec<Result<String, String>> {
        let llm = self.llm;
        std::thread::scope(|s| {
            let handles: Vec<_> = prompts
                .iter()
                .map(|p| s.spawn(move || llm.complete_with_format(p, schema)))
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err("transport: worker thread panicked".into()))
                })
                .collect()
        })
    }

    fn llm_ok(&mut self) -> Result<(), GraphRagError> {
        self.consecutive_transport = 0;
        let now = self.db.now_utc()?;
        self.db.set_worker_state("last_llm_ok_at", &now)
    }

    /// Record a failed LLM call. Transport failures back off and count toward
    /// an outage; anything else is retried up to `MAX_MALFORMED` times.
    fn llm_failed(
        &mut self,
        kind: SubjectKind,
        id: i64,
        step: Step,
        fp: &str,
        err: &str,
    ) -> Result<Outcome, GraphRagError> {
        self.db.set_worker_state("last_llm_error", err)?;
        if classify_error(err) == LlmErrorKind::Transport {
            let before = self.db.step_attempts(kind, id, step)?;
            let delay = 2u64.saturating_pow(before + 1).min(MAX_BACKOFF_SECS);
            let next = self.db.utc_after_secs(delay)?;
            self.db
                .record_transient_failure(kind, id, step, err, &next)?;
            self.consecutive_transport += 1;
            if self.consecutive_transport >= self.cfg.outage_threshold {
                eprintln!(
                    "LLM outage after {} consecutive transport failures: {err}",
                    self.consecutive_transport
                );
                return Ok(Outcome::Stop(ExitStatus::Outage));
            }
            return Ok(Outcome::Continue);
        }
        let now = self.db.now_utc()?;
        let attempts = self
            .db
            .record_transient_failure(kind, id, step, err, &now)?;
        if attempts >= MAX_MALFORMED {
            self.db
                .record_step(kind, id, step, fp, StepState::Failed, Some(err))?;
        }
        Ok(Outcome::Continue)
    }

    fn handle_extraction(
        &mut self,
        w: &WorkItem,
        resp: Result<String, String>,
    ) -> Result<Outcome, GraphRagError> {
        let raw = match resp {
            Ok(raw) => raw,
            Err(e) => {
                return self.llm_failed(
                    SubjectKind::Chunk,
                    w.id,
                    Step::Extract,
                    &w.expected_fp,
                    &e,
                );
            }
        };
        let (outcome, rejected) = match paper_json::parse(&raw) {
            Ok(p) => p,
            Err(e) => {
                return self.llm_failed(
                    SubjectKind::Chunk,
                    w.id,
                    Step::Extract,
                    &w.expected_fp,
                    &e,
                );
            }
        };
        self.llm_ok()?;
        // Map new class names onto existing classes by embedding before the
        // commit resolves them exactly.
        if let AttemptOutcome::Ok { entities, .. } = &outcome {
            let emb = self.emb;
            let embed = |s: &str| emb.embed(s).unwrap_or_default();
            for e in entities {
                for name in [&e.kind, &e.broader].into_iter().flatten() {
                    self.db.resolve_class_similar(
                        &self.cfg.store,
                        name,
                        &embed,
                        self.cfg.class_min_cos,
                    )?;
                }
            }
        }
        self.db.commit_attempt(&AttemptRecord {
            chunk_id: w.id,
            attempt_fp: &w.expected_fp,
            model: &self.cfg.model,
            profile: paper_json::PROFILE,
            prompt_version: &self.cfg.prompt_version,
            schema_version: paper_json::SCHEMA_VERSION,
            glean_max_rounds: 0,
            raw_response: &raw,
            outcome,
            rejected_entities: rejected,
        })?;
        Ok(Outcome::Continue)
    }

    /// Propose broader relations for unmapped ones (best effort), lift class
    /// edges, refresh salience.
    fn generalize(&mut self) -> Result<(), GraphRagError> {
        let unmapped = self.db.unmapped_relation_names(&self.cfg.store)?;
        if !unmapped.is_empty() {
            let canon = canonical_relations();
            let prompt = ChatPrompt {
                system: "You map relation names from a knowledge graph to a broader relation. \
                         For each relation, give the closest broader relation from the allowed list, \
                         or omit it when none fits. Return JSON only."
                    .into(),
                user: format!(
                    "Allowed broader relations: {}\n\nRelations to map:\n{}",
                    canon.join(", "),
                    unmapped.iter().map(|r| format!("- {r}")).collect::<Vec<_>>().join("\n")
                ),
                max_tokens: 4000,
            };
            let schema = json!({"type": "object", "properties": {"mappings": {"type": "array", "items": {
                "type": "object", "properties": {"relation": {"type": "string"}, "broader": {"type": "string"}},
                "required": ["relation", "broader"]}}}, "required": ["mappings"]});
            if let Ok(raw) = self.llm.complete_with_format(&prompt, Some(&schema))
                && let Some(maps) = serde_json::from_str::<Value>(&raw)
                    .ok()
                    .and_then(|v| v["mappings"].as_array().cloned())
            {
                for m in maps {
                    let (Some(r), Some(b)) = (m["relation"].as_str(), m["broader"].as_str()) else {
                        continue;
                    };
                    if unmapped.iter().any(|u| u == r) && canon.contains(&b) && r != b {
                        self.db.set_relation_broader(r, b, "llm")?;
                    }
                }
            }
        }
        self.db.lift(&self.cfg.store, &self.cfg.lift)?;
        let now = self.db.now_utc()?;
        self.db
            .refresh_salience(&self.cfg.store, &self.cfg.salience, &now)?;
        Ok(())
    }

    fn summaries(&mut self) -> Result<Outcome, GraphRagError> {
        let limit = self.cfg.concurrency.max(1).min(self.budget_left());
        let plans = self
            .db
            .plan_summaries(&self.cfg.store, &self.cfg.model, limit)?;
        let calls: Vec<(i64, String, ChatPrompt)> = plans
            .iter()
            .filter_map(|p| match p {
                SummaryPlan::Call {
                    community_id,
                    fp,
                    prompt,
                } => Some((*community_id, fp.clone(), prompt.clone())),
                SummaryPlan::Skip { .. } => None,
            })
            .collect();
        if plans.is_empty() {
            // Remaining summaries wait on children that failed: nothing to do.
            return Ok(Outcome::Stop(ExitStatus::Done));
        }
        let prompts: Vec<ChatPrompt> = calls.iter().map(|c| c.2.clone()).collect();
        let responses = self.parallel(&prompts, None);
        for ((id, fp, _), resp) in calls.into_iter().zip(responses) {
            self.processed += 1;
            let text = match resp {
                Ok(t) => t,
                Err(e) => {
                    if let Outcome::Stop(s) =
                        self.llm_failed(SubjectKind::Community, id, Step::Summary, &fp, &e)?
                    {
                        return Ok(Outcome::Stop(s));
                    }
                    continue;
                }
            };
            self.llm_ok()?;
            match self.db.apply_summary(id, &fp, &text, &self.cfg.model) {
                Ok(()) => {}
                Err(SummaryGateError::Db(e)) => return Err(e),
                Err(gate) => {
                    self.llm_failed(
                        SubjectKind::Community,
                        id,
                        Step::Summary,
                        &fp,
                        &format!("gate: {gate}"),
                    )?;
                }
            }
        }
        Ok(Outcome::Continue)
    }
}
