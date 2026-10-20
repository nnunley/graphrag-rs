//! GraphRAG MCP Server
//!
//! A Model Context Protocol server providing knowledge graph memory tools.
//! Implements JSON-RPC 2.0 over stdio with sampling support for LLM-powered features.

use graphrag_core::documents::{AddDocument, AddOutcome, Origin};
use graphrag_core::fingerprints::ChunkerProfile;
use graphrag_core::status::{Step, SubjectKind};
use graphrag_core::{
    BruteForceVectorSource, Database, Embedder, EmbedderConfig, EmbedderModel, LexicalIndex,
    VectorCandidate, VectorCandidateSource, default_embedder_cache_dir,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::OnceLock;

/// Global embedder instance (lazy initialized)
static EMBEDDER: OnceLock<Result<Embedder, String>> = OnceLock::new();

/// The embedding model, from `GRAPHRAG_EMBED_MODEL` (default Nomic).
fn embedder_model() -> EmbedderModel {
    std::env::var("GRAPHRAG_EMBED_MODEL")
        .ok()
        .and_then(|m| match m.to_lowercase().as_str() {
            "minilm" | "mini" => Some(EmbedderModel::MiniLM),
            "nomic" | "nomic-embed-text" => Some(EmbedderModel::NomicEmbedText),
            _ => None,
        })
        .unwrap_or(EmbedderModel::NomicEmbedText)
}

/// Get or initialize the embedder
fn get_embedder() -> Result<&'static Embedder, String> {
    EMBEDDER
        .get_or_init(|| {
            eprintln!("Initializing embedder (this may download the model on first use)...");
            let config = EmbedderConfig {
                model: embedder_model(),
                show_download_progress: true,
                cache_dir: Some(default_embedder_cache_dir()),
                remote: None,
            };
            Embedder::new(config).map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|e| e.clone())
}

/// Get the data directory from environment or default
fn get_data_dir() -> PathBuf {
    std::env::var("GRAPHRAG_DATA_DIR")
        .map(PathBuf::from)
        .ok()
        .unwrap_or_else(|| {
            dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("graphrag")
        })
}

/// JSON-RPC 2.0 Request
#[derive(Debug, Deserialize, Clone)]
struct JsonRpcRequest {
    /// JSON-RPC version. Required on the wire; we don't act on it post-decode
    /// because we only support 2.0, but serde must accept the field.
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

/// JSON-RPC 2.0 Response
#[derive(Debug, Serialize, Deserialize, Clone)]
struct JsonRpcResponse {
    jsonrpc: String,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct JsonRpcError {
    code: i32,
    message: String,
}

impl JsonRpcResponse {
    fn success(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    fn error(id: Value, code: i32, message: String) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError { code, message }),
        }
    }
}

/// MCP Server implementation
struct McpServer {
    data_dir: PathBuf,
    default_store: String,
    /// Per-store BM25 indexes, built lazily on first recall, retained for
    /// the server's lifetime, invalidated on add_document/add_code. leit
    /// Phase 1 has no incremental adds, so any mutation rebuilds the whole
    /// thing on the next recall.
    ///
    /// Known limitation: external `graphrag note` (CLI) writes the
    /// `.leitseg` sidecar but does NOT signal this in-process cache, so a
    /// long-lived MCP server will serve a stale lexical index after a
    /// CLI-side note. Acceptable today — agents use MCP, users restart.
    /// Fix when we add a filesystem-watch hook or mtime-stamp check.
    lexical_cache: std::sync::Mutex<HashMap<String, std::sync::Arc<LexicalIndex>>>,
}

impl McpServer {
    fn new() -> Self {
        Self {
            data_dir: get_data_dir(),
            default_store: std::env::var("GRAPHRAG_STORE").unwrap_or_else(|_| "memory".to_string()),
            lexical_cache: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Borrow (and lazily build) the cached LexicalIndex for `store`.
    /// Returns `None` when the store is empty so callers can fall through
    /// to vector-only recall cleanly.
    // Arc (not Rc) despite LexicalIndex being !Send/!Sync: the server is
    // single-threaded stdio, and the cache map is already Mutex-guarded.
    #[allow(clippy::arc_with_non_send_sync)]
    fn cached_lexical_for(
        &self,
        db: &Database,
        store: &str,
    ) -> Result<Option<std::sync::Arc<LexicalIndex>>, String> {
        {
            let cache = self.lexical_cache.lock().unwrap();
            if let Some(idx) = cache.get(store) {
                return Ok(Some(idx.clone()));
            }
        }
        // Build outside the lock so concurrent recalls on different stores
        // don't serialize on each other's index-build. Cost of a duplicate
        // build on the same store under contention is acceptable.
        let chunks = db
            .list_chunks(store)
            .map_err(|e| format!("list chunks for lexical build: {e}"))?;
        let Some(index) = LexicalIndex::build_from_chunks(&chunks)
            .map_err(|e| format!("build lexical index: {e}"))?
        else {
            return Ok(None);
        };
        // LexicalIndex is only ever accessed under `lexical_cache`'s Mutex.
        #[allow(clippy::arc_with_non_send_sync)]
        let arc = std::sync::Arc::new(index);
        let mut cache = self.lexical_cache.lock().unwrap();
        // Last-writer-wins on race; both Arc'd indexes are equivalent.
        cache.insert(store.to_string(), arc.clone());
        Ok(Some(arc))
    }

    /// Drop the cached LexicalIndex for `store`. Called from any path that
    /// mutates chunks (add_document, add_code). The next recall rebuilds.
    fn invalidate_lexical(&self, store: &str) {
        let mut cache = self.lexical_cache.lock().unwrap();
        cache.remove(store);
    }

    fn db_path(&self) -> PathBuf {
        self.data_dir.join("graphrag.db")
    }

    /// Handle incoming message - could be a request or a response to our sampling request
    fn handle_message(&mut self, line: &str, stdout: &mut impl Write) {
        // Otherwise parse as a request
        let request: JsonRpcRequest = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Parse error: {} - line: {}", e, line);
                let response =
                    JsonRpcResponse::error(Value::Null, -32700, format!("Parse error: {}", e));
                let _ = writeln!(stdout, "{}", serde_json::to_string(&response).unwrap());
                let _ = stdout.flush();
                return;
            }
        };

        // Skip notifications
        if request.id.is_none() && request.method.starts_with("notifications/") {
            return;
        }

        let response = self.handle_request(request, stdout);

        // Only send response if we have one (some operations are async)
        if let Some(resp) = response {
            if let Err(e) = writeln!(stdout, "{}", serde_json::to_string(&resp).unwrap()) {
                eprintln!("Write error: {}", e);
            }
            let _ = stdout.flush();
        }
    }

    fn handle_request(
        &mut self,
        req: JsonRpcRequest,
        stdout: &mut impl Write,
    ) -> Option<JsonRpcResponse> {
        let id = req.id.clone().unwrap_or(Value::Null);

        match req.method.as_str() {
            "initialize" => Some(self.handle_initialize(id)),
            "tools/list" => Some(self.handle_list_tools(id)),
            "tools/call" => self.handle_call_tool(id, req.params, stdout),
            "notifications/initialized" => Some(JsonRpcResponse::success(id, Value::Null)),
            _ => Some(JsonRpcResponse::error(
                id,
                -32601,
                format!("Method not found: {}", req.method),
            )),
        }
    }

    fn handle_initialize(&self, id: Value) -> JsonRpcResponse {
        JsonRpcResponse::success(
            id,
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "sampling": {}
                },
                "serverInfo": {
                    "name": "graphrag",
                    "version": env!("CARGO_PKG_VERSION")
                }
            }),
        )
    }

    fn handle_list_tools(&self, id: Value) -> JsonRpcResponse {
        let tools = json!({
            "tools": [
                {
                    "name": "graphrag_recall",
                    "description": "Search memory with natural language: exact-cosine vector search + BM25, fused with RRF, plus a graph lane that follows shared entities across groups. Hits with unfinished ingestion are moved to the front of the background work.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": { "type": "string", "description": "Natural language query" },
                            "top": { "type": "number", "description": "Number of results (default: 5)" },
                            "in": { "type": "array", "items": { "type": "string" }, "description": "Restrict to these groups (and subgroups); default: everything" },
                            "store": { "type": "string", "description": "Deprecated alias for `in`" }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "graphrag_add_document",
                    "description": "Add a document. It is stored once (identical text is deduplicated), split into sections and chunks, and handed to the background worker, which embeds, extracts entities and relations, clusters and summarizes. Returns immediately.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "content": { "type": "string", "description": "The document content" },
                            "source": { "type": "string", "description": "Optional source identifier (filename, URL)" },
                            "doc_key": { "type": "string", "description": "Stable key (path, page id); new text under the same key replaces the old version" },
                            "groups": { "type": "array", "items": { "type": "string" }, "description": "Groups it belongs to (default: conversations)" },
                            "store": { "type": "string", "description": "Deprecated alias for a group" }
                        },
                        "required": ["content"]
                    }
                },
                {
                    "name": "graphrag_add_code",
                    "description": "Add source code, chunked as code (no markdown sections), and hand it to the background worker. Returns immediately.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "content": { "type": "string", "description": "The source code" },
                            "file_path": { "type": "string", "description": "File path, used as the source" },
                            "source": { "type": "string", "description": "Optional source identifier" },
                            "doc_key": { "type": "string", "description": "Stable key; new text under it replaces the old version" },
                            "groups": { "type": "array", "items": { "type": "string" }, "description": "Groups it belongs to (default: code)" },
                            "store": { "type": "string", "description": "Deprecated alias for a group" }
                        },
                        "required": ["content"]
                    }
                },
                {
                    "name": "graphrag_summarize_community",
                    "description": "Return a community's stored summary and its status. Summaries are written by the background worker.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "community_id": { "type": "number", "description": "The community ID" }
                        },
                        "required": ["community_id"]
                    }
                },
                {
                    "name": "graphrag_status",
                    "description": "Ingestion status: per group and step, how many are current, stale, missing and failed; backlog; whether a worker is running; last LLM success or error.",
                    "inputSchema": { "type": "object", "properties": {} }
                },
                {
                    "name": "graphrag_stores",
                    "description": "List all GraphRAG stores with their dimensions.",
                    "inputSchema": { "type": "object", "properties": {} }
                },
                {
                    "name": "graphrag_entities",
                    "description": "List entities in the knowledge graph.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "store": { "type": "string" },
                            "limit": { "type": "number", "description": "Max entities to return (default: 100)" }
                        }
                    }
                },
                {
                    "name": "graphrag_relations",
                    "description": "Get relations for an entity.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "entity": { "type": "string", "description": "Entity name" },
                            "store": { "type": "string" }
                        },
                        "required": ["entity"]
                    }
                },
                {
                    "name": "graphrag_communities",
                    "description": "List detected communities with summaries.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "store": { "type": "string" }
                        }
                    }
                },
                {
                    "name": "graphrag_search",
                    "description": "Search using pre-computed embedding vector. Use graphrag_recall for text queries.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "embedding": { "type": "array", "items": { "type": "number" } },
                            "top": { "type": "number" },
                            "expand": { "type": "number" },
                            "in": { "type": "array", "items": { "type": "string" }, "description": "Restrict to these groups" },
                            "store": { "type": "string", "description": "Deprecated alias for `in`" }
                        },
                        "required": ["embedding"]
                    }
                },
                {
                    "name": "graphrag_query_global",
                    "description": "Query across all communities for global sensemaking. Returns community contexts for map-reduce style answering. Use this for questions that span the entire knowledge base like 'What are the main themes?' or 'Summarize key concepts'. Each community context can be processed by a subagent, then results synthesized.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The global question to answer"
                            },
                            "max_communities": {
                                "type": "number",
                                "description": "Maximum number of communities to include (default: all)"
                            },
                            "min_entities": {
                                "type": "number",
                                "description": "Minimum entities for a community to be included (default: 2)"
                            },
                            "store": { "type": "string", "description": format!("Store name (default: {})", self.default_store) }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "graphrag_find_merge_candidates",
                    "description": "Find entity pairs that might be duplicates based on embedding similarity. Returns candidates above the threshold for review before merging.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "threshold": {
                                "type": "number",
                                "description": "Similarity threshold 0.0-1.0 (default: 0.85). Higher = stricter matching."
                            },
                            "store": { "type": "string", "description": format!("Store name (default: {})", self.default_store) }
                        }
                    }
                },
                {
                    "name": "graphrag_merge_entities",
                    "description": "Merge two entities: records a retractable same_as assertion from source to target. Nothing is deleted; reads and clustering resolve the source to the target.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "target_id": {
                                "type": "number",
                                "description": "ID of the entity to keep (target)"
                            },
                            "source_id": {
                                "type": "number",
                                "description": "ID of the entity to merge into target"
                            }
                        },
                        "required": ["target_id", "source_id"]
                    }
                },
                {
                    "name": "graphrag_embed_entities",
                    "description": "Generate and store embeddings for all entities in a store. Required before finding merge candidates.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "store": { "type": "string", "description": format!("Store name (default: {})", self.default_store) }
                        }
                    }
                }
            ]
        });
        JsonRpcResponse::success(id, tools)
    }

    fn handle_call_tool(
        &mut self,
        id: Value,
        params: Value,
        _stdout: &mut impl Write,
    ) -> Option<JsonRpcResponse> {
        let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let args = params.get("arguments").cloned().unwrap_or(json!({}));

        match name {
            "graphrag_recall" => Some(self.tool_recall(&args, id)),
            "graphrag_add_document" => Some(self.tool_add_document(&args, id)),
            "graphrag_add_code" => Some(self.tool_add_code(&args, id)),
            "graphrag_summarize_community" => Some(self.tool_summarize_community(&args, id)),
            "graphrag_status" => Some(self.tool_status(id)),
            "graphrag_stores" => Some(self.tool_stores(id)),
            "graphrag_entities" => Some(self.tool_entities(&args, id)),
            "graphrag_relations" => Some(self.tool_relations(&args, id)),
            "graphrag_communities" => Some(self.tool_communities(&args, id)),
            "graphrag_search" => Some(self.tool_search(&args, id)),
            "graphrag_query_global" => Some(self.tool_query_global(&args, id)),
            "graphrag_find_merge_candidates" => Some(self.tool_find_merge_candidates(&args, id)),
            "graphrag_merge_entities" => Some(self.tool_merge_entities(&args, id)),
            "graphrag_embed_entities" => Some(self.tool_embed_entities(&args, id)),
            _ => Some(self.tool_error(id, format!("Unknown tool: {}", name))),
        }
    }

    fn tool_error(&self, id: Value, message: String) -> JsonRpcResponse {
        JsonRpcResponse::success(
            id,
            json!({
                "content": [{ "type": "text", "text": format!("Error: {}", message) }],
                "isError": true
            }),
        )
    }

    fn tool_success(&self, id: Value, text: String) -> JsonRpcResponse {
        JsonRpcResponse::success(
            id,
            json!({
                "content": [{ "type": "text", "text": text }]
            }),
        )
    }

    /// Everything lives in one store; a `store` argument is a deprecated
    /// group alias and doesn't select a store.
    fn get_store_name(&self, _args: &Value) -> String {
        self.default_store.clone()
    }

    /// Groups from `in`/`groups` (arrays) plus a deprecated `store` alias.
    fn groups_arg(args: &Value, key: &str) -> Vec<String> {
        let mut groups: Vec<String> = args
            .get(key)
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        if let Some(s) = args.get("store").and_then(|v| v.as_str()) {
            eprintln!("warning: `store` is deprecated; use `in`");
            groups.push(s.to_string());
        }
        groups
    }

    fn spawn_worker(&self) {
        if let Err(e) = graphrag_worker::spawn_detached(&self.data_dir) {
            eprintln!("warning: could not start the worker: {e}");
        }
    }

    /// Store, chunk and hand off to the worker; never calls back to the agent.
    fn add(
        &self,
        args: &Value,
        id: Value,
        profile: ChunkerProfile,
        default_group: &str,
    ) -> JsonRpcResponse {
        let content = match args.get("content").and_then(|v| v.as_str()) {
            Some(c) => c,
            None => return self.tool_error(id, "Missing 'content' argument".to_string()),
        };
        let source = args
            .get("source")
            .or_else(|| args.get("file_path"))
            .and_then(|v| v.as_str());
        let doc_key = args.get("doc_key").and_then(|v| v.as_str());
        let mut groups = Self::groups_arg(args, "groups");
        if groups.is_empty() {
            groups.push(default_group.to_string());
        }
        let groups: Vec<&str> = groups.iter().map(String::as_str).collect();
        let db = match Database::open(&self.db_path()) {
            Ok(d) => d,
            Err(e) => return self.tool_error(id, e.to_string()),
        };
        if db.get_store(&self.default_store).is_err() {
            let dim = embedder_model().dimension();
            if let Err(e) = db.create_store(&self.default_store, dim) {
                return self.tool_error(id, e.to_string());
            }
        }
        let outcome = match db.add_document(&AddDocument {
            store: &self.default_store,
            content,
            origin: Origin::Source,
            source,
            doc_key,
            title: None,
            groups: &groups,
            surface: "mcp",
            profile: &profile,
        }) {
            Ok(o) => o,
            Err(e) => return self.tool_error(id, e.to_string()),
        };
        let (document_id, existing) = match outcome {
            AddOutcome::Created { document_id } | AddOutcome::Replaced { document_id, .. } => {
                (document_id, false)
            }
            AddOutcome::Existing { document_id, .. } => (document_id, true),
        };
        let chunks = db
            .document_chunks(document_id)
            .map(|c| c.len())
            .unwrap_or(0);
        self.invalidate_lexical(&self.default_store);
        self.spawn_worker();
        self.tool_success(
            id,
            json!({
                "document_id": document_id,
                "chunks": chunks,
                "existing": existing,
                "groups": groups,
                "note": "Stored. Embedding, extraction and clustering run in the background worker; see graphrag_status."
            })
            .to_string(),
        )
    }

    fn tool_add_document(&self, args: &Value, id: Value) -> JsonRpcResponse {
        self.add(
            args,
            id,
            ChunkerProfile::default_markdown(),
            "conversations",
        )
    }

    fn tool_add_code(&self, args: &Value, id: Value) -> JsonRpcResponse {
        let profile = ChunkerProfile {
            kind: "code",
            ..ChunkerProfile::default_markdown()
        };
        self.add(args, id, profile, "code")
    }

    /// The stored summary and its status. Summaries are written by the
    /// worker; this never samples the client.
    fn tool_summarize_community(&self, args: &Value, id: Value) -> JsonRpcResponse {
        let community_id = match args.get("community_id").and_then(|v| v.as_i64()) {
            Some(c) => c,
            None => return self.tool_error(id, "Missing 'community_id' argument".to_string()),
        };
        let db = match Database::open(&self.db_path()) {
            Ok(d) => d,
            Err(e) => return self.tool_error(id, e.to_string()),
        };
        let summary = db
            .list_communities(&self.default_store)
            .ok()
            .and_then(|cs| cs.into_iter().find(|c| c.id == community_id))
            .and_then(|c| c.summary);
        let status = db
            .step_fp(SubjectKind::Community, community_id, Step::Summary)
            .ok()
            .flatten()
            .map(|(_, state)| state);
        self.tool_success(
            id,
            json!({
                "community_id": community_id,
                "summary": summary,
                "status": status.unwrap_or_else(|| "pending".into()),
            })
            .to_string(),
        )
    }

    fn tool_status(&self, id: Value) -> JsonRpcResponse {
        let db = match Database::open(&self.db_path()) {
            Ok(d) => d,
            Err(e) => return self.tool_error(id, e.to_string()),
        };
        let mut cfg = graphrag_worker::WorkerConfig::from_env(self.data_dir.clone());
        cfg.store = self.default_store.clone();
        let model = embedder_model();
        let sc = cfg.static_config_with(graphrag_core::embedder_model_id(model), model.dimension());
        let report = match db.status_report(&sc) {
            Ok(r) => r,
            Err(e) => return self.tool_error(id, e.to_string()),
        };
        let worker_running = matches!(graphrag_worker::lock::acquire(&self.data_dir), Ok(None));
        let state = db.worker_state().unwrap_or_default();
        self.tool_success(
            id,
            json!({
                "report": report,
                "complete": report.is_complete(),
                "worker_running": worker_running,
                "worker_state": state,
            })
            .to_string(),
        )
    }

    fn tool_recall(&self, args: &Value, id: Value) -> JsonRpcResponse {
        let query = match args.get("query").and_then(|v| v.as_str()) {
            Some(q) => q,
            None => return self.tool_error(id, "Missing 'query' argument".to_string()),
        };
        let top = args.get("top").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        let groups = Self::groups_arg(args, "in");
        let groups: Vec<&str> = groups.iter().map(String::as_str).collect();
        let db = match Database::open(&self.db_path()) {
            Ok(d) => d,
            Err(e) => return self.tool_error(id, e.to_string()),
        };
        let store = match db.get_store(&self.default_store) {
            Ok(s) => s,
            Err(_) => return self.tool_success(id, "No memory yet.".to_string()),
        };
        let embedder = match get_embedder() {
            Ok(e) => e,
            Err(e) => return self.tool_error(id, e),
        };
        let embedding = match embedder.embed(query) {
            Ok(e) => e,
            Err(e) => return self.tool_error(id, e.to_string()),
        };
        if embedding.len() != store.dim {
            return self.tool_error(
                id,
                format!(
                    "Dimension mismatch: embedder={}, store={}",
                    embedding.len(),
                    store.dim
                ),
            );
        }
        let lexical = match self.cached_lexical_for(&db, &self.default_store) {
            Ok(l) => l,
            Err(e) => return self.tool_error(id, e),
        };
        let now = db.now_utc().unwrap_or_default();
        let hits = match db.hybrid_recall(
            &self.default_store,
            query,
            &embedding,
            &groups,
            top,
            lexical.as_deref(),
            &now,
        ) {
            Ok(h) => h,
            Err(e) => return self.tool_error(id, e.to_string()),
        };
        let ids: Vec<i64> = hits.iter().map(|h| h.chunk_id).collect();
        if db.escalate_chunks(&ids).unwrap_or(0) > 0 {
            self.spawn_worker();
        }
        if hits.is_empty() {
            return self.tool_success(id, "No results.".to_string());
        }
        let mut text = String::from("## Recall results\n\n");
        for (i, h) in hits.iter().enumerate() {
            let opt = |v: Option<String>| v.unwrap_or_else(|| "-".into());
            text.push_str(&format!(
                "### {}. chunk {} · rrf {:.4} · vec {} · lex {} · graph {}\n",
                i + 1,
                h.chunk_id,
                h.rrf,
                opt(h.vec.map(|s| format!("{s:.3}"))),
                opt(h.lex.map(|s| format!("{s:.3}"))),
                opt(h.graph.map(|s| format!("{s:.3}"))),
            ));
            text.push_str(&format!(
                "groups: {} · section: {} · source: {}\n\n{}\n\n",
                if h.groups.is_empty() {
                    "-".into()
                } else {
                    h.groups.join(", ")
                },
                if h.heading_path.is_empty() {
                    "-".into()
                } else {
                    h.heading_path.join(" › ")
                },
                h.source.as_deref().unwrap_or("-"),
                h.content
            ));
        }
        self.tool_success(id, text)
    }

    fn tool_stores(&self, id: Value) -> JsonRpcResponse {
        let db = match Database::open(&self.db_path()) {
            Ok(d) => d,
            Err(e) => return self.tool_error(id, e.to_string()),
        };

        let stores = match db.list_stores() {
            Ok(s) => s,
            Err(e) => return self.tool_error(id, e.to_string()),
        };

        let mut text = String::from(
            "## GraphRAG Stores\n\n| Name | Dimension | Created |\n|------|-----------|----------|\n",
        );
        for store in &stores {
            text.push_str(&format!(
                "| {} | {} | {} |\n",
                store.name, store.dim, store.created_at
            ));
        }
        text.push_str(&format!("\n*Total: {} stores*", stores.len()));

        self.tool_success(id, text)
    }

    fn tool_entities(&self, args: &Value, id: Value) -> JsonRpcResponse {
        let store_name = self.get_store_name(args);
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(100) as usize;

        let db = match Database::open(&self.db_path()) {
            Ok(d) => d,
            Err(e) => return self.tool_error(id, e.to_string()),
        };

        let entities = match db.list_entities(&store_name) {
            Ok(e) => e,
            Err(e) => return self.tool_error(id, e.to_string()),
        };

        let mut text = format!(
            "## Entities in \"{}\"\n\n| Name | Type |\n|------|------|\n",
            store_name
        );
        for e in entities.iter().take(limit) {
            let entity_type = e.entity_type.as_deref().unwrap_or("-");
            text.push_str(&format!("| {} | {} |\n", e.name, entity_type));
        }
        text.push_str(&format!(
            "\n*Showing {} of {} entities*",
            limit.min(entities.len()),
            entities.len()
        ));

        self.tool_success(id, text)
    }

    fn tool_relations(&self, args: &Value, id: Value) -> JsonRpcResponse {
        let store_name = self.get_store_name(args);
        let entity_name = match args.get("entity").and_then(|v| v.as_str()) {
            Some(e) => e,
            None => return self.tool_error(id, "Missing 'entity' argument".to_string()),
        };

        let db = match Database::open(&self.db_path()) {
            Ok(d) => d,
            Err(e) => return self.tool_error(id, e.to_string()),
        };

        let entity = match db.get_entity_by_name(&store_name, entity_name) {
            Ok(e) => e,
            Err(e) => return self.tool_error(id, e.to_string()),
        };

        let relations = match db.get_relations_for_entity(entity.id) {
            Ok(r) => r,
            Err(e) => return self.tool_error(id, e.to_string()),
        };

        let mut text = format!(
            "## Relations for \"{}\"\n\n| Direction | Relation | Other Entity | Type |\n|-----------|----------|--------------|------|\n",
            entity_name
        );

        for r in &relations {
            let (direction, other_id) = if r.head_id == entity.id {
                ("outgoing", r.tail_id)
            } else {
                ("incoming", r.head_id)
            };

            if let Ok(other) = db.get_entity_by_id(other_id) {
                let other_type = other.entity_type.as_deref().unwrap_or("-");
                text.push_str(&format!(
                    "| {} | {} | {} | {} |\n",
                    direction, r.relation, other.name, other_type
                ));
            }
        }
        text.push_str(&format!("\n*Total: {} relations*", relations.len()));

        self.tool_success(id, text)
    }

    fn tool_communities(&self, args: &Value, id: Value) -> JsonRpcResponse {
        let store_name = self.get_store_name(args);

        let db = match Database::open(&self.db_path()) {
            Ok(d) => d,
            Err(e) => return self.tool_error(id, e.to_string()),
        };

        let communities = match db.list_communities(&store_name) {
            Ok(c) => c,
            Err(e) => return self.tool_error(id, e.to_string()),
        };

        let mut text = format!("## Communities in \"{}\"\n\n", store_name);
        for c in &communities {
            text.push_str(&format!("### Community {} (Level {})\n", c.id, c.level));

            if let Ok(entities) = db.get_community_entities(c.id) {
                text.push_str(&format!("- **Size**: {} entities\n", entities.len()));
                let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
                text.push_str(&format!("- **Entities**: {}\n", names.join(", ")));
            }

            if let Some(parent) = c.parent_id {
                text.push_str(&format!("- **Parent**: {}\n", parent));
            }
            if let Some(ref summary) = c.summary {
                text.push_str(&format!("- **Summary**: {}\n", summary));
            }
            text.push('\n');
        }

        self.tool_success(id, text)
    }

    /// Global query tool - returns community contexts for map-reduce answering
    fn tool_query_global(&self, args: &Value, id: Value) -> JsonRpcResponse {
        let query = match args.get("query").and_then(|v| v.as_str()) {
            Some(q) => q,
            None => return self.tool_error(id, "Missing 'query' argument".to_string()),
        };

        let store_name = self.get_store_name(args);
        let max_communities = args
            .get("max_communities")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let min_entities = args
            .get("min_entities")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;

        let db = match Database::open(&self.db_path()) {
            Ok(d) => d,
            Err(e) => return self.tool_error(id, e.to_string()),
        };

        let communities = match db.list_communities(&store_name) {
            Ok(c) => c,
            Err(e) => return self.tool_error(id, e.to_string()),
        };

        if communities.is_empty() {
            return self.tool_error(
                id,
                format!(
                    "No communities found in store '{}'. Run community detection first.",
                    store_name
                ),
            );
        }

        // Build community contexts
        let mut community_contexts: Vec<Value> = Vec::new();

        for community in &communities {
            let entities = match db.get_community_entities(community.id) {
                Ok(e) => e,
                Err(_) => continue,
            };

            // Skip small communities
            if entities.len() < min_entities {
                continue;
            }

            // Build context for this community
            let mut context = String::new();

            // Add summary if available
            if let Some(ref summary) = community.summary {
                context.push_str(&format!("## Community Summary\n{}\n\n", summary));
            }

            // Add entities
            context.push_str("## Entities\n");
            let entity_names: Vec<String> = entities.iter().map(|e| e.name.clone()).collect();

            for entity in &entities {
                let entity_type = entity.entity_type.as_deref().unwrap_or("unknown");
                context.push_str(&format!("- **{}** ({})\n", entity.name, entity_type));
            }

            // Add relations within this community
            context.push_str("\n## Relationships\n");
            let mut relation_count = 0;

            for entity in &entities {
                if let Ok(relations) = db.get_relations_for_entity(entity.id) {
                    for rel in relations {
                        let other_id = if rel.head_id == entity.id {
                            rel.tail_id
                        } else {
                            rel.head_id
                        };
                        if let Ok(other) = db.get_entity_by_id(other_id) {
                            // Only include relations within this community
                            if entity_names.contains(&other.name) && rel.head_id == entity.id {
                                context.push_str(&format!(
                                    "- {} --[{}]--> {}\n",
                                    entity.name, rel.relation, other.name
                                ));
                                relation_count += 1;
                            }
                        }
                    }
                }
            }

            // Add relevant chunks (sample)
            let mut chunk_sample = String::new();
            let mut chunks_added = 0;
            for entity in &entities {
                if chunks_added >= 3 {
                    break;
                }
                if let Ok(chunk_ids) = db.get_chunks_for_entity(entity.id) {
                    for chunk_id in chunk_ids.iter().take(1) {
                        if let Ok(chunks) = db.get_chunks_by_ids(&[*chunk_id])
                            && let Some(chunk) = chunks.first()
                        {
                            // Truncate long chunks
                            let content = if chunk.content.len() > 500 {
                                format!("{}...", &chunk.content[..500])
                            } else {
                                chunk.content.clone()
                            };
                            chunk_sample.push_str(&format!("\n---\n{}\n", content));
                            chunks_added += 1;
                        }
                    }
                }
            }

            if !chunk_sample.is_empty() {
                context.push_str("\n## Sample Content");
                context.push_str(&chunk_sample);
            }

            community_contexts.push(json!({
                "community_id": community.id,
                "level": community.level,
                "entity_count": entities.len(),
                "relation_count": relation_count,
                "summary": community.summary,
                "context": context
            }));

            // Apply max limit
            if let Some(max) = max_communities
                && community_contexts.len() >= max
            {
                break;
            }
        }

        if community_contexts.is_empty() {
            return self.tool_error(
                id,
                format!(
                    "No communities with >= {} entities found in store '{}'",
                    min_entities, store_name
                ),
            );
        }

        // Return structured response
        let response = json!({
            "query": query,
            "store": store_name,
            "total_communities": community_contexts.len(),
            "communities": community_contexts,
            "instructions": {
                "map_phase": "For each community, spawn a subagent with: 'Answer this question using ONLY the provided context: [query]' and provide the community's context field.",
                "reduce_phase": "Collect all subagent responses and synthesize into a comprehensive answer.",
                "suggested_prompt": format!("Answer this question using ONLY the context below. Be specific and cite entities when relevant.\n\nQuestion: {}\n\nContext:\n{{context}}", query)
            }
        });

        // Format as readable output with JSON data
        let mut text = format!(
            "## Global Query: {}\n\n**Store**: {}\n**Communities**: {}\n\n",
            query,
            store_name,
            community_contexts.len()
        );

        text.push_str("### Community Contexts for Map-Reduce\n\n");
        text.push_str("Each community below can be processed by a subagent:\n\n");

        for (i, ctx) in community_contexts.iter().enumerate() {
            let c_id = ctx
                .get("community_id")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let entities = ctx
                .get("entity_count")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let summary = ctx
                .get("summary")
                .and_then(|v| v.as_str())
                .unwrap_or("(no summary)");

            text.push_str(&format!(
                "{}. **Community {}** ({} entities): {}\n",
                i + 1,
                c_id,
                entities,
                summary
            ));
        }

        text.push_str(&format!(
            "\n---\n\n<graphrag_global_query_data>\n{}\n</graphrag_global_query_data>",
            serde_json::to_string_pretty(&response).unwrap_or_default()
        ));

        self.tool_success(id, text)
    }

    /// Embed all entities in a store for similarity search
    fn tool_embed_entities(&self, args: &Value, id: Value) -> JsonRpcResponse {
        let store_name = self.get_store_name(args);

        let embedder = match get_embedder() {
            Ok(e) => e,
            Err(e) => return self.tool_error(id, e),
        };

        let db = match Database::open(&self.db_path()) {
            Ok(d) => d,
            Err(e) => return self.tool_error(id, e.to_string()),
        };

        let entities = match db.list_entities(&store_name) {
            Ok(e) => e,
            Err(e) => return self.tool_error(id, e.to_string()),
        };

        if entities.is_empty() {
            return self.tool_error(id, format!("No entities in store '{}'", store_name));
        }

        let mut embedded_count = 0;
        let mut errors = 0;

        for entity in &entities {
            match embedder.embed(&entity.name) {
                Ok(embedding) => {
                    if let Err(e) = db.set_entity_embedding(entity.id, &embedding) {
                        eprintln!("Error embedding entity '{}': {}", entity.name, e);
                        errors += 1;
                    } else {
                        embedded_count += 1;
                    }
                }
                Err(e) => {
                    eprintln!("Error embedding entity '{}': {}", entity.name, e);
                    errors += 1;
                }
            }
        }

        self.tool_success(
            id,
            format!(
                "Embedded {} entities in '{}' ({} errors)",
                embedded_count, store_name, errors
            ),
        )
    }

    /// Find entity pairs that might be duplicates
    fn tool_find_merge_candidates(&self, args: &Value, id: Value) -> JsonRpcResponse {
        let store_name = self.get_store_name(args);
        let threshold = args
            .get("threshold")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.85) as f32;

        let db = match Database::open(&self.db_path()) {
            Ok(d) => d,
            Err(e) => return self.tool_error(id, e.to_string()),
        };

        let candidates = match db.find_merge_candidates(&store_name, threshold) {
            Ok(c) => c,
            Err(e) => return self.tool_error(id, e.to_string()),
        };

        if candidates.is_empty() {
            return self.tool_success(
                id,
                format!(
                    "No merge candidates found in '{}' with threshold >= {:.2}",
                    store_name, threshold
                ),
            );
        }

        let mut text = format!(
            "## Entity Merge Candidates in '{}'\n\nThreshold: {:.2}\n\n| Entity 1 (ID) | Entity 2 (ID) | Similarity |\n|---------------|---------------|------------|\n",
            store_name, threshold
        );

        for (id1, name1, id2, name2, sim) in &candidates {
            text.push_str(&format!(
                "| {} ({}) | {} ({}) | {:.3} |\n",
                name1, id1, name2, id2, sim
            ));
        }

        text.push_str(&format!(
            "\n*Found {} candidate pairs*\n\n",
            candidates.len()
        ));
        text.push_str("Use `graphrag_merge_entities` with target_id and source_id to merge pairs.");

        self.tool_success(id, text)
    }

    /// Merge two entities
    fn tool_merge_entities(&self, args: &Value, id: Value) -> JsonRpcResponse {
        let target_id = match args.get("target_id").and_then(|v| v.as_i64()) {
            Some(id) => id,
            None => return self.tool_error(id, "Missing 'target_id' argument".to_string()),
        };

        let source_id = match args.get("source_id").and_then(|v| v.as_i64()) {
            Some(id) => id,
            None => return self.tool_error(id, "Missing 'source_id' argument".to_string()),
        };

        if target_id == source_id {
            return self.tool_error(id, "Cannot merge entity with itself".to_string());
        }

        let db = match Database::open(&self.db_path()) {
            Ok(d) => d,
            Err(e) => return self.tool_error(id, e.to_string()),
        };

        // Get entity names for the response
        let target_name: String = match db.get_entity_by_id(target_id) {
            Ok(e) => e.name,
            Err(e) => return self.tool_error(id, format!("Target entity not found: {}", e)),
        };

        let source_name: String = match db.get_entity_by_id(source_id) {
            Ok(e) => e.name,
            Err(e) => return self.tool_error(id, format!("Source entity not found: {}", e)),
        };

        match db.merge_entities(target_id, source_id) {
            Ok(()) => self.tool_success(
                id,
                format!(
                    "Merged '{}' (id:{}) into '{}' (id:{}). Source entity deleted.",
                    source_name, source_id, target_name, target_id
                ),
            ),
            Err(e) => self.tool_error(id, format!("Merge failed: {}", e)),
        }
    }

    fn tool_search(&self, args: &Value, id: Value) -> JsonRpcResponse {
        let embedding: Vec<f32> = match args.get("embedding").and_then(|v| v.as_array()) {
            Some(arr) => arr
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect(),
            None => return self.tool_error(id, "Missing 'embedding' argument".to_string()),
        };

        let store_name = self.get_store_name(args);
        let top = args.get("top").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        let expand = args.get("expand").and_then(|v| v.as_u64()).unwrap_or(1) as usize;

        let db = match Database::open(&self.db_path()) {
            Ok(d) => d,
            Err(e) => return self.tool_error(id, e.to_string()),
        };

        let store = match db.get_store(&store_name) {
            Ok(s) => s,
            Err(e) => return self.tool_error(id, e.to_string()),
        };

        if embedding.len() != store.dim {
            return self.tool_error(
                id,
                format!(
                    "Dimension mismatch: expected {}, got {}",
                    store.dim,
                    embedding.len()
                ),
            );
        }

        let source = match BruteForceVectorSource::for_store(&db, &store_name) {
            Ok(s) => s,
            Err(e) => return self.tool_error(id, e.to_string()),
        };

        let groups = Self::groups_arg(args, "in");
        let groups: Vec<&str> = groups.iter().map(String::as_str).collect();
        let mut allowed = match db.active_chunk_ids(&store_name) {
            Ok(a) => a,
            Err(e) => return self.tool_error(id, e.to_string()),
        };
        if !groups.is_empty() {
            match db.chunk_ids_in_groups(&store_name, &groups) {
                Ok(g) => allowed.retain(|c| g.contains(c)),
                Err(e) => return self.tool_error(id, e.to_string()),
            }
        }
        let results: Vec<VectorCandidate> =
            match source.top_candidates(&embedding, source.len().max(1)) {
                Ok(r) => r
                    .into_iter()
                    .filter(|h| allowed.contains(&h.chunk_id))
                    .take(top)
                    .collect(),
                Err(e) => return self.tool_error(id, e.to_string()),
            };
        let ids: Vec<i64> = results.iter().map(|r| r.chunk_id).collect();
        if db.escalate_chunks(&ids).unwrap_or(0) > 0 {
            self.spawn_worker();
        }

        match self.format_search_results(&db, &results, expand) {
            Ok(text) => self.tool_success(id, text),
            Err(e) => self.tool_error(id, e),
        }
    }

    fn format_search_results(
        &self,
        db: &Database,
        results: &[VectorCandidate],
        expand: usize,
    ) -> Result<String, String> {
        let mut text = String::from("## Search Results\n\n");
        let mut seen_chunks = std::collections::HashSet::new();
        let mut chunk_ids: Vec<i64> = Vec::new();

        for VectorCandidate { chunk_id, score: _ } in results {
            seen_chunks.insert(*chunk_id);
            chunk_ids.push(*chunk_id);
        }

        let chunks = db
            .get_chunks_by_ids(&chunk_ids)
            .map_err(|e| e.to_string())?;

        for (chunk, result) in chunks.iter().zip(results.iter()) {
            let source = chunk.source.as_deref().unwrap_or("unknown");
            // `score` is higher-is-better (cosine similarity for direct
            // search, RRF score for hybrid recall).
            text.push_str(&format!(
                "### [{}] (score: {:.3})\n\n{}\n\n---\n\n",
                source, result.score, chunk.content
            ));
        }

        if expand > 0 && !chunk_ids.is_empty() {
            let mut expanded_chunk_ids = std::collections::HashSet::new();

            for chunk_id in &chunk_ids {
                if let Ok(entities) = db.get_entities_for_chunk(*chunk_id) {
                    for entity in entities {
                        if let Ok(related_chunks) = db.get_chunks_for_entity(entity.id) {
                            for related_id in related_chunks {
                                if !seen_chunks.contains(&related_id) {
                                    expanded_chunk_ids.insert(related_id);
                                }
                            }
                        }
                    }
                }
            }

            if !expanded_chunk_ids.is_empty() {
                let expanded_ids: Vec<i64> = expanded_chunk_ids.into_iter().collect();
                if let Ok(expanded_chunks) = db.get_chunks_by_ids(&expanded_ids) {
                    for chunk in expanded_chunks {
                        let source = chunk.source.as_deref().unwrap_or("unknown");
                        text.push_str(&format!(
                            "### [{}] (via graph)\n\n{}\n\n---\n\n",
                            source, chunk.content
                        ));
                    }
                }
            }
        }

        Ok(text)
    }
}

fn main() {
    eprintln!("GraphRAG MCP server running via stdio (with sampling support)");

    let mut server = McpServer::new();
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("Read error: {}", e);
                continue;
            }
        };

        if line.is_empty() {
            continue;
        }

        server.handle_message(&line, &mut stdout);
    }
}
