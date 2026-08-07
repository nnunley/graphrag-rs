//! GraphRAG MCP Server
//!
//! A Model Context Protocol server providing knowledge graph memory tools.
//! Implements JSON-RPC 2.0 over stdio with sampling support for LLM-powered features.

use graphrag_core::{
    BruteForceVectorSource, ChunkerConfig, CodeChunkerConfig, CodeLanguage, Database, Embedder,
    EmbedderConfig, EmbedderModel, LexicalIndex, VectorCandidate, VectorCandidateSource,
    chunk_code, chunk_text as core_chunk_text, default_embedder_cache_dir, leit_fusion,
};
use graphrag_llm::{
    EntityTriple, continuation_extraction_prompt, entity_extraction_prompt,
    parse_entity_extraction, reflection_prompt,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// Default chunk size in characters
const DEFAULT_CHUNK_SIZE: usize = 2000;
/// Default overlap between chunks in characters
const DEFAULT_CHUNK_OVERLAP: usize = 200;

/// Chunk text using semantic text-splitter
fn chunk_text(text: &str, chunk_size: usize, overlap: usize, markdown: bool) -> Vec<String> {
    let config = ChunkerConfig::new(chunk_size)
        .with_markdown(markdown)
        .with_overlap(overlap);
    core_chunk_text(text, &config)
}

/// Global embedder instance (lazy initialized)
static EMBEDDER: OnceLock<Result<Embedder, String>> = OnceLock::new();

/// Request ID counter for sampling requests
static REQUEST_ID: AtomicU64 = AtomicU64::new(1000);

/// Get or initialize the embedder
fn get_embedder() -> Result<&'static Embedder, String> {
    EMBEDDER
        .get_or_init(|| {
            eprintln!("Initializing embedder (this may download the model on first use)...");
            let model = std::env::var("GRAPHRAG_EMBED_MODEL")
                .ok()
                .and_then(|m| match m.to_lowercase().as_str() {
                    "minilm" | "mini" => Some(EmbedderModel::MiniLM),
                    "nomic" | "nomic-embed-text" => Some(EmbedderModel::NomicEmbedText),
                    _ => None,
                })
                .unwrap_or(EmbedderModel::NomicEmbedText);

            let config = EmbedderConfig {
                model,
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

// MCP sampling-protocol structs. These describe the wire shape used when
// requesting LLM completions from the connected MCP client. They're declared
// here to document the contract and to be ready when sampling is wired into
// pending-operation handling; for now no constructor exists (see the
// `pending_operations` map and the request/response handler in
// `handle_message`). Drop the allow once sampling is invoked from code.
/// Sampling message content
#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize, Clone)]
struct SamplingContent {
    #[serde(rename = "type")]
    content_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
}

/// Sampling message
#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize, Clone)]
struct SamplingMessage {
    role: String,
    content: SamplingContent,
}

/// Sampling request params
#[allow(dead_code)]
#[derive(Debug, Serialize)]
struct SamplingParams {
    messages: Vec<SamplingMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "systemPrompt")]
    system_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "maxTokens")]
    max_tokens: Option<u32>,
}

/// Sampling response result
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct SamplingResult {
    content: SamplingContent,
    model: Option<String>,
    #[serde(rename = "stopReason")]
    stop_reason: Option<String>,
}

/// Maximum number of self-reflection iterations for entity extraction
const MAX_REFLECTION_ITERATIONS: usize = 2;

/// Pending operation that requires sampling
#[derive(Debug)]
enum PendingOperation {
    /// Process a single chunk for entity extraction
    AddChunk {
        original_id: Value,
        store_name: String,
        chunk_content: String,
        chunk_index: usize,
        total_chunks: usize,
        source: Option<String>,
        /// Remaining chunks to process after this one
        remaining_chunks: Vec<String>,
        /// Accumulated results
        chunks_added: usize,
        entities_extracted: usize,
        /// Entities found so far in this chunk (for reflection)
        current_chunk_entities: Vec<EntityTriple>,
        /// Current reflection iteration (0 = initial extraction)
        reflection_iteration: usize,
    },
    /// Self-reflection: ask if entities were missed
    ReflectionCheck {
        original_id: Value,
        store_name: String,
        chunk_content: String,
        chunk_index: usize,
        total_chunks: usize,
        source: Option<String>,
        remaining_chunks: Vec<String>,
        chunks_added: usize,
        entities_extracted: usize,
        current_chunk_entities: Vec<EntityTriple>,
        reflection_iteration: usize,
    },
    SummarizeCommunity {
        original_id: Value,
        community_id: i64,
        #[allow(dead_code)]
        entities: Vec<String>,
    },
}

/// MCP Server implementation
struct McpServer {
    data_dir: PathBuf,
    default_store: String,
    pending_operations: HashMap<u64, PendingOperation>,
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

/// Per-chunk batch context: where in the chunk stream we are and how much
/// work has accumulated. Bundled so process_chunk and its async continuations
/// don't need 5-7 individual positional arguments threaded through them.
struct ChunkBatchProgress {
    chunk_index: usize,
    total_chunks: usize,
    /// Chunks not yet started.
    remaining_chunks: Vec<String>,
    /// Total chunks successfully written across the batch so far.
    chunks_added: usize,
    /// Total entity-relations extracted across the batch so far.
    entities_extracted: usize,
}

impl McpServer {
    fn new() -> Self {
        Self {
            data_dir: get_data_dir(),
            default_store: std::env::var("GRAPHRAG_STORE")
                .unwrap_or_else(|_| "conversations".to_string()),
            pending_operations: HashMap::new(),
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

    fn next_request_id() -> u64 {
        REQUEST_ID.fetch_add(1, Ordering::SeqCst)
    }

    /// Handle incoming message - could be a request or a response to our sampling request
    fn handle_message(&mut self, line: &str, stdout: &mut impl Write) {
        // Try to parse as a response first (to our sampling requests)
        if let Ok(response) = serde_json::from_str::<JsonRpcResponse>(line)
            && (response.result.is_some() || response.error.is_some())
            && let Some(id) = response.id.as_u64()
        {
            self.handle_sampling_response(id, response, stdout);
            return;
        }

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
                    "description": "Search knowledge graph memory using natural language. Embeds the query locally and returns semantically similar content plus related entities discovered through graph traversal.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language query to search for"
                            },
                            "top": { "type": "number", "description": "Number of results (default: 5)" },
                            "expand": { "type": "number", "description": "Graph expansion depth (default: 1)" },
                            "store": { "type": "string", "description": format!("Store name (default: {})", self.default_store) }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "graphrag_add_document",
                    "description": "Add a document to the knowledge graph. Automatically chunks long documents, uses LLM to extract entities and relations from each chunk, then embeds and stores the content.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The document content to add"
                            },
                            "source": {
                                "type": "string",
                                "description": "Optional source identifier (e.g., filename, URL)"
                            },
                            "store": { "type": "string", "description": format!("Store name (default: {})", self.default_store) },
                            "chunk_size": {
                                "type": "number",
                                "description": format!("Chunk size in characters (default: {})", DEFAULT_CHUNK_SIZE)
                            },
                            "chunk_overlap": {
                                "type": "number",
                                "description": format!("Overlap between chunks in characters (default: {})", DEFAULT_CHUNK_OVERLAP)
                            }
                        },
                        "required": ["content"]
                    }
                },
                {
                    "name": "graphrag_add_code",
                    "description": "Add source code to the knowledge graph with AST-aware chunking. Uses tree-sitter to split code at semantic boundaries (functions, classes, modules), extracts entities and relations, then embeds and stores the content.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The source code content to add"
                            },
                            "file_path": {
                                "type": "string",
                                "description": "File path (used for language detection from extension, e.g., 'main.rs', 'app.py')"
                            },
                            "language": {
                                "type": "string",
                                "description": "Programming language (optional, auto-detected from file_path). Supported: rust, python, javascript, typescript, go, c, cpp"
                            },
                            "source": {
                                "type": "string",
                                "description": "Optional source identifier (e.g., repository URL, module name)"
                            },
                            "store": { "type": "string", "description": format!("Store name (default: {})", self.default_store) },
                            "chunk_size": {
                                "type": "number",
                                "description": format!("Chunk size in characters (default: {})", DEFAULT_CHUNK_SIZE)
                            }
                        },
                        "required": ["content"]
                    }
                },
                {
                    "name": "graphrag_summarize_community",
                    "description": "Generate a summary for a community using LLM. The summary captures the key themes and relationships among the community's entities.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "community_id": {
                                "type": "number",
                                "description": "The community ID to summarize"
                            },
                            "store": { "type": "string", "description": format!("Store name (default: {})", self.default_store) }
                        },
                        "required": ["community_id"]
                    }
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
                            "store": { "type": "string" }
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
                    "description": "Merge two entities into one. All relations and chunk links from source entity are transferred to target. Source entity is deleted.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "target_id": {
                                "type": "number",
                                "description": "ID of the entity to keep (target)"
                            },
                            "source_id": {
                                "type": "number",
                                "description": "ID of the entity to merge into target (will be deleted)"
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
        stdout: &mut impl Write,
    ) -> Option<JsonRpcResponse> {
        let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let args = params.get("arguments").cloned().unwrap_or(json!({}));

        match name {
            "graphrag_recall" => Some(self.tool_recall(&args, id)),
            "graphrag_add_document" => self.tool_add_document(&args, id, stdout),
            "graphrag_add_code" => self.tool_add_code(&args, id, stdout),
            "graphrag_summarize_community" => self.tool_summarize_community(&args, id, stdout),
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

    fn get_store_name(&self, args: &Value) -> String {
        args.get("store")
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or_else(|| self.default_store.clone())
    }

    /// Send a sampling request to the client
    fn send_sampling_request(
        &mut self,
        system_prompt: &str,
        user_message: &str,
        max_tokens: u32,
        operation: PendingOperation,
        stdout: &mut impl Write,
    ) -> u64 {
        let request_id = Self::next_request_id();

        let request = json!({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "sampling/createMessage",
            "params": {
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": user_message
                        }
                    }
                ],
                "systemPrompt": system_prompt,
                "maxTokens": max_tokens
            }
        });

        self.pending_operations.insert(request_id, operation);

        if let Err(e) = writeln!(stdout, "{}", serde_json::to_string(&request).unwrap()) {
            eprintln!("Failed to send sampling request: {}", e);
        }
        let _ = stdout.flush();

        request_id
    }

    /// Handle response to our sampling request
    fn handle_sampling_response(
        &mut self,
        request_id: u64,
        response: JsonRpcResponse,
        stdout: &mut impl Write,
    ) {
        let operation = match self.pending_operations.remove(&request_id) {
            Some(op) => op,
            None => {
                eprintln!("Received response for unknown request: {}", request_id);
                return;
            }
        };

        // Extract text from sampling response
        let text = response
            .result
            .as_ref()
            .and_then(|r| r.get("content"))
            .and_then(|c| c.get("text"))
            .and_then(|t| t.as_str())
            .unwrap_or("");

        match operation {
            PendingOperation::AddChunk {
                original_id,
                store_name,
                chunk_content,
                chunk_index,
                total_chunks,
                source,
                remaining_chunks,
                chunks_added,
                entities_extracted,
                current_chunk_entities,
                reflection_iteration,
            } => {
                // Process extraction result and potentially trigger reflection
                self.handle_extraction_result(
                    original_id,
                    store_name,
                    chunk_content,
                    chunk_index,
                    total_chunks,
                    source,
                    remaining_chunks,
                    chunks_added,
                    entities_extracted,
                    current_chunk_entities,
                    reflection_iteration,
                    text,
                    stdout,
                );
            }
            PendingOperation::ReflectionCheck {
                original_id,
                store_name,
                chunk_content,
                chunk_index,
                total_chunks,
                source,
                remaining_chunks,
                chunks_added,
                entities_extracted,
                current_chunk_entities,
                reflection_iteration,
            } => {
                // Handle reflection response (yes/no to "did you miss entities?")
                self.handle_reflection_response(
                    original_id,
                    store_name,
                    chunk_content,
                    chunk_index,
                    total_chunks,
                    source,
                    remaining_chunks,
                    chunks_added,
                    entities_extracted,
                    current_chunk_entities,
                    reflection_iteration,
                    text,
                    stdout,
                );
            }
            PendingOperation::SummarizeCommunity {
                original_id,
                community_id,
                entities: _,
            } => {
                let final_response =
                    self.complete_summarize_community(original_id, community_id, text);
                if let Err(e) = writeln!(
                    stdout,
                    "{}",
                    serde_json::to_string(&final_response).unwrap()
                ) {
                    eprintln!("Write error: {}", e);
                }
                let _ = stdout.flush();
            }
        };
    }

    // ============ Tool implementations ============

    fn tool_add_document(
        &mut self,
        args: &Value,
        id: Value,
        stdout: &mut impl Write,
    ) -> Option<JsonRpcResponse> {
        let content = match args.get("content").and_then(|v| v.as_str()) {
            Some(c) => c.to_string(),
            None => return Some(self.tool_error(id, "Missing 'content' argument".to_string())),
        };

        let store_name = self.get_store_name(args);
        let source = args
            .get("source")
            .and_then(|v| v.as_str())
            .map(String::from);

        // Get chunk parameters from args or use defaults
        let chunk_size = args
            .get("chunk_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_CHUNK_SIZE);
        let chunk_overlap = args
            .get("chunk_overlap")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_CHUNK_OVERLAP);

        // Check if content looks like markdown
        let is_markdown = args
            .get("markdown")
            .and_then(|v| v.as_bool())
            .unwrap_or_else(|| {
                // Auto-detect markdown
                content.contains("# ") || content.contains("## ") || content.contains("```")
            });

        // Split content into chunks using semantic text-splitter
        let mut chunks = chunk_text(&content, chunk_size, chunk_overlap, is_markdown);
        let total_chunks = chunks.len();

        if chunks.is_empty() {
            return Some(self.tool_error(id, "No content to process".to_string()));
        }

        eprintln!(
            "Processing document: {} chars -> {} chunks",
            content.len(),
            total_chunks
        );

        // Take first chunk to process, rest go in remaining_chunks
        let first_chunk = chunks.remove(0);

        // Start processing first chunk
        self.process_chunk(
            id,
            store_name,
            first_chunk,
            source,
            ChunkBatchProgress {
                chunk_index: 0,
                total_chunks,
                remaining_chunks: chunks,
                chunks_added: 0,
                entities_extracted: 0,
            },
            stdout,
        );

        None // Response will come later
    }

    /// Add source code to the knowledge graph with AST-aware chunking
    fn tool_add_code(
        &mut self,
        args: &Value,
        id: Value,
        stdout: &mut impl Write,
    ) -> Option<JsonRpcResponse> {
        let content = match args.get("content").and_then(|v| v.as_str()) {
            Some(c) => c.to_string(),
            None => return Some(self.tool_error(id, "Missing 'content' argument".to_string())),
        };

        // Determine programming language
        let language = if let Some(lang_str) = args.get("language").and_then(|v| v.as_str()) {
            match CodeLanguage::from_name(lang_str) {
                Some(lang) => lang,
                None => {
                    return Some(self.tool_error(
                        id,
                        format!(
                            "Unsupported language: '{}'. Supported: rust, python, javascript, typescript, go, c, cpp",
                            lang_str
                        ),
                    ))
                }
            }
        } else if let Some(file_path) = args.get("file_path").and_then(|v| v.as_str()) {
            let ext = file_path.rsplit('.').next().unwrap_or("");
            match CodeLanguage::from_extension(ext) {
                Some(lang) => lang,
                None => {
                    return Some(self.tool_error(
                        id,
                        format!(
                            "Cannot detect language from extension '.{}'. Supported extensions: rs, py, js, ts, tsx, go, c, h, cpp, cc, cxx, hpp",
                            ext
                        ),
                    ))
                }
            }
        } else {
            return Some(
                self.tool_error(
                    id,
                    "Either 'language' or 'file_path' (for auto-detection) must be provided"
                        .to_string(),
                ),
            );
        };

        let store_name = self.get_store_name(args);
        let source = args
            .get("source")
            .and_then(|v| v.as_str())
            .or_else(|| args.get("file_path").and_then(|v| v.as_str()))
            .map(String::from);

        let chunk_size = args
            .get("chunk_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_CHUNK_SIZE);

        // Use AST-aware code chunking
        let config = CodeChunkerConfig::new(language).with_chunk_size(chunk_size);
        let mut chunks = match chunk_code(&content, &config) {
            Ok(c) => c,
            Err(e) => return Some(self.tool_error(id, format!("Code chunking failed: {}", e))),
        };

        let total_chunks = chunks.len();

        if chunks.is_empty() {
            return Some(self.tool_error(id, "No code content to process".to_string()));
        }

        eprintln!(
            "Processing code ({:?}): {} chars -> {} chunks",
            language,
            content.len(),
            total_chunks
        );

        // Take first chunk to process
        let first_chunk = chunks.remove(0);

        // Use same processing pipeline as documents
        self.process_chunk(
            id,
            store_name,
            first_chunk,
            source,
            ChunkBatchProgress {
                chunk_index: 0,
                total_chunks,
                remaining_chunks: chunks,
                chunks_added: 0,
                entities_extracted: 0,
            },
            stdout,
        );

        None // Response will come later
    }

    /// Start processing a chunk by sending entity extraction request
    fn process_chunk(
        &mut self,
        original_id: Value,
        store_name: String,
        chunk_content: String,
        source: Option<String>,
        batch: ChunkBatchProgress,
        stdout: &mut impl Write,
    ) {
        let ChunkBatchProgress {
            chunk_index,
            total_chunks,
            remaining_chunks,
            chunks_added,
            entities_extracted,
        } = batch;
        let prompt = entity_extraction_prompt(&chunk_content, chunk_index, total_chunks);

        eprintln!(
            "Processing chunk {}/{} ({} chars)",
            chunk_index + 1,
            total_chunks,
            chunk_content.len()
        );

        self.send_sampling_request(
            &prompt.system,
            &prompt.user,
            prompt.max_tokens,
            PendingOperation::AddChunk {
                original_id,
                store_name,
                chunk_content,
                chunk_index,
                total_chunks,
                source,
                remaining_chunks,
                chunks_added,
                entities_extracted,
                current_chunk_entities: Vec::new(),
                reflection_iteration: 0,
            },
            stdout,
        );
    }

    /// Handle extraction result - either trigger reflection or finalize chunk
    #[allow(clippy::too_many_arguments)]
    fn handle_extraction_result(
        &mut self,
        original_id: Value,
        store_name: String,
        chunk_content: String,
        chunk_index: usize,
        total_chunks: usize,
        source: Option<String>,
        remaining_chunks: Vec<String>,
        chunks_added: usize,
        entities_extracted: usize,
        mut current_chunk_entities: Vec<EntityTriple>,
        reflection_iteration: usize,
        llm_response: &str,
        stdout: &mut impl Write,
    ) {
        // Parse new entities from LLM response
        let new_entities = parse_entity_extraction(llm_response)
            .map(|extraction| extraction.entities)
            .unwrap_or_default();

        // Merge new entities with existing ones
        current_chunk_entities.extend(new_entities.clone());

        eprintln!(
            "Chunk {}/{} iteration {}: found {} new entities (total: {})",
            chunk_index + 1,
            total_chunks,
            reflection_iteration,
            new_entities.len(),
            current_chunk_entities.len()
        );

        // Decide whether to do reflection
        // Only reflect if we found entities and haven't hit max iterations
        if !new_entities.is_empty() && reflection_iteration < MAX_REFLECTION_ITERATIONS {
            // Ask if we missed any entities
            self.send_reflection_check(
                original_id,
                store_name,
                chunk_content,
                chunk_index,
                total_chunks,
                source,
                remaining_chunks,
                chunks_added,
                entities_extracted,
                current_chunk_entities,
                reflection_iteration,
                stdout,
            );
        } else {
            // No more reflection needed - finalize this chunk
            self.finalize_chunk(
                original_id,
                store_name,
                chunk_content,
                chunk_index,
                total_chunks,
                source,
                remaining_chunks,
                chunks_added,
                entities_extracted,
                current_chunk_entities,
                stdout,
            );
        }
    }

    /// Send a reflection check: "Did you miss any entities?"
    #[allow(clippy::too_many_arguments)]
    fn send_reflection_check(
        &mut self,
        original_id: Value,
        store_name: String,
        chunk_content: String,
        chunk_index: usize,
        total_chunks: usize,
        source: Option<String>,
        remaining_chunks: Vec<String>,
        chunks_added: usize,
        entities_extracted: usize,
        current_chunk_entities: Vec<EntityTriple>,
        reflection_iteration: usize,
        stdout: &mut impl Write,
    ) {
        let prompt = reflection_prompt(&chunk_content, &current_chunk_entities);

        self.send_sampling_request(
            &prompt.system,
            &prompt.user,
            prompt.max_tokens,
            PendingOperation::ReflectionCheck {
                original_id,
                store_name,
                chunk_content,
                chunk_index,
                total_chunks,
                source,
                remaining_chunks,
                chunks_added,
                entities_extracted,
                current_chunk_entities,
                reflection_iteration,
            },
            stdout,
        );
    }

    /// Handle reflection response - either extract more or finalize
    #[allow(clippy::too_many_arguments)]
    fn handle_reflection_response(
        &mut self,
        original_id: Value,
        store_name: String,
        chunk_content: String,
        chunk_index: usize,
        total_chunks: usize,
        source: Option<String>,
        remaining_chunks: Vec<String>,
        chunks_added: usize,
        entities_extracted: usize,
        current_chunk_entities: Vec<EntityTriple>,
        reflection_iteration: usize,
        response: &str,
        stdout: &mut impl Write,
    ) {
        let missed = response.to_lowercase().contains("yes");

        if missed {
            eprintln!(
                "Chunk {}/{}: reflection says entities were missed, extracting more...",
                chunk_index + 1,
                total_chunks
            );

            // Do another extraction pass
            self.send_continuation_extraction(
                original_id,
                store_name,
                chunk_content,
                chunk_index,
                total_chunks,
                source,
                remaining_chunks,
                chunks_added,
                entities_extracted,
                current_chunk_entities,
                reflection_iteration + 1,
                stdout,
            );
        } else {
            eprintln!(
                "Chunk {}/{}: reflection says extraction complete",
                chunk_index + 1,
                total_chunks
            );

            // Finalize this chunk
            self.finalize_chunk(
                original_id,
                store_name,
                chunk_content,
                chunk_index,
                total_chunks,
                source,
                remaining_chunks,
                chunks_added,
                entities_extracted,
                current_chunk_entities,
                stdout,
            );
        }
    }

    /// Send a continuation extraction request (after reflection said "yes")
    #[allow(clippy::too_many_arguments)]
    fn send_continuation_extraction(
        &mut self,
        original_id: Value,
        store_name: String,
        chunk_content: String,
        chunk_index: usize,
        total_chunks: usize,
        source: Option<String>,
        remaining_chunks: Vec<String>,
        chunks_added: usize,
        entities_extracted: usize,
        current_chunk_entities: Vec<EntityTriple>,
        reflection_iteration: usize,
        stdout: &mut impl Write,
    ) {
        let prompt = continuation_extraction_prompt(&chunk_content, &current_chunk_entities);

        self.send_sampling_request(
            &prompt.system,
            &prompt.user,
            prompt.max_tokens,
            PendingOperation::AddChunk {
                original_id,
                store_name,
                chunk_content,
                chunk_index,
                total_chunks,
                source,
                remaining_chunks,
                chunks_added,
                entities_extracted,
                current_chunk_entities,
                reflection_iteration,
            },
            stdout,
        );
    }

    /// Finalize a chunk - store entities and embeddings, then move to next chunk
    #[allow(clippy::too_many_arguments)]
    fn finalize_chunk(
        &mut self,
        original_id: Value,
        store_name: String,
        chunk_content: String,
        chunk_index: usize,
        total_chunks: usize,
        source: Option<String>,
        mut remaining_chunks: Vec<String>,
        chunks_added: usize,
        entities_extracted: usize,
        entities: Vec<EntityTriple>,
        stdout: &mut impl Write,
    ) {
        // Get embedder
        let embedder = match get_embedder() {
            Ok(e) => e,
            Err(e) => {
                self.send_final_response(self.tool_error(original_id, e), stdout);
                return;
            }
        };

        // Embed the chunk
        let embedding = match embedder.embed(&chunk_content) {
            Ok(e) => e,
            Err(e) => {
                self.send_final_response(self.tool_error(original_id, e.to_string()), stdout);
                return;
            }
        };

        // Open database
        let db = match Database::open(&self.db_path()) {
            Ok(d) => d,
            Err(e) => {
                self.send_final_response(self.tool_error(original_id, e.to_string()), stdout);
                return;
            }
        };

        // Ensure store exists with correct dimension
        let store = match db.get_store(&store_name) {
            Ok(s) => s,
            Err(_) => match db.create_store(&store_name, embedder.dimension()) {
                Ok(s) => s,
                Err(e) => {
                    self.send_final_response(self.tool_error(original_id, e.to_string()), stdout);
                    return;
                }
            },
        };

        if store.dim != embedding.len() {
            self.send_final_response(
                self.tool_error(
                    original_id,
                    format!(
                        "Dimension mismatch: store has {}, embedder produces {}",
                        store.dim,
                        embedding.len()
                    ),
                ),
                stdout,
            );
            return;
        }

        // Add chunk with source indicating chunk index
        let chunk_source = source.as_ref().map(|s| {
            if total_chunks > 1 {
                format!("{}#chunk{}", s, chunk_index + 1)
            } else {
                s.clone()
            }
        });

        let chunk_id =
            match db.add_chunk(&store_name, &chunk_content, chunk_source.as_deref(), None) {
                Ok(id) => id,
                Err(e) => {
                    self.send_final_response(self.tool_error(original_id, e.to_string()), stdout);
                    return;
                }
            };
        // Lexical cache is now stale for this store; next recall rebuilds.
        // Cheaper to invalidate here than after the whole batch finishes —
        // multi-chunk add_document writes one chunk per sampling callback,
        // so we'd otherwise serve N-1 stale recalls during a single ingest.
        self.invalidate_lexical(&store_name);

        // Store the canonical embedding
        if let Err(e) = db.set_chunk_embedding(chunk_id, &embedding) {
            self.send_final_response(self.tool_error(original_id, e.to_string()), stdout);
            return;
        }

        // Process entities
        let mut entity_count = 0;
        for entity in &entities {
            if let Ok(head_id) = db.get_or_create_entity(
                &store_name,
                &entity.head,
                entity.head_type.as_deref(),
                None,
            ) && let Ok(tail_id) = db.get_or_create_entity(
                &store_name,
                &entity.tail,
                entity.tail_type.as_deref(),
                None,
            ) {
                // Add relation
                let _ = db.add_relation(&store_name, head_id, tail_id, &entity.relation, None);

                // Link to chunk
                let _ = db.link_chunk_entity(chunk_id, head_id);
                let _ = db.link_chunk_entity(chunk_id, tail_id);

                entity_count += 1;
            }
        }

        let new_chunks_added = chunks_added + 1;
        let new_entities_extracted = entities_extracted + entity_count;

        eprintln!(
            "Chunk {}/{} complete: {} entities extracted",
            chunk_index + 1,
            total_chunks,
            entity_count
        );

        // Check if there are more chunks to process
        if remaining_chunks.is_empty() {
            // All done - send final success response
            let response = self.tool_success(
                original_id,
                format!(
                    "Added document to '{}': {} chunks, {} entity relations extracted",
                    store_name, new_chunks_added, new_entities_extracted
                ),
            );
            self.send_final_response(response, stdout);
        } else {
            // Process next chunk
            let next_chunk = remaining_chunks.remove(0);
            self.process_chunk(
                original_id,
                store_name,
                next_chunk,
                source,
                ChunkBatchProgress {
                    chunk_index: chunk_index + 1,
                    total_chunks,
                    remaining_chunks,
                    chunks_added: new_chunks_added,
                    entities_extracted: new_entities_extracted,
                },
                stdout,
            );
        }
    }

    /// Helper to send a final response to stdout
    fn send_final_response(&self, response: JsonRpcResponse, stdout: &mut impl Write) {
        if let Err(e) = writeln!(stdout, "{}", serde_json::to_string(&response).unwrap()) {
            eprintln!("Write error: {}", e);
        }
        let _ = stdout.flush();
    }

    fn tool_summarize_community(
        &mut self,
        args: &Value,
        id: Value,
        stdout: &mut impl Write,
    ) -> Option<JsonRpcResponse> {
        let community_id = match args.get("community_id").and_then(|v| v.as_i64()) {
            Some(c) => c,
            None => {
                return Some(self.tool_error(id, "Missing 'community_id' argument".to_string()));
            }
        };

        // store_name is captured but not needed after community lookup; the
        // community_id is store-scoped via FK on the chunks table.
        let _ = self.get_store_name(args);

        // Get community entities
        let db = match Database::open(&self.db_path()) {
            Ok(d) => d,
            Err(e) => return Some(self.tool_error(id, e.to_string())),
        };

        let entities = match db.get_community_entities(community_id) {
            Ok(e) => e,
            Err(e) => return Some(self.tool_error(id, e.to_string())),
        };

        if entities.is_empty() {
            return Some(self.tool_error(id, "Community has no entities".to_string()));
        }

        // Build context with entities and their relations
        let mut context = String::from("Entities in this community:\n");
        let entity_names: Vec<String> = entities.iter().map(|e| e.name.clone()).collect();

        for entity in &entities {
            let entity_type = entity.entity_type.as_deref().unwrap_or("unknown");
            context.push_str(&format!("- {} ({})\n", entity.name, entity_type));

            // Get relations for this entity
            if let Ok(relations) = db.get_relations_for_entity(entity.id) {
                for rel in relations {
                    let other_id = if rel.head_id == entity.id {
                        rel.tail_id
                    } else {
                        rel.head_id
                    };
                    if let Ok(other) = db.get_entity_by_id(other_id)
                        && entity_names.contains(&other.name)
                    {
                        context.push_str(&format!("  -> {} -> {}\n", rel.relation, other.name));
                    }
                }
            }
        }

        let system_prompt = "You are a summarization system. Given a list of entities and their relationships, write a concise 1-2 sentence summary that captures the key theme or topic of this cluster. Output ONLY the summary text, nothing else.";

        self.send_sampling_request(
            system_prompt,
            &context,
            200,
            PendingOperation::SummarizeCommunity {
                original_id: id,
                community_id,
                entities: entity_names,
            },
            stdout,
        );

        None // Response will come later
    }

    fn complete_summarize_community(
        &self,
        id: Value,
        community_id: i64,
        summary: &str,
    ) -> JsonRpcResponse {
        let db = match Database::open(&self.db_path()) {
            Ok(d) => d,
            Err(e) => return self.tool_error(id, e.to_string()),
        };

        if let Err(e) = db.update_community_summary(community_id, summary) {
            return self.tool_error(id, e.to_string());
        }

        self.tool_success(
            id,
            format!(
                "Updated community {} summary: {}",
                community_id,
                summary.trim()
            ),
        )
    }

    fn tool_recall(&self, args: &Value, id: Value) -> JsonRpcResponse {
        let query = match args.get("query").and_then(|v| v.as_str()) {
            Some(q) => q,
            None => return self.tool_error(id, "Missing 'query' argument".to_string()),
        };

        let embedder = match get_embedder() {
            Ok(e) => e,
            Err(e) => return self.tool_error(id, e),
        };

        let embedding = match embedder.embed(query) {
            Ok(e) => e,
            Err(e) => return self.tool_error(id, e.to_string()),
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
                    "Dimension mismatch: embedder={}, store={}",
                    embedding.len(),
                    store.dim
                ),
            );
        }

        // Hybrid recall: brute-force cosine (semantic) ∪ leit BM25 (lexical), fused via RRF.
        // Pull candidate_k per lane so RRF has overlap to fuse over.
        let candidate_k = top.max(5) * 4;

        let vector_hits = match BruteForceVectorSource::for_store(&db, &store_name) {
            Ok(source) => match source.top_candidates(&embedding, candidate_k) {
                Ok(r) => r,
                Err(e) => return self.tool_error(id, e.to_string()),
            },
            Err(e) => return self.tool_error(id, e.to_string()),
        };

        let lexical_hits = match self.cached_lexical_for(&db, &store_name) {
            Ok(Some(idx)) => match idx.search(query, candidate_k) {
                Ok(h) => h,
                Err(e) => return self.tool_error(id, e.to_string()),
            },
            Ok(None) => Vec::new(),
            Err(e) => return self.tool_error(id, e),
        };

        if vector_hits.is_empty() && lexical_hits.is_empty() {
            return self.tool_success(id, String::new());
        }

        let vector_ranked: Vec<leit_fusion::RankedResult> = vector_hits
            .iter()
            .enumerate()
            .map(|(i, h)| leit_fusion::RankedResult::new(h.chunk_id.to_string(), i + 1))
            .collect();
        let lexical_ranked: Vec<leit_fusion::RankedResult> = lexical_hits
            .iter()
            .enumerate()
            .map(|(i, (cid, _))| leit_fusion::RankedResult::new(cid.to_string(), i + 1))
            .collect();
        let fused = leit_fusion::fuse_default(&[vector_ranked, lexical_ranked]);

        // Translate fused output back into VectorCandidate so the existing
        // format_search_results + graph-expansion path keeps working unchanged.
        // VectorCandidate.score carries the RRF score here (higher = better).
        let results: Vec<VectorCandidate> = fused
            .iter()
            .take(top)
            .filter_map(|f| {
                f.id.parse::<i64>().ok().map(|chunk_id| VectorCandidate {
                    chunk_id,
                    score: f.score as f32,
                })
            })
            .collect();

        match self.format_search_results(&db, &results, expand) {
            Ok(text) => self.tool_success(id, text),
            Err(e) => self.tool_error(id, e),
        }
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

        let results = match source.top_candidates(&embedding, top) {
            Ok(r) => r,
            Err(e) => return self.tool_error(id, e.to_string()),
        };

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
