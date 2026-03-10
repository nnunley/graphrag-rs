# graphrag-rs

A Rust workspace combining HNSW vector search with knowledge graph storage for GraphRAG-style retrieval. Provides both a Nushell plugin and an MCP server.

## Features

- **Vector Search**: Fast approximate nearest neighbor search via HNSW (usearch)
- **Knowledge Graph**: Entity and relation storage in SQLite
- **Community Detection**: Leiden algorithm for finding entity clusters
- **Graph Expansion**: Retrieve related chunks by traversing entity connections

## Installation

```bash
cargo build --release
plugin add target/release/nu_plugin_graphrag
plugin use graphrag
```

## Commands

### Store Management

```nushell
# Create a store with embedding dimension
graphrag create mystore --dim 768

# List all stores with stats
graphrag list

# Delete a store
graphrag delete mystore
```

### Adding Data

```nushell
# Add a chunk with embedding (required: content, embedding)
{
    content: "The quick brown fox"
    embedding: [0.1, 0.2, ...]  # Must match store dimension
    source: "doc:123"           # Optional
    metadata: '{"key": "val"}'  # Optional JSON
    entities: [                 # Optional entity extraction
        {head: "fox", head_type: "animal", relation: "is", tail: "quick", tail_type: "property"}
    ]
} | graphrag add mystore
```

### Searching

```nushell
# Vector similarity search
graphrag search mystore $embedding --top 5

# GraphRAG query: vector search + graph expansion
graphrag query mystore $embedding --top 5 --expand 1
```

### Knowledge Graph

```nushell
# List entities
graphrag entities mystore

# List relations
graphrag relations mystore
```

### Community Detection

```nushell
# Detect and store communities using Leiden algorithm
graphrag store-communities mystore --clear

# List communities with their entities
graphrag list-communities mystore

# Get community details
graphrag get-community mystore 1

# Update community summary (typically from LLM)
graphrag update-summary 1 "These entities relate to..."
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        graphrag-rs                       │
├─────────────────────────────────────────────────────────┤
│  HNSW Index (usearch)     │  SQLite Database            │
│  - Fast ANN search        │  - stores, chunks           │
│  - Per-store indexes      │  - entities, relations      │
│  - Persistent on disk     │  - communities              │
├─────────────────────────────────────────────────────────┤
│  Leiden Algorithm                                        │
│  - Modularity optimization                               │
│  - Refinement for well-connected communities             │
│  - Hierarchical detection                                │
└─────────────────────────────────────────────────────────┘
```

## Data Model

- **Store**: Named collection with fixed embedding dimension
- **Chunk**: Text content with embedding vector and optional metadata
- **Entity**: Named node with optional type and properties
- **Relation**: Directed edge between entities with relation type
- **Community**: Cluster of related entities detected by Leiden algorithm

## Integration with ai.nu

This plugin provides the "hard layer" for ai.nu's memory system:

```nushell
use ai
ai ai-new-session

# Initialize memory (creates graphrag store)
ai ai-memory init

# Add conversation messages
{role: 'user', content: 'How does GraphRAG work?'} | ai ai-memory add

# Semantic recall
ai ai-memory recall 'vector search' --include-communities
```

See `ai/memory.nu` for the soft layer implementation that handles:
- Embedding generation via ollama
- Entity extraction via LLM
- Community summarization via LLM

## References

- [Microsoft GraphRAG](https://github.com/microsoft/graphrag) - Original GraphRAG implementation
- [Leiden Algorithm](https://www.nature.com/articles/s41598-019-41695-z) - Community detection paper
- [usearch](https://github.com/unum-cloud/usearch) - HNSW implementation
