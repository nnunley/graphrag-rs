# GraphRAG MCP Server

MCP (Model Context Protocol) server for the nu_plugin_graphrag knowledge graph memory system.

## Features

- **recall** - Semantic search with graph expansion
- **entities** - List all entities with types
- **relations** - Get relations for an entity
- **communities** - List detected communities with summaries
- **stats** - Store statistics

## Requirements

- Node.js >= 18
- nushell with nu_plugin_graphrag installed
- Ollama running with `nomic-embed-text` model (for semantic search)

## Installation

### As a Claude Code plugin

```bash
# From the mcp-plugin directory
npm install

# Add to Claude Code plugins (manual setup required)
```

### Manual MCP server

```bash
cd mcp-plugin
npm install
node mcp-server.js
```

## Configuration

Environment variables:

- `GRAPHRAG_STORE` - Default store name (default: "conversations")
- `NU_PATH` - Path to nushell binary (default: "nu")
- `OLLAMA_HOST` - Ollama server URL (default: localhost:11434)

## Usage

The MCP server exposes these tools:

### recall
Search knowledge graph memory for relevant context.

```json
{
  "query": "authentication",
  "top": 5,
  "store": "conversations"
}
```

### entities
List all entities in the knowledge graph.

```json
{
  "store": "conversations"
}
```

### relations
Get all relations involving an entity.

```json
{
  "entity": "JWT",
  "store": "conversations"
}
```

### communities
List detected communities with their summaries.

```json
{
  "store": "conversations"
}
```

### stats
Get statistics about a store.

```json
{
  "store": "conversations"
}
```
