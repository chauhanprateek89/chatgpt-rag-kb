# chatgpt-rag-kb

## Overview

This project converts a ChatGPT data export (`data/raw/conversations.json`) into
Obsidian-compatible markdown notes in `notes/`.

It then builds a local RAG (Retrieval-Augmented Generation) pipeline using:

- ChromaDB for the persistent vector store
- Ollama embeddings for chunk indexing
- Ollama chat models for grounded local answers

## Folder Structure

```text
chatgpt-rag-kb/
├── data/
│   ├── raw/              # ChatGPT export drop zone (gitignored)
│   └── processed/        # cleaned intermediate files
├── notes/                # Obsidian-ready markdown output
├── scripts/
│   ├── parse_chatgpt.py       # converts conversations.json → .md notes
│   ├── build_vectorstore.py   # ingests notes into ChromaDB
│   └── query.py               # RAG query CLI interface
├── mcp_server.py         # MCP server exposing query_kb and search_kb tools
├── .env.example          # template for environment variables
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup Instructions

1. Create and activate a conda environment with Python 3.10+.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy environment template and adjust values if needed:

   ```bash
   cp .env.example .env
   ```

## Configuration

The main runtime settings live in `.env`:

```bash
OLLAMA_MODEL=llama3
EMBED_MODEL=nomic-embed-text
CHROMA_DB_PATH=./chroma_db
NOTES_PATH=./notes
CHUNK_SIZE=500
CHUNK_OVERLAP=50
OLLAMA_BASE_URL=http://localhost:11434
```

## Usage

1. Parse ChatGPT export into markdown notes:

   ```bash
   python scripts/parse_chatgpt.py
   ```

2. Build ChromaDB vector store from notes:

   ```bash
   python scripts/build_vectorstore.py
   ```

3. Query the local RAG pipeline:

   ```bash
   python scripts/query.py
   ```

## MCP Server (Claude Code integration)

`mcp_server.py` exposes two tools to Claude Code:

- `query_kb` — embeds a question, retrieves relevant chunks, and returns a grounded answer via Ollama
- `search_kb` — returns raw matching chunks with source titles and similarity scores

### Prerequisites

- Ollama must be running (`ollama serve`) with the embed model and chat model pulled
- The ChromaDB vector store must already be built (`python scripts/build_vectorstore.py`)
- Dependencies must be installed into `.py312pkgs` (used as the standalone package dir for the server)

### Registering with Claude Code

Claude Code uses `~/.claude.json` as its MCP registry (not `~/.claude/settings.json`). Use the
Claude CLI to register the server at user scope so it is available in every project:

```bash
claude mcp add -s user chatgpt-kb \
  -e PYTHONPATH=/home/pc/src/chatgpt-rag-kb/.py312pkgs \
  -- python3.12 /home/pc/src/chatgpt-rag-kb/mcp_server.py
```

Verify the server is connected:

```bash
claude mcp list
# chatgpt-kb: python3.12 ... - ✓ Connected
```

Reload the VS Code window (`Ctrl+Shift+P` → "Developer: Reload Window") for the server to appear
in the Claude Code MCP panel.

### Troubleshooting

If the server fails to start, test it directly:

```bash
PYTHONPATH=/home/pc/src/chatgpt-rag-kb/.py312pkgs \
  python3.12 /home/pc/src/chatgpt-rag-kb/mcp_server.py
```

A clean exit (no output, no traceback) means the server initialises correctly.
