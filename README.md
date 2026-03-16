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
