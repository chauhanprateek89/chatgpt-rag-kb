"""Ingest markdown notes into a ChromaDB vector store for local retrieval."""

from pathlib import Path

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def main() -> None:
    # TODO: Implement note ingestion and vectorstore persistence.
    pass


if __name__ == "__main__":
    main()
