#!/usr/bin/env python3
"""MCP server exposing the chatgpt-rag-kb knowledge base to Claude."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib import error, request

ROOT = Path(__file__).resolve().parent
PKGS = ROOT / ".py312pkgs"
if str(PKGS) not in sys.path:
    sys.path.insert(0, str(PKGS))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from mcp.server.fastmcp import FastMCP  # noqa: E402

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None  # type: ignore[assignment]

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[assignment]

from query import (  # noqa: E402
    COLLECTION_NAME,
    DEFAULT_EMBED_MODEL,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_TOP_K,
    OllamaConnectionError,
    OllamaError,
    build_prompt,
    embed_text,
    query_collection,
)

# ---------------------------------------------------------------------------
# Server init
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "chatgpt-kb",
    instructions="Personal ChatGPT knowledge base — search and query your past conversations",
)

_state: dict[str, Any] = {"collection": None}


def _load_env() -> None:
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")


def _resolve(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else ROOT / p


def _open_collection() -> Any:
    if chromadb is None:
        raise RuntimeError("chromadb is not installed.")
    _load_env()
    db_path = _resolve(os.getenv("CHROMA_DB_PATH", "./chroma_db"))
    if not db_path.exists():
        raise RuntimeError(
            f"ChromaDB not found at {db_path}. "
            "Run `python scripts/build_vectorstore.py` first."
        )
    chroma_settings = ChromaSettings(
        chroma_api_impl="chromadb.api.segment.SegmentAPI",
        is_persistent=True,
        persist_directory=str(db_path),
        allow_reset=True,
    )
    client = chromadb.Client(chroma_settings)
    return client.get_collection(COLLECTION_NAME)


def _get_collection() -> Any:
    if _state["collection"] is None:
        _state["collection"] = _open_collection()
    return _state["collection"]


def _get_settings() -> dict[str, Any]:
    _load_env()
    return {
        "model": os.getenv("OLLAMA_MODEL", DEFAULT_MODEL),
        "embed_model": os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL),
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/"),
    }


def _chat(client_url: str, model: str, prompt: str) -> str:
    """Non-streaming Ollama chat — stdout-safe for use inside MCP."""
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer using the retrieved notes. "
                        "Do not invent details not present in the provided context."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
    ).encode("utf-8")
    req = request.Request(
        f"{client_url}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("message", {}).get("content", "").strip()
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise OllamaError(details or str(exc)) from exc
    except error.URLError as exc:
        raise OllamaConnectionError(str(exc)) from exc
    except TimeoutError as exc:
        raise OllamaConnectionError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def query_kb(question: str, top_k: int = DEFAULT_TOP_K) -> str:
    """
    Query the personal ChatGPT knowledge base and get a grounded answer.

    Embeds the question, retrieves the most relevant chunks from past
    ChatGPT conversations stored in ChromaDB, then generates a concise
    answer using a local Ollama LLM. Use this when you need a synthesized
    answer drawn from the user's past work and conversations.

    Args:
        question: Natural-language question to answer from the knowledge base.
        top_k: Number of note chunks to retrieve (default 5).
    """
    try:
        collection = _get_collection()
    except Exception as exc:
        return f"[kb error] Could not load knowledge base: {exc}"

    settings = _get_settings()
    url = settings["ollama_base_url"]

    try:
        embedding = embed_text(url, question, settings["embed_model"])
        results = query_collection(collection, embedding, top_k)
    except OllamaConnectionError:
        return "[kb error] Cannot reach Ollama. Make sure `ollama serve` is running."
    except OllamaError as exc:
        return f"[kb error] Embedding failed: {exc}"

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]

    if not documents:
        return "No relevant chunks found in the knowledge base for that question."

    prompt = build_prompt(question, documents, metadatas)

    try:
        return _chat(url, settings["model"], prompt)
    except OllamaConnectionError:
        return "[kb error] Cannot reach Ollama for answer generation. Make sure `ollama serve` is running."
    except OllamaError as exc:
        return f"[kb error] Chat generation failed: {exc}"


@mcp.tool()
def search_kb(query: str, top_k: int = DEFAULT_TOP_K) -> str:
    """
    Search the personal ChatGPT knowledge base and return raw matching chunks.

    Embeds the query and retrieves the top-k most similar note chunks from
    past ChatGPT conversations. Returns the raw text with source titles and
    similarity distances — no LLM generation. Useful when you want to read
    the source context directly rather than a synthesized answer.

    Args:
        query: Search query to find relevant note chunks.
        top_k: Number of chunks to return (default 5).
    """
    try:
        collection = _get_collection()
    except Exception as exc:
        return f"[kb error] Could not load knowledge base: {exc}"

    settings = _get_settings()
    url = settings["ollama_base_url"]

    try:
        embedding = embed_text(url, query, settings["embed_model"])
        results = query_collection(collection, embedding, top_k)
    except OllamaConnectionError:
        return "[kb error] Cannot reach Ollama. Make sure `ollama serve` is running."
    except OllamaError as exc:
        return f"[kb error] Embedding failed: {exc}"

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    if not documents:
        return "No relevant chunks found."

    parts: list[str] = []
    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), 1):
        title = meta.get("title", "Untitled")
        source = meta.get("source_file", "unknown")
        parts.append(f"[{i}] {title} ({source}) — distance {dist:.4f}\n\n{doc}")

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
