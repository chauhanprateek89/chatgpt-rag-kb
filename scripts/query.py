"""Query the local ChatGPT knowledge base with retrieval-augmented generation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from urllib import error, request

try:
    import chromadb
except ImportError:  # pragma: no cover - runtime dependency check
    chromadb = None
else:
    from chromadb.config import Settings

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - runtime dependency check
    load_dotenv = None

try:
    from rich.console import Console
except ImportError:  # pragma: no cover - optional formatting
    Console = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTION_NAME = "chatgpt_kb"
DEFAULT_MODEL = "llama3"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_DB_PATH = "./chroma_db"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_TOP_K = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the local ChatGPT knowledge base."
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Natural-language question to ask the knowledge base.",
    )
    parser.add_argument(
        "--db",
        help="Path to the ChromaDB directory relative to the project root.",
    )
    parser.add_argument(
        "--model",
        help="Ollama chat model to use for answer generation.",
    )
    parser.add_argument(
        "--embed-model",
        help="Ollama embedding model to use for retrieval.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"How many note chunks to retrieve (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Show the retrieved note sources after the answer.",
    )
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be greater than 0.")
    return args


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def check_runtime_dependencies() -> bool:
    missing = []
    if chromadb is None:
        missing.append("chromadb")
    if load_dotenv is None:
        missing.append("python-dotenv")
    if missing:
        print(f"Missing Python dependencies: {', '.join(missing)}")
        print("Install the project dependencies, then re-run:")
        print("    pip install -r requirements.txt")
        return False
    return True


def load_settings(args: argparse.Namespace) -> dict[str, Any]:
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")

    return {
        "db_path": resolve_project_path(
            args.db or os.getenv("CHROMA_DB_PATH", DEFAULT_DB_PATH)
        ),
        "model": args.model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL),
        "embed_model": args.embed_model or os.getenv(
            "EMBED_MODEL", DEFAULT_EMBED_MODEL
        ),
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
    }


def create_console() -> Any:
    if Console is not None:
        return Console()
    return None


def print_line(console: Any, message: str) -> None:
    if console is not None:
        console.print(message)
    else:
        print(message)


def prompt_for_question() -> str:
    return input("Question: ").strip()


def create_ollama_client(base_url: str) -> str:
    return base_url.rstrip("/")


class OllamaError(RuntimeError):
    """Raised when Ollama returns an unexpected or unsuccessful response."""


class OllamaConnectionError(OllamaError):
    """Raised when Ollama cannot be reached."""


def print_ollama_error(chat_model: str, embed_model: str) -> None:
    print("✗ Could not connect to Ollama. Make sure Ollama is running:")
    print("    ollama serve")
    print("Then ensure the required models are available:")
    print(f"    ollama pull {chat_model}")
    print(f"    ollama pull {embed_model}")


def embed_text(client: str, text: str, model: str) -> list[float]:
    try:
        for endpoint, payload in (
            ("/api/embed", {"model": model, "input": text}),
            ("/api/embeddings", {"model": model, "prompt": text}),
        ):
            body = json.dumps(payload).encode("utf-8")
            req = request.Request(
                f"{client}{endpoint}",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with request.urlopen(req, timeout=120) as response:
                    data = json.loads(response.read().decode("utf-8"))
            except error.HTTPError as exc:
                if exc.code == 404 and endpoint == "/api/embed":
                    continue
                details = exc.read().decode("utf-8", errors="replace")
                raise OllamaError(details or str(exc)) from exc

            if "embeddings" in data and data["embeddings"]:
                return data["embeddings"][0]
            if "embedding" in data and data["embedding"]:
                return data["embedding"]
        raise OllamaError("Unexpected embedding response from Ollama.")
    except error.URLError as exc:
        raise OllamaConnectionError(str(exc)) from exc
    except TimeoutError as exc:
        raise OllamaConnectionError(str(exc)) from exc
    except OllamaError:
        raise
    except Exception as exc:
        raise OllamaConnectionError(str(exc)) from exc


def query_collection(collection: Any, embedding: list[float], top_k: int) -> dict[str, Any]:
    return collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )


def build_prompt(question: str, documents: list[str], metadatas: list[dict[str, Any]]) -> str:
    context_sections = []
    for index, document in enumerate(documents, start=1):
        metadata = metadatas[index - 1] if index - 1 < len(metadatas) else {}
        title = metadata.get("title", "Untitled")
        source_file = metadata.get("source_file", "unknown")
        context_sections.append(
            f"[Source {index}] {title} ({source_file})\n{document}"
        )

    context = "\n\n".join(context_sections)
    return (
        "You are answering questions using a private ChatGPT knowledge base.\n"
        "Use only the retrieved context below. If the answer is not supported by the "
        "context, say so clearly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def stream_answer(client: str, model: str, prompt: str, console: Any) -> None:
    try:
        body = json.dumps(
            {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Answer using the retrieved notes. Do not invent details that are "
                            "not present in the provided context."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": True,
            }
        ).encode("utf-8")
        req = request.Request(
            f"{client}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=300) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                chunk = json.loads(line)
                message = chunk.get("message", {})
                content = message.get("content", "")
                if not content:
                    continue
                if console is not None:
                    console.print(content, end="")
                else:
                    print(content, end="", flush=True)
        print()
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise OllamaError(details or str(exc)) from exc
    except error.URLError as exc:
        raise OllamaConnectionError(str(exc)) from exc
    except TimeoutError as exc:
        raise OllamaConnectionError(str(exc)) from exc
    except Exception as exc:
        raise OllamaConnectionError(str(exc)) from exc


def print_sources(console: Any, metadatas: list[dict[str, Any]], distances: list[float]) -> None:
    print_line(console, "\nSources:")
    for index, metadata in enumerate(metadatas, start=1):
        distance = distances[index - 1] if index - 1 < len(distances) else None
        title = metadata.get("title", "Untitled")
        source_file = metadata.get("source_file", "unknown")
        if distance is None:
            print_line(console, f"{index}. {title} [{source_file}]")
        else:
            print_line(console, f"{index}. {title} [{source_file}] distance={distance:.4f}")


def main() -> None:
    args = parse_args()
    if not check_runtime_dependencies():
        raise SystemExit(1)

    settings = load_settings(args)
    question = (args.question or prompt_for_question()).strip()
    if not question:
        print("A question is required.")
        raise SystemExit(1)

    db_path = settings["db_path"]
    if not db_path.exists():
        print(f"ChromaDB path not found: {db_path}")
        print("Build the vector store first with: python scripts/build_vectorstore.py")
        raise SystemExit(1)

    chroma_settings = Settings(
        chroma_api_impl="chromadb.api.segment.SegmentAPI",
        is_persistent=True,
        persist_directory=str(db_path),
        allow_reset=True,
    )
    client = chromadb.Client(chroma_settings)
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        print("Collection 'chatgpt_kb' was not found.")
        print("Build the vector store first with: python scripts/build_vectorstore.py")
        raise SystemExit(1)

    ollama_client = create_ollama_client(settings["ollama_base_url"])
    console = create_console()

    try:
        question_embedding = embed_text(
            ollama_client,
            question,
            settings["embed_model"],
        )
        results = query_collection(collection, question_embedding, args.top_k)
    except OllamaConnectionError:
        print_ollama_error(settings["model"], settings["embed_model"])
        raise SystemExit(1)
    except OllamaError:
        print_ollama_error(settings["model"], settings["embed_model"])
        raise SystemExit(1)

    documents = results.get("documents", [[]])
    metadatas = results.get("metadatas", [[]])
    distances = results.get("distances", [[]])
    top_documents = documents[0] if documents else []
    top_metadatas = metadatas[0] if metadatas else []
    top_distances = distances[0] if distances else []

    if not top_documents:
        print("No relevant note chunks were found in the vector store.")
        raise SystemExit(0)

    prompt = build_prompt(question, top_documents, top_metadatas)

    print_line(console, "Answer:\n")
    try:
        stream_answer(ollama_client, settings["model"], prompt, console)
    except OllamaConnectionError:
        print_ollama_error(settings["model"], settings["embed_model"])
        raise SystemExit(1)
    except OllamaError:
        print_ollama_error(settings["model"], settings["embed_model"])
        raise SystemExit(1)

    if args.show_sources:
        print_sources(console, top_metadatas, top_distances)


if __name__ == "__main__":
    main()
