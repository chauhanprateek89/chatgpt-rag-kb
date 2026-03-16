"""Ingest markdown notes into a persistent ChromaDB vector store."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
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
    import tiktoken
except ImportError:  # pragma: no cover - runtime dependency check
    tiktoken = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - runtime dependency check
    load_dotenv = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for bare Python environments
    def tqdm(iterable: list[Path], **_: Any) -> list[Path]:
        return iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTION_NAME = "chatgpt_kb"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_DB_PATH = "./chroma_db"
DEFAULT_NOTES_PATH = "./notes"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
MAX_EMBED_CHARS = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a persistent ChromaDB vector store from Obsidian markdown notes."
    )
    parser.add_argument(
        "--notes",
        help="Path to the markdown notes directory relative to the project root.",
    )
    parser.add_argument(
        "--db",
        help="Path to the ChromaDB directory relative to the project root.",
    )
    parser.add_argument(
        "--model",
        help="Ollama embedding model to use.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the ChromaDB collection before ingesting notes.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only process the first N markdown notes.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be 0 or greater.")
    return args


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_int_env(name: str, fallback: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return fallback
    try:
        parsed = int(value)
    except ValueError:
        return fallback
    return parsed if parsed >= 0 else fallback


def load_settings(args: argparse.Namespace) -> dict[str, Any]:
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")

    settings = {
        "model": args.model or os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL),
        "db_path": resolve_project_path(
            args.db or os.getenv("CHROMA_DB_PATH", DEFAULT_DB_PATH)
        ),
        "notes_path": resolve_project_path(
            args.notes or os.getenv("NOTES_PATH", DEFAULT_NOTES_PATH)
        ),
        "chunk_size": get_int_env("CHUNK_SIZE", DEFAULT_CHUNK_SIZE),
        "chunk_overlap": get_int_env("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP),
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
    }
    if settings["chunk_size"] <= 0:
        settings["chunk_size"] = DEFAULT_CHUNK_SIZE
    if settings["chunk_overlap"] >= settings["chunk_size"]:
        settings["chunk_overlap"] = min(DEFAULT_CHUNK_OVERLAP, settings["chunk_size"] - 1)
    if settings["chunk_overlap"] < 0:
        settings["chunk_overlap"] = 0
    return settings


def check_runtime_dependencies() -> bool:
    missing = []
    if chromadb is None:
        missing.append("chromadb")
    if tiktoken is None:
        missing.append("tiktoken")
    if load_dotenv is None:
        missing.append("python-dotenv")
    if missing:
        joined = ", ".join(missing)
        print(f"Missing Python dependencies: {joined}")
        print("Install the project dependencies, then re-run:")
        print("    pip install -r requirements.txt")
        return False
    return True


def find_note_files(notes_path: Path, limit: int | None) -> list[Path]:
    if not notes_path.exists():
        print(f"Notes path not found: {notes_path}")
        print("Generate notes first with: python scripts/parse_chatgpt.py")
        return []

    if not notes_path.is_dir():
        print(f"Notes path is not a directory: {notes_path}")
        return []

    note_files = sorted(path for path in notes_path.glob("*.md") if path.is_file())
    if not note_files:
        print(f"No markdown notes were found in: {notes_path}")
        return []
    if limit is not None:
        note_files = note_files[:limit]
    return note_files


def parse_scalar(value: str) -> str:
    scalar = value.strip()
    if len(scalar) >= 2 and scalar[0] == scalar[-1] and scalar[0] in {"'", '"'}:
        scalar = scalar[1:-1]
        scalar = scalar.replace("''", "'").replace('\\"', '"')
    return scalar


def parse_tags(value: str) -> list[str]:
    stripped = value.strip()
    if not stripped:
        return []
    if stripped.startswith("[") and stripped.endswith("]"):
        inner = stripped[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part) for part in inner.split(",") if parse_scalar(part)]
    return [parse_scalar(stripped)]


def parse_markdown_note(note_path: Path) -> tuple[dict[str, Any], str]:
    text = note_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    metadata: dict[str, Any] = {
        "title": note_path.stem,
        "created": "",
        "updated": "",
        "tags": [],
    }

    if lines and lines[0].strip() == "---":
        for index in range(1, len(lines)):
            if lines[index].strip() != "---":
                continue
            frontmatter = lines[1:index]
            body = "\n".join(lines[index + 1 :]).strip()
            for line in frontmatter:
                if ":" not in line:
                    continue
                key, raw_value = line.split(":", 1)
                key = key.strip()
                value = raw_value.strip()
                if key == "tags":
                    metadata["tags"] = parse_tags(value)
                elif key in {"title", "created", "updated"}:
                    metadata[key] = parse_scalar(value)
            return metadata, body

    return metadata, text.strip()


def sanitize_identifier(value: str) -> str:
    slug = value.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_") or "untitled"


def preferred_split_index(text: str) -> int | None:
    if not text:
        return None

    minimum = max(1, int(len(text) * 0.5))
    for separator in ("\n\n", "\n"):
        index = text.rfind(separator)
        if index >= minimum:
            return index

    sentence_match = None
    for match in re.finditer(r"(?<=[.!?])\s+", text):
        if match.start() >= minimum:
            sentence_match = match
    if sentence_match is not None:
        return sentence_match.start()

    index = text.rfind(" ")
    if index >= minimum:
        return index
    return None


def choose_chunk_end(
    tokens: list[int],
    start: int,
    desired_end: int,
    encoder: Any,
) -> int:
    if desired_end >= len(tokens):
        return len(tokens)

    window_text = encoder.decode(tokens[start:desired_end])
    split_at = preferred_split_index(window_text)
    if split_at is None:
        return desired_end

    candidate = window_text[:split_at].rstrip()
    if not candidate:
        return desired_end

    candidate_length = len(encoder.encode(candidate))
    if candidate_length <= 0:
        return desired_end

    return start + candidate_length


def enforce_chunk_character_limit(
    tokens: list[int],
    start: int,
    end: int,
    encoder: Any,
) -> tuple[int, str]:
    chunk_text_value = encoder.decode(tokens[start:end]).strip()
    if len(chunk_text_value) <= MAX_EMBED_CHARS:
        return end, chunk_text_value

    current_end = end
    while current_end - start > 1:
        reduced_end = start + max(1, (current_end - start) // 2)
        reduced_end = choose_chunk_end(tokens, start, reduced_end, encoder)
        if reduced_end <= start:
            reduced_end = start + max(1, (current_end - start) // 2)

        reduced_text = encoder.decode(tokens[start:reduced_end]).strip()
        if reduced_text and len(reduced_text) <= MAX_EMBED_CHARS:
            return reduced_end, reduced_text

        current_end = reduced_end
        chunk_text_value = reduced_text

    return current_end, chunk_text_value


def chunk_text(text: str, chunk_size: int, chunk_overlap: int, encoder: Any) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []

    tokens = encoder.encode(normalized)
    if not tokens:
        return []

    chunks: list[str] = []
    start = 0
    total_tokens = len(tokens)

    while start < total_tokens:
        desired_end = min(start + chunk_size, total_tokens)
        end = choose_chunk_end(tokens, start, desired_end, encoder)
        if end <= start:
            end = desired_end

        end, chunk = enforce_chunk_character_limit(tokens, start, end, encoder)
        if end <= start:
            end = min(start + 1, total_tokens)
            chunk = encoder.decode(tokens[start:end]).strip()
        if chunk:
            chunks.append(chunk)

        if end >= total_tokens:
            break

        start = max(end - chunk_overlap, start + 1)

    return chunks


def prepare_note_batches(
    note_files: list[Path],
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[dict[str, Any]], int]:
    encoder = tiktoken.get_encoding("cl100k_base")
    batches: list[dict[str, Any]] = []
    total_chunks = 0

    for note_path in note_files:
        metadata, body = parse_markdown_note(note_path)
        chunks = chunk_text(body, chunk_size, chunk_overlap, encoder)
        total_chunks += len(chunks)
        batches.append(
            {
                "note_path": note_path,
                "metadata": metadata,
                "chunks": chunks,
            }
        )

    return batches, total_chunks


def create_ollama_client(base_url: str) -> str:
    return base_url.rstrip("/")


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


class OllamaError(RuntimeError):
    """Raised when Ollama returns an unexpected or unsuccessful response."""


class OllamaConnectionError(OllamaError):
    """Raised when Ollama cannot be reached."""


def print_ollama_error(model: str) -> None:
    print("✗ Could not connect to Ollama. Make sure Ollama is running:")
    print("    ollama serve")
    print("Then ensure the embedding model is available:")
    print(f"    ollama pull {model}")


def reset_database_dir(db_path: Path) -> None:
    if db_path.exists():
        shutil.rmtree(db_path)


def get_collection(db_path: Path, reset: bool) -> Any:
    if reset:
        reset_database_dir(db_path)
    db_path.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        chroma_api_impl="chromadb.api.segment.SegmentAPI",
        is_persistent=True,
        persist_directory=str(db_path),
        allow_reset=True,
    )
    client = chromadb.Client(settings)
    if reset:
        client.reset()
    return client.get_or_create_collection(COLLECTION_NAME)


def note_source_path(note_path: Path, notes_root: Path) -> str:
    try:
        return str(note_path.relative_to(notes_root))
    except ValueError:
        return note_path.name


def note_chunks_changed(
    existing_map: dict[str, dict[str, Any]],
    chunk_id: str,
    chunk_text_value: str,
    total_chunks: int,
    title: str,
    created: str,
) -> bool:
    existing = existing_map.get(chunk_id)
    if existing is None:
        return True

    if existing.get("document") != chunk_text_value:
        return True

    metadata = existing.get("metadata") or {}
    return (
        metadata.get("title") != title
        or metadata.get("created") != created
        or metadata.get("total_chunks") != total_chunks
    )


def ingest_notes(
    note_batches: list[dict[str, Any]],
    notes_root: Path,
    collection: Any,
    ollama_client: Any,
    model: str,
    total_chunks: int,
) -> tuple[int, int, int]:
    notes_processed = len(note_batches)
    chunks_created = 0
    chunks_skipped = 0
    total_notes = len(note_batches)

    progress = tqdm(total=total_chunks, desc="Chunks", unit="chunk")

    for note_index, note_batch in enumerate(note_batches, start=1):
        note_path = note_batch["note_path"]
        metadata = note_batch["metadata"]
        chunks = note_batch["chunks"]
        if not chunks:
            continue

        source_file = note_source_path(note_path, notes_root)
        existing = collection.get(
            where={"source_file": source_file},
            include=["documents", "metadatas"],
        )
        existing_map: dict[str, dict[str, Any]] = {}
        for index, chunk_id in enumerate(existing.get("ids", [])):
            existing_map[chunk_id] = {
                "document": existing.get("documents", [])[index],
                "metadata": existing.get("metadatas", [])[index],
            }

        base_id = sanitize_identifier(note_path.stem)
        note_total_chunks = len(chunks)
        current_ids: list[str] = []
        note_label = note_path.stem[:32]

        for index, chunk in enumerate(chunks):
            progress.set_postfix(
                note=f"{note_index}/{total_notes}",
                file=note_label,
                created=chunks_created,
                skipped=chunks_skipped,
            )
            chunk_id = f"{base_id}_chunk_{index}"
            current_ids.append(chunk_id)
            if not note_chunks_changed(
                existing_map,
                chunk_id,
                chunk,
                note_total_chunks,
                metadata["title"],
                metadata["created"],
            ):
                chunks_skipped += 1
                progress.update(1)
                continue

            try:
                embedding = embed_text(ollama_client, chunk, model)
            except OllamaConnectionError:
                progress.close()
                print_ollama_error(model)
                raise SystemExit(1)
            except OllamaError:
                progress.close()
                print_ollama_error(model)
                raise SystemExit(1)

            collection.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[
                    {
                        "source_file": source_file,
                        "title": metadata["title"],
                        "created": metadata["created"],
                        "chunk_index": index,
                        "total_chunks": note_total_chunks,
                    }
                ],
            )
            chunks_created += 1
            progress.update(1)

        stale_ids = [chunk_id for chunk_id in existing_map if chunk_id not in current_ids]
        if stale_ids:
            collection.delete(ids=stale_ids)

    progress.close()

    return notes_processed, chunks_created, chunks_skipped


def main() -> None:
    args = parse_args()
    if not check_runtime_dependencies():
        raise SystemExit(1)

    settings = load_settings(args)
    note_files = find_note_files(settings["notes_path"], args.limit)
    if not note_files:
        raise SystemExit(1)

    note_batches, total_chunks = prepare_note_batches(
        note_files=note_files,
        chunk_size=settings["chunk_size"],
        chunk_overlap=settings["chunk_overlap"],
    )
    print(f"Notes discovered : {len(note_batches)}")
    print(f"Chunks planned   : {total_chunks:,}")

    collection = get_collection(settings["db_path"], args.reset)
    ollama_client = create_ollama_client(settings["ollama_base_url"])
    notes_processed, chunks_created, chunks_skipped = ingest_notes(
        note_batches=note_batches,
        notes_root=settings["notes_path"],
        collection=collection,
        ollama_client=ollama_client,
        model=settings["model"],
        total_chunks=total_chunks,
    )

    print("✓ Done.")
    print(f"  Notes processed : {notes_processed}")
    print(f"  Chunks created  : {chunks_created:,}")
    print(f"  Chunks skipped  : {chunks_skipped:,}  (already in DB)")
    print(f"  Collection size : {collection.count():,} vectors")


if __name__ == "__main__":
    main()
