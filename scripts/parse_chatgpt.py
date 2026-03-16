"""Convert ChatGPT export conversations.json into Obsidian markdown notes."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for bare Python environments
    def tqdm(iterable: list[dict[str, Any]], **_: Any) -> list[dict[str, Any]]:
        return iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = "data/raw/conversations.json"
DEFAULT_OUTPUT = "notes"
VALID_ROLES = {"user", "assistant"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a ChatGPT data export into Obsidian markdown notes."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=(
            "Path to a conversation JSON file or a raw export directory relative to the "
            f"project root (default: {DEFAULT_INPUT})."
        ),
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output directory relative to the project root (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only process the first N conversations.",
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


def discover_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        files = sorted(input_path.glob("conversations*.json"))
        return [file for file in files if file.is_file()]

    return []


def load_conversations(input_paths: list[Path]) -> list[dict[str, Any]]:
    conversations: list[dict[str, Any]] = []

    for input_path in input_paths:
        with input_path.open(encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError(
                f"Expected {input_path} to contain a list of conversations."
            )
        conversations.extend(item for item in data if isinstance(item, dict))

    return conversations


def isoformat_timestamp(timestamp: Any) -> str:
    if timestamp is None:
        return ""
    try:
        return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return ""


def sanitize_filename(title: Any) -> str:
    if not isinstance(title, str) or not title.strip():
        title = "untitled"
    slug = title.strip().lower()
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"[^a-z0-9-]", "", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    slug = slug[:60].rstrip("-")
    return slug or "untitled"


def unique_output_path(output_dir: Path, base_name: str) -> Path:
    candidate = output_dir / f"{base_name}.md"
    if not candidate.exists():
        return candidate

    suffix_index = 2
    while True:
        suffix = f"_{suffix_index}"
        trimmed = base_name[: max(1, 60 - len(suffix))].rstrip("-") or "untitled"
        candidate = output_dir / f"{trimmed}{suffix}.md"
        if not candidate.exists():
            return candidate
        suffix_index += 1


def yaml_quote(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ").strip()
    return "'" + normalized.replace("'", "''") + "'"


def extract_message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if not isinstance(content, dict):
        return ""

    parts = content.get("parts")
    if not isinstance(parts, list) or not parts:
        return ""

    text_parts = []
    for part in parts:
        if isinstance(part, str) and part:
            text_parts.append(part)

    if not text_parts:
        return ""

    return "\n\n".join(text_parts).strip()


def conversation_nodes(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    mapping = conversation.get("mapping")
    if not isinstance(mapping, dict):
        return []

    current_node = conversation.get("current_node")
    if isinstance(current_node, str) and current_node in mapping:
        node_chain = []
        seen: set[str] = set()
        node_id: str | None = current_node
        while node_id and node_id in mapping and node_id not in seen:
            seen.add(node_id)
            node = mapping[node_id]
            if isinstance(node, dict):
                node_chain.append(node)
                parent = node.get("parent")
                node_id = parent if isinstance(parent, str) else None
            else:
                node_id = None
        return list(reversed(node_chain))

    return [node for node in mapping.values() if isinstance(node, dict)]


def extract_messages(conversation: dict[str, Any]) -> list[dict[str, str]]:
    nodes = conversation_nodes(conversation)
    messages: list[dict[str, str]] = []

    for node in nodes:
        message = node.get("message")
        if not isinstance(message, dict):
            continue

        author = message.get("author")
        if not isinstance(author, dict):
            continue

        role = author.get("role")
        if role not in VALID_ROLES:
            continue

        text = extract_message_text(message)
        if not text:
            continue

        messages.append({"role": role, "text": text})

    return messages


def build_markdown(conversation: dict[str, Any]) -> str | None:
    messages = extract_messages(conversation)
    if not messages:
        return None

    title = conversation.get("title")
    if not isinstance(title, str) or not title.strip():
        title = "untitled"

    created = isoformat_timestamp(conversation.get("create_time"))
    updated = isoformat_timestamp(conversation.get("update_time"))

    frontmatter = "\n".join(
        [
            "---",
            f"title: {yaml_quote(title)}",
            f"created: {yaml_quote(created)}",
            f"updated: {yaml_quote(updated)}",
            "tags: [chatgpt, imported]",
            f"message_count: {len(messages)}",
            "---",
        ]
    )

    rendered_messages = []
    for message in messages:
        speaker = "User" if message["role"] == "user" else "Assistant"
        rendered_messages.append(f"**{speaker}:** {message['text']}")

    body = "\n\n".join(rendered_messages)
    return f"{frontmatter}\n\n## Conversation\n\n{body}\n"


def write_notes(
    conversations: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[int, int, int]:
    processed = len(conversations)
    skipped = 0
    written = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    for conversation in tqdm(conversations, desc="Conversations", unit="conv"):
        markdown = build_markdown(conversation)
        if markdown is None:
            skipped += 1
            continue

        file_stem = sanitize_filename(conversation.get("title"))
        output_path = unique_output_path(output_dir, file_stem)
        output_path.write_text(markdown, encoding="utf-8")
        written += 1

    return processed, skipped, written


def main() -> None:
    args = parse_args()
    input_path = resolve_project_path(args.input)
    output_dir = resolve_project_path(args.output)
    input_files = discover_input_files(input_path)

    if not input_files:
        print(
            "No conversation dump files were found. Place your ChatGPT export file at "
            "data/raw/conversations.json and re-run with: python scripts/parse_chatgpt.py"
        )
        return

    conversations = load_conversations(input_files)
    if args.limit is not None:
        conversations = conversations[: args.limit]

    processed, skipped, written = write_notes(conversations, output_dir)

    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Written: {written}")


if __name__ == "__main__":
    main()
