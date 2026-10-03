"""Load source files and split them into chunks that keep retrieval metadata."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.settings import CHUNK_OVERLAP, CHUNK_SIZE, SEPARATORS


def load_documents(data_directory: str | Path) -> list[Document]:
    """Load PDF, text, and Markdown files from a directory.

    PDF pages stay separate so page numbers survive into the chunk metadata.
    Text and Markdown files are one document each and use page number 1.
    """

    documents: list[Document] = []
    root = Path(data_directory)

    for file_path in sorted(root.iterdir()):
        if not file_path.is_file() or file_path.name.startswith("."):
            continue

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            loaded = loader.load()
        elif suffix in {".txt", ".md"}:
            loader = TextLoader(str(file_path), encoding="utf-8")
            loaded = loader.load()
        else:
            continue

        for document in loaded:
            _annotate_source(document, file_path)
            documents.append(document)

    return documents


def _annotate_source(document: Document, file_path: Path) -> None:
    document.metadata["source"] = file_path.as_posix()
    document.metadata["file_name"] = file_path.name
    document.metadata["document_type"] = file_path.suffix.lower().lstrip(".")
    raw_page = document.metadata.get("page")
    if raw_page is None:
        document.metadata["page_number"] = 1
    else:
        document.metadata["page_number"] = int(raw_page) + 1


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents on paragraph, line, and sentence boundaries.

    Each chunk keeps the source file name, page number, document type, and a
    stable chunk identifier used later by reciprocal rank fusion.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
    )
    chunks = [
        chunk for chunk in splitter.split_documents(documents) if chunk.page_content.strip()
    ]

    counts: dict[str, int] = {}
    for chunk in chunks:
        file_name = str(chunk.metadata.get("file_name") or Path(str(chunk.metadata.get("source", "unknown"))).name)
        page_number = int(chunk.metadata.get("page_number") or 1)
        document_type = str(chunk.metadata.get("document_type") or Path(file_name).suffix.lower().lstrip(".") or "unknown")
        local_index = counts.get(file_name, 0)
        counts[file_name] = local_index + 1
        chunk.metadata["file_name"] = file_name
        chunk.metadata["page_number"] = page_number
        chunk.metadata["document_type"] = document_type
        chunk.metadata["chunk_id"] = f"{Path(file_name).stem}-p{page_number}-c{local_index:04d}"

    return chunks


def save_chunks(chunks: list[Document], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for chunk in chunks:
        payload = {
            "page_content": chunk.page_content,
            "metadata": _json_safe_metadata(chunk.metadata),
        }
        lines.append(json.dumps(payload, ensure_ascii=False))
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_chunks(path: str | Path) -> list[Document]:
    documents = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        documents.append(
            Document(
                page_content=payload["page_content"],
                metadata=payload.get("metadata") or {},
            )
        )
    return documents


def _json_safe_metadata(metadata: dict) -> dict:
    safe: dict = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe


if __name__ == "__main__":
    from src.paths import DATA_DIR

    loaded = load_documents(DATA_DIR)
    print(f"Loaded {len(loaded)} documents")
    if loaded:
        print(loaded[0].metadata)
    pieces = split_documents(loaded)
    print(f"Created {len(pieces)} chunks")
    if pieces:
        print(pieces[0].metadata)
