"""Metadata filters applied to both the dense and sparse indexes."""

from __future__ import annotations


def _active_items(metadata_filter: dict | None) -> list[tuple[str, object]]:
    if not metadata_filter:
        return []
    items = []
    for key, expected in metadata_filter.items():
        if expected is None:
            continue
        if str(expected).strip().lower() in {"", "all"}:
            continue
        items.append((key, expected))
    return items


def matches_metadata(metadata: dict, metadata_filter: dict | None) -> bool:
    """Return True when every active filter field matches the chunk metadata."""

    for key, expected in _active_items(metadata_filter):
        actual = metadata.get(key)
        if str(actual).lower() != str(expected).lower():
            return False
    return True


def chroma_where(metadata_filter: dict | None) -> dict | None:
    """Build a Chroma `where` clause. One field stays a plain equality filter."""

    clauses = [{key: expected} for key, expected in _active_items(metadata_filter)]
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
