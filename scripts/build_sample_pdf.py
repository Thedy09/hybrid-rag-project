"""Write the two-page operations sheet used as the PDF sample."""

from __future__ import annotations

from pathlib import Path


PAGE_ONE = [
    "Vector Error Codes",
    "Operations sheet for the Aurora Search Appliance.",
    "This sheet lists terse operational codes that the narrative manual does not spell out.",
    "Collection rebuild service targets depend on corpus size.",
    "Large collections take longer than small ones.",
    "The small-collection target is named on the next page of this sheet.",
    "Do not guess the code from the narrative manual.",
    "Use this sheet when an operator asks for the named service target.",
]

PAGE_TWO = [
    "Small-collection rebuild target",
    "Code SLA-15M names a 15 minute collection rebuild service target for corpora under 50000 chunks.",
    "If a rebuild exceeds that target, capture the appliance log and open a ticket with the collection name.",
    "SLA-15M applies only under the stated corpus size.",
    "Above that size, schedule the rebuild overnight instead of using this target.",
    "Quote SLA-15M in the ticket so the queue can match the service target.",
]


def write_vector_error_codes(path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_render_pdf([_wrap(PAGE_ONE), _wrap(PAGE_TWO)]))


def _wrap(lines: list[str], width: int = 78) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        words = line.split()
        current = ""
        for word in words:
            trial = word if not current else f"{current} {word}"
            if len(trial) <= width:
                current = trial
            else:
                if current:
                    wrapped.append(current)
                current = word
        if current:
            wrapped.append(current)
        wrapped.append("")
    return wrapped


def _render_pdf(pages: list[list[str]]) -> bytes:
    streams = [_page_stream(lines) for lines in pages]
    page_count = len(streams)

    def page_object_id(index: int) -> int:
        return 4 + index * 2

    def content_object_id(index: int) -> int:
        return 5 + index * 2

    kids = " ".join(f"{page_object_id(index)} 0 R" for index in range(page_count))
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode("ascii"),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for index, stream in enumerate(streams):
        objects[page_object_id(index)] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_object_id(index)} 0 R "
            f"/Resources << /Font << /F1 3 0 R >> >> >>"
        ).encode("ascii")
        objects[content_object_id(index)] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"\nendstream"
        )

    max_id = content_object_id(page_count - 1)
    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    pieces = [header]
    offsets = [0]
    for object_id in range(1, max_id + 1):
        offsets.append(sum(len(piece) for piece in pieces))
        pieces.append(f"{object_id} 0 obj\n".encode("ascii") + objects[object_id] + b"\nendobj\n")

    xref_offset = sum(len(piece) for piece in pieces)
    xref_lines = [f"xref\n0 {max_id + 1}\n", "0000000000 65535 f \n"]
    xref_lines.extend(f"{offsets[object_id]:010d} 00000 n \n" for object_id in range(1, max_id + 1))
    trailer = (
        f"trailer\n<< /Size {max_id + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    )
    return b"".join(pieces) + "".join(xref_lines).encode("ascii") + trailer.encode("ascii")


def _page_stream(lines: list[str]) -> bytes:
    commands = ["BT", "/F1 11 Tf", "14 TL", "54 740 Td"]
    for index, line in enumerate(lines):
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        if index == 0:
            commands.append(f"({escaped}) Tj")
        else:
            commands.append(f"T* ({escaped}) Tj")
    commands.append("ET")
    return "\n".join(commands).encode("latin-1")


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "data" / "vector-error-codes.pdf"
    write_vector_error_codes(output)
    print(output)
