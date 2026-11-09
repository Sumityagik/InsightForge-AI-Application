"""
Document loader: turns an uploaded file into plain text the AI can work with.

Supported now: PDF, DOCX, TXT, XLSX, XLS.
(Images need OCR and are added in a later sprint.)

Every failure becomes a DocumentError with a short message that is safe to
show to the user. Raw library errors are logged on the server only.
"""

import logging

logger = logging.getLogger("insightforge")

# Upper limit on how much extracted text we keep in the database per file.
MAX_EXTRACTED_CHARS = 300_000

# Spreadsheet preview limits (keeps the text small and useful).
MAX_PREVIEW_ROWS = 30
MAX_SHEET_CHARS = 6_000
MAX_NUMERIC_COLUMNS = 15


class DocumentError(Exception):
    """Raised when a file cannot be read. The message is safe to show the user."""


def load_document(path: str, file_type: str) -> dict:
    """
    Extracts text from the file at `path`.

    Returns: {"text": str, "truncated": bool, "meta": dict}
    Raises:  DocumentError
    """
    loaders = {
        "pdf": _load_pdf,
        "docx": _load_docx,
        "txt": _load_txt,
        "xlsx": _load_excel,
        "xls": _load_excel,
    }
    loader = loaders.get(file_type)
    if loader is None:
        raise DocumentError(f"'.{file_type}' files are not supported yet.")

    try:
        result = loader(path)
    except DocumentError:
        raise
    except ImportError:
        logger.exception("Missing library while reading %s file", file_type)
        raise DocumentError("A library needed to read this file type is not installed on the server.")
    except Exception:
        logger.exception("Failed to read %s file", file_type)
        raise DocumentError("Could not read this file. It may be corrupted or in an unexpected format.")

    text = (result.get("text") or "").strip()
    if not text:
        raise DocumentError("This file doesn't contain any readable text.")

    truncated = len(text) > MAX_EXTRACTED_CHARS
    if truncated:
        text = text[:MAX_EXTRACTED_CHARS]

    return {"text": text, "truncated": truncated, "meta": result.get("meta", {})}


# ---------------------------------------------------------------- PDF
def _load_pdf(path: str) -> dict:
    try:
        import pymupdf as fitz  # current PyMuPDF import name
    except ImportError:
        import fitz  # older PyMuPDF versions

    with fitz.open(path) as doc:
        if doc.needs_pass:
            raise DocumentError("This PDF is password-protected. Please remove the password and try again.")
        pages = [page.get_text("text") for page in doc]
        page_count = len(pages)

    text = "\n\n".join(p.strip() for p in pages if p.strip())
    if not text:
        raise DocumentError(
            "No text could be found in this PDF. It may be a scanned document or a set of images. "
            "Scanned PDFs need OCR, which will be added in a later update."
        )
    return {"text": text, "meta": {"pages": page_count}}


# ---------------------------------------------------------------- DOCX
def _load_docx(path: str) -> dict:
    import docx

    document = docx.Document(path)
    lines = []

    try:
        # Keeps paragraphs and tables in the order they appear in the file.
        for block in document.iter_inner_content():
            if hasattr(block, "rows"):  # it's a table
                lines.extend(_table_to_lines(block))
            else:
                if block.text.strip():
                    lines.append(block.text.strip())
    except AttributeError:
        # Older python-docx versions: paragraphs first, then tables.
        lines = [p.text.strip() for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            lines.extend(_table_to_lines(table))

    return {"text": "\n".join(lines), "meta": {"paragraphs": len(document.paragraphs)}}


def _table_to_lines(table) -> list:
    out = []
    for row in table.rows:
        cells = [c.text.strip().replace("\n", " ") for c in row.cells]
        if any(cells):
            out.append(" | ".join(cells))
    return out


# ---------------------------------------------------------------- TXT
def _load_txt(path: str) -> dict:
    with open(path, "rb") as f:
        raw = f.read()

    if b"\x00" in raw[:2048] and not raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise DocumentError("This doesn't look like a plain text file.")

    # UTF-16 decoding accepts almost any bytes, so only use it when the file
    # actually starts with a UTF-16 byte-order mark.
    encodings = ["utf-8-sig", "cp1252", "latin-1"]
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        encodings.insert(0, "utf-16")

    for encoding in encodings:
        try:
            text = raw.decode(encoding)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        raise DocumentError("Could not decode this text file.")

    return {"text": text, "meta": {"characters": len(text)}}


# ---------------------------------------------------------------- EXCEL
def _load_excel(path: str) -> dict:
    import pandas as pd

    sheets = pd.read_excel(path, sheet_name=None)  # {sheet name: DataFrame}
    if not sheets:
        raise DocumentError("This workbook has no sheets.")

    names = list(sheets.keys())
    parts = [f"Workbook with {len(names)} sheet(s): {', '.join(str(n) for n in names)}"]

    for name, df in sheets.items():
        df = df.dropna(how="all").dropna(axis=1, how="all")
        rows, cols = df.shape
        section = [f"\n=== Sheet: {name} ({rows} rows x {cols} columns) ==="]

        if df.empty:
            section.append("(this sheet is empty)")
            parts.append("\n".join(section))
            continue

        section.append("Columns: " + ", ".join(str(c) for c in df.columns))

        numeric = df.select_dtypes(include="number")
        stats = []
        for col in list(numeric.columns)[:MAX_NUMERIC_COLUMNS]:
            s = numeric[col].dropna()
            if len(s):
                stats.append(f"  {col}: min={s.min():g}, max={s.max():g}, average={s.mean():.4g}")
        if stats:
            section.append("Numeric summary:\n" + "\n".join(stats))

        shown = min(rows, MAX_PREVIEW_ROWS)
        preview = df.head(MAX_PREVIEW_ROWS).fillna("").to_csv(index=False)
        if len(preview) > MAX_SHEET_CHARS:
            preview = preview[:MAX_SHEET_CHARS] + "\n...(cut short)"
        section.append(f"First {shown} row(s) as CSV:\n{preview}")

        parts.append("\n".join(section))

    return {"text": "\n".join(parts), "meta": {"sheets": len(names)}}