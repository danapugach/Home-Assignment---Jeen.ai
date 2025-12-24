"""
index_documents.py

Home assignment - Part 2 (Python modules):

- Input -> PDF / DOCX file
- Extract clean text from file
- Split text into chunks using the fixed-size with overlap strategy
- Create embeddings for each chunk using Google Gemini API
- Store the chunks + embeddings in a PostgreSQL DB with predefined columns (using pgvector)

Env vars (.env):
- GEMINI_API_KEY= ...
- POSTGRES_URL = postgresql://user:pass@host:5432/dbname

Run:
    python index_documents.py  /path/to/file.pdf
    python index_documents.py  /path/to/folder --recursive

keep API keys/connectio details in another place
use a .env file with according variables => GEMINI_API_KEY,POSTGRE_URL
keep script in GitHub + define readme file with clear explanations about everything
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Iterable, Tuple

from dotenv import load_dotenv

import psycopg2
from psycopg2.extras import execute_batch

from pypdf import PdfReader
from pypdf.errors import EmptyFileError
from docx import Document
from google import genai  # Google GenAI SDK

SPLIT_STRATEGY = "fixed_size_overlap"

# Embedding model (Gemini API)
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 3072


# ------- PART 1 -> Text extraction ----
# Keep both forms of text files
def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        reader = PdfReader(str(pdf_path))
    except EmptyFileError:
        return ""
    parts = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(parts).strip()


def extract_text_from_docx(docx_path: Path) -> str:
    doc = Document(str(docx_path))
    parts: List[str] = []
    for p in doc.paragraphs:
        if p.text:
            parts.append(p.text)
    return "\n".join(parts).strip()


# The main function for this
def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    if suffix == ".docx":
        return extract_text_from_docx(file_path)
    raise ValueError(f"Unsupported file type: {suffix}. Only PDF/DOCX are supported.")


# ------- PART 2 -> Text chunking ----
def split_text_fixed_size(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """
    Fixed-size chunking with overlap.
    Works reliably even when PDF/DOCX formatting is messy.
    """
    # Normalize whitespace
    text = " ".join(text.split())
    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


# ------- PART 3 -> Embeddings (Gemini API) ----
def embed_texts_gemini(texts: List[str], api_key: str) -> List[List[float]]:
    """
    Create embeddings for each chunk using Gemini API
    """
    client = genai.Client(api_key=api_key)
    vectors: List[List[float]] = []
    for t in texts:
        resp = client.models.embed_content(
            model= EMBEDDING_MODEL,
            contents=t,
        )
        # resp.embeddings is a list; take the first embedding
        vectors.append(resp.embeddings[0].values)
    return vectors

# ------- PART 4 -> PostgreSQL (pgvector) ----
DDL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
  id BIGSERIAL PRIMARY KEY,
  chunk_text TEXT NOT NULL,
  embedding vector({EMBEDDING_DIM}) NOT NULL,
  filename TEXT NOT NULL,
  split_strategy TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

INSERT_SQL = """
INSERT INTO document_chunks (chunk_text, embedding, filename, split_strategy)
VALUES (%s, %s, %s, %s)
"""


def ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()


def insert_chunks(
    conn,
    chunks: List[str],
    embeddings: List[List[float]],
    filename: str,
    split_strategy: str,
    batch_size: int = 50
) -> int:
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings length mismatch")
    rows = [(ch, emb, filename, split_strategy) for ch, emb in zip(chunks, embeddings)]
    with conn.cursor() as cur:
        execute_batch(cur, INSERT_SQL, rows, page_size=batch_size)
    conn.commit()
    return len(rows)


# ---------------------------
# File discovery
# ---------------------------
def iter_input_files(path: Path, recursive: bool) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        raise FileNotFoundError(f"Path not found: {path}")
    pattern = "**/*" if recursive else "*"
    for p in path.glob(pattern):
        if p.is_file() and p.suffix.lower() in {".pdf", ".docx"}:
            yield p


# ---------------------------
# Main
# ---------------------------
def main() -> int:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    pg_url = os.getenv("POSTGRES_URL")

    if not api_key:
        print("ERROR: Missing GEMINI_API_KEY in environment/.env", file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser(description="Index PDF/DOCX documents into PostgreSQL using Gemini embeddings.")
    parser.add_argument("input_path", help="PDF/DOCX file path or a folder containing documents")
    parser.add_argument("--recursive", action="store_true", help="Recursively search for documents in folders")
    parser.add_argument("--chunk-size", type=int, default=900, help="Chunk size in characters (default: 900)")
    parser.add_argument("--overlap", type=int, default=150, help="Overlap in characters (default: 150)")
    parser.add_argument("--dry-run", action="store_true", help="Run extraction+chunking and sample embeddings (no DB)")
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()

    # DRY RUN: no DB required
    if args.dry_run:
        for file_path in iter_input_files(input_path, args.recursive):
            text = extract_text(file_path)
            chunks = split_text_fixed_size(text, chunk_size=args.chunk_size, overlap=args.overlap)

            if not chunks:
                print(f"Skipped (no text/chunks): {file_path.name}")
                continue

            print(f"[DRY RUN] File: {file_path.name} | Chunks: {len(chunks)} | Strategy: {SPLIT_STRATEGY}")

            sample = chunks[:5]
            embeddings = embed_texts_gemini(sample, api_key=api_key)
            print(f"[DRY RUN] Created {len(embeddings)} embeddings. First dim={len(embeddings[0])}")

        print("\nDry run completed (no DB).")
        return 0

    # Normal run: requires DB
    if not pg_url:
        print("ERROR: Missing POSTGRES_URL in environment/.env", file=sys.stderr)
        return 1

    conn = psycopg2.connect(pg_url)
    try:
        ensure_schema(conn)

        total_inserted = 0
        for file_path in iter_input_files(input_path, args.recursive):
            try:
                text = extract_text(file_path)
                chunks = split_text_fixed_size(text, chunk_size=args.chunk_size, overlap=args.overlap)

                if not chunks:
                    print(f"Skipped (no text/chunks): {file_path.name}")
                    continue

                print(f"File: {file_path.name} | Chunks: {len(chunks)} | Strategy: {SPLIT_STRATEGY}")

                embeddings = embed_texts_gemini(chunks, api_key=api_key)

                if embeddings and len(embeddings[0]) != EMBEDDING_DIM:
                    raise ValueError(
                        f"Embedding dim mismatch. Got {len(embeddings[0])}, expected {EMBEDDING_DIM}. "
                        "Update EMBEDDING_DIM / DB schema accordingly."
                    )

                inserted = insert_chunks(
                    conn=conn,
                    chunks=chunks,
                    embeddings=embeddings,
                    filename=file_path.name,
                    split_strategy=SPLIT_STRATEGY
                )
                total_inserted += inserted

            except Exception as e:
                print(f"ERROR processing {file_path}: {e}", file=sys.stderr)

        print(f"\nDone. Inserted total chunks: {total_inserted}")
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())