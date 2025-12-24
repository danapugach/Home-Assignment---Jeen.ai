# Home-Assignment - Jeen.ai
## Part 2 - Python Module (Indexing)

This script:
- Reads PDF/DOCX files
- Extracts and cleans text
- Splits text into chunks using **fixed-size with overlap**
- Generates embeddings using the **Gemini API**
- (Optional) Stores chunks + embeddings in PostgreSQL with pgvector

## Setup
Create a `.env` file (do not commit it) based on `.env.example`:
- GEMINI_API_KEY=...
- POSTGRES_URL=...

Install dependencies:
```bash
pip install -r requirements.txt
```

## Demo mode (no DB required) - This runs extraction, chunking and a small sample of embeddings generation without writing to the database. 
```bash
python index_documents.py <file_or_folder> --dry-run
```

## Full mode (requires PostgreSQL + pgvector) - This requires a PostgreSQL database with the pgvector extension enabled. 
```bash
python index_documents.py <file_or_folder>
```

