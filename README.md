# Document Q&A Bot with RAG

A Retrieval-Augmented Generation (RAG) system that answers questions based on custom documents (PDF & DOCX) using Google Gemini and ChromaDB.

## Features
- Document ingestion (PDF + DOCX)
- Semantic search using embeddings
- Grounded answers with source citations
- Simple Streamlit web interface

## Project Structure
- `src/` - Main Python code
- `data/` - Documents to be indexed
- `db/` - Vector database storage
- `main.py` - Streamlit application

## How to Run
1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate it and install requirements: `pip install -r requirements.txt`
4. Add your Gemini API key in `.env`
5. Run: `python -m src.ingest`
6. Run the app: `streamlit run src/main.py`

## Note
There was a package compatibility issue with `google-generativeai` and `chromadb` during development. The core structure and UI are complete.

## Tech Stack
- Python
- Google Gemini
- ChromaDB
- Streamlit
- pypdf + python-docx