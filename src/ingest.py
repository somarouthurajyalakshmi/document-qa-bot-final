def ingest_documents():
    from dotenv import load_dotenv          # Add this line
    load_dotenv()                           # Add this line
    
    # ... rest of the code stays the same
import os
import glob
from tqdm import tqdm
from pypdf import PdfReader
from docx import Document
from src.config import DATA_DIR, DB_PATH, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP
import chromadb
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

def extract_pdf_pages(file_path: str) -> list[dict]:
    extracted_data = []
    file_name = os.path.basename(file_path)
    try:
        reader = PdfReader(file_path)
        for index, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                clean_text = " ".join(text.split())
                extracted_data.append({
                    "text": clean_text,
                    "metadata": {
                        "source": file_name,
                        "page": index + 1
                    }
                })
    except Exception as e:
        print(f"Error reading PDF {file_name}: {e}")
    return extracted_data

def extract_docx_pages(file_path: str) -> list[dict]:
    extracted_data = []
    file_name = os.path.basename(file_path)
    try:
        doc = Document(file_path)
        full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        if full_text:
            extracted_data.append({
                "text": " ".join(full_text.split()),
                "metadata": {
                    "source": file_name,
                    "page": 1
                }
            })
    except Exception as e:
        print(f"Error reading DOCX {file_name}: {e}")
    return extracted_data

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - chunk_overlap)
    return chunks

def chunk_extracted_pages(pages: list[dict], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[dict]:
    chunks = []
    for page in pages:
        text = page["text"]
        metadata = page["metadata"]
        text_chunks = chunk_text(text, chunk_size, chunk_overlap)
        for i, chunk in enumerate(text_chunks):           # <-- Changed from chunk_text to chunk
            chunks.append({
                "text": chunk,
                "metadata": {
                    **metadata,
                    "chunk_id": i
                }
            })
    return chunks

def ingest_documents():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created {DATA_DIR} directory. Please add your documents there.")
        return

    all_chunks = []
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    docx_files = glob.glob(os.path.join(DATA_DIR, "*.docx"))

    for file_path in tqdm(pdf_files + docx_files, desc="Processing documents"):
        if file_path.endswith(".pdf"):
            pages = extract_pdf_pages(file_path)
        elif file_path.endswith(".docx"):
            pages = extract_docx_pages(file_path)
        else:
            continue
        chunks = chunk_extracted_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No documents found or no text extracted.")
        return

    client = chromadb.PersistentClient(path=DB_PATH)
    embedding_fn = GoogleGenerativeAiEmbeddingFunction(
        api_key=os.getenv("GEMINI_API_KEY"),
        model_name="models/text-embedding-004"
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    documents = [chunk["text"] for chunk in all_chunks]
    metadatas = [chunk["metadata"] for chunk in all_chunks]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    print(f"Successfully indexed {len(all_chunks)} chunks from {len(pdf_files + docx_files)} documents.")

if __name__ == "__main__":
    ingest_documents()