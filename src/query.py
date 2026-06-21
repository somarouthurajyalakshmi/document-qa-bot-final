import os
import google.generativeai as genai
import chromadb
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
from dotenv import load_dotenv
from src.config import DB_PATH, COLLECTION_NAME, TOP_K

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def query_rag_pipeline(user_query: str, k: int = TOP_K):
    client = chromadb.PersistentClient(path=DB_PATH)
    embedding_fn = GoogleGenerativeAiEmbeddingFunction(
        api_key=os.getenv("GEMINI_API_KEY"),
        model_name="models/text-embedding-004"
    )

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    results = collection.query(
        query_texts=[user_query],
        n_results=k
    )

    context_blocks = []
    citations = []

    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        source = meta.get('source', 'Unknown')
        page = meta.get('page', 'N/A')
        citation_str = f"Source: {source}, Page: {page}"
        context_blocks.append(f"[{citation_str}]\n{doc}")
        citations.append(citation_str)

    context_payload = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "You are a professional, accurate document Q&A assistant. "
        "Answer the user's question using ONLY the provided document context below. "
        "Cite the sources (filenames and pages) inline next to facts you cite. "
        "If the answer cannot be found in the context, clearly state: "
        "'I am sorry, but the provided documents do not contain the answer to your question.' "
        "Do not make up facts or use external knowledge."
    )

    prompt = (
        f"{system_prompt}\n\n"
        f"CONTEXT INFORMATION:\n{context_payload}\n\n"
        f"USER QUESTION: {user_query}\n\n"
        f"GROUNDED ANSWER:"
    )

    model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
    response = model.generate_content(prompt)

    return {
        "answer": response.text,
        "citations": citations,
        "raw_context": results['documents'][0]
    }

if __name__ == "__main__":
    query = input("Enter your question: ")
    result = query_rag_pipeline(query)
    print("\nAnswer:\n", result["answer"])
    print("\nCitations:", result["citations"])