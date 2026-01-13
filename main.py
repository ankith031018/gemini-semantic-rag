import os
import chromadb
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv

load_dotenv()

def process_documents(data_path, chunk_size=1000, chunk_overlap=200):
    """
    Loads PDFs and splits them into chunks.
    Increasing chunk_size to 1000 helps the model understand full contexts.
    """
    print(f"Loading documents from {data_path}...")
    loader = PyPDFDirectoryLoader(data_path)
    raw_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(raw_documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

def setup_vector_db(chunks, db_path, collection_name="ESG"):
    """
    Initializes ChromaDB and upserts documents.
    """
    if not chunks:
        print("\n⚠️  WARNING: No data found! The 'chunks' list is empty.")
        print("   Please check that your 'data' folder contains valid PDF files.")
        return None

    documents = []
    metadata = []
    ids = []

    for i, chunk in enumerate(chunks):
        documents.append(chunk.page_content)
        ids.append(f"ID_{i}") 
        metadata.append(chunk.metadata)

    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection(name=collection_name)

    collection.upsert(
        documents=documents,
        metadatas=metadata,
        ids=ids
    )
    print(f"Database '{collection_name}' ready at {db_path} with {len(ids)} chunks.")
    return collection

def get_rag_response(query, collection, openai_client):
    """
    Retrieves context and queries OpenAI.
    Includes error handling for billing issues.
    """
    results = collection.query(
        query_texts=[query],
        n_results=5
    )
    
    context_text = "\n\n---\n\n".join(results['documents'][0])

    system_prompt = f"""
    You are an expert financial researcher specializing in ESG investing.
    
    CONTEXT:
    {context_text}
    
    INSTRUCTIONS:
    1. Answer using ONLY the provided context.
    2. Distinguish strictly between "ESG Investing" and "Impact Investing".
    3. Use the "Spectrum of Capital" framework where relevant.
    4. Cite sources as [Source: Page X].
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )
        return response.choices[0].message.content

    except RateLimitError:
        return "🔴 BILLING ERROR: Quota exceeded. Please check OpenAI billing."
    except Exception as e:
        return f"🔴 ERROR: {e}"


if __name__ == "__main__":
    DATA_PATH = r'data'
    CHROMA_PATH = r'chroma_db'
    
    client = OpenAI()

    pdf_chunks = process_documents(DATA_PATH)

    vector_collection = setup_vector_db(pdf_chunks, CHROMA_PATH)

    while True:
        user_input = input("\nAsk an ESG question (or type 'quit'): ")
        
        if user_input.lower() == 'quit':
            break
        answer = get_rag_response(user_input, vector_collection, client)
        print(f"\nAnswer:\n{answer}\n{'='*30}")