import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_experimental.text_splitter import SemanticChunker
from langchain_milvus import Milvus
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain import PromptTemplate

# --- CONFIGURATION ---
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"  # Replace with your actual key
PDF_PATH = "your_large_document.pdf"    # Replace with your PDF file path
MILVUS_DB_PATH = "./milvus_rag.db"      # Persistence file location

# Set env variable for LangChain to use
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

def main():
    # 1. Initialize Gemini Embeddings
    # We use 'text-embedding-004' which is optimized for retrieval tasks
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    # 2. Setup Vector Database (Milvus Lite)
    # This creates a persistent database file. If it exists, it loads it.
    vector_db = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": MILVUS_DB_PATH},
        collection_name="rag_collection",
        auto_id=True
    )

    # 3. Check if data exists, if not, Load and Chunk
    # (Simple check: usually you'd check collection stats, here we assume empty means load)
    # Note: In production, you might want more robust logic to avoid re-loading.
    print("Checking database...")
    
    # We will attempt a dummy search. If it returns nothing (or collection is new), we ingest.
    # For simplicity in this script, we'll ask the user if they want to ingest.
    user_input = input("Do you want to ingest the PDF? (yes/no): ").lower()

    if user_input == "yes":
        print(f"Loading {PDF_PATH}...")
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()

        print("Splitting text semantically (this takes time as it computes embeddings)...")
        # --- THE SECRET SAUCE: SEMANTIC CHUNKING ---
        # This splitter calculates the cosine distance between sentences.
        # If the distance > threshold (breakpoint), it starts a new chunk.
        text_splitter = SemanticChunker(
            embeddings, 
            breakpoint_threshold_type="percentile" # Splits at statistical outliers in similarity
        )
        chunks = text_splitter.split_documents(docs)
        
        print(f"Created {len(chunks)} semantic chunks. Indexing to Milvus...")
        vector_db.add_documents(chunks)
        print("Indexing Complete!")
    else:
        print("Skipping ingestion. Using existing database.")

    # 4. Setup Retrieval System
    # We use Gemini 1.5 Flash for its speed and 1M token window
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    # Custom Prompt to force context usage
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep the answer concise and strictly based on the provided context.

    Context: {context}

    Question: {question}
    Helpful Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 5}), # Retrieve top 5 semantic chunks
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    # 5. Interactive Loop
    print("\n--- RAG System Ready (Type 'exit' to quit) ---")
    while True:
        query = input("\nAsk a question: ")
        if query.lower() == "exit":
            break
        
        result = qa_chain.invoke({"query": query})
        print(f"\nAnswer: {result['result']}")
        
        # Optional: Show which chunk was used
        # print(f"\n[Source Chunk]: {result['source_documents'][0].page_content[:200]}...")

if __name__ == "__main__":
    main()