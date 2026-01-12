import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv

load_dotenv()

def get_models():
    """Initializes the embedding engine and the LLM with a valid model name."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set in .env")
    
    # Use text-embedding-004 for the 768-dimension Pinecone index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # Update to a valid production model name
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        temperature=0,
    )
    return embeddings, llm

def indexing_pinecone(index_name="learning-pdf-index"):
    """Ensures the Pinecone index is ready on AWS us-east-1."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    if index_name not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=768,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    return pc

def document_loading(file_path, embeddings, index_name):
    """Loads and persists PDF data only if index is empty."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(index_name)
    stats = index.describe_index_stats()

    if stats['total_vector_count'] > 0:
        print(f"✅ Data exists. Skipping ingestion.")
        return PineconeVectorStore(index_name=index_name, embedding=embeddings)

    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=95)
    chunks = text_splitter.split_documents(docs)
    
    return PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)

def get_answer(query, manual_context, embeddings, index_name, llm):
    """Retrieves PDF context and combines it with User-provided context."""
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 7})
    relevant_docs = retriever.invoke(query)
    
    # Format retrieved PDF context
    pdf_context = "\n\n".join([f"[Source: Page {d.metadata.get('page')}] {d.page_content}" for d in relevant_docs])
    
    # PROMPT TEMPLATE: Incorporates both sources
    prompt = f"""
    You are a professional researcher. Use the following TWO sources of information to answer:
    1. PROVIDED PDF CONTEXT (Retrieved from the document)
    2. USER-PROVIDED MANUAL CONTEXT (Context provided by the user in this chat)

    STRICT RULES:
    - Base your answer ONLY on these two sources. 
    - If information is missing from both, say: "I cannot find this in the document or the provided manual context."
    - Cite [Source: Page X] when using PDF data.
    - Explicitly mention "Based on manual context" when using user-provided info.

    --- PDF CONTEXT ---
    {pdf_context}

    --- USER-PROVIDED MANUAL CONTEXT ---
    {manual_context}

    --- USER QUERY ---
    {query}
    
    ANSWER:"""
    
    response = llm.invoke(prompt)
    return response.content

def main():
    # PATH HANDLING (Using your provided Windows path)
    file_to_process = r"C:\Users\ankit\Downloads\ESG 2021 Chapter 1.pdf"
    INDEX_NAME = "esg-2021-chapter-1" 

    embeddings, llm = get_models()
    indexing_pinecone(INDEX_NAME)
    document_loading(file_to_process, embeddings, INDEX_NAME)

    print("\n" + "="*50)
    print("HYBRID RAG SYSTEM ACTIVE")
    print("Instruction: You can provide context along with your query.")
    print("Example: 'Regarding the 2021 climate goal, use the context that the budget was doubled.'")
    print("="*50)

    while True:
        print("\n--- NEW QUERY ---")
        query = input("👤 Question: ")
        if query.lower() in ["exit", "quit"]: break
        
        manual_context = input("📝 Manual Context (Leave empty if none): ")
        if not manual_context: manual_context = "No manual context provided."

        print("🤖 Researching...")
        answer = get_answer(query, manual_context, embeddings, INDEX_NAME, llm)
        print(f"\n📢 ANSWER:\n{answer}")

if __name__ == "__main__":
    main()