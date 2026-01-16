import os
import streamlit as st
import chromadb
from tempfile import NamedTemporaryFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv

load_dotenv()

def process_uploaded_file(uploaded_file, chunk_size=1000, chunk_overlap=200):
    """Saves uploaded file to a temp path and splits into chunks."""
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    raw_documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(raw_documents)
    os.remove(tmp_path)  # Cleanup temp file
    return chunks

def setup_vector_db(chunks, collection_name="ESG_Collection"):
    """Initializes an ephemeral ChromaDB collection for the session."""
    chroma_client = chromadb.Client()  # Use ephemeral client for Streamlit sessions
    collection = chroma_client.get_or_create_collection(name=collection_name)
    
    documents = [chunk.page_content for chunk in chunks]
    ids = [f"id_{i}" for i in range(len(chunks))]
    metadatas = [chunk.metadata for chunk in chunks]
    
    collection.upsert(documents=documents, ids=ids, metadatas=metadatas)
    return collection

def get_rag_response(query, manual_context, collection, openai_client):
    """Retrieves context and queries OpenAI with hybrid grounding rules."""
    results = collection.query(query_texts=[query], n_results=5)
    context_text = "\n\n---\n\n".join(results['documents'][0])

    system_prompt = f"""
    You are a professional assistant and rigorous researcher. 
    Your task is to provide accurate answers based solely on the data provided.

    CONTEXT FROM PDF:
    {context_text}
    
    USER-PROVIDED MANUAL CONTEXT:
    {manual_context if manual_context else "None provided."}

    STRICT OPERATIONAL RULES:
    1. Absolute Grounding: Answer using ONLY the provided context and manual context.
    2. No Hallucinations: Do not use internal knowledge.
    3. Source Citation: Cite specific pages using [Source: Page X].
    4. Manual Context: If manual context is used, state: "Based on manual context".
    5. Admitting Ignorance: If the answer isn't there, say "I don't know".
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
        return "🔴 BILLING ERROR: Quota exceeded."
    except Exception as e:
        return f"🔴 ERROR: {e}"

# --- STREAMLIT UI ---

st.set_page_config(page_title="Research Assistant", layout="wide")
st.title("📚 Research Assistant (RAG)")

# Sidebar for Setup
with st.sidebar:
    st.header("1. Data Ingestion")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file:
        if "collection" not in st.session_state:
            with st.spinner("Processing PDF and building vector database..."):
                chunks = process_uploaded_file(uploaded_file)
                st.session_state.collection = setup_vector_db(chunks)
            st.success("✅ PDF Indexed Successfully!")

# Main Panel
st.header("2. Analysis Terminal")

col1, col2 = st.columns(2)

with col1:
    user_query = st.text_area("Question Terminal", placeholder="Enter your ESG query here...", height=150)

with col2:
    manual_context = st.text_area("Context Terminal (Optional)", placeholder="Provide extra facts or subject guidance...", height=150)

if st.button("Generate Grounded Response"):
    if not uploaded_file:
        st.error("Please upload a PDF first!")
    elif not user_query:
        st.warning("Please enter a question.")
    else:
        client = OpenAI()
        with st.spinner("Analyzing document..."):
            answer = get_rag_response(user_query, manual_context, st.session_state.collection, client)
        
        st.subheader("Output")
        st.markdown(answer)