import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise EnvironmentError("GOOGLE_API_KEY not found. Please add it to your .env file")

def Load_documents():
    loader = PyPDFLoader(file_path="")
    docs = loader.load()
    return docs

def document_chunking(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    splits = text_splitter.split_documents(docs)
    return splits

def get_embeddings(splits):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_1 = embeddings.embed_query(splits[0].page_content)
    vector_2 = embeddings.embed_query(splits[1].page_content)

    assert len(vector_1) == len(vector_2)
    print(f"Generated vectors of length {len(vector_1)}\n")
    print(vector_1[:10])


def upsert_splits(pc, index_name, splits):
    """Create embeddings for each split and upsert them into Pinecone.

    This function is defensive about the Pinecone client's available methods and
    stores the raw split text under the `chunk_text` metadata key to match the
    `field_map` used at index creation.
    """
    if not splits:
        raise ValueError("No document splits provided to upsert.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    vectors = []
    for i, doc in enumerate(splits):
        text = doc.page_content
        vec = embeddings.embed_query(text)
        vectors.append({"id": f"{index_name}-{i}", "values": vec, "metadata": {"chunk_text": text}})

    try:
        # Prefer a simple upsert signature if available
        if hasattr(pc, "upsert"):
            # Some Pinecone clients accept (index_name, items) while others require an Index object.
            try:
                pc.upsert(index_name, vectors)
            except TypeError:
                # Fall back to creating an Index client if provided by the SDK
                idx = getattr(pc, "Index", None)
                if idx:
                    idx_client = pc.Index(index_name)
                    idx_client.upsert(vectors)
                else:
                    raise
        elif hasattr(pc, "Index"):
            idx_client = pc.Index(index_name)
            idx_client.upsert(vectors)
        else:
            raise RuntimeError("Unable to find a supported upsert method on the Pinecone client.")
    except Exception as e:
        raise RuntimeError(f"Failed to upsert vectors: {e}")

    print(f"Upserted {len(vectors)} vectors to index '{index_name}'")


def vector_db():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise EnvironmentError("PINECONE_API_KEY not found. Please add it to your .env file")

    pc = Pinecone(api_key=api_key)
    index_name = "developer-quickstart-py"

    # Create the index if it doesn't already exist
    if not pc.has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": "models/gemini-embedding-001",
                "field_map": {"text": "chunk_text"}
            },
        )
        print(f"Created index '{index_name}'")
    else:
        print(f"Index '{index_name}' already exists")

    return pc
