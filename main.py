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

def vector_db():
    pc = Pinecone(api_key="pcsk_2pNUSg_8NwUfVPXewYvGPohMjRNQaMQ4sSaWK9FsFEayYkv6Ac181tqXmCxSFWEUYAN2gC")    
    index_name = "developer-quickstart-py"

    # Create the index if it doesn't already exist
    if not pc.has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": "llama-text-embed-v2",
                "field_map": {"text": "chunk_text"}
            }
        )
        print(f"Created index '{index_name}'")
    else:
        print(f"Index '{index_name}' already exists")

    return pc    