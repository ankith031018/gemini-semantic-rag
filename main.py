import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Read the Google API key directly from the environment (populated by .env)
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise EnvironmentError("GOOGLE_API_KEY not found. Please add it to your .env file")

from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector = embeddings.embed_query("Hello world")
print(vector[:5])  