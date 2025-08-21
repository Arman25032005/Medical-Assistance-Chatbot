from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_splitter, get_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

extracted_data = load_pdf_files("/Users/armanchaturvedi/Desktop/Medical chatbot /Medical-Assistance-Chatbot/data")
minimal_docs = filter_to_minimal_docs(extracted_data)
docs_chunks = text_splitter(minimal_docs)

embedding = get_embeddings()

Pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=Pinecone_api_key)

index_name = "medical-assistant-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,  # dimension of the embeddings
        metric="cosine", # similarity metric
        spec=ServerlessSpec(
            cloud = "aws",
            region = "us-east-1",  # specify the region
        )
    )
    index = pc.Index(index_name)

vector_search = PineconeVectorStore.from_documents(
    index_name=index_name,
    documents=docs_chunks,
    embedding=embedding,
)

