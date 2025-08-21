
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
import torch
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_pdf_files(data):
    loader = DirectoryLoader(data, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Filters documents to only include source and original page content:
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        minimal_docs.append(Document(
            page_content=doc.page_content,
            metadata={"source": doc.metadata.get("source", "unknown"),}
        ))
    return minimal_docs


def text_splitter(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
    )
    docs_chunk = text_splitter.split_documents(minimal_docs)
    return docs_chunk

def get_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
                                        )
    return embeddings



