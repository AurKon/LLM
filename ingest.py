import os
import sys
from typing import List
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, WebBaseLoader

# Force UTF-8 for stdout/stderr to handle non-ASCII filenames
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Configuration
SOURCE_DIRECTORY = os.path.join(os.path.dirname(__file__), "data")
PERSIST_DIRECTORY = os.path.join(os.path.dirname(__file__), "db")
LINKS_FILE = "links.txt"

import re

def normalize_text(text: str) -> str:
    """
    Normalizes text by:
    1. Removing excessive whitespace (newlines, tabs converted to single spaces).
    2. Stripping leading/trailing whitespace.
    """
    if not text:
        return ""
    # Replace newlines and tabs with a space
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    # Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_documents(source_dir: str) -> List[Document]:
    documents = []
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
        print(f"Created data directory at {source_dir}")
        return []

    print(f"Loading documents from {source_dir}...")
    
    for file in os.listdir(source_dir):
        file_path = os.path.join(source_dir, file)
        loaded_docs = []
        if file.endswith(".pdf"):
            print(f"Loading PDF: {file}")
            loader = PyPDFLoader(file_path)
            loaded_docs = loader.load()
        elif file.endswith(".docx"):
            print(f"Loading DOCX: {file}")
            loader = Docx2txtLoader(file_path)
            loaded_docs = loader.load()
        elif file == LINKS_FILE:
            print(f"Loading Links from: {file}")
            try:
                with open(file_path, "r") as f:
                    urls = [line.strip() for line in f if line.strip()]
                if urls:
                    loader = WebBaseLoader(urls)
                    loaded_docs = loader.load()
            except Exception as e:
                print(f"Error loading links: {e}")

        # specific normalization for each document
        if loaded_docs:
            print(f"  - Loaded {len(loaded_docs)} pages/sections. Normalizing...")
            for doc in loaded_docs:
                doc.page_content = normalize_text(doc.page_content)
            documents.extend(loaded_docs)

    return documents

import shutil

def ingest():
    # 0. Clear existing database to avoid duplicates
    if os.path.exists(PERSIST_DIRECTORY):
        print(f"Clearing existing database at {PERSIST_DIRECTORY}...")
        shutil.rmtree(PERSIST_DIRECTORY)

    # 1. Load Documents
    documents = load_documents(SOURCE_DIRECTORY)
    if not documents:
        print("No documents found in 'data' directory.")
        return
    
    print(f"Total loaded documents/pages: {len(documents)}")

    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks from {len(documents)} original documents/pages.")

    # 3. Create Embeddings & Vector Store
    print("Creating embeddings and vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create and persist database
    db = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings, 
        persist_directory=PERSIST_DIRECTORY
    )
    # db.persist() # Chroma 0.4+ persists automatically or needs explicit handling depending on version, 
                 # langchain-chroma handles it.
    
    print(f"Ingestion complete! Database saved to {PERSIST_DIRECTORY}")

if __name__ == "__main__":
    ingest()
