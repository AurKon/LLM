import argparse
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import sys

# Force UTF-8 for stdout/stderr to handle non-ASCII filenames
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Configuration
PERSIST_DIRECTORY = os.path.join(os.path.dirname(__file__), "db")

def list_sources():
    if not os.path.exists(PERSIST_DIRECTORY):
        print("Database not found.")
        return

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    
    # This is a bit of a hack since Chroma doesn't have a simple "list all docs"
    # We will fetch all IDs and their metadata
    print("Fetching documents from DB...")
    data = db.get() # fetch all
    metadatas = data['metadatas']
    
    sources = set()
    for meta in metadatas:
        if meta and 'source' in meta:
            sources.add(os.path.basename(meta['source']))
    
    print("\n--- Ingested Documents ---")
    for s in sorted(sources):
        print(f" - {s}")
    print("--------------------------")

def view_chunks(source_name: str, limit: int = 5):
    if not os.path.exists(PERSIST_DIRECTORY):
        print("Database not found.")
        return

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

    print(f"Searching for chunks from: {source_name}")
    data = db.get(where={"source": {"$like": f"%{source_name}%"}})
    
    ids = data['ids']
    documents = data['documents']
    metadatas = data['metadatas']
    
    count = 0
    for i, content in enumerate(documents):
        if count >= limit:
            break
        print(f"\n[Chunk {i+1}]")
        print(f"Metadata: {metadatas[i]}")
        print(f"Content (First 200 chars): {content[:200]}...")
        count += 1
    
    print(f"\nTotal chunks found for this source: {len(documents)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect ChromaDB chunks")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    subparsers.add_parser("list", help="List all ingested document sources")
    
    view_parser = subparsers.add_parser("view", help="View chunks for a source")
    view_parser.add_argument("--source", type=str, required=True, help="Filename of the source document")
    view_parser.add_argument("--limit", type=int, default=5, help="Number of chunks to view")

    args = parser.parse_args()
    
    if args.command == "list":
        list_sources()
    elif args.command == "view":
        view_chunks(args.source, args.limit)
    else:
        parser.print_help()
