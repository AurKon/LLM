import os
import argparse
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Configuration
PERSIST_DIRECTORY = os.path.join(os.path.dirname(__file__), "db")

def get_qa_chain():
    # 1. Initialize Embeddings (Must match ingestion)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 2. Load Vector Store
    if not os.path.exists(PERSIST_DIRECTORY):
        raise FileNotFoundError(f"Database not found at {PERSIST_DIRECTORY}. Please run ingest.py first.")
    
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 4})

    # 3. Initialize LLM (Ollama)
    # Ensure you have Ollama installed and have run `ollama pull llama3.2:1b`
    llm = ChatOllama(model="llama3.2:1b", temperature=0)

    # 4. Create QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

def query(question: str):
    try:
        qa_chain = get_qa_chain()
        result = qa_chain.invoke({"query": question})
        
        answer = result["result"]
        source_docs = result["source_documents"]

        print(f"\nQuestion: {question}")
        print(f"Answer: {answer}\n")
        print("Sources:")
        for i, doc in enumerate(source_docs):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            print(f"{i+1}. {os.path.basename(source)} (Page {page})")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the RAG system.")
    parser.add_argument("question", type=str, help="The question to answer")
    args = parser.parse_args()
    query(args.question)
