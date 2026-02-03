# RAG Document QA System

This project is a Retrieval-Augmented Generation (RAG) system capable of ingesting PDF/DOCX documents and answering questions based on their content.

## Features
- **Multi-Document Ingestion**: Supports PDF, DOCX, and Text files.
- **Normalization**: Cleans and normalizes text for better retrieval.
- **Chunking**: Splits documents into manageable chunks (1000 chars).
- **Offline LLM**: Uses `Ollama` with the `llama3.2:1b` model (configurable).
- **Inspection**: Includes `inspect_chunks.py` to verify database contents.

## Setup
1.  Install Python 3.9+.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Install [Ollama](https://ollama.com/) and pull the model:
    ```bash
    ollama pull llama3.2:1b
    ```

## Usage
1.  **Ingest Documents**: Put files in `data/` and run:
    ```bash
    python ingest.py
    ```
2.  **Ask Questions**:
    ```bash
    python main.py ask "Your question here"
    ```
3.  **Inspect Database**:
    ```bash
    python inspect_chunks.py list
    ```
