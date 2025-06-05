# RAG PDF Chat API

A FastAPI-based REST API for chatting with PDF documents using Retrieval-Augmented Generation (RAG) with LangChain, Ollama, and FAISS.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure Ollama is running locally:
```bash
# Install Ollama first, then pull required models
ollama pull nomic-embed-text
ollama pull qwen2.5:3b
```

3. Start the API server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Check Status
```bash
curl -X GET "http://localhost:8000/status"
```

### 2. Upload PDF Files
```bash
curl -X POST "http://localhost:8000/upload-pdfs" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf"
```

### 3. Load PDFs from Directory
```bash
curl -X POST "http://localhost:8000/load-directory" \
  -H "Content-Type: application/json" \
  -d '{"directory_path": "/path/to/pdf/directory"}'
```

### 4. Ask Questions
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main topics discussed in the documents?",
    "search_type": "mmr",
    "num_results": 3
  }'
```

### 5. Search Documents
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "contract terms",
    "search_type": "similarity",
    "num_results": 5
  }'
```

### 6. Reset System
```bash
curl -X DELETE "http://localhost:8000/reset"
```
  

## Configuration

You can customize the RAG system by modifying the initialization parameters in `main.py`:

```python
rag_chat = RAGPDFChat(
    embedding_model='nomic-embed-text',  # Ollama embedding model
    llm_model='qwen2.5:3b',            # Ollama LLM model
    ollama_base_url='http://localhost:11434',  # Ollama server URL
    chunk_size=1000,                    # Text chunk size
    chunk_overlap=100                   # Chunk overlap
)
```

## Environment Variables

Create a `.env` file for configuration:
```env
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
LLM_MODEL=qwen2.5:3b
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
```

## Features

- **Multi-file Upload**: Upload multiple PDF files at once
- **Directory Loading**: Load all PDFs from a directory
- **Real-time Q&A**: Ask questions and get answers based on document content
- **Document Search**: Search for specific information across documents
- **Persistent Storage**: Vector store is saved and loaded automatically
- **CORS Support**: Can be used from web frontends
- **Async Processing**: Non-blocking file processing
- **Arabic Support**: Supports Arabic text in PDFs and queries

## Swagger Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

## Health Check

Visit `http://localhost:8000/status` to check system status and readiness.