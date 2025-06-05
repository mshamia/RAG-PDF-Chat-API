"""
FastAPI RAG PDF Chat System
RAG with LangChain, Ollama, and FAISS Vector Store
"""

import os
import warnings
from typing import List, Optional
from contextlib import asynccontextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor
import tiktoken
import faiss

# FastAPI imports
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

# Pydantic models for API
class QuestionRequest(BaseModel):
    question: str
    search_type: Optional[str] = "mmr"
    num_results: Optional[int] = 3

class QuestionResponse(BaseModel):
    question: str
    answer: str
    status: str

class DocumentUploadResponse(BaseModel):
    message: str
    files_processed: int
    chunks_created: int
    status: str

class SearchRequest(BaseModel):
    query: str
    search_type: Optional[str] = "similarity"
    num_results: Optional[int] = 4

class SearchResponse(BaseModel):
    query: str
    results: List[dict]
    status: str

class StatusResponse(BaseModel):
    system_status: str
    documents_loaded: int
    vector_store_ready: bool

class RAGPDFChat:
    def __init__(self, 
                 embedding_model='nomic-embed-text',
                 llm_model='qwen2.5:3b',
                 ollama_base_url='http://localhost:11434',
                 chunk_size=1000,
                 chunk_overlap=100):
        """Initialize the RAG PDF Chat system"""
        # Setup environment
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        warnings.filterwarnings("ignore")
        load_dotenv()
        
        # Initialize models
        self.embeddings = OllamaEmbeddings(
            model=embedding_model, 
            base_url=ollama_base_url
        )
        self.llm = ChatOllama(
            model=llm_model, 
            base_url=ollama_base_url
        )
        
        # Text splitter configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        
        # Vector store
        self.vector_store = None
        self.retriever = None
        self.rag_chain = None
        self.documents_count = 0
        
        # Setup prompt template
        self._setup_prompt()
        
    def _setup_prompt(self):
        """Setup the RAG prompt template"""
        prompt_text = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Answer in bullet points. Make sure your answer is relevant to the question and it is answered from the context only.
        Question: {question} 
        Context: {context} 
        Answer:
        """
        self.prompt = ChatPromptTemplate.from_template(prompt_text)
    
    def load_pdf_from_bytes(self, pdf_bytes: bytes, filename: str):
        """Load a PDF from bytes"""
        # Save temporarily
        temp_path = f"/tmp/{filename}"
        with open(temp_path, "wb") as f:
            f.write(pdf_bytes)
        
        loader = PyMuPDFLoader(temp_path)
        docs = loader.load()
        
        # Clean up
        os.remove(temp_path)
        return docs
    
    def load_pdf(self, pdf_path):
        """Load a single PDF file"""
        loader = PyMuPDFLoader(pdf_path)
        return loader.load()
    
    def load_pdfs_from_directory(self, directory_path):
        """Load all PDF files from a directory recursively"""
        pdfs = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.pdf'):
                    pdfs.append(os.path.join(root, file))
        
        docs = []
        for pdf in pdfs:
            try:
                loader = PyMuPDFLoader(pdf)
                pages = loader.load()
                docs.extend(pages)
            except Exception as e:
                print(f"Error loading {pdf}: {e}")
        
        return docs
    
    def process_documents(self, docs):
        """Process documents by chunking them"""
        chunks = self.text_splitter.split_documents(docs)
        return chunks
    
    def create_vector_store(self, chunks):
        """Create FAISS vector store from document chunks"""
        # Get embedding dimension
        single_vector = self.embeddings.embed_query("test text")
        embedding_dim = len(single_vector)
        
        # Create FAISS index
        index = faiss.IndexFlatL2(embedding_dim)
        
        # Create vector store
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        
        # Add documents to vector store
        ids = self.vector_store.add_documents(documents=chunks)
        self.documents_count = len(ids)
        
        # Setup retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr", 
            search_kwargs={
                'k': 3, 
                'fetch_k': 100,
                'lambda_mult': 1
            }
        )
        
        # Setup RAG chain
        self._setup_rag_chain()
        
        return len(ids)
    
    def _setup_rag_chain(self):
        """Setup the RAG chain for question answering"""
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])
        
        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def search_documents(self, query, search_type='similarity', num_results=4):
        """Search documents using the vector store"""
        if not self.vector_store:
            raise ValueError("No vector store available")
        
        docs = self.vector_store.search(
            query=query, 
            search_type=search_type,
            k=num_results
        )
        return docs
    
    def ask_question(self, question):
        """Ask a question using the RAG system"""
        if not self.rag_chain:
            raise ValueError("RAG system not initialized")
        
        answer = self.rag_chain.invoke(question)
        return answer
    
    def is_ready(self):
        """Check if the system is ready"""
        return self.rag_chain is not None and self.vector_store is not None

# Global RAG instance
rag_chat = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global rag_chat
    
    # Startup
    print("Starting RAG PDF Chat API...")
    rag_chat = RAGPDFChat()
    
    # Try to load existing vector store
    try:
        if os.path.exists("default_vector_store"):
            rag_chat.load_vector_store("default_vector_store")
            print("Loaded existing vector store")
    except Exception as e:
        print(f"No existing vector store found: {e}")
    
    yield
    
    # Shutdown
    print("Shutting down RAG PDF Chat API...")

# Create FastAPI app
app = FastAPI(
    title="RAG PDF Chat API",
    description="API for chatting with PDF documents using RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=4)

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "RAG PDF Chat API",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get system status"""
    global rag_chat
    
    return StatusResponse(
        system_status="ready" if rag_chat and rag_chat.is_ready() else "not_ready",
        documents_loaded=rag_chat.documents_count if rag_chat else 0,
        vector_store_ready=rag_chat.is_ready() if rag_chat else False
    )

@app.post("/upload-pdfs", response_model=DocumentUploadResponse)
async def upload_pdfs(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """Upload and process PDF files"""
    global rag_chat
    
    if not rag_chat:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    pdf_files = [f for f in files if f.content_type == "application/pdf"]
    
    if not pdf_files:
        raise HTTPException(status_code=400, detail="No PDF files provided")
    
    try:
        # Process files in background
        def process_files():
            docs = []
            for file in pdf_files:
                content = file.file.read()
                file_docs = rag_chat.load_pdf_from_bytes(content, file.filename)
                docs.extend(file_docs)
            
            chunks = rag_chat.process_documents(docs)
            chunks_created = rag_chat.create_vector_store(chunks)
            
            # Save vector store
            try:
                rag_chat.save_vector_store("default_vector_store")
            except Exception as e:
                print(f"Error saving vector store: {e}")
            
            return chunks_created
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        chunks_created = await loop.run_in_executor(executor, process_files)
        
        return DocumentUploadResponse(
            message="Files processed successfully",
            files_processed=len(pdf_files),
            chunks_created=chunks_created,
            status="success"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@app.post("/load-directory")
async def load_directory(directory_path: str):
    """Load all PDFs from a directory"""
    global rag_chat
    
    if not rag_chat:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    if not os.path.exists(directory_path):
        raise HTTPException(status_code=404, detail="Directory not found")
    
    try:
        def process_directory():
            docs = rag_chat.load_pdfs_from_directory(directory_path)
            chunks = rag_chat.process_documents(docs)
            chunks_created = rag_chat.create_vector_store(chunks)
            
            # Save vector store
            try:
                rag_chat.save_vector_store("default_vector_store")
            except Exception as e:
                print(f"Error saving vector store: {e}")
            
            return len(docs), chunks_created
        
        loop = asyncio.get_event_loop()
        docs_count, chunks_created = await loop.run_in_executor(executor, process_directory)
        
        return {
            "message": "Directory processed successfully",
            "documents_loaded": docs_count,
            "chunks_created": chunks_created,
            "status": "success"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing directory: {str(e)}")

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question to the RAG system"""
    global rag_chat
    
    if not rag_chat or not rag_chat.is_ready():
        raise HTTPException(status_code=400, detail="RAG system not ready. Please upload documents first.")
    
    try:
        def get_answer():
            return rag_chat.ask_question(request.question)
        
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(executor, get_answer)
        
        return QuestionResponse(
            question=request.question,
            answer=answer,
            status="success"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search documents in the vector store"""
    global rag_chat
    
    if not rag_chat or not rag_chat.vector_store:
        raise HTTPException(status_code=400, detail="Vector store not ready. Please upload documents first.")
    
    try:
        def search():
            docs = rag_chat.search_documents(
                request.query, 
                request.search_type, 
                request.num_results
            )
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } 
                for doc in docs
            ]
        
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(executor, search)
        
        return SearchResponse(
            query=request.query,
            results=results,
            status="success"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")

@app.delete("/reset")
async def reset_system():
    """Reset the RAG system"""
    global rag_chat
    
    try:
        rag_chat = RAGPDFChat()
        return {"message": "System reset successfully", "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting system: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )