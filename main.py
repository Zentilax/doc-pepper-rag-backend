from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import faiss
import numpy as np
import pickle
import os
from pathlib import Path
import PyPDF2
import io
import uvicorn
from openai import OpenAI

app = FastAPI(title="Knowledge Base Service")

# Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your GitHub Pages URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
VOLUME_PATH = os.getenv("VOLUME_PATH", "./data")
FAISS_INDEX_PATH = os.path.join(VOLUME_PATH, "faiss_index.bin")
METADATA_PATH = os.path.join(VOLUME_PATH, "metadata.pkl")
DOCS_PATH = os.path.join(VOLUME_PATH, "documents")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# Ensure directories exist
Path(VOLUME_PATH).mkdir(parents=True, exist_ok=True)
Path(DOCS_PATH).mkdir(parents=True, exist_ok=True)

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class QueryResponse(BaseModel):
    results: List[dict]
    count: int

def get_embedding(text: str) -> List[float]:
    """Get embedding from OpenAI API"""
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for multiple texts from OpenAI API"""
    response = client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL
    )
    return [item.embedding for item in response.data]

def load_or_create_index():
    """Load existing FAISS index or create a new one"""
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, 'rb') as f:
            metadata = pickle.load(f)
    else:
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        metadata = []
    return index, metadata

def save_index(index, metadata):
    """Save FAISS index and metadata"""
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)

def extract_text_from_pdf(pdf_bytes):
    """Extract text from PDF bytes"""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

@app.get("/")
async def root():
    return {
        "message": "Knowledge Base Service",
        "embedding_model": EMBEDDING_MODEL,
        "endpoints": {
            "upload": "/upload - POST PDF files",
            "query": "/query - POST questions",
            "documents": "/documents - GET list of documents",
            "delete": "/documents/{doc_id} - DELETE a document"
        }
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and embed a PDF document"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Read PDF content
        pdf_bytes = await file.read()
        text = extract_text_from_pdf(pdf_bytes)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found in PDF")
        
        # Chunk the text
        chunks = chunk_text(text)
        
        # Generate embeddings using OpenAI
        embeddings = get_embeddings_batch(chunks)
        
        # Load existing index and metadata
        index, metadata = load_or_create_index()
        
        # Generate document ID
        doc_id = file.filename + "_" + str(len([m for m in metadata if m['filename'] == file.filename]))
        
        # Save PDF file
        pdf_path = os.path.join(DOCS_PATH, doc_id + ".pdf")
        with open(pdf_path, 'wb') as f:
            f.write(pdf_bytes)
        
        # Add to index
        start_idx = index.ntotal
        index.add(np.array(embeddings).astype('float32'))
        
        # Add metadata
        for i, chunk in enumerate(chunks):
            metadata.append({
                'doc_id': doc_id,
                'filename': file.filename,
                'chunk_id': i,
                'text': chunk,
                'index_id': start_idx + i
            })
        
        # Save index and metadata
        save_index(index, metadata)
        
        return {
            "message": "Document uploaded and embedded successfully",
            "doc_id": doc_id,
            "filename": file.filename,
            "chunks": len(chunks),
            "total_embeddings": index.ntotal
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """Query the knowledge base with a question"""
    try:
        # Load index and metadata
        index, metadata = load_or_create_index()
        
        if index.ntotal == 0:
            return QueryResponse(results=[], count=0)
        
        # Embed the question using OpenAI
        question_embedding = get_embedding(request.question)
        
        # Search in FAISS
        k = min(request.top_k, index.ntotal)
        distances, indices = index.search(
            np.array([question_embedding]).astype('float32'), k
        )
        
        # Retrieve metadata for results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(metadata):
                meta = metadata[idx]
                results.append({
                    'doc_id': meta['doc_id'],
                    'filename': meta['filename'],
                    'chunk_id': meta['chunk_id'],
                    'text': meta['text'],
                    'distance': float(distances[0][i]),
                    'similarity': float(1 / (1 + distances[0][i]))  # Convert distance to similarity
                })
        
        return QueryResponse(results=results, count=len(results))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying knowledge base: {str(e)}")

@app.get("/documents")
async def list_documents():
    """List all documents in the knowledge base"""
    try:
        _, metadata = load_or_create_index()
        
        # Group by document
        docs = {}
        for meta in metadata:
            doc_id = meta['doc_id']
            if doc_id not in docs:
                docs[doc_id] = {
                    'doc_id': doc_id,
                    'filename': meta['filename'],
                    'chunks': 0
                }
            docs[doc_id]['chunks'] += 1
        
        return {
            "total_documents": len(docs),
            "total_chunks": len(metadata),
            "documents": list(docs.values())
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and its embeddings"""
    try:
        # Load index and metadata
        index, metadata = load_or_create_index()
        
        # Find indices to remove
        indices_to_remove = [i for i, meta in enumerate(metadata) if meta['doc_id'] == doc_id]
        
        if not indices_to_remove:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Remove from metadata
        new_metadata = [meta for meta in metadata if meta['doc_id'] != doc_id]
        
        # Rebuild FAISS index without the removed embeddings
        new_index = faiss.IndexFlatL2(EMBEDDING_DIM)
        
        if new_metadata:
            # Get embeddings for remaining chunks
            remaining_texts = [meta['text'] for meta in new_metadata]
            remaining_embeddings = get_embeddings_batch(remaining_texts)
            new_index.add(np.array(remaining_embeddings).astype('float32'))
            
            # Update index_id in metadata
            for i, meta in enumerate(new_metadata):
                meta['index_id'] = i
        
        # Delete PDF file
        pdf_path = os.path.join(DOCS_PATH, doc_id + ".pdf")
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        
        # Save new index and metadata
        save_index(new_index, new_metadata)
        
        return {
            "message": "Document deleted successfully",
            "doc_id": doc_id,
            "chunks_removed": len(indices_to_remove),
            "remaining_chunks": len(new_metadata)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    index, metadata = load_or_create_index()
    return {
        "status": "healthy",
        "embedding_model": EMBEDDING_MODEL,
        "total_embeddings": index.ntotal,
        "total_documents": len(set(meta['doc_id'] for meta in metadata))
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))