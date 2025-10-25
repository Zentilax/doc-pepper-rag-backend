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
import pdfplumber
from sentence_transformers import SentenceTransformer
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
    print("ERROR: OPENAI_API_KEY environment variable is not set!")
    print("Please set it in Railway dashboard under Variables tab")
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Initialize OpenAI client
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    print("✅ OpenAI client initialized successfully")
except Exception as e:
    print(f"ERROR: Failed to initialize OpenAI client: {e}")
    raise

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
    try:
        response = client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ ERROR in get_embedding: {e}")
        print(f"Text length: {len(text)}")
        raise

def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for multiple texts from OpenAI API"""
    try:
        print(f"📊 Getting embeddings for {len(texts)} chunks...")
        response = client.embeddings.create(
            input=texts,
            model=EMBEDDING_MODEL
        )
        print(f"✅ Successfully got {len(response.data)} embeddings")
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"❌ ERROR in get_embeddings_batch: {e}")
        print(f"Number of texts: {len(texts)}")
        print(f"Model: {EMBEDDING_MODEL}")
        raise

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
    """Extract text from PDF bytes using multiple methods"""
    text = ""
    pages_with_text = 0
    
    # Try pdfplumber first (better for complex PDFs)
    try:
        print(f"📄 Trying pdfplumber...")
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            total_pages = len(pdf.pages)
            print(f"📄 PDF has {total_pages} pages")
            
            for page_num, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text += page_text + "\n"
                        pages_with_text += 1
                        print(f"  ✅ Page {page_num + 1}: {len(page_text)} characters")
                    else:
                        print(f"  ⚠️ Page {page_num + 1}: No text (might be image/scanned)")
                except Exception as e:
                    print(f"  ❌ Page {page_num + 1}: Error: {e}")
        
        if text.strip():
            print(f"✅ pdfplumber: Extracted text from {pages_with_text}/{total_pages} pages")
            print(f"📝 Total text length: {len(text)} characters")
            return text
    except Exception as e:
        print(f"⚠️ pdfplumber failed: {e}")
    
    # Fallback to PyPDF2
    print(f"📄 Trying PyPDF2 as fallback...")
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(pdf_reader.pages)
        print(f"📄 PDF has {total_pages} pages")
        
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text += page_text + "\n"
                    pages_with_text += 1
                    print(f"  ✅ Page {page_num + 1}: {len(page_text)} characters")
                else:
                    print(f"  ⚠️ Page {page_num + 1}: No text (might be image/scanned)")
            except Exception as e:
                print(f"  ❌ Page {page_num + 1}: Error: {e}")
        
        print(f"✅ PyPDF2: Extracted text from {pages_with_text}/{total_pages} pages")
        print(f"📝 Total text length: {len(text)} characters")
    except Exception as e:
        print(f"❌ PyPDF2 also failed: {e}")
    
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
    print(f"\n{'='*50}")
    print(f"📤 UPLOAD REQUEST: {file.filename}")
    print(f"{'='*50}")
    
    if not file.filename.endswith('.pdf'):
        print(f"❌ ERROR: Not a PDF file")
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Read PDF content
        print(f"📖 Reading PDF content...")
        pdf_bytes = await file.read()
        print(f"✅ PDF read successfully: {len(pdf_bytes)} bytes")
        
        try:
            print(f"📝 Extracting text from PDF...")
            text = extract_text_from_pdf(pdf_bytes)
            print(f"✅ Text extracted: {len(text)} characters")
            
            # Check if we have meaningful text (more than just whitespace/short fragments)
            if len(text.strip()) < 100:
                print(f"❌ ERROR: Insufficient text in PDF (only {len(text.strip())} characters)")
                print(f"ℹ️ This PDF might be:")
                print(f"   - Scanned images without OCR")
                print(f"   - Protected/encrypted")
                print(f"   - Mostly images/graphics")
                raise HTTPException(
                    status_code=400, 
                    detail=f"Insufficient text found in PDF. Only {len(text.strip())} characters extracted. The PDF might contain scanned images. Please use a PDF with extractable text or apply OCR first."
                )
        except HTTPException:
            raise
        except Exception as e:
            print(f"❌ ERROR extracting text: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {str(e)}")
        
        if not text.strip():
            print(f"❌ ERROR: No text found in PDF after extraction")
            raise HTTPException(
                status_code=400, 
                detail="No text found in PDF. The document might contain only images or scanned pages without OCR."
            )
        
        # Chunk the text
        print(f"✂️ Chunking text...")
        chunks = chunk_text(text)
        print(f"✅ Created {len(chunks)} chunks")
        
        if not chunks:
            print(f"❌ ERROR: Failed to create chunks")
            raise HTTPException(status_code=400, detail="Failed to create text chunks from PDF")
        
        # Generate embeddings using OpenAI
        try:
            print(f"🤖 Calling OpenAI API for embeddings...")
            embeddings = get_embeddings_batch(chunks)
            print(f"✅ Embeddings received: {len(embeddings)} vectors")
        except Exception as e:
            print(f"❌ ERROR calling OpenAI API: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500, 
                detail=f"OpenAI API error: {str(e)}. Check your OPENAI_API_KEY environment variable."
            )
        
        # Load existing index and metadata
        print(f"💾 Loading existing index...")
        index, metadata = load_or_create_index()
        print(f"✅ Index loaded: {index.ntotal} existing embeddings")
        
        # Generate document ID
        doc_id = file.filename + "_" + str(len([m for m in metadata if m['filename'] == file.filename]))
        print(f"🆔 Document ID: {doc_id}")
        
        # Save PDF file
        print(f"💾 Saving PDF file...")
        pdf_path = os.path.join(DOCS_PATH, doc_id + ".pdf")
        with open(pdf_path, 'wb') as f:
            f.write(pdf_bytes)
        print(f"✅ PDF saved to: {pdf_path}")
        
        # Add to index
        print(f"📊 Adding embeddings to FAISS index...")
        start_idx = index.ntotal
        index.add(np.array(embeddings).astype('float32'))
        print(f"✅ Embeddings added. Total now: {index.ntotal}")
        
        # Add metadata
        print(f"📝 Adding metadata...")
        for i, chunk in enumerate(chunks):
            metadata.append({
                'doc_id': doc_id,
                'filename': file.filename,
                'chunk_id': i,
                'text': chunk,
                'index_id': start_idx + i
            })
        print(f"✅ Metadata added: {len(chunks)} entries")
        
        # Save index and metadata
        print(f"💾 Saving index and metadata to disk...")
        save_index(index, metadata)
        print(f"✅ Index and metadata saved")
        
        print(f"{'='*50}")
        print(f"✅ SUCCESS: Document uploaded")
        print(f"{'='*50}\n")
        
        return {
            "message": "Document uploaded and embedded successfully",
            "doc_id": doc_id,
            "filename": file.filename,
            "chunks": len(chunks),
            "total_embeddings": index.ntotal
        }
    
    except HTTPException:
        raise
    except Exception as e:
        # Log the full error for debugging
        print(f"\n{'='*50}")
        print(f"❌ UNEXPECTED ERROR")
        print(f"{'='*50}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*50}\n")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """Query the knowledge base with a question"""
    print(f"\n{'='*50}")
    print(f"🔍 QUERY REQUEST: {request.question[:100]}...")
    print(f"{'='*50}")
    
    try:
        # Load index and metadata
        print(f"💾 Loading index...")
        index, metadata = load_or_create_index()
        print(f"✅ Index loaded: {index.ntotal} embeddings, {len(metadata)} metadata entries")
        
        if index.ntotal == 0:
            print(f"⚠️ No embeddings in index")
            return QueryResponse(results=[], count=0)
        
        # Embed the question using OpenAI
        try:
            print(f"🤖 Getting embedding for question...")
            question_embedding = get_embedding(request.question)
            print(f"✅ Question embedded")
        except Exception as e:
            print(f"❌ ERROR getting question embedding: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to embed question: {str(e)}")
        
        # Search in FAISS
        k = min(request.top_k, index.ntotal)
        print(f"🔎 Searching for top {k} results...")
        distances, indices = index.search(
            np.array([question_embedding]).astype('float32'), k
        )
        print(f"✅ Search complete")
        
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
                    'similarity': float(1 / (1 + distances[0][i]))
                })
        
        print(f"✅ Returning {len(results)} results")
        print(f"{'='*50}\n")
        
        return QueryResponse(results=results, count=len(results))
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n{'='*50}")
        print(f"❌ QUERY ERROR")
        print(f"{'='*50}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*50}\n")
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
    print(f"\n{'='*50}")
    print(f"🗑️ DELETE REQUEST: {doc_id}")
    print(f"{'='*50}")
    
    try:
        # Load index and metadata
        print(f"💾 Loading index...")
        index, metadata = load_or_create_index()
        print(f"✅ Index loaded: {index.ntotal} embeddings")
        
        # Find indices to remove
        indices_to_remove = [i for i, meta in enumerate(metadata) if meta['doc_id'] == doc_id]
        
        if not indices_to_remove:
            print(f"❌ Document not found: {doc_id}")
            raise HTTPException(status_code=404, detail="Document not found")
        
        print(f"📊 Found {len(indices_to_remove)} chunks to remove")
        
        # Remove from metadata
        new_metadata = [meta for meta in metadata if meta['doc_id'] != doc_id]
        print(f"✅ New metadata size: {len(new_metadata)} entries")
        
        # Rebuild FAISS index without the removed embeddings
        print(f"🔨 Rebuilding FAISS index...")
        new_index = faiss.IndexFlatL2(EMBEDDING_DIM)
        
        if new_metadata:
            # Get embeddings for remaining chunks
            remaining_texts = [meta['text'] for meta in new_metadata]
            print(f"🤖 Re-embedding {len(remaining_texts)} remaining chunks...")
            remaining_embeddings = get_embeddings_batch(remaining_texts)
            new_index.add(np.array(remaining_embeddings).astype('float32'))
            
            # Update index_id in metadata
            for i, meta in enumerate(new_metadata):
                meta['index_id'] = i
            print(f"✅ Index rebuilt with {new_index.ntotal} embeddings")
        else:
            print(f"ℹ️ No documents remaining, index is empty")
        
        # Delete PDF file
        pdf_path = os.path.join(DOCS_PATH, doc_id + ".pdf")
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            print(f"✅ PDF file deleted: {pdf_path}")
        else:
            print(f"⚠️ PDF file not found: {pdf_path}")
        
        # Save new index and metadata
        print(f"💾 Saving updated index and metadata...")
        save_index(new_index, new_metadata)
        print(f"✅ Index and metadata saved")
        
        print(f"{'='*50}")
        print(f"✅ SUCCESS: Document deleted")
        print(f"{'='*50}\n")
        
        return {
            "message": "Document deleted successfully",
            "doc_id": doc_id,
            "chunks_removed": len(indices_to_remove),
            "remaining_chunks": len(new_metadata)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n{'='*50}")
        print(f"❌ DELETE ERROR")
        print(f"{'='*50}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*50}\n")
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