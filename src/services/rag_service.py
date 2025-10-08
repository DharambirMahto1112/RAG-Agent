"""
RAG (Retrieval-Augmented Generation) service for document processing and Q&A.
"""
import os
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
import PyPDF2
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from openai import OpenAI
from .config import get_settings

class RAGService:
    """Service for document processing, embedding generation, and retrieval."""
    
    def __init__(self):
        self.settings = get_settings()
        # Defer embedding model load to first use to speed up app startup
        self.embedding_model = None
        # Use Groq OpenAI-compatible API
        self.openai_client = OpenAI(
            api_key=self.settings.groq_api_key,
            base_url=self.settings.groq_base_url,
        )
        self.groq_model = self.settings.groq_model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key
        )
        
        self.collection_name = "rag_documents"
        self._ensure_collection_exists()

    def _ensure_embeddings_model(self):
        """Lazily initialize the embedding model on first use."""
        if self.embedding_model is None:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

    def has_documents(self) -> bool:
        """Return True if the vector collection has at least one point."""
        try:
            count_result = self.qdrant_client.count(self.collection_name, exact=True)
            # qdrant-client returns CountReply with .count
            return getattr(count_result, "count", 0) > 0
        except Exception:
            return False
    
    def _ensure_collection_exists(self):
        """Ensure the Qdrant collection exists."""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=384,  # all-MiniLM-L6-v2 embedding size
                        distance=Distance.COSINE
                    )
                )
                print(f"Created collection: {self.collection_name}")
        except Exception as e:
            print(f"Error ensuring collection exists: {e}")
    
    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Process a PDF file and extract text chunks with improved text extraction.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of text chunks with metadata
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                all_text = ""
                page_texts = []
                
                # Extract text from each page
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():  # Only add non-empty pages
                            all_text += page_text + "\n"
                            page_texts.append({
                                "page_num": page_num + 1,
                                "text": page_text.strip()
                            })
                    except Exception as e:
                        print(f"Warning: Could not extract text from page {page_num + 1}: {e}")
                        continue
                
                if not all_text.strip():
                    print(f"Warning: No text extracted from PDF {pdf_path}")
                    return []
                
                # Clean and preprocess text
                all_text = self._clean_text(all_text)
                
                # Split text into chunks
                chunks = self.text_splitter.split_text(all_text)
                
                # Create chunk metadata with better page tracking
                chunks_with_metadata = []
                for i, chunk in enumerate(chunks):
                    # Find which page this chunk likely came from
                    page_range = self._find_chunk_page_range(chunk, page_texts)
                    
                    chunk_data = {
                        "id": str(uuid.uuid4()),
                        "text": chunk.strip(),
                        "source": os.path.basename(pdf_path),
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "page_range": page_range,
                        "total_pages": len(pdf_reader.pages),
                        "file_size": os.path.getsize(pdf_path)
                    }
                    chunks_with_metadata.append(chunk_data)
                
                print(f"Successfully processed PDF: {len(chunks)} chunks from {len(pdf_reader.pages)} pages")
                return chunks_with_metadata
                
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess extracted text."""
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers (common patterns)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
        
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def _find_chunk_page_range(self, chunk: str, page_texts: List[Dict[str, Any]]) -> str:
        """Find which page(s) a chunk likely came from."""
        chunk_words = set(chunk.lower().split())
        
        best_match_score = 0
        best_page = 1
        
        for page_info in page_texts:
            page_words = set(page_info["text"].lower().split())
            # Calculate overlap
            overlap = len(chunk_words.intersection(page_words))
            if overlap > best_match_score:
                best_match_score = overlap
                best_page = page_info["page_num"]
        
        # If we found a good match, return single page
        if best_match_score > 3:  # Threshold for good match
            return str(best_page)
        else:
            # Return range if uncertain
            return f"{max(1, best_page-1)}-{best_page+1}"
    
    def _estimate_page_range(self, chunk_index: int, total_chunks: int, total_pages: int) -> str:
        """Estimate page range for a chunk."""
        pages_per_chunk = total_pages / total_chunks
        start_page = int(chunk_index * pages_per_chunk) + 1
        end_page = int((chunk_index + 1) * pages_per_chunk)
        return f"{start_page}-{end_page}"
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for text chunks.
        
        Args:
            chunks: List of text chunks with metadata
            
        Returns:
            List of chunks with embeddings
        """
        try:
            # Ensure embeddings model is ready
            self._ensure_embeddings_model()
            texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedding_model.embed_documents(texts)
            
            # Add embeddings to chunks
            for i, chunk in enumerate(chunks):
                chunk["embedding"] = embeddings[i]
            
            return chunks
            
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return chunks
    
    def store_in_qdrant(self, chunks_with_embeddings: List[Dict[str, Any]]) -> bool:
        """
        Store chunks with embeddings in Qdrant vector database.
        
        Args:
            chunks_with_embeddings: List of chunks with embeddings
            
        Returns:
            True if successful, False otherwise
        """
        try:
            points = []
            for chunk in chunks_with_embeddings:
                point = PointStruct(
                    id=chunk["id"],
                    vector=chunk["embedding"],
                    payload={
                        "text": chunk["text"],
                        "source": chunk["source"],
                        "chunk_index": chunk["chunk_index"],
                        "total_chunks": chunk["total_chunks"],
                        "page_range": chunk["page_range"]
                    }
                )
                points.append(point)
            
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            print(f"Stored {len(points)} chunks in Qdrant")
            return True
            
        except Exception as e:
            print(f"Error storing in Qdrant: {e}")
            return False
    
    def search_similar_chunks(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of similar chunks with scores
        """
        try:
            # Ensure embeddings model is ready
            self._ensure_embeddings_model()
            # Generate embedding for query
            query_embedding = self.embedding_model.embed_query(query)
            
            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )
            
            results = []
            for result in search_results:
                chunk_data = {
                    "id": result.id,
                    "text": result.payload["text"],
                    "source": result.payload["source"],
                    "chunk_index": result.payload["chunk_index"],
                    "page_range": result.payload["page_range"],
                    "score": result.score
                }
                results.append(chunk_data)
            
            return results
            
        except Exception as e:
            print(f"Error searching similar chunks: {e}")
            return []
    
    def answer_question(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate an answer using LLM with retrieved context.
        
        Args:
            question: User question
            context_chunks: Retrieved context chunks
            
        Returns:
            Generated answer
        """
        try:
            # Combine context chunks
            context = "\n\n".join([chunk["text"] for chunk in context_chunks])
            
            # Create prompt
            prompt = f"""
            Based on the following context, please answer the question. If the answer is not in the context, say so clearly.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:
            """
            
            # Generate response using LLM
            result = self.openai_client.responses.create(
                input=prompt,
                model=self.groq_model,
            )
            response = getattr(result, "output_text", str(result))
            
            return response.strip()
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"Sorry, I encountered an error while generating the answer: {str(e)}"
    
    def process_document_and_store(self, pdf_path: str) -> bool:
        """
        Complete pipeline: process PDF, generate embeddings, and store in Qdrant.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Processing PDF: {pdf_path}")
            
            # Step 1: Process PDF
            chunks = self.process_pdf(pdf_path)
            if not chunks:
                return False
            
            # Step 2: Generate embeddings
            chunks_with_embeddings = self.generate_embeddings(chunks)
            
            # Step 3: Store in Qdrant
            success = self.store_in_qdrant(chunks_with_embeddings)
            
            if success:
                print(f"Successfully processed and stored {len(chunks)} chunks from {pdf_path}")
            
            return success
            
        except Exception as e:
            print(f"Error in document processing pipeline: {e}")
            return False
    
    def query_documents(self, question: str, limit: int = 3) -> Dict[str, Any]:
        """
        Query documents and return answer with sources.
        
        Args:
            question: User question
            limit: Number of context chunks to retrieve
            
        Returns:
            Dictionary with answer and sources
        """
        try:
            # Search for relevant chunks
            similar_chunks = self.search_similar_chunks(question, limit=limit)
            
            if not similar_chunks:
                return {
                    "answer": "I couldn't find any relevant information in the documents to answer your question.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Generate answer
            answer = self.answer_question(question, similar_chunks)
            
            # Prepare sources
            sources = []
            for chunk in similar_chunks:
                source = {
                    "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                    "source": chunk["source"],
                    "page_range": chunk["page_range"],
                    "score": chunk["score"]
                }
                sources.append(source)
            
            # Calculate confidence based on scores
            avg_score = sum(chunk["score"] for chunk in similar_chunks) / len(similar_chunks)
            confidence = min(avg_score * 100, 100)  # Convert to percentage
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence
            }
            
        except Exception as e:
            print(f"Error querying documents: {e}")
            return {
                "answer": f"Sorry, I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }
