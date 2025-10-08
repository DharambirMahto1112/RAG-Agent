"""
Unit tests for RAG Service.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.services.rag_service import RAGService

class TestRAGService:
    """Test cases for RAGService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('services.rag_service.QdrantClient'):
            self.rag_service = RAGService()
    
    @patch('PyPDF2.PdfReader')
    def test_process_pdf_success(self, mock_pdf_reader):
        """Test successful PDF processing."""
        # Mock PDF reader
        mock_page = Mock()
        mock_page.extract_text.return_value = "Sample text content from PDF"
        mock_reader = Mock()
        mock_reader.pages = [mock_page, mock_page]  # 2 pages
        mock_pdf_reader.return_value = mock_reader
        
        # Test the method
        result = self.rag_service.process_pdf("test.pdf")
        
        # Assertions
        assert len(result) > 0
        assert all("id" in chunk for chunk in result)
        assert all("text" in chunk for chunk in result)
        assert all("source" in chunk for chunk in result)
        assert all(chunk["source"] == "test.pdf" for chunk in result)
    
    @patch('PyPDF2.PdfReader')
    def test_process_pdf_error(self, mock_pdf_reader):
        """Test PDF processing error handling."""
        # Mock PDF reader error
        mock_pdf_reader.side_effect = Exception("PDF Error")
        
        result = self.rag_service.process_pdf("test.pdf")
        
        # Assertions
        assert result == []
    
    @patch('services.rag_service.RAGService.embedding_model')
    def test_generate_embeddings(self, mock_embedding_model):
        """Test embedding generation."""
        # Mock embedding model
        mock_embedding_model.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        chunks = [
            {"id": "1", "text": "Sample text 1"},
            {"id": "2", "text": "Sample text 2"}
        ]
        
        result = self.rag_service.generate_embeddings(chunks)
        
        # Assertions
        assert len(result) == 2
        assert all("embedding" in chunk for chunk in result)
        assert result[0]["embedding"] == [0.1, 0.2, 0.3]
        assert result[1]["embedding"] == [0.4, 0.5, 0.6]
    
    @patch('services.rag_service.RAGService.qdrant_client')
    def test_store_in_qdrant_success(self, mock_client):
        """Test successful storage in Qdrant."""
        chunks_with_embeddings = [
            {
                "id": "1",
                "text": "Sample text",
                "embedding": [0.1, 0.2, 0.3],
                "source": "test.pdf",
                "chunk_index": 0,
                "total_chunks": 1,
                "page_range": "1-1"
            }
        ]
        
        result = self.rag_service.store_in_qdrant(chunks_with_embeddings)
        
        # Assertions
        assert result is True
        mock_client.upsert.assert_called_once()
    
    @patch('services.rag_service.RAGService.qdrant_client')
    def test_store_in_qdrant_error(self, mock_client):
        """Test Qdrant storage error handling."""
        # Mock Qdrant error
        mock_client.upsert.side_effect = Exception("Qdrant Error")
        
        chunks_with_embeddings = [
            {
                "id": "1",
                "text": "Sample text",
                "embedding": [0.1, 0.2, 0.3],
                "source": "test.pdf",
                "chunk_index": 0,
                "total_chunks": 1,
                "page_range": "1-1"
            }
        ]
        
        result = self.rag_service.store_in_qdrant(chunks_with_embeddings)
        
        # Assertions
        assert result is False
    
    @patch('services.rag_service.RAGService.embedding_model')
    @patch('services.rag_service.RAGService.qdrant_client')
    def test_search_similar_chunks(self, mock_client, mock_embedding_model):
        """Test similar chunk search."""
        # Mock embedding model
        mock_embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]
        
        # Mock Qdrant search result
        mock_result = Mock()
        mock_result.id = "1"
        mock_result.score = 0.95
        mock_result.payload = {
            "text": "Sample text",
            "source": "test.pdf",
            "chunk_index": 0,
            "page_range": "1-1"
        }
        mock_client.search.return_value = [mock_result]
        
        result = self.rag_service.search_similar_chunks("test query")
        
        # Assertions
        assert len(result) == 1
        assert result[0]["id"] == "1"
        assert result[0]["score"] == 0.95
        assert result[0]["text"] == "Sample text"
    
    @patch('services.rag_service.RAGService.openai_client')
    def test_answer_question(self, mock_openai):
        """Test answer generation."""
        # Mock Groq response
        mock_resp = Mock()
        mock_resp.output_text = "This is a test answer."
        mock_openai.responses.create.return_value = mock_resp
        
        context_chunks = [
            {"text": "Context text 1"},
            {"text": "Context text 2"}
        ]
        
        result = self.rag_service.answer_question("Test question", context_chunks)
        
        # Assertions
        assert result == "This is a test answer."
        mock_openai.responses.create.assert_called_once()
    
    def test_estimate_page_range(self):
        """Test page range estimation."""
        result = self.rag_service._estimate_page_range(0, 4, 8)
        assert result == "1-2"
        
        result = self.rag_service._estimate_page_range(1, 4, 8)
        assert result == "3-4"
