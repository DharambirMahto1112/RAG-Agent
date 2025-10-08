"""
Unit tests for Decision Node.
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.nodes.decision_node import DecisionNode

class TestDecisionNode:
    """Test cases for DecisionNode."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.decision_node = DecisionNode()
    
    def test_classify_weather_query(self):
        """Test weather query classification."""
        weather_queries = [
            "What's the weather in London?",
            "Temperature in Paris",
            "Is it raining in New York?",
            "Weather forecast for tomorrow",
            "How hot is it in Tokyo?"
        ]
        
        for query in weather_queries:
            result = self.decision_node.classify_query(query)
            assert result == "weather", f"Failed for query: {query}"
    
    def test_classify_document_query(self):
        """Test document query classification."""
        document_queries = [
            "What does the document say about AI?",
            "Explain the content in the PDF",
            "Summarize the information",
            "What is mentioned about machine learning?",
            "Find information about neural networks"
        ]
        
        for query in document_queries:
            result = self.decision_node.classify_query(query)
            assert result == "document", f"Failed for query: {query}"
    
    def test_classify_unknown_query(self):
        """Test unknown query classification."""
        unknown_queries = [
            "Hello",
            "How are you?",
            "What time is it?",
            "Random text without keywords"
        ]
        
        for query in unknown_queries:
            result = self.decision_node.classify_query(query)
            assert result == "unknown", f"Failed for query: {query}"
    
    def test_process_weather_state(self):
        """Test processing weather query state."""
        state = {
            "query": "What's the weather in London?",
            "messages": []
        }
        
        result = self.decision_node.process(state)
        
        # Assertions
        assert result["query_classification"] == "weather"
        assert result["routing_decision"] == "weather"
        assert "reasoning" in result
    
    def test_process_document_state(self):
        """Test processing document query state."""
        state = {
            "query": "What does the document say about AI?",
            "messages": []
        }
        
        result = self.decision_node.process(state)
        
        # Assertions
        assert result["query_classification"] == "document"
        assert result["routing_decision"] == "document"
        assert "reasoning" in result
    
    def test_should_continue_weather(self):
        """Test routing for weather queries."""
        state = {"query_classification": "weather"}
        result = self.decision_node.should_continue(state)
        assert result == "weather"
    
    def test_should_continue_document(self):
        """Test routing for document queries."""
        state = {"query_classification": "document"}
        result = self.decision_node.should_continue(state)
        assert result == "rag"
    
    def test_should_continue_unknown(self):
        """Test routing for unknown queries."""
        state = {"query_classification": "unknown", "has_documents": False}
        result = self.decision_node.should_continue(state)
        assert result == "fallback"
