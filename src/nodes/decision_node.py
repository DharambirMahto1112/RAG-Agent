"""
Decision node for routing queries to appropriate handlers.
"""
import re
from typing import Dict, Any, Literal
from langchain.schema import BaseMessage

class DecisionNode:
    """Node for deciding whether a query is weather-related or document-related."""
    
    def __init__(self):
        # Weather-related keywords
        self.weather_keywords = [
            'weather', 'temperature', 'forecast', 'rain', 'snow', 'sunny', 'cloudy',
            'humidity', 'wind', 'storm', 'climate', 'temperature', 'degrees',
            'celsius', 'fahrenheit', 'hot', 'cold', 'warm', 'cool'
        ]
        
        # Document-related keywords
        self.document_keywords = [
            'document', 'pdf', 'file', 'text', 'content', 'information',
            'what does', 'what is', 'define', 'definition', 'explain', 'describe',
            'tell me about', 'summarize', 'summary', 'overview', 'find', 'purpose',
            'objective', 'objectives', 'core tasks', 'according to', 'in the document',
            'based on the document'
        ]
    
    def classify_query(self, query: str) -> Literal["weather", "document", "unknown"]:
        """
        Classify a query as weather-related, document-related, or unknown.
        
        Args:
            query: User query string
            
        Returns:
            Classification result
        """
        query_lower = query.lower()
        
        # Check for weather keywords
        weather_score = sum(1 for keyword in self.weather_keywords if keyword in query_lower)
        
        # Check for document keywords
        document_score = sum(1 for keyword in self.document_keywords if keyword in query_lower)
        
        # Check for location indicators (common in weather queries)
        location_patterns = [
            r'\bweather\s+(in|at|for)\s+\w+',  # "weather in Paris"
            r'\btemperature\s+(in|at|for)\s+\w+',  # "temperature in London"
            r'\b\w+\s+(weather|temperature)',  # "London weather"
            r'\b(in|at|for)\s+[A-Z][a-z]+',  # "in London", "at Paris" (capitalized cities)
        ]
        
        location_score = sum(1 for pattern in location_patterns if re.search(pattern, query_lower))
        
        # Adjust scores: only boost weather if explicit weather terms exist to avoid false positives
        if weather_score > 0:
            weather_score += location_score * 2
        
        # Debug logging
        print(f"DEBUG - Query: {query}")
        print(f"DEBUG - Weather score: {weather_score}, Document score: {document_score}, Location score: {location_score}")
        
        # Decision logic
        if weather_score > document_score and weather_score > 0:
            print("DEBUG - Routing to WEATHER")
            return "weather"
        elif document_score >= weather_score and document_score > 0:
            print("DEBUG - Routing to DOCUMENT")
            return "document"
        else:
            # If not clearly weather, prefer document for general knowledge queries
            # This helps route questions like "what is ..." to RAG when docs exist
            print("DEBUG - Routing to DOCUMENT (default)")
            return "document"
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the current state and determine the next action.
        
        Args:
            state: Current state dictionary
            
        Returns:
            Updated state with routing decision
        """
        query = state.get("query", "")
        classification = self.classify_query(query)
        
        # Update state with classification
        state["query_classification"] = classification
        state["routing_decision"] = classification
        
        # Add reasoning for debugging
        state["reasoning"] = f"Query classified as '{classification}' based on content analysis"
        
        return state
    
    def should_continue(self, state: Dict[str, Any]) -> str:
        """
        Determine which node to execute next based on classification.
        
        Args:
            state: Current state dictionary
            
        Returns:
            Next node name
        """
        classification = state.get("query_classification", "unknown")
        has_documents = state.get("has_documents", False)
        
        print(f"DEBUG - should_continue: classification={classification}, has_documents={has_documents}")
        
        # Map classification to graph node names defined in RAGAgent
        if classification == "weather":
            print("DEBUG - Routing to WEATHER node")
            return "weather"
        elif classification == "document":
            print("DEBUG - Routing to RAG node")
            return "rag"
        else:
            # Unknown: prefer RAG only if we have documents indexed; otherwise fallback
            result = "rag" if has_documents else "fallback"
            print(f"DEBUG - Routing to {result.upper()} node (unknown classification)")
            return result
