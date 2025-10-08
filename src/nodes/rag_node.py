"""
RAG node for handling document-related queries.
"""
from typing import Dict, Any
from ..services.rag_service import RAGService

class RAGNode:
    """Node for processing document-related queries using RAG."""
    
    def __init__(self):
        self.rag_service = RAGService()
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process document-related query using RAG.
        
        Args:
            state: Current state dictionary
            
        Returns:
            Updated state with RAG response
        """
        query = state.get("query", "")
        
        try:
            # Query documents using RAG
            rag_result = self.rag_service.query_documents(query, limit=3)
            
            # Format response
            if rag_result["confidence"] > 0.3:  # Threshold for confidence
                response = self._format_rag_response(rag_result)
            else:
                response = "I couldn't find relevant information in the documents to answer your question. Please try rephrasing or ask about a different topic."
            
            # Update state
            state["rag_result"] = rag_result
            state["response"] = response
            state["response_type"] = "document"
            state["confidence"] = rag_result["confidence"]
            state["sources"] = rag_result["sources"]
            
            return state
            
        except Exception as e:
            error_response = f"❌ Error processing document query: {str(e)}"
            state["response"] = error_response
            state["response_type"] = "error"
            state["error"] = str(e)
            
            return state
    
    def _format_rag_response(self, rag_result: Dict[str, Any]) -> str:
        """
        Format RAG result into a readable response.
        
        Args:
            rag_result: RAG query result
            
        Returns:
            Formatted response string
        """
        answer = rag_result["answer"]
        sources = rag_result["sources"]
        confidence = rag_result["confidence"]
        
        response = f"""
📚 **Document Answer** (Confidence: {confidence:.1f}%)

{answer}

**Sources:**
"""
        
        for i, source in enumerate(sources, 1):
            response += f"""
{i}. **{source['source']}** (Page {source['page_range']}, Score: {source['score']:.3f})
   {source['text']}
"""
        
        return response.strip()
