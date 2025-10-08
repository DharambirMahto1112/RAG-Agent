"""
Fallback node for handling unrecognized queries.
"""
from typing import Dict, Any

class FallbackNode:
    """Node for handling queries that don't fit weather or document categories."""
    
    def __init__(self):
        self.suggestions = [
            "Try asking about weather in a specific city (e.g., 'What's the weather in London?')",
            "Ask questions about documents you've uploaded (e.g., 'What does the document say about...?')",
            "Be more specific about what you're looking for"
        ]
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process unrecognized query with helpful suggestions.
        
        Args:
            state: Current state dictionary
            
        Returns:
            Updated state with fallback response
        """
        query = state.get("query", "")
        
        # Generate helpful response
        response = f"""
🤔 I'm not sure how to help with that query.

**Your query:** "{query}"

**I can help you with:**
• Weather information for any city
• Questions about documents you've uploaded

**Try these instead:**
"""
        
        for suggestion in self.suggestions:
            response += f"\n• {suggestion}"
        
        response += "\n\nPlease rephrase your question and try again!"
        
        # Update state
        state["response"] = response
        state["response_type"] = "fallback"
        state["suggestions"] = self.suggestions
        
        return state
